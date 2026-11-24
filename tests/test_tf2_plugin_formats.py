"""Exercise TF2 plugin output without importing the producer libraries."""

import base64
import json
from pathlib import Path

import pytest

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2

CORPUS = Path(__file__).parent / "data" / "producer-cases" / "tf2-plugin-formats"


@pytest.mark.parametrize(
    ("tag", "step", "weights"),
    [
        ("native_pr", 4, (1.0,) * 6),
        ("weighted_pr", 5, (2.0, 0.5, 1.5, 3.0, 0.25, 1.25)),
    ],
)
def test_classifier_pr(tag: str, step: int, weights: tuple[float, ...]) -> None:
    fs = TensorBoardFS(CORPUS)
    rows = json.loads(fs.read(f"/pr_curves/{tag}/pr_curves/{step:06}.json", 4096, 0))
    positives = (0.95, 0.65, 0.25)
    negatives = (0.85, 0.45, 0.05)
    for row, threshold in zip(rows, (0, 0.25, 0.5, 0.75, 1), strict=True):
        tp = sum(
            w for p, w in zip(positives, weights[::2], strict=True) if p >= threshold
        )
        fp = sum(
            w for p, w in zip(negatives, weights[1::2], strict=True) if p >= threshold
        )
        assert row == pytest.approx(
            {
                "threshold": threshold,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": sum(weights[1::2]) - fp,
                "false_negatives": sum(weights[::2]) - tp,
                "precision": tp / (tp + fp) if tp + fp else 0,
                "recall": tp / sum(weights[::2]),
            }
        )


def test_precomputed_pr() -> None:
    fs = TensorBoardFS(CORPUS)
    rows = json.loads(fs.read("/pr_curves/raw_pr/pr_curves/000006.json", 4096, 0))
    for row, (threshold, tp, fp) in zip(
        rows, [(0, 4, 3), (0.5, 3, 1), (1, 1, 0)], strict=True
    ):
        assert row == pytest.approx(
            {
                "threshold": threshold,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": 3 - fp,
                "false_negatives": 4 - tp,
                "precision": tp / (tp + fp),
                "recall": tp / 4,
            }
        )


def test_distribution_interpolates_in_populated_buckets() -> None:
    fs = TensorBoardFS(CORPUS)
    rows = json.loads(fs.read("/distributions/distribution/000007.json", 4096, 0))
    percentiles = [0, 6.68, 15.87, 30.85, 50, 69.15, 84.13, 93.32, 100]
    assert [row["percentile"] for row in rows] == percentiles
    ranks = [p * 6 / 100 for p in percentiles]
    assert [row["value"] for row in rows] == pytest.approx(
        [-4 + 2 * rank if rank < 2 else rank - 2 for rank in ranks]
    )


def test_generic_strings_keep_shape_and_exact_binary_in_raw_view() -> None:
    fs = TensorBoardFS(CORPUS, raw=True)
    text = json.loads(fs.read("/text/words/000008.json", 4096, 0))
    assert text["shape"] == [2, 2]
    assert text["values"] == [["café", ""], ["南", "a\nb"]]
    binary = json.loads(fs.read("/text/bytes/000009.json", 4096, 0))
    assert binary["values"] == ["a\x00b", "\ufffd\ufffd"]
    directory = "/raw/events.out.tfevents.tf2-plugin-formats"
    for name in fs.readdir(directory):
        if name.endswith(".tfrecord"):
            frame = fs.read(f"{directory}/{name}", 4096, 0)
            event = compat_pb2.Event.FromString(frame[12:-4])
            if event.step == 9:
                (value,) = event.summary.value
                assert value.tag == "bytes"
                assert list(value.tensor.string_val) == [b"a\x00b", b"\xff\xfe"]
                return
    pytest.fail("no raw record for binary string tensor")


def test_resource_and_variant_keep_native_tensor_content() -> None:
    fs = TensorBoardFS(CORPUS)

    def content(tag: str, step: int, dtype: int) -> bytes:
        record = json.loads(fs.read(f"/tensors/{tag}/{step:06}.json", 4096, 0))
        tensor = record["tensor"]
        assert tensor["dtype"] == dtype and tensor["shape"] == []
        payload = base64.b64decode(tensor["tensor_content"]["base64"], validate=True)
        assert payload[0] == len(payload) - 1 < 128
        return payload[1:]

    handle = compat_pb2.ResourceHandleProto.FromString(content("handle", 10, 20))
    assert handle.name == "plugin_weights"
    (spec,) = handle.dtypes_and_shapes
    assert spec.dtype == 1 and [dim.size for dim in spec.shape.dim] == [2]
    variant = compat_pb2.VariantTensorDataProto.FromString(content("list", 11, 21))
    assert variant.type_name == "tensorflow::TensorList"
    assert [list(tensor.int_val) for tensor in variant.tensors] == [[7, -2], [8, 11]]
