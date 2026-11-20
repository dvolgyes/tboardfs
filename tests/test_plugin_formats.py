"""Read TensorFlow and TensorBoard plugin writer output without either writer."""

import base64
import json
import struct
from pathlib import Path

import pytest

from tboardfs import TensorBoardFS

RUN = Path(__file__).parent / "data/producer-cases/plugins"


@pytest.fixture(scope="module")
def fs() -> TensorBoardFS:
    return TensorBoardFS(RUN)


def test_custom_scalar_layout_keeps_margin_and_multiline_charts(
    fs: TensorBoardFS,
) -> None:
    data = json.loads(fs.read("/custom_scalars/layout.json", 65536, 0))

    assert data == {
        "category": [
            {
                "title": "comparisons",
                "closed": True,
                "chart": [
                    {
                        "title": "range",
                        "margin": {
                            "series": [
                                {"value": "score", "lower": "low", "upper": "high"}
                            ]
                        },
                    },
                    {
                        "title": "traces",
                        "multiline": {"tag": ["train/loss", "eval/loss"]},
                    },
                ],
            }
        ]
    }


def test_hparams_domains_and_trial_values(fs: TensorBoardFS) -> None:
    data = json.loads(fs.read("/hparams/hparams.json", 65536, 0))
    infos = {entry["name"]: entry for entry in data["experiment"]["hparam_infos"]}

    assert {k: v["type"] for k, v in infos.items()} == {
        "optimizer": "DATA_TYPE_STRING",
        "enabled": "DATA_TYPE_BOOL",
        "rate": "DATA_TYPE_FLOAT64",
    }
    assert infos["optimizer"]["domain_discrete"] == ["adam", "sgd"]
    assert infos["enabled"]["domain_discrete"] == [False, True]
    assert infos["rate"]["domain_interval"] == {"min_value": 0.125, "max_value": 0.5}
    assert data["session_start"] == {
        "group_name": "trial",
        "start_time_secs": 11,
        "hparams": {"optimizer": "adam", "enabled": False, "rate": 0.25},
    }
    assert data["session_end"] == {"status": "STATUS_RUNNING", "end_time_secs": 12}


def test_v2_text_keeps_cells_separate_from_markdown(fs: TensorBoardFS) -> None:
    text = fs.read("/text/labels/000002.txt", 65536, 0).decode()
    table = json.loads(fs.read("/text/labels/000002.json", 65536, 0))
    alpha = "\N{GREEK SMALL LETTER ALPHA}"
    assert text == f"|   |   |\n| --- | --- |\n| {alpha} | x\\|y |\n| one<br>two |  |\n"
    assert table["shape"] == [2, 2]
    assert table["values"] == [[alpha, "x|y"], ["one\ntwo", ""]]


def test_projector_config_references_remain_readable(fs: TensorBoardFS) -> None:
    config = (RUN / "projector_config.pbtxt").read_bytes()
    assert fs.read("/projector/projector_config.pbtxt", 65536, 0) == config
    for name in ("vectors.tsv", "labels.tsv", "views.json", "sprite.png"):
        assert f'"assets/{name}"'.encode() in config
        assert (
            fs.read(f"/projector/assets/{name}", 65536, 0)
            == (RUN / "assets" / name).read_bytes()
        )


def test_profile_exposes_original_xplane_bytes(fs: TensorBoardFS) -> None:
    path = "profile/capture/cpu.xplane.pb"
    data = (RUN / "plugins" / path).read_bytes()
    assert b"plugin_case" in data
    assert fs.read("/" + path, len(data), 0) == data


def test_unrecognized_summary_retains_numeric_tensor_and_metadata(
    fs: TensorBoardFS,
) -> None:
    record = json.loads(fs.read("/plugins/example_array/points/000002.json", 65536, 0))
    assert base64.b64decode(record["plugin_content"]["base64"]) == b"\x00\xffpayload"
    tensor = record["tensor"]
    assert tensor["dtype"] == 3
    assert base64.b64decode(tensor["tensor_content"]["base64"]) == struct.pack(
        "<ii", 7, -3
    )
