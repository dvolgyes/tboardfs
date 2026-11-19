"""Check filenames and exact bytes from native image and MPEG audio encoders."""

import pytest

from tboardfs import TensorBoardFS
from test_tf_producer_cases import CORPUS, _events


def _payloads(tag: str) -> tuple[str, bytes, bytes]:
    """Compare both native summary representations before checking their exports."""
    values = {v.tag: v for e in _events("native-media") for v in e.summary.value}
    legacy, plugin = values[f"legacy-{tag}"], values[f"tf2-{tag}"]
    tab = plugin.metadata.plugin_data.plugin_name
    if tag in ("mp2", "mp3"):
        assert tab == "audio"
        assert legacy.audio.content_type == "audio/mpeg"
        encoded, label = plugin.tensor.string_val
        assert label == b"A4"
        return tab, legacy.audio.encoded_audio_string, encoded
    assert tab == "images"
    width, height, encoded = plugin.tensor.string_val
    assert (width, height) == (b"16", b"12")
    return tab, legacy.image.encoded_image_string, encoded


@pytest.mark.parametrize(
    ("tag", "suffix", "signature"),
    [
        ("jp2", "jp2", b"\x00\x00\x00\x0cjP  \r\n\x87\n"),
        ("j2k", "j2k", b"\xff\x4f\xff\x51"),
        ("jxl-codestream", "jxl", b"\xff\x0a"),
        ("jxl-container", "jxl", b"\x00\x00\x00\x0cJXL \r\n\x87\n"),
        ("exr", "exr", b"\x76\x2f\x31\x01"),
        ("mp2", "mp2", b"\xff\xfd"),
        ("mp3", "mp3", b"\xff\xfb"),
    ],
)
def test_native_exports(tag: str, suffix: str, signature: bytes) -> None:
    """Both low-level summary layouts must export the original encoded stream."""
    tab, blob, tensor = _payloads(tag)
    assert blob == tensor
    assert blob.startswith(signature)
    fs = TensorBoardFS(CORPUS / "native-media")
    for layout in ("legacy", "tf2"):
        directory = f"/{tab}/{layout}-{tag}"
        name = f"000001.{suffix}"
        assert fs.readdir(directory)[2:] == [name]
        assert fs.read(f"{directory}/{name}", len(blob) + 1, 0) == blob
