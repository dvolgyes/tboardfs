"""Check Open3D TensorFlow assets and auxiliary inference ranges end to end."""

import json
from pathlib import Path
import shutil
import struct

from click.testing import CliRunner
import pytest

from tboardfs import TensorBoardFS
from tboardfs.crc32c import masked_crc32c
from tboardfs.file_cli import main

RUN = Path(__file__).parent / "data" / "producer-cases" / "open3d-advanced"
PLUGIN = Path("plugins", "Open3D")
TAGS = ("detections", "line_set", "segmentation", "textured_mesh")
MANIFEST = "000005.manifest.json"


@pytest.mark.parametrize("tag", TAGS)
def test_native_open3d_manifest_and_exact_assets(tag: str, tmp_path: Path) -> None:
    """The CLI copies the original native sidecar and the manifest indexes it."""
    output = tmp_path / "output"
    result = CliRunner().invoke(main, ["copy-all", str(RUN), str(output)])
    assert result.exit_code == 0, result.output
    manifest = json.loads((output / PLUGIN / tag / MANIFEST).read_bytes())
    name = f"{tag}.msgpack"
    asset = PLUGIN / "~assets" / name
    assert manifest["asset"] == asset.as_posix()
    assert manifest["filename"] == name
    assert manifest["version"] == 14
    assert manifest["property_references"] == {}
    contents = (output / asset).read_bytes()
    assert contents == (RUN / PLUGIN / name).read_bytes()
    (entry,) = manifest["start_size"]
    assert entry["start"] == 0 < entry["size"]
    assert entry["size"] + entry["aux_size"] == len(contents)
    assert masked_crc32c(contents[: entry["size"]]) == entry["masked_crc32c"]
    if tag == "detections":
        assert entry["aux_start"] == entry["size"]
        assert entry["aux_size"] == 18
        aux = contents[entry["aux_start"] :]
        assert masked_crc32c(aux) == entry["aux_masked_crc32c"]
        assert aux == (
            b"\x0a\x07\x08\x02\x15"
            + struct.pack("<f", 0.875)
            + b"\x0a\x07\x08\x07\x15"
            + struct.pack("<f", 0.625)
        )
    else:
        assert (
            entry["aux_start"] == entry["aux_size"] == entry["aux_masked_crc32c"] == 0
        )


def test_open3d_manifest_does_not_link_asset_shorter_than_aux_range(
    tmp_path: Path,
) -> None:
    """Valid geometry bytes alone are not sufficient to link a detection asset."""
    run = shutil.copytree(RUN, tmp_path / "run")
    sidecar = run / PLUGIN / "detections.msgpack"
    sidecar.write_bytes(sidecar.read_bytes()[:-1])
    fs = TensorBoardFS(run)
    path = "/plugins/Open3D/detections/" + MANIFEST
    manifest = json.loads(fs.read(path, 4096, 0))
    assert manifest["asset"] is None
    (entry,) = manifest["start_size"]
    assert entry["aux_size"] == 18
    assert (
        entry["size"] < sidecar.stat().st_size < entry["aux_start"] + entry["aux_size"]
    )
