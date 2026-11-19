#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "h5py==3.14.0", "numpy==2.2.6", "plyfile==1.1.3", "pyarrow==20.0.0",
#   "pypdf==5.8.0", "reportlab==4.4.2", "trimesh==4.7.1",
# ]
# ///
"""Generate and reopen small geometry, document and scientific table fixtures."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from plyfile import PlyData, PlyElement
from pypdf import PdfReader
from reportlab.pdfgen.canvas import Canvas


def _geometry(output: Path) -> None:
    """Store a small terrain triangle as a mesh and three portable point clouds."""
    xyz = [(0.0, 0.0, 1.5), (1.0, 0.0, 2.0), (0.0, 1.0, 1.75)]
    mesh = trimesh.Trimesh(vertices=xyz, faces=[[0, 1, 2]], process=False)
    path = output / "terrain.glb"
    mesh.export(path)
    recovered = trimesh.load_scene(path).to_mesh()
    if len(recovered.faces) != 1 or not np.array_equal(recovered.vertices, xyz):
        raise ValueError(f"{path}: terrain triangle did not round-trip")
    points = np.array(xyz, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    variants = (("ascii", True, "<"), ("little", False, "<"), ("big", False, ">"))
    for label, text, order in variants:
        path = output / f"{label}.ply"
        point_cloud = PlyData(
            [PlyElement.describe(points, "vertex")], text=text, byte_order=order
        )
        point_cloud.write(path)
        if not np.array_equal(PlyData.read(path)["vertex"].data, points):
            raise ValueError(f"{path}: point coordinates did not round-trip")


def _tables(output: Path) -> None:
    """Round-trip station measurements in each columnar or array container."""
    table = pa.table({"station": ["Ridge", "Marsh"], "height_m": [105.5, 11.75]})
    parquet = output / "stations.parquet"
    pq.write_table(table, parquet, compression="NONE")
    if not pq.read_table(parquet).equals(table):
        raise ValueError(f"{parquet}: station table did not round-trip")
    for suffix, create, reopen in (
        ("arrow", pa.ipc.new_file, pa.ipc.open_file),
        ("stream", pa.ipc.new_stream, pa.ipc.open_stream),
    ):
        ipc_path = str(output / f"stations.{suffix}")
        with pa.OSFile(ipc_path, "wb") as sink, create(sink, table.schema) as writer:
            writer.write_table(table)
        with pa.memory_map(ipc_path, "r") as source:
            if not reopen(source).read_all().equals(table):
                raise ValueError(f"{ipc_path}: station table did not round-trip")
    for name, block in (("profile", 0), ("user-block", 512)):
        path = output / f"{name}.h5"
        with h5py.File(path, "w", userblock_size=block) as store:
            store.create_dataset("height_m", data=[105.5, 11.75], track_times=False)
        with h5py.File(path, "r") as store:
            if not np.array_equal(store["height_m"][:], [105.5, 11.75]):
                raise ValueError(f"{path}: station heights did not round-trip")


def main() -> None:
    """Write scientific files in a caller-selected new or empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    output = parser.parse_args().output
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _geometry(output)
    _tables(output)
    path = output / "survey.pdf"
    canvas = Canvas(str(path), invariant=1)
    canvas.drawString(72, 720, "Station survey: Ridge 105.5 m; Marsh 11.75 m")
    canvas.save()
    if "Ridge 105.5 m" not in PdfReader(path).pages[0].extract_text():
        raise ValueError(f"{path}: survey page is missing its ridge measurement")


if __name__ == "__main__":
    main()
