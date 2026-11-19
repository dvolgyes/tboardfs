#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["netCDF4==1.7.4", "numpy==2.4.6", "safetensors==0.7.0", "scipy==1.17.1", "trimesh==4.11.3"]
# ///
"""Write and independently reopen compact scientific and mesh fixtures."""

import argparse
import bz2
import gzip
import lzma
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import trimesh
from netCDF4 import Dataset
from safetensors.numpy import load_file, save_file
from scipy.io import loadmat, savemat


class _Cases:
    @staticmethod
    def arrays(output: Path) -> None:
        """Write MAT v5 matrices and Safetensors arrays and compare their values."""
        values = np.array([[2, 3], [5, 7], [11, 13]], dtype=np.int32)
        for compressed in (False, True):
            mat = output / ("compressed.mat" if compressed else "counts.mat")
            savemat(mat, {"counts": values}, do_compression=compressed)
            np.testing.assert_array_equal(loadmat(mat)["counts"], values)
        tensors = output / "counts.safetensors"
        save_file(
            {"counts": values, "error": np.array([0.25, 0.5, 1], np.float32)}, tensors
        )
        readback = load_file(tensors)
        np.testing.assert_array_equal(readback["counts"], values)
        np.testing.assert_array_equal(readback["error"], [0.25, 0.5, 1])

    @staticmethod
    def measurements(output: Path) -> None:
        """Check real SQLite rows and NetCDF measurements."""
        values = [12.5, 14.75, 20.25]
        db = output / "depths.sqlite"
        with closing(sqlite3.connect(db)) as connection, connection:
            connection.execute(
                "CREATE TABLE depths (station INTEGER PRIMARY KEY, meters REAL)"
            )
            connection.executemany(
                "INSERT INTO depths VALUES (?, ?)", enumerate(values, 1)
            )
        with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as connection:
            rows = connection.execute(
                "SELECT meters FROM depths ORDER BY station"
            ).fetchall()
            np.testing.assert_array_equal(rows, np.array(values)[:, None])
        for name, version in (
            ("classic.nc", "NETCDF3_CLASSIC"),
            ("offset64.nc", "NETCDF3_64BIT_OFFSET"),
            ("data64.nc", "NETCDF3_64BIT_DATA"),
            ("netcdf4.nc", "NETCDF4"),
        ):
            path = output / name
            with Dataset(path, "w", format=version) as dataset:
                dataset.createDimension("station", len(values))
                depth = dataset.createVariable("depth", "f4", ("station",))
                depth.units = "m"
                depth[:] = values
            with Dataset(path) as dataset:
                depth = dataset.variables["depth"]
                if dataset.data_model != version or depth.units != "m":
                    raise ValueError(f"unexpected measurement encoding: {path}")
                np.testing.assert_array_equal(depth[:], values)

    @staticmethod
    def empty_records(output: Path) -> None:
        """Write two unlimited variables without any allocated records."""
        path = output / "empty-records.nc"
        with Dataset(path, "w", format="NETCDF3_CLASSIC") as dataset:
            dataset.createDimension("station", None)
            dataset.createVariable("depth", "f4", ("station",))
            dataset.createVariable("error", "f4", ("station",))
        with Dataset(path) as dataset:
            if not dataset.dimensions["station"].isunlimited():
                raise ValueError(f"expected an unlimited dimension: {path}")
            for name in ("depth", "error"):
                np.testing.assert_array_equal(dataset.variables[name][:], [])

    @staticmethod
    def compression(output: Path) -> None:
        """Round-trip the same CSV through three standard compression formats."""
        expected = b"station,depth_m\nA,12.5\nB,14.75\nC,20.25\n"
        for suffix, compress, decompress in (
            ("gz", gzip.compress, gzip.decompress),
            ("bz2", bz2.compress, bz2.decompress),
            ("xz", lzma.compress, lzma.decompress),
        ):
            path = output / f"depths.csv.{suffix}"
            path.write_bytes(compress(expected))
            if decompress(path.read_bytes()) != expected:
                raise ValueError(f"unexpected measurements: {path}")

    @staticmethod
    def meshes(output: Path) -> None:
        """Write and independently reopen two triangles in STL and OBJ."""
        vertices = [[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]]
        faces = [[0, 1, 2], [0, 2, 3]]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        for suffix in ("stl", "obj"):
            path = output / f"slope.{suffix}"
            if suffix == "obj":
                text = trimesh.exchange.obj.export_obj(mesh)
                path.write_bytes(text.rstrip("\r\n").encode("utf-8") + b"\n")
            else:
                mesh.export(path)
            readback = trimesh.load_mesh(path, process=False)
            np.testing.assert_array_equal(
                readback.triangles, np.asarray(vertices)[faces]
            )
            np.testing.assert_array_equal(readback.area_faces, [3, 6])


def main() -> None:
    """Populate a new or empty output directory with verified fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output: Path = parser.parse_args().output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _Cases.arrays(output)
    _Cases.measurements(output)
    _Cases.empty_records(output)
    _Cases.compression(output)
    _Cases.meshes(output)


if __name__ == "__main__":
    main()
