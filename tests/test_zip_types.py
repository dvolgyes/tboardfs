"""Exercise the ZIP package sniffer independently of the TensorBoard reader."""

from io import BytesIO
from random import Random
import struct
from unittest import TestCase
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from tboardfs.zip_types import zip_extension


def _zip(members, method=ZIP_DEFLATED, comment=b""):
    buffer = BytesIO()
    with ZipFile(buffer, "w", method) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        archive.comment = comment
    return buffer.getvalue()


def _npy(version, values=struct.pack("<hh", -17, 67)):
    width = 2 if version == 1 else 4
    header = f"{{'descr': '<i2', 'fortran_order': False, 'shape': ({len(values) // 2},), }}".encode()
    header = header.ljust(128 - 8 - width - 1) + b"\n"
    return (
        b"\x93NUMPY"
        + bytes((version, 0))
        + len(header).to_bytes(width, "little")
        + header
        + values
    )


def _office(part, kind):
    types = "http://schemas.openxmlformats.org/package/2006/content-types"
    rels = "http://schemas.openxmlformats.org/package/2006/relationships"
    root = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    return {
        "[Content_Types].xml": (
            f'<Types xmlns="{types}"><Override PartName="/{part}" ContentType="application/'
            f'vnd.openxmlformats-officedocument.{kind}.main+xml"/></Types>'
        ).encode(),
        "_rels/.rels": (
            f'<Relationships xmlns="{rels}"><Relationship Id="rId1" '
            f'Type="{root}" Target="/{part}"/></Relationships>'
        ).encode(),
        part: b"<root/>",
    }


class TestZipTypes(TestCase):
    def test_zip_types_stored_deflated_numpy(self):
        arrays = {f"v{v}.npy": _npy(v) for v in (1, 2, 3)}
        for method in (ZIP_STORED, ZIP_DEFLATED):
            with self.subTest(method=method):
                self.assertEqual(zip_extension(_zip(arrays, method)), "npz")
        self.assertEqual(zip_extension(_zip(arrays | {"readme": b"x"})), "zip")
        for data in (b"text", _npy(1)[:8], _npy(1).replace(b"'shape'", b"'wrong'")):
            self.assertEqual(zip_extension(_zip(arrays | {"bad.npy": data})), "zip")

    def test_zip_types_office_packages(self):
        for ext, part, kind in (
            ("docx", "word/document.xml", "wordprocessingml.document"),
            ("xlsx", "xl/workbook.xml", "spreadsheetml.sheet"),
            ("pptx", "ppt/presentation.xml", "presentationml.presentation"),
        ):
            members = _office(part, kind)
            for method in (ZIP_STORED, ZIP_DEFLATED):
                with self.subTest(ext=ext, method=method):
                    self.assertEqual(zip_extension(_zip(members, method)), ext)
            for missing in members:
                subset = members.copy()
                del subset[missing]
                self.assertEqual(zip_extension(_zip(subset)), "zip")

    def test_zip_types_opc_encoding_and_comments(self):
        members = _office("word/document.xml", "wordprocessingml.document")
        manifest = members["[Content_Types].xml"].decode()
        for encoding in ("utf-8", "iso-8859-1", "utf-16"):
            xml = f'<?xml version="1.0" encoding="{encoding}"?><!--café-->{manifest}'
            members["[Content_Types].xml"] = xml.encode(encoding)
            self.assertEqual(zip_extension(_zip(members)), "docx")

    def test_zip_types_opc_reject_unsafe_or_oversized_metadata(self):
        members = _office("word/document.xml", "wordprocessingml.document")
        manifest = members["[Content_Types].xml"]
        for data in (
            b'<!DOCTYPE Types SYSTEM "urn:unused">' + manifest,
            b'<!DOCTYPE Types [<!ENTITY x "unused">]>' + manifest,
            b'<?xml version="1.0" encoding="nonexistent"?>' + manifest,
            manifest.replace(b"</Types>", b"<a><b/></a></Types>"),
            manifest.replace(b"</Types>", b"<a/>" * 1024 + b"</Types>"),
            manifest[:-1],
        ):
            members["[Content_Types].xml"] = data
            self.assertEqual(zip_extension(_zip(members)), "zip")

    def test_zip_types_require_unambiguous_office_evidence(self):
        members = _office("word/document.xml", "wordprocessingml.document")
        manifest = members["[Content_Types].xml"]
        rels = members["_rels/.rels"]
        for alternate in (
            {"xl/workbook.xml": b"<xml/>"},
            {"numbers.npy": _npy(1)},
            {"word/vbaProject.bin": b"x"},
            {
                "[Content_Types].xml": manifest.replace(
                    b"document.main", b"template.main"
                )
            },
            {
                "[Content_Types].xml": manifest.replace(
                    b"document.main", b"macroEnabled.main"
                )
            },
            {"[Content_Types].xml": b"<!DOCTYPE Types>" + manifest},
            {
                "[Content_Types].xml": ("<!DOCTYPE Types>" + manifest.decode()).encode(
                    "utf-16"
                )
            },
            {"[Content_Types].xml": manifest.replace(b"content-types", b"other")},
            {"_rels/.rels": rels.replace(b"Target=", b'TargetMode="External" Target=')},
            {"_rels/.rels": rels.replace(b"officeDocument/2006", b"other/2006")},
        ):
            self.assertEqual(zip_extension(_zip(members | alternate)), "zip")

    def test_zip_types_encrypted_and_malformed(self):
        archive = _zip({"data.npy": _npy(1)}, ZIP_STORED, b"test comment")
        encrypted = bytearray(archive)
        central = archive.index(b"PK\x01\x02")
        encrypted[6] |= 1
        encrypted[central + 8] |= 1
        self.assertEqual(zip_extension(bytes(encrypted)), "zip")
        for invalid in (
            b"PK\x03\x04",
            archive[:-1],
            archive[:central],
            archive.replace(b"PK\x01\x02", b"not!"),
            archive.replace(b"data.npy", b"name.npy", 1),
        ):
            self.assertEqual(zip_extension(invalid), "bin")
        self.assertEqual(zip_extension(_zip({})), "zip")
        self.assertEqual(zip_extension(_zip({"data.csv": b"0,1"})), "zip")

    def test_zip_types_bound_inspection(self):
        many = _zip({f"{n}.npy": _npy(1) for n in range(129)})
        with patch("tboardfs.zip_types.ZipFile", side_effect=AssertionError):
            self.assertEqual(zip_extension(many), "zip")
            forged = bytearray(many)
            struct.pack_into("<HH", forged, len(many) - 14, 1, 1)
            self.assertEqual(zip_extension(bytes(forged)), "bin")
        members = _office("word/document.xml", "wordprocessingml.document")
        members["[Content_Types].xml"] += b" " * (1 << 16)
        self.assertEqual(zip_extension(_zip(members)), "zip")

    def test_zip_types_large_array_prefix(self):
        array = _npy(1, Random(17).randbytes(2 << 20))
        for method in (ZIP_STORED, ZIP_DEFLATED):
            archive = _zip({"noise.npy": array}, method)
            self.assertGreater(len(archive), 1 << 20)
            self.assertEqual(zip_extension(archive), "npz")

    def test_zip_types_deflate_input_budget(self):
        array = _npy(1)
        archive = bytearray(_zip({"data.npy": array}))
        central = archive.index(b"PK\x01\x02")
        end = len(archive) - 22
        padding = b"\x00\x00\x00\xff\xff" * 1700
        compressed = struct.unpack_from("<I", archive, 18)[0] + len(padding)
        struct.pack_into("<I", archive, 18, compressed)
        struct.pack_into("<I", archive, central + 20, compressed)
        struct.pack_into("<I", archive, end + 16, central + len(padding))
        start = 30 + len("data.npy")
        archive[start:start] = padding
        with ZipFile(BytesIO(archive)) as reader:
            self.assertEqual(reader.read("data.npy"), array)
        self.assertEqual(zip_extension(bytes(archive)), "zip")

    def test_zip_types_zip64_end_record(self):
        archive = _zip({"data.npy": _npy(1)})
        end = len(archive) - 22
        size, offset = struct.unpack_from("<II", archive, end + 12)
        record = struct.pack(
            "<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, 1, 1, size, offset
        )
        locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, end, 1)
        with patch("tboardfs.zip_types.ZipFile", side_effect=AssertionError):
            for expected, pair in (("zip", record + locator), ("bin", locator)):
                self.assertEqual(
                    zip_extension(archive[:end] + pair + archive[end:]), expected
                )
