"""Recognize Office and OpenDocument packages from bounded XML metadata."""

import pyexpat
from dataclasses import dataclass, field
from types import MappingProxyType
from zipfile import ZIP_STORED, ZipFile

_MAX_XML = 1 << 16
_TYPES = "http://schemas.openxmlformats.org/package/2006/content-types}"
_RELS = "http://schemas.openxmlformats.org/package/2006/relationships}"
_OFFICE_RELS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument",
    "http://purl.oclc.org/ooxml/officeDocument/relationships/officeDocument",
)
_MAIN_TYPES = MappingProxyType(
    {
        "word/document.xml": ("wordprocessingml.document", "docx"),
        "xl/workbook.xml": ("spreadsheetml.sheet", "xlsx"),
        "ppt/presentation.xml": ("presentationml.presentation", "pptx"),
    }
)
_MANIFEST = "urn:oasis:names:tc:opendocument:xmlns:manifest:1.0}"
_ODF_TYPES = MappingProxyType(
    {
        b"application/vnd.oasis.opendocument.text": "odt",
        b"application/vnd.oasis.opendocument.spreadsheet": "ods",
        b"application/vnd.oasis.opendocument.presentation": "odp",
    }
)


def office_extension(archive: ZipFile, names: set[str]) -> str:
    """Recognize a supported ODF package or a standard OPC main part."""
    if "mimetype" in names:
        return _opendocument(archive, names)
    candidates = names.intersection(_MAIN_TYPES)
    if len(candidates) != 1 or not {"[Content_Types].xml", "_rels/.rels"} <= names:
        return "zip"
    if any(name.lower().endswith((".npy", "/vbaproject.bin")) for name in names):
        return "zip"
    part = candidates.pop()
    if not archive.getinfo(part).file_size or not _main_relationship(archive, part):
        return "zip"
    kind, extension = _MAIN_TYPES[part]
    overrides = _Metadata(_TYPES + "Types", _TYPES + "Override").read(
        archive, "[Content_Types].xml"
    )
    types = [
        node.get("ContentType")
        for node in overrides
        if node.get("PartName") == "/" + part
    ]
    expected = f"application/vnd.openxmlformats-officedocument.{kind}.main+xml"
    return extension if types == [expected] else "zip"


def _opendocument(archive: ZipFile, names: set[str]) -> str:
    """Require a stored first mimetype and a matching root and content manifest entry."""
    if not {"content.xml", "META-INF/manifest.xml"} <= names:
        return "zip"
    if names & {"[Content_Types].xml", "_rels/.rels"} or any(
        name.lower().endswith(".npy") for name in names
    ):
        return "zip"
    info = archive.getinfo("mimetype")
    if (
        info.header_offset != 0
        or info.compress_type != ZIP_STORED
        or max(info.file_size, info.compress_size) > 64
        or not archive.getinfo("content.xml").file_size
    ):
        return "zip"
    media = archive.read(info)
    extension = _ODF_TYPES.get(media)
    if not extension:
        return "zip"
    entries = _Metadata(_MANIFEST + "manifest", _MANIFEST + "file-entry").read(
        archive, "META-INF/manifest.xml"
    )
    types = [
        (node.get(_MANIFEST + "full-path"), node.get(_MANIFEST + "media-type"))
        for node in entries
    ]
    root = [value for path, value in types if path == "/"]
    content = [value for path, value in types if path == "content.xml"]
    return (
        extension
        if root == [media.decode("ascii")] and content == ["text/xml"]
        else "zip"
    )


def _main_relationship(archive: ZipFile, part: str) -> bool:
    relationships = _Metadata(_RELS + "Relationships", _RELS + "Relationship").read(
        archive, "_rels/.rels"
    )
    roots = [node for node in relationships if node.get("Type") in _OFFICE_RELS]
    if len(roots) != 1:
        return False
    return roots[0].get("TargetMode", "Internal") == "Internal" and roots[0].get(
        "Target"
    ) in (part, "/" + part)


@dataclass(slots=True)
class _Metadata:
    """Collect only direct children of an expected OPC root, without XML entities."""

    root: str
    child: str
    depth: int = 0
    count: int = 0
    matches: list[dict[str, str]] = field(default_factory=list)

    def read(self, archive: ZipFile, name: str) -> list[dict[str, str]]:
        """Return no candidates for forbidden XML features or unsupported metadata."""
        info = archive.getinfo(name)
        if max(info.file_size, info.compress_size) > _MAX_XML:
            return []
        with archive.open(info) as member:
            data = member.read(_MAX_XML + 1)
        parser = pyexpat.ParserCreate(namespace_separator="}")
        parser.SetParamEntityParsing(pyexpat.XML_PARAM_ENTITY_PARSING_NEVER)
        parser.StartDoctypeDeclHandler = self._reject
        parser.EntityDeclHandler = self._reject
        parser.ExternalEntityRefHandler = self._reject
        parser.XmlDeclHandler = self._declaration
        parser.StartElementHandler = self._start
        parser.EndElementHandler = self._end
        try:
            parser.Parse(data, True)
        except (pyexpat.ExpatError, ValueError, LookupError):
            return []
        return self.matches

    @staticmethod
    def _reject(*_args: object) -> int:
        raise ValueError("Entity declarations and references are unsupported")

    @staticmethod
    def _declaration(_version: str, encoding: str | None, _standalone: int) -> None:
        encodings = (
            "utf-8",
            "us-ascii",
            "iso-8859-1",
            "utf-16",
            "utf-16le",
            "utf-16be",
        )
        if encoding and encoding.lower() not in encodings:
            raise ValueError("Unsupported XML encoding")

    def _start(self, name: str, attrs: dict[str, str]) -> None:
        self.count += 1
        if self.depth > 1 or self.count > 1024:
            raise ValueError("OPC metadata exceeds structural limits")
        if self.depth == 0 and name != self.root:
            raise ValueError("Unexpected metadata root")
        if self.depth == 1 and name == self.child:
            self.matches.append(attrs)
        self.depth += 1

    def _end(self, _name: str) -> None:
        self.depth -= 1
