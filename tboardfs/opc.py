"""Recognize Office packages from bounded, namespace-aware OPC metadata."""

from dataclasses import dataclass, field
import pyexpat
from types import MappingProxyType
from zipfile import ZipFile

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


def office_extension(archive: ZipFile, names: set[str]) -> str:
    """Match one standard main part, its override and an internal root relationship."""
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
