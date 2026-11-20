#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["odfdo==3.19.0", "numpy==2.5.4", "python-docx==1.2.0"]
# ///
"""Produce small field-survey documents and arrays, including real ZIP64 directories."""

import argparse
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from docx import Document as WordDocument
from odfdo import Document, Frame, Paragraph

_READING = "Upstream water temperature: 12.5 C."


class _FieldSurvey:
    """Create and reopen the editable survey with native document producers."""

    @staticmethod
    def odf(kind: str) -> Document:
        """Fill a native text, spreadsheet or presentation template."""
        doc = Document(kind)
        if kind == "text":
            doc.body.append(Paragraph(_READING))
        elif kind == "spreadsheet":
            table = doc.body.get_table()
            if table is None:
                raise ValueError("spreadsheet template lacks a table")
            table.set_value("A1", "Sampling site")
            table.set_value("B1", "Temperature (C)")
            table.set_value("A2", "Upstream")
            table.set_value("B2", 12.5)
        else:
            page = doc.body.get_draw_page()
            if page is None:
                raise ValueError("presentation template lacks a slide")
            page.append(
                Frame.text_frame(
                    _READING, position=("2cm", "3cm"), size=("19cm", "4cm")
                )
            )
        return doc

    @staticmethod
    def check_odf(blob: bytes, kind: str) -> None:
        """Verify the reading in the final packaged document, not the source template."""
        body = Document(BytesIO(blob)).body
        if kind == "spreadsheet":
            table = body.get_table()
            if table is None or table.get_value("B2") != 12.5:
                raise ValueError("spreadsheet lost the numeric reading")
        elif _READING not in body.text_recursive:
            raise ValueError("document lost the survey reading")


def _package(data: BytesIO, zip64: bool) -> bytes:
    """Fix timestamps and request genuine ZIP64 end records at a small member count."""
    result = BytesIO()
    limit = 0 if zip64 else zipfile.ZIP_FILECOUNT_LIMIT
    with (
        patch.object(zipfile, "ZIP_FILECOUNT_LIMIT", limit),
        zipfile.ZipFile(data) as source,
        zipfile.ZipFile(result, "w") as dest,
    ):
        for name in sorted(
            source.namelist(), key=lambda item: (item != "mimetype", item)
        ):
            method = zipfile.ZIP_STORED if name == "mimetype" else zipfile.ZIP_DEFLATED
            dest.writestr(
                zipfile.ZipInfo(name), source.read(name), compress_type=method
            )
    return result.getvalue()


def _zip64_examples(output: Path) -> None:
    """Reopen the small ZIP64 Office and array examples before publishing them."""
    source = BytesIO()
    document = WordDocument()
    document.add_paragraph(_READING)
    document.save(source)
    blob = _package(source, True)
    if WordDocument(BytesIO(blob)).paragraphs[0].text != _READING:
        raise ValueError("Word report lost the survey reading")
    (output / "survey64.docx").write_bytes(blob)
    source = BytesIO()
    values = np.array([12.5, 13.0], dtype="<f4")
    np.savez(source, temperature=values)
    blob = _package(source, True)
    with np.load(BytesIO(blob), allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["temperature"], values)
    (output / "temperatures64.npz").write_bytes(blob)


def main() -> None:
    """Write standalone documents and arrays to a new or empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    for kind, ext in (("text", "odt"), ("spreadsheet", "ods"), ("presentation", "odp")):
        data = BytesIO()
        _FieldSurvey.odf(kind).save(data)
        for zip64 in (False, True) if kind == "text" else (False,):
            blob = _package(data, zip64)
            _FieldSurvey.check_odf(blob, kind)
            (output / f"survey{'64' if zip64 else ''}.{ext}").write_bytes(blob)
    _zip64_examples(output)


if __name__ == "__main__":
    main()
