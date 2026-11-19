#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "python-docx==1.2.0", "openpyxl==3.1.5", "python-pptx==1.0.2",
# ]
# ///
"""Generate standalone Office documents for file type detection, not TensorBoard events."""

import argparse
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from docx import Document
from openpyxl import Workbook, load_workbook
from openpyxl.writer.excel import ExcelWriter
from pptx import Presentation


class _FieldNotes:
    """Make three views of a field survey with the native document libraries."""

    @staticmethod
    def report() -> bytes:
        """Write a short report with an attributed reading."""
        data = BytesIO()
        doc = Document()
        doc.add_heading("River survey", level=1)
        doc.add_paragraph("Upstream water temperature: 12.5 C.")
        doc.save(data)
        if Document(data).paragraphs[-1].text != "Upstream water temperature: 12.5 C.":
            raise ValueError("report lost the temperature reading")
        return data.getvalue()

    @staticmethod
    def workbook() -> bytes:
        """Write the two sampling sites and their numeric readings."""
        data = BytesIO()
        book = Workbook()
        sheet = book.active
        sheet.title = "Temperatures"
        sheet.append(["Site", "Water temperature (C)"])
        sheet.append(["Upstream", 12.5])
        sheet.append(["Downstream", 13.0])
        book.properties.created = book.properties.modified = datetime(
            2000, 1, 1, tzinfo=UTC
        )
        with ZipFile(data, "w", ZIP_DEFLATED) as archive:
            ExcelWriter(book, archive).save()
        if load_workbook(data, data_only=True)["Temperatures"]["B2"].value != 12.5:
            raise ValueError("workbook lost the numeric reading")
        return data.getvalue()

    @staticmethod
    def slides() -> bytes:
        """Write a one-slide field briefing."""
        data = BytesIO()
        deck = Presentation()
        slide = deck.slides.add_slide(deck.slide_layouts[0])
        slide.shapes.title.text = "River survey"
        slide.placeholders[1].text = "Water warmed by 0.5 C downstream."
        deck.save(data)
        if Presentation(data).slides[0].shapes.title.text != "River survey":
            raise ValueError("slide lost its title")
        return data.getvalue()


def _normalize(blob: bytes) -> bytes:
    """Give the producer's ZIP entries fixed timestamps without changing their contents."""
    out = BytesIO()
    with ZipFile(BytesIO(blob)) as src, ZipFile(out, "w") as dst:
        for name in sorted(src.namelist()):
            dst.writestr(ZipInfo(name), src.read(name), compress_type=ZIP_DEFLATED)
    return out.getvalue()


def main() -> None:
    """Write three reproducible Office packages to a new or empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    for name, build in (
        ("survey.docx", _FieldNotes.report),
        ("readings.xlsx", _FieldNotes.workbook),
        ("briefing.pptx", _FieldNotes.slides),
    ):
        (output / name).write_bytes(_normalize(build()))


if __name__ == "__main__":
    main()
