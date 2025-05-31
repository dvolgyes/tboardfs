#!/usr/bin/env python3
"""Demonstration of tboardfs functionality."""

import tempfile
from pathlib import Path
from tboardfs.test_generator import generate_test_tensorboard_log
from tboardfs.parser import TensorBoardParser
from loguru import logger
import sys


def main():
    """Run demonstration."""
    # Setup logging for demo
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
        colorize=True,
    )

    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        print("=== TensorBoard FS Demo ===\n")

        # Generate test data
        print("1. Generating test TensorBoard log with all data types...")
        log_dir = temp_path / "demo_log"
        event_file = generate_test_tensorboard_log(str(log_dir), num_iterations=5)
        print(f"   Created: {event_file}\n")

        # Parse and display content
        print("2. Parsing TensorBoard event file...")
        parser = TensorBoardParser(event_file, show_progress=True)
        content = parser.list_all_content()

        print("   Found data types:")
        for dtype, tags in content.items():
            if tags:
                print(f"   - {dtype}: {len(tags)} tag(s)")

        # Show virtual paths
        print("\n3. Virtual filesystem paths:")
        paths = parser.get_virtual_paths()
        for path in paths[:10]:  # Show first 10
            print(f"   {path}")
        if len(paths) > 10:
            print(f"   ... and {len(paths) - 10} more paths")

        # Extract some data
        print("\n4. Extracting sample data:")

        # Scalar data
        if content["scalars"]:
            tag = content["scalars"][0]
            data = parser.export_scalar_to_text(tag)
            lines = data.strip().split("\n")
            print(f"   Scalar '{tag}' (first 3 values):")
            for line in lines[:3]:
                print(f"     {line}")

        # Image info
        if content["images"]:
            tag = content["images"][0]
            images = parser.get_image_data(tag)
            print(f"\n   Images '{tag}':")
            print(f"     Count: {len(images)}")
            print(f"     Size: {images[0].width}x{images[0].height}")

        # Extract all to directory
        print("\n5. Extracting all data to directory structure...")
        output_dir = temp_path / "extracted"
        parser.extract_all_to_directory(str(output_dir))

        # Show extracted structure
        print("   Created directory structure:")
        for subdir in sorted(output_dir.iterdir()):
            if subdir.is_dir():
                print(f"   - {subdir.name}/")
                files = list(subdir.iterdir())
                if subdir.name == "scalars":
                    for f in sorted(files)[:3]:
                        print(f"     {f.name}")
                else:
                    # For other types, show subdirectories
                    subdirs = [f for f in files if f.is_dir()]
                    for sd in sorted(subdirs)[:2]:
                        print(f"     {sd.name}/")
                        subfiles = list(sd.iterdir())
                        if subfiles:
                            print(f"       ({len(subfiles)} files)")

        print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
