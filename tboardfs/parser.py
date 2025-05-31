"""TensorBoard event file parser."""

import os
import struct
import io
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.util import tensor_util
import numpy as np


@dataclass
class ScalarData:
    """Container for scalar data points."""
    step: int
    value: float
    wall_time: float


@dataclass
class ImageData:
    """Container for image data."""
    step: int
    encoded_image_string: bytes
    width: int
    height: int
    wall_time: float


class TensorBoardParser:
    """Parser for TensorBoard event files."""
    
    def __init__(self, event_file_path: str):
        """Initialize parser with event file path."""
        self.event_file_path = event_file_path
        self.ea = event_accumulator.EventAccumulator(
            event_file_path,
            size_guidance={
                event_accumulator.SCALARS: 0,
                event_accumulator.IMAGES: 0,
                event_accumulator.HISTOGRAMS: 0,
                event_accumulator.TENSORS: 0,
            }
        )
        self.ea.Reload()
        
    def list_scalars(self) -> List[str]:
        """List all scalar tags in the event file."""
        return list(self.ea.Tags()['scalars'])
    
    def list_images(self) -> List[str]:
        """List all image tags in the event file."""
        return list(self.ea.Tags()['images'])
    
    def list_histograms(self) -> List[str]:
        """List all histogram tags in the event file."""
        return list(self.ea.Tags()['histograms'])
    
    def list_tensors(self) -> List[str]:
        """List all tensor tags in the event file."""
        return list(self.ea.Tags()['tensors'])
    
    def get_scalar_data(self, tag: str) -> List[ScalarData]:
        """Get all scalar data for a given tag."""
        scalar_events = self.ea.Scalars(tag)
        return [
            ScalarData(
                step=event.step,
                value=event.value,
                wall_time=event.wall_time
            )
            for event in scalar_events
        ]
    
    def get_image_data(self, tag: str) -> List[ImageData]:
        """Get all image data for a given tag."""
        image_events = self.ea.Images(tag)
        return [
            ImageData(
                step=event.step,
                encoded_image_string=event.encoded_image_string,
                width=event.width,
                height=event.height,
                wall_time=event.wall_time
            )
            for event in image_events
        ]
    
    def export_scalar_to_text(self, tag: str) -> str:
        """Export scalar data to text format (iteration, value)."""
        scalar_data = self.get_scalar_data(tag)
        lines = []
        for data in scalar_data:
            lines.append(f"{data.step}\t{data.value}")
        return '\n'.join(lines)
    
    def export_image(self, tag: str, step: int) -> Optional[bytes]:
        """Export a specific image by tag and step."""
        image_data = self.get_image_data(tag)
        for data in image_data:
            if data.step == step:
                return data.encoded_image_string
        return None
    
    def get_image_extension(self, image_bytes: bytes) -> str:
        """Determine image extension from bytes."""
        if image_bytes.startswith(b'\x89PNG'):
            return 'png'
        elif image_bytes.startswith(b'\xff\xd8\xff'):
            return 'jpg'
        else:
            return 'bin'
    
    def list_all_content(self) -> Dict[str, List[str]]:
        """List all content organized by type."""
        return {
            'scalars': self.list_scalars(),
            'images': self.list_images(),
            'histograms': self.list_histograms(),
            'tensors': self.list_tensors(),
        }
    
    def get_virtual_paths(self, digits: int = 6) -> List[str]:
        """Get all virtual paths that would exist in the filesystem."""
        paths = []
        
        # Scalar paths
        for tag in self.list_scalars():
            safe_tag = tag.replace('/', '_')
            paths.append(f"scalars/{safe_tag}.txt")
        
        # Image paths
        for tag in self.list_images():
            safe_tag = tag.replace('/', '_')
            image_data = self.get_image_data(tag)
            for data in image_data:
                ext = self.get_image_extension(data.encoded_image_string)
                padded_step = str(data.step).zfill(digits)
                paths.append(f"images/{safe_tag}/{padded_step}.{ext}")
        
        # Add directories
        paths.append("scalars/")
        paths.append("images/")
        
        image_tags = self.list_images()
        for tag in image_tags:
            safe_tag = tag.replace('/', '_')
            paths.append(f"images/{safe_tag}/")
        
        return sorted(set(paths))
    
    def extract_all_to_directory(self, output_dir: str, sort_scalars: bool = True, digits: int = 6):
        """Extract all data to a directory structure."""
        import os
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        scalars_dir = output_path / "scalars"
        images_dir = output_path / "images"
        scalars_dir.mkdir(exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        
        scalar_files = {}
        
        # Extract scalars - append data as we go
        for tag in self.list_scalars():
            safe_tag = tag.replace('/', '_')
            scalar_file = scalars_dir / f"{safe_tag}.txt"
            scalar_data = self.get_scalar_data(tag)
            
            with open(scalar_file, 'w') as f:
                for data in scalar_data:
                    f.write(f"{data.step}\t{data.value}\n")
            
            if sort_scalars:
                scalar_files[scalar_file] = True
        
        # Extract images
        for tag in self.list_images():
            safe_tag = tag.replace('/', '_')
            tag_dir = images_dir / safe_tag
            tag_dir.mkdir(exist_ok=True)
            
            image_data = self.get_image_data(tag)
            for data in image_data:
                ext = self.get_image_extension(data.encoded_image_string)
                padded_step = str(data.step).zfill(digits)
                image_file = tag_dir / f"{padded_step}.{ext}"
                with open(image_file, 'wb') as f:
                    f.write(data.encoded_image_string)
        
        # Sort scalar files if requested
        if sort_scalars:
            self._sort_scalar_files(scalar_files.keys())
    
    def _sort_scalar_files(self, scalar_files):
        """Sort scalar files by iteration number (first column)."""
        for file_path in scalar_files:
            # Read the file
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse and sort by step (first column)
            data_points = []
            for line in lines:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        try:
                            step = int(parts[0])
                            value = float(parts[1])
                            data_points.append((step, value))
                        except ValueError:
                            continue
            
            # Sort by step
            data_points.sort(key=lambda x: x[0])
            
            # Write back sorted data
            with open(file_path, 'w') as f:
                for step, value in data_points:
                    f.write(f"{step}\t{value}\n")