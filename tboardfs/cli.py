"""CLI interface for tboardfs."""

import os
import sys
import click
from pathlib import Path
from typing import Optional

from .parser import TensorBoardParser


@click.group()
def main():
    """TensorBoard filesystem interface CLI."""
    pass


@main.command()
@click.argument('tensorboard_path', type=click.Path(exists=True))
@click.option('--recursive', '-r', is_flag=True, help='List recursively')
@click.option('--digits', type=int, default=6, help='Number of digits for padding iteration numbers (default: 6)')
def list(tensorboard_path: str, recursive: bool, digits: int):
    """List contents of TensorBoard log file(s)."""
    path = Path(tensorboard_path)
    
    if path.is_file() and 'tfevents' in path.name:
        _list_single_file(path, digits)
    elif path.is_dir():
        if recursive:
            _list_directory_recursive(path, digits)
        else:
            _list_directory(path)
    else:
        click.echo(f"Error: {tensorboard_path} is not a valid TensorBoard log file or directory", err=True)
        sys.exit(1)


def _list_single_file(file_path: Path, digits: int = 6):
    """List contents of a single TensorBoard event file."""
    try:
        parser = TensorBoardParser(str(file_path))
        content = parser.list_all_content()
        
        click.echo(f"\nContents of {file_path}:")
        click.echo("=" * 60)
        
        if content['scalars']:
            click.echo("\nScalars:")
            for tag in content['scalars']:
                click.echo(f"  - {tag}")
        
        if content['images']:
            click.echo("\nImages:")
            for tag in content['images']:
                image_data = parser.get_image_data(tag)
                click.echo(f"  - {tag} ({len(image_data)} steps)")
        
        if content['histograms']:
            click.echo("\nHistograms:")
            for tag in content['histograms']:
                click.echo(f"  - {tag}")
        
        if content['tensors']:
            click.echo("\nTensors:")
            for tag in content['tensors']:
                click.echo(f"  - {tag}")
        
        click.echo("\nVirtual filesystem paths:")
        for path in parser.get_virtual_paths(digits=digits):
            click.echo(f"  {path}")
            
    except Exception as e:
        click.echo(f"Error processing {file_path}: {e}", err=True)


def _list_directory(directory: Path):
    """List TensorBoard files in a directory."""
    event_files = list(directory.glob("*.tfevents.*"))
    if not event_files:
        click.echo(f"No TensorBoard event files found in {directory}")
        return
    
    click.echo(f"Found {len(event_files)} TensorBoard event file(s) in {directory}:")
    for file in event_files:
        click.echo(f"  - {file.name}")


def _list_directory_recursive(directory: Path, digits: int = 6):
    """List TensorBoard files recursively in a directory."""
    event_files = list(directory.rglob("*.tfevents.*"))
    if not event_files:
        click.echo(f"No TensorBoard event files found in {directory}")
        return
    
    for file in sorted(event_files):
        _list_single_file(file, digits)


@main.command()
@click.argument('tensorboard_path', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(), required=True, help='Output directory path')
@click.option('--no-sort', is_flag=True, help='Disable sorting of scalar files by iteration number')
@click.option('--digits', type=int, default=6, help='Number of digits for padding iteration numbers (default: 6)')
def extract(tensorboard_path: str, output: str, no_sort: bool, digits: int):
    """Extract all data from TensorBoard log to directory structure."""
    path = Path(tensorboard_path)
    
    if not path.is_file() or 'tfevents' not in path.name:
        click.echo(f"Error: {tensorboard_path} is not a valid TensorBoard event file", err=True)
        sys.exit(1)
    
    try:
        parser = TensorBoardParser(str(path))
        
        # Extract all data
        sort_scalars = not no_sort
        parser.extract_all_to_directory(output, sort_scalars=sort_scalars, digits=digits)
        
        # Report what was extracted
        content = parser.list_all_content()
        click.echo(f"Extracted TensorBoard data to: {output}")
        
        if content['scalars']:
            click.echo(f"  - {len(content['scalars'])} scalar tag(s)")
        
        if content['images']:
            total_images = sum(len(parser.get_image_data(tag)) for tag in content['images'])
            click.echo(f"  - {len(content['images'])} image tag(s) ({total_images} total images)")
        
        if content['histograms']:
            click.echo(f"  - {len(content['histograms'])} histogram tag(s)")
        
        if content['tensors']:
            click.echo(f"  - {len(content['tensors'])} tensor tag(s)")
        
        if content['scalars'] and sort_scalars:
            click.echo("  - Scalar files sorted by iteration number")
        elif content['scalars'] and no_sort:
            click.echo("  - Scalar files not sorted (--no-sort specified)")
            
    except Exception as e:
        click.echo(f"Error extracting data: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('tensorboard_path', type=click.Path(exists=True))
@click.argument('virtual_path', type=str)
@click.option('-o', '--output', type=click.Path(), help='Output file path')
def export(tensorboard_path: str, virtual_path: str, output: Optional[str]):
    """Export a specific item from TensorBoard log."""
    path = Path(tensorboard_path)
    
    if not path.is_file() or 'tfevents' not in path.name:
        click.echo(f"Error: {tensorboard_path} is not a valid TensorBoard event file", err=True)
        sys.exit(1)
    
    try:
        parser = TensorBoardParser(str(path))
        
        # Parse the virtual path
        parts = virtual_path.strip('/').split('/')
        
        if len(parts) < 2:
            click.echo(f"Error: Invalid virtual path format: {virtual_path}", err=True)
            sys.exit(1)
        
        data_type = parts[0]
        
        if data_type == 'scalars':
            # Format: scalars/tag_name.txt
            if not parts[1].endswith('.txt'):
                click.echo(f"Error: Scalar path must end with .txt: {virtual_path}", err=True)
                sys.exit(1)
            
            tag = parts[1][:-4].replace('_', '/')  # Remove .txt and restore slashes
            scalars = parser.list_scalars()
            
            if tag not in scalars:
                click.echo(f"Error: Scalar tag '{tag}' not found", err=True)
                click.echo(f"Available scalar tags: {', '.join(scalars)}", err=True)
                sys.exit(1)
            
            data = parser.export_scalar_to_text(tag)
            
            if output:
                Path(output).write_text(data)
                click.echo(f"Exported scalar data to {output}")
            else:
                click.echo(data)
        
        elif data_type == 'images':
            # Format: images/tag_name/step.ext
            if len(parts) != 3:
                click.echo(f"Error: Image path must be in format 'images/tag/step.ext': {virtual_path}", err=True)
                sys.exit(1)
            
            tag = parts[1].replace('_', '/')
            step_file = parts[2]
            # Remove padding zeros and extract step
            step = int(step_file.split('.')[0])
            
            image_bytes = parser.export_image(tag, step)
            
            if image_bytes is None:
                click.echo(f"Error: Image not found for tag '{tag}' at step {step}", err=True)
                sys.exit(1)
            
            if output:
                Path(output).write_bytes(image_bytes)
                click.echo(f"Exported image to {output}")
            else:
                # For binary data without output file, just report success
                click.echo(f"Image data available ({len(image_bytes)} bytes). Use -o to save to file.")
        
        else:
            click.echo(f"Error: Unsupported data type: {data_type}", err=True)
            click.echo("Supported types: scalars, images", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error exporting data: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()