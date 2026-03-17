import os
import json
import argparse
import sys
from pathlib import Path
from typing import List

# Add the current directory to Python path to find poselib module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poselib.skeleton.skeleton3d import SkeletonMotion


def find_fbx_files(input_dir: str) -> List[Path]:
    """Find all .fbx files in the given directory."""
    dir_path = Path(input_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    # Recursively find all .fbx files
    fbx_files = sorted(dir_path.rglob("*.fbx"))
    return fbx_files


def get_output_path(input_path: Path, output_dir: str, preserve_structure: bool = True) -> str:
    """Generate output path for the converted motion.

    Args:
        input_path: Path to the input FBX file
        output_dir: Root output directory
        preserve_structure: If True, preserve the subdirectory structure relative to input root
    """
    if preserve_structure:
        # Get relative path from input directory root
        rel_path = input_path.relative_to(input_path.parent.parent)
        # Replace .fbx extension with nothing (or another extension if needed)
        output_path = Path(output_dir) / rel_path.with_suffix('')
    else:
        # Just use the filename
        output_path = Path(output_dir) / input_path.stem

    return str(output_path)


def convert_single_file(
    input_path: str,
    output_path: str,
    root_joint: str,
    fps: int
) -> bool:
    """Convert a single FBX file to PoseLib format.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"[red]Error: Input file '{input_path}' does not exist[/red]")
            return False

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Import fbx file
        motion = SkeletonMotion.from_fbx(
            fbx_file_path=input_path,
            root_joint=root_joint,
            fps=fps
        )

        # Save motion in the specified format
        motion.to_retarget_motion_file(output_path)
        return True

    except Exception as e:
        print(f"[red]Error converting '{input_path}': {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Batch import FBX files and convert to PoseLib format'
    )

    parser.add_argument(
        '--input_dir', '-i',
        required=True,
        help='Directory containing FBX files to convert'
    )

    parser.add_argument(
        '--output_dir', '-o',
        required=True,
        help='Output directory for the converted motion files'
    )

    parser.add_argument(
        '--root-joint', '-r',
        default='Hips',
        help='Root joint name (default: pelvis)'
    )

    parser.add_argument(
        '--fps', '-f',
        type=int,
        default=120,
        help='FPS for the motion (default: 120)'
    )

    parser.add_argument(
        '--preserve-structure',
        action='store_true',
        help='Preserve the subdirectory structure from input to output'
    )

    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search for FBX files recursively in subdirectories'
    )

    parser.add_argument(
        '--extension',
        default='pkl',
        help='Output file extension (e.g., "pkl" or "npz"). If empty, uses default format'
    )

    args = parser.parse_args()

    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"[red]Error: Input directory '{args.input_dir}' does not exist[/red]")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")

    # Find all FBX files
    input_dir_path = Path(args.input_dir)

    if args.recursive:
        fbx_files = sorted(input_dir_path.rglob("*.fbx"))
        print(f"Searching recursively for FBX files...")
    else:
        fbx_files = sorted(input_dir_path.glob("*.FBX"))
        print(f"Searching for FBX files in {args.input_dir}...")

    if not fbx_files:
        print("[yellow]No FBX files found in the specified directory.[/yellow]")
        return

    print(f"Found {len(fbx_files)} FBX file(s) to convert:")
    for idx, f in enumerate(fbx_files[:10], 1):
        rel_path = f.relative_to(input_dir_path)
        print(f"  {idx}. {rel_path}")
    if len(fbx_files) > 10:
        print(f"  ... and {len(fbx_files) - 10} more")

    # Batch conversion
    print(f"\n[blue]Starting batch conversion...[/blue]")
    print(f"Root joint: {args.root_joint}")
    print(f"FPS: {args.fps}")
    print(f"{'=' * 60}")

    success_count = 0
    fail_count = 0

    for file_idx, fbx_file in enumerate(fbx_files, start=1):
        try:
            # Get relative path for display
            rel_path = fbx_file.relative_to(input_dir_path)
            print(f"\n[{file_idx}/{len(fbx_files)}] Processing: {rel_path}")

            # Generate output path
            if args.preserve_structure:
                # Preserve subdirectory structure
                output_path = Path(args.output_dir) / rel_path
                # Remove .fbx extension
                output_path = output_path.with_suffix('')
                # Add custom extension if specified
                if args.extension:
                    output_path = output_path.with_suffix(f'.{args.extension}')
            else:
                # Flat structure - just use filename
                output_path = Path(args.output_dir) / fbx_file.stem
                if args.extension:
                    output_path = output_path.with_suffix(f'.{args.extension}')

            # Convert the file
            if convert_single_file(
                input_path=str(fbx_file),
                output_path=str(output_path),
                root_joint=args.root_joint,
                fps=args.fps
            ):
                print(f"[green]✓ Successfully converted to: {output_path}[/green]")
                success_count += 1
            else:
                fail_count += 1

        except Exception as e:
            print(f"[red]✗ Failed: {e}[/red]")
            fail_count += 1
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"[blue]Batch conversion completed![/blue]")
    print(f"Total files: {len(fbx_files)}")
    print(f"[green]Successfully converted: {success_count}[/green]")
    if fail_count > 0:
        print(f"[red]Failed: {fail_count}[/red]")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
