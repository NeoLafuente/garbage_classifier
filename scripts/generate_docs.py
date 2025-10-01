#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Documentation Generation Script for Garbage Classifier Project.

This script uses pdoc to automatically generate HTML documentation
from NumPy-style docstrings in the project modules. The documentation
is generated for training scripts and utility classes.

Usage
-----
Run from the project root directory:
    $ python scripts/generate_docs.py

Or from the scripts directory:
    $ cd scripts
    $ python generate_docs.py

The generated documentation will be saved in the docs/ directory.
"""
import os
import sys
import subprocess
from pathlib import Path

# Project root directory (parent of scripts folder)
PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_DIR = PROJECT_ROOT / "source"
OUTPUT_DIR = PROJECT_ROOT / "docs"

# Files to document (using file paths instead of module names)
FILES_TO_DOCUMENT = [
    SOURCE_DIR / "train.py",
    SOURCE_DIR / "predict.py",
    SOURCE_DIR / "utils" / "config.py",
    SOURCE_DIR / "utils" / "custom_classes" / "GarbageClassifier.py",
    SOURCE_DIR / "utils" / "custom_classes" / "GarbageDataModule.py",
    SOURCE_DIR / "utils" / "custom_classes" / "LossCurveCallback.py",
]


def check_pdoc_installed():
    """
    Check if pdoc is installed.
    
    Returns
    -------
    bool
        True if pdoc is installed, False otherwise.
    """
    try:
        subprocess.run(
            ["pdoc", "--version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def generate_documentation():
    """
    Generate HTML documentation using pdoc.
    
    Creates HTML documentation for all specified modules and saves
    them to the docs/ directory. The function handles module path
    resolution and provides detailed feedback about the generation process.
    
    Notes
    -----
    - Requires pdoc to be installed (pip install pdoc)
    - Overwrites existing documentation files
    - Changes working directory to source/ for proper module resolution
    """
    print("=" * 60)
    print("Garbage Classifier - Documentation Generation")
    print("=" * 60)
    
    # Check if pdoc is installed
    if not check_pdoc_installed():
        print("\n✗ Error: pdoc is not installed")
        print("  Install it with: pip install pdoc")
        return False
    
    print("\n✓ pdoc found")
    
    # Create docs directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"✓ Output directory ready: {OUTPUT_DIR.absolute()}")
    
    print(f"\n📝 Generating documentation for {len(FILES_TO_DOCUMENT)} files...")
    print("-" * 60)
    
    # Verify all files exist
    missing_files = [f for f in FILES_TO_DOCUMENT if not f.exists()]
    if missing_files:
        print("\n✗ Error: Some files do not exist:")
        for f in missing_files:
            print(f"  ✗ {f.relative_to(PROJECT_ROOT)}")
        return False
    
    # Build pdoc command for modern pdoc (v13+)
    # Set PYTHONPATH environment variable to include source directory
    env = os.environ.copy()
    env['PYTHONPATH'] = str(SOURCE_DIR)
    
    cmd = [
        "pdoc",
        "-o", str(OUTPUT_DIR),  # Output directory
        "-d", "numpy",  # NumPy-style docstrings
        "--show-source",  # Include source code
    ] + [str(f) for f in FILES_TO_DOCUMENT]
    
    try:
        # Run pdoc with modified environment
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=str(SOURCE_DIR)  # Run from source directory
        )
        
        print("\n✓ Documentation generated successfully!")
        print("-" * 60)
        print("\nGenerated files:")
        
        # List generated files
        for file_path in FILES_TO_DOCUMENT:
            # pdoc generates HTML files with the same name as the source file
            html_file = OUTPUT_DIR / file_path.relative_to(SOURCE_DIR).with_suffix('.html')
            if html_file.exists():
                print(f"  ✓ {html_file.relative_to(PROJECT_ROOT)}")
            else:
                # Sometimes pdoc may organize files differently
                module_name = file_path.stem
                possible_paths = list(OUTPUT_DIR.rglob(f"{module_name}.html"))
                if possible_paths:
                    print(f"  ✓ {possible_paths[0].relative_to(PROJECT_ROOT)}")
                else:
                    print(f"  ? {html_file.relative_to(PROJECT_ROOT)} (expected but not found)")
        
        print("\n" + "=" * 60)
        print("🎉 Documentation generation complete!")
        print("=" * 60)
        print(f"\nTo view the documentation:")
        print(f"  1. Navigate to the 'docs/' directory")
        print(f"  2. Open any HTML file in your web browser")
        print(f"\nQuick start:")
        print(f"  Open: {OUTPUT_DIR.absolute()}/index.html")
        print("=" * 60)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("\n✗ Error generating documentation")
        print("-" * 60)
        print("Error output:")
        print(e.stderr if e.stderr else e.stdout)
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


def main():
    """
    Main entry point for the documentation generation script.
    """
    success = generate_documentation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()