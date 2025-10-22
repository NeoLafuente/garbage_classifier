#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Documentation Generation Script for Garbage Classifier Project.

This script uses pdoc to automatically generate HTML documentation
from NumPy-style docstrings across the entire project, including
training scripts, utility classes, prediction modules, and the
Gradio web application interface.

The script generates comprehensive documentation for:
- Core training and prediction functionality
- Utility modules and configuration
- Custom PyTorch Lightning classes
- Gradio application and UI sections
- Carbon emissions tracking utilities

Usage
-----
Run from the project root directory:
    $ python scripts/generate_docs.py

Or using uv:
    $ uv run python scripts/generate_docs.py

The generated documentation will be saved in the docs/ directory
with an index.html entry point for easy navigation.

Notes
-----
Requires pdoc to be installed:
    $ pip install pdoc

or with uv:
    $ uv pip install pdoc
"""
__docformat__ = "numpy"

import os
import sys
import subprocess
from pathlib import Path

# Project root directory (parent of scripts folder)
PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_DIR = PROJECT_ROOT / "source"
APP_DIR = PROJECT_ROOT / "app"
OUTPUT_DIR = PROJECT_ROOT / "docs"

# Files to document (organized by category)
FILES_TO_DOCUMENT = [
    # === Core Training & Prediction ===
    SOURCE_DIR / "__init__.py",
    SOURCE_DIR / "train.py",
    SOURCE_DIR / "predict.py",
    
    # === Configuration & Utilities ===
    SOURCE_DIR / "utils" / "__init__.py",
    SOURCE_DIR / "utils" / "config.py",
    SOURCE_DIR / "utils" / "carbon_utils.py",
    
    # === Custom PyTorch Lightning Classes ===
    SOURCE_DIR / "utils" / "custom_classes" / "__init__.py",
    SOURCE_DIR / "utils" / "custom_classes" / "GarbageClassifier.py",
    SOURCE_DIR / "utils" / "custom_classes" / "GarbageDataModule.py",
    SOURCE_DIR / "utils" / "custom_classes" / "LossCurveCallback.py",
    SOURCE_DIR / "utils" / "custom_classes" / "EdaAnalyzer.py",
    SOURCE_DIR / "utils" / "custom_classes" / "EvalAnalyzer.py",
    
    # === Gradio Application ===
    APP_DIR / "__init__.py",
    APP_DIR / "main.py",
    
    # === Gradio UI Sections ===
    APP_DIR / "sections" / "__init__.py",
    APP_DIR / "sections" / "data_exploration.py",
    APP_DIR / "sections" / "model_training.py",
    APP_DIR / "sections" / "model_evaluation.py",
]


def check_pdoc_installed():
    """
    Check if pdoc is installed and accessible.
    
    Returns
    -------
    bool
        True if pdoc is installed and can be executed, False otherwise.
    
    Notes
    -----
    Attempts to run `pdoc --version` to verify installation. This is
    more reliable than checking import paths, as it confirms the CLI
    tool is properly configured.
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
    Generate HTML documentation using pdoc for all project modules.
    
    Creates comprehensive HTML documentation with the following features:
    - NumPy-style docstring parsing
    - Source code inclusion for reference
    - Automatic cross-linking between modules
    - Hierarchical organization by package structure
    
    The function performs these steps:
    1. Verifies pdoc installation
    2. Creates output directory
    3. Validates all source files exist
    4. Configures PYTHONPATH for proper imports
    5. Executes pdoc with appropriate flags
    6. Reports generation results
    
    Returns
    -------
    bool
        True if documentation was generated successfully, False otherwise.
    
    Notes
    -----
    **Environment Setup:**
    - Sets PYTHONPATH to include both source/ and app/ directories
    - Runs pdoc from the project root for correct module resolution
    
    **pdoc Configuration:**
    - Output format: HTML
    - Docstring style: NumPy
    - Includes source code in documentation
    - Generates index.html for easy navigation
    
    **File Organization:**
    The generated documentation mirrors the source structure:
    - source/ modules → docs/source/
    - app/ modules → docs/app/
    - Index page links to all modules
    
    **Error Handling:**
    Returns False and prints detailed error messages if:
    - pdoc is not installed
    - Source files are missing
    - pdoc execution fails
    
    Examples
    --------
    >>> success = generate_documentation()
    >>> if success:
    ...     print("Documentation ready in docs/")
    """
    print("=" * 60)
    print("Garbage Classifier - Documentation Generation")
    print("=" * 60)
    
    # Check if pdoc is installed
    if not check_pdoc_installed():
        print("\n✗ Error: pdoc is not installed")
        print("  Install it with: pip install pdoc")
        print("  Or with uv: uv pip install pdoc")
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
    
    print("✓ All source files found\n")
    
    # Build pdoc command for modern pdoc (v13+)
    # Set PYTHONPATH environment variable to include both source and app directories
    env = os.environ.copy()
    pythonpath_parts = [str(SOURCE_DIR), str(APP_DIR)]
    # Preserve existing PYTHONPATH if present
    if 'PYTHONPATH' in env:
        pythonpath_parts.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)
    
    cmd = [
        "pdoc",
        "-o", str(OUTPUT_DIR),  # Output directory
        "-d", "numpy",           # NumPy-style docstrings
        "--show-source",         # Include source code
    ] + [str(f) for f in FILES_TO_DOCUMENT]
    
    try:
        # Run pdoc with modified environment
        print("Running pdoc...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=str(PROJECT_ROOT)  # Run from project root
        )
        
        print("\n✓ Documentation generated successfully!")
        print("-" * 60)
        print("\nGenerated documentation structure:")
        
        # List generated files organized by category
        categories = {
            "Core Modules": [SOURCE_DIR / "__init__.py", SOURCE_DIR / "train.py", SOURCE_DIR / "predict.py"],
            "Utilities": [
                SOURCE_DIR / "utils" / "__init__.py",
                SOURCE_DIR / "utils" / "config.py",
                SOURCE_DIR / "utils" / "carbon_utils.py"
            ],
            "Custom Classes": [
                SOURCE_DIR / "utils" / "custom_classes" / "__init__.py",
                SOURCE_DIR / "utils" / "custom_classes" / "GarbageClassifier.py",
                SOURCE_DIR / "utils" / "custom_classes" / "GarbageDataModule.py",
                SOURCE_DIR / "utils" / "custom_classes" / "LossCurveCallback.py",
                SOURCE_DIR / "utils" / "custom_classes" / "EdaAnalyzer.py",
                SOURCE_DIR / "utils" / "custom_classes" / "EvalAnalyzer.py",
            ],
            "Gradio Application": [
                APP_DIR / "__init__.py",
                APP_DIR / "main.py",
                APP_DIR / "sections" / "__init__.py",
                APP_DIR / "sections" / "data_exploration.py",
                APP_DIR / "sections" / "model_training.py",
                APP_DIR / "sections" / "model_evaluation.py",
            ],
        }
        
        for category, files in categories.items():
            print(f"\n  {category}:")
            for file_path in files:
                # Try to find the generated HTML file
                rel_path = file_path.relative_to(PROJECT_ROOT)
                html_file = OUTPUT_DIR / rel_path.with_suffix('.html')
                
                if html_file.exists():
                    print(f"    ✓ {rel_path}")
                else:
                    # Search for it in case pdoc organized it differently
                    module_name = file_path.stem
                    possible_paths = list(OUTPUT_DIR.rglob(f"*{module_name}.html"))
                    if possible_paths:
                        found_path = possible_paths[0].relative_to(OUTPUT_DIR)
                        print(f"    ✓ {rel_path} → {found_path}")
                    else:
                        print(f"    ? {rel_path} (expected but not found)")
        
        print("\n" + "=" * 60)
        print("🎉 Documentation generation complete!")
        print("=" * 60)
        print(f"\nDocumentation location: {OUTPUT_DIR.absolute()}")
        print("\nTo view the documentation:")
        print("  1. Navigate to the 'docs/' directory")
        print("  2. Open 'index.html' in your web browser")
        print(f"\nQuick start:")
        
        # Check if index.html exists
        index_file = OUTPUT_DIR / "index.html"
        if index_file.exists():
            print(f"  open {index_file.absolute()}")
        else:
            # Suggest the first HTML file found
            html_files = list(OUTPUT_DIR.rglob("*.html"))
            if html_files:
                print(f"  open {html_files[0].absolute()}")
            else:
                print(f"  Check files in: {OUTPUT_DIR.absolute()}")
        
        print("=" * 60)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("\n✗ Error generating documentation")
        print("-" * 60)
        print("Error output:")
        print(e.stderr if e.stderr else e.stdout)
        print("\nCommand that failed:")
        print(" ".join(cmd))
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main entry point for the documentation generation script.
    
    Executes the documentation generation process and exits with
    appropriate status code (0 for success, 1 for failure).
    
    Returns
    -------
    None
        Exits the process with status code.
    
    Notes
    -----
    Exit codes:
    - 0: Documentation generated successfully
    - 1: Documentation generation failed
    
    This allows the script to be used in CI/CD pipelines or
    automated build systems.
    """
    success = generate_documentation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()