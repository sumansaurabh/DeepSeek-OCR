#!/usr/bin/env python3
"""
Compatibility checker for DeepSeek-OCR
Validates that the environment has compatible versions of required packages.

This script addresses GitHub Issue #7:
ImportError: cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama.modeling_llama'

Usage:
    python check_compatibility.py
"""

import sys
import importlib.metadata
from packaging import version


def check_package_version(package_name, required_version=None, max_version=None):
    """
    Check if a package is installed and optionally verify its version.
    
    Args:
        package_name: Name of the package to check
        required_version: Minimum required version (optional)
        max_version: Maximum compatible version (optional)
    
    Returns:
        tuple: (is_compatible, installed_version, message)
    """
    try:
        installed_version = importlib.metadata.version(package_name)
        
        if required_version and version.parse(installed_version) < version.parse(required_version):
            return False, installed_version, f"Version too old (minimum: {required_version})"
        
        if max_version and version.parse(installed_version) > version.parse(max_version):
            return False, installed_version, f"Version too new (maximum: {max_version})"
        
        return True, installed_version, "OK"
    
    except importlib.metadata.PackageNotFoundError:
        return False, None, "Not installed"


def check_transformers_import():
    """
    Check if LlamaFlashAttention2 can be imported from transformers.
    This is the specific issue reported in GitHub Issue #7.
    
    Returns:
        tuple: (can_import, error_message)
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaFlashAttention2
        return True, None
    except ImportError as e:
        return False, str(e)


def main():
    """Main compatibility check function."""
    print("=" * 70)
    print("DeepSeek-OCR Compatibility Checker")
    print("=" * 70)
    print()
    
    all_compatible = True
    
    # Check Python version
    print("1. Checking Python version...")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"   Installed: Python {python_version}")
    if sys.version_info < (3, 8):
        print("   ❌ ERROR: Python 3.8 or higher is required")
        all_compatible = False
    else:
        print("   ✓ OK")
    print()
    
    # Check transformers version (critical for Issue #7)
    print("2. Checking transformers version...")
    is_compatible, installed_ver, msg = check_package_version(
        "transformers",
        required_version="4.46.0",
        max_version="4.51.9"
    )
    
    if installed_ver:
        print(f"   Installed: transformers {installed_ver}")
    else:
        print(f"   ❌ transformers is not installed")
        all_compatible = False
    
    if not is_compatible and installed_ver:
        print(f"   ❌ ERROR: {msg}")
        print(f"   Recommended: transformers==4.46.3")
        print()
        print("   To fix this issue:")
        print("   pip uninstall transformers -y")
        print("   pip install transformers==4.46.3")
        all_compatible = False
    elif is_compatible:
        print(f"   ✓ OK")
    print()
    
    # Check if LlamaFlashAttention2 can be imported
    print("3. Checking LlamaFlashAttention2 import (Issue #7)...")
    can_import, error_msg = check_transformers_import()
    
    if can_import:
        print("   ✓ LlamaFlashAttention2 can be imported successfully")
    else:
        print(f"   ❌ ERROR: Cannot import LlamaFlashAttention2")
        print(f"   Error: {error_msg}")
        print()
        print("   This is the issue reported in GitHub Issue #7.")
        print("   Solution: Install transformers==4.46.3")
        print()
        print("   Run these commands:")
        print("   pip uninstall transformers -y")
        print("   pip install transformers==4.46.3")
        all_compatible = False
    print()
    
    # Check other required packages
    print("4. Checking other required packages...")
    other_packages = [
        ("torch", "2.0.0", None),
        ("tokenizers", None, None),
        ("einops", None, None),
        ("Pillow", None, None),
        ("numpy", None, None),
    ]
    
    for pkg_name, min_ver, max_ver in other_packages:
        is_compat, inst_ver, msg = check_package_version(pkg_name, min_ver, max_ver)
        if inst_ver:
            status = "✓" if is_compat else "❌"
            print(f"   {status} {pkg_name}: {inst_ver} ({msg})")
            if not is_compat:
                all_compatible = False
        else:
            print(f"   ❌ {pkg_name}: Not installed")
            all_compatible = False
    print()
    
    # Check flash-attn (optional but recommended)
    print("5. Checking flash-attn (optional but recommended)...")
    is_compat, inst_ver, msg = check_package_version("flash-attn")
    if inst_ver:
        print(f"   ✓ flash-attn: {inst_ver}")
    else:
        print(f"   ⚠ flash-attn: Not installed (optional, but improves performance)")
        print(f"   To install: pip install flash-attn==2.7.3 --no-build-isolation")
    print()
    
    # Final summary
    print("=" * 70)
    if all_compatible:
        print("✓ All compatibility checks passed!")
        print("You can now run DeepSeek-OCR without issues.")
        return 0
    else:
        print("❌ Compatibility issues detected!")
        print()
        print("Please fix the issues above before running DeepSeek-OCR.")
        print()
        print("Quick fix for Issue #7 (LlamaFlashAttention2 ImportError):")
        print("  pip uninstall transformers -y")
        print("  pip install transformers==4.46.3")
        print()
        print("Or install all requirements:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
