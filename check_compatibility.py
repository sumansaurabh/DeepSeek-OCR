#!/usr/bin/env python3
"""
Compatibility checker for DeepSeek-OCR
Verifies that the installed transformers version is compatible with DeepSeek-OCR.

This script checks for the LlamaFlashAttention2 import issue that occurs
with transformers >= 4.47.0

Usage:
    python check_compatibility.py
"""

import sys
import importlib.util


def check_transformers_version():
    """Check if transformers is installed and get version."""
    try:
        import transformers
        version = transformers.__version__
        print(f"✓ transformers version: {version}")
        return version
    except ImportError:
        print("✗ transformers is not installed")
        print("  Install with: pip install transformers==4.46.3")
        return None


def check_llama_flash_attention():
    """Check if LlamaFlashAttention2 can be imported."""
    try:
        from transformers.models.llama.modeling_llama import LlamaFlashAttention2
        print("✓ LlamaFlashAttention2 is available")
        return True
    except ImportError as e:
        print(f"✗ LlamaFlashAttention2 import failed: {e}")
        return False


def check_torch():
    """Check if PyTorch is installed."""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        print(f"✓ torch version: {version}")
        if cuda_available:
            print(f"✓ CUDA is available: {torch.version.cuda}")
        else:
            print("⚠ CUDA is not available (CPU-only mode)")
        return True
    except ImportError:
        print("✗ torch is not installed")
        return False


def check_flash_attn():
    """Check if flash-attn is installed (optional)."""
    try:
        import flash_attn
        version = flash_attn.__version__
        print(f"✓ flash-attn version: {version}")
        return True
    except ImportError:
        print("⚠ flash-attn is not installed (optional, but recommended for performance)")
        print("  Install with: pip install flash-attn==2.7.3 --no-build-isolation")
        return False


def main():
    """Run all compatibility checks."""
    print("=" * 60)
    print("DeepSeek-OCR Compatibility Check")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # Check transformers
    print("[1/4] Checking transformers...")
    transformers_version = check_transformers_version()
    if transformers_version is None:
        all_passed = False
    else:
        # Parse version
        try:
            major, minor, patch = transformers_version.split('.')[:3]
            version_tuple = (int(major), int(minor), int(patch.split('+')[0]))
            
            # Check if version is compatible
            if version_tuple >= (4, 47, 0):
                print("⚠ WARNING: transformers version >= 4.47.0 may cause compatibility issues")
                print("  Recommended version: 4.46.3")
                print("  Downgrade with: pip install transformers==4.46.3")
        except:
            pass
    print()
    
    # Check LlamaFlashAttention2
    print("[2/4] Checking LlamaFlashAttention2...")
    if not check_llama_flash_attention():
        all_passed = False
        print("  FIX: Install compatible transformers version:")
        print("       pip install transformers==4.46.3")
    print()
    
    # Check PyTorch
    print("[3/4] Checking PyTorch...")
    if not check_torch():
        all_passed = False
    print()
    
    # Check flash-attn (optional)
    print("[4/4] Checking flash-attn (optional)...")
    check_flash_attn()
    print()
    
    # Summary
    print("=" * 60)
    if all_passed:
        print("✓ All required dependencies are compatible!")
        print("  You can now use DeepSeek-OCR with transformers.")
    else:
        print("✗ Some compatibility issues detected.")
        print("  Please follow the instructions above to fix them.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
