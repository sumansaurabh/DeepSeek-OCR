#!/usr/bin/env python3
"""
Fix for transformers compatibility issue with DeepSeek-OCR
Addresses GitHub Issue #7: ImportError: cannot import name 'LlamaFlashAttention2'

This script patches the downloaded model files from Hugging Face to work with
newer versions of transformers (4.51.0+) where LlamaFlashAttention2 was removed.

Usage:
    python fix_transformers_compatibility.py [--model-path MODEL_PATH]
"""

import os
import sys
import argparse
from pathlib import Path
import re


def find_model_cache_dir(model_name="deepseek-ai/DeepSeek-OCR"):
    """Find the Hugging Face cache directory for the model."""
    # Common cache locations
    cache_dirs = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules",
    ]
    
    model_dirs = []
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            # Look for model directories
            for item in cache_dir.iterdir():
                if item.is_dir() and "deepseek" in item.name.lower() and "ocr" in item.name.lower():
                    model_dirs.append(item)
    
    return model_dirs


def patch_modeling_file(file_path):
    """Patch a modeling file to handle missing LlamaFlashAttention2."""
    print(f"Patching file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern 1: Direct import of LlamaFlashAttention2
        pattern1 = r'from transformers\.models\.llama\.modeling_llama import ([^)]+LlamaFlashAttention2[^)]*)'
        if re.search(pattern1, content):
            # Replace with try-except block
            replacement1 = r'''try:
    from transformers.models.llama.modeling_llama import \1
except ImportError:
    # LlamaFlashAttention2 not available in transformers >= 4.51.0
    LlamaFlashAttention2 = None'''
            content = re.sub(pattern1, replacement1, content)
            print("  ✓ Patched LlamaFlashAttention2 import")
        
        # Pattern 2: Multi-line import statement
        pattern2 = r'from transformers\.models\.llama\.modeling_llama import \(\s*([^)]*LlamaFlashAttention2[^)]*)\s*\)'
        if re.search(pattern2, content, re.MULTILINE | re.DOTALL):
            def replace_multiline(match):
                imports = match.group(1)
                return f'''try:
    from transformers.models.llama.modeling_llama import (
        {imports}
    )
except ImportError:
    # LlamaFlashAttention2 not available in transformers >= 4.51.0
    from transformers.models.llama.modeling_llama import (
        {imports.replace('LlamaFlashAttention2,', '').replace(', LlamaFlashAttention2', '').replace('LlamaFlashAttention2', '')}
    )
    LlamaFlashAttention2 = None'''
            
            content = re.sub(pattern2, replace_multiline, content, flags=re.MULTILINE | re.DOTALL)
            print("  ✓ Patched multi-line LlamaFlashAttention2 import")
        
        # Pattern 3: Check for usage of LlamaFlashAttention2 and add None check
        if 'LlamaFlashAttention2' in content and original_content != content:
            # Add a helper check at the beginning of the file if not already present
            if 'def _get_attention_class' not in content:
                helper_code = '''

# Compatibility helper for missing LlamaFlashAttention2
def _get_attention_class(attn_implementation):
    """Get appropriate attention class based on implementation type."""
    if attn_implementation == "flash_attention_2":
        if LlamaFlashAttention2 is None:
            import warnings
            warnings.warn(
                "flash_attention_2 is not available in this transformers version. "
                "Falling back to eager attention. Consider using transformers<4.51.0 "
                "or install flash-attn separately.",
                UserWarning
            )
            from transformers.models.llama.modeling_llama import LlamaAttention
            return LlamaAttention
        return LlamaFlashAttention2
    from transformers.models.llama.modeling_llama import LlamaAttention
    return LlamaAttention

'''
                # Insert after imports
                import_end = content.rfind('import ')
                if import_end != -1:
                    next_newline = content.find('\n\n', import_end)
                    if next_newline != -1:
                        content = content[:next_newline] + helper_code + content[next_newline:]
                        print("  ✓ Added compatibility helper function")
        
        # Only write if changes were made
        if content != original_content:
            # Create backup
            backup_path = str(file_path) + '.backup'
            if not os.path.exists(backup_path):
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                print(f"  ✓ Created backup: {backup_path}")
            
            # Write patched content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ Successfully patched {file_path}")
            return True
        else:
            print(f"  ℹ No LlamaFlashAttention2 imports found in {file_path}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error patching {file_path}: {e}")
        return False


def find_and_patch_files(model_path=None):
    """Find and patch all modeling files in the model cache."""
    patched_count = 0
    
    if model_path:
        model_dirs = [Path(model_path)]
    else:
        model_dirs = find_model_cache_dir()
    
    if not model_dirs:
        print("⚠ Could not find DeepSeek-OCR model cache directory.")
        print("Please specify the model path manually using --model-path")
        return 0
    
    print(f"Found {len(model_dirs)} potential model directory(ies)")
    
    for model_dir in model_dirs:
        print(f"\nSearching in: {model_dir}")
        
        # Look for Python files that might contain the problematic import
        for py_file in model_dir.rglob("*.py"):
            if py_file.name.startswith("modeling_"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'LlamaFlashAttention2' in content:
                            if patch_modeling_file(py_file):
                                patched_count += 1
                except Exception as e:
                    print(f"  ✗ Error reading {py_file}: {e}")
    
    return patched_count


def check_transformers_version():
    """Check the installed transformers version."""
    try:
        import transformers
        version = transformers.__version__
        print(f"Installed transformers version: {version}")
        
        # Parse version
        major, minor = map(int, version.split('.')[:2])
        
        if major > 4 or (major == 4 and minor >= 51):
            print("⚠ Warning: transformers >= 4.51.0 detected.")
            print("  LlamaFlashAttention2 is not available in this version.")
            print("  This script will patch the model files to work around this issue.")
            return True
        else:
            print("✓ transformers version is compatible (< 4.51.0)")
            return False
            
    except ImportError:
        print("✗ transformers is not installed")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Fix transformers compatibility for DeepSeek-OCR"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the model directory (optional, will auto-detect if not provided)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force patching even if transformers version is compatible"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DeepSeek-OCR Transformers Compatibility Patcher")
    print("=" * 60)
    print()
    
    needs_patch = check_transformers_version()
    
    if not needs_patch and not args.force:
        print("\n✓ No patching needed. Your transformers version is compatible.")
        print("  Use --force to patch anyway.")
        return 0
    
    print("\nSearching for model files to patch...")
    patched_count = find_and_patch_files(args.model_path)
    
    print("\n" + "=" * 60)
    if patched_count > 0:
        print(f"✓ Successfully patched {patched_count} file(s)")
        print("\nYou can now use DeepSeek-OCR with your transformers version.")
        print("Note: Backups were created with .backup extension")
    else:
        print("ℹ No files needed patching or model not found.")
        print("\nIf you're still experiencing issues:")
        print("1. Make sure you've downloaded the model at least once")
        print("2. Try specifying the model path with --model-path")
        print("3. Consider downgrading transformers: pip install 'transformers<4.51.0'")
    print("=" * 60)
    
    return 0 if patched_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
