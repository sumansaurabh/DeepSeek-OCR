#!/usr/bin/env python3
"""
Fix for GitHub Issue #7: ImportError with LlamaFlashAttention2

This script patches the DeepSeek-OCR model files downloaded from Hugging Face
to handle the missing LlamaFlashAttention2 import in newer transformers versions.

Usage:
    python fix_flash_attention_import.py [--model-name MODEL_NAME] [--cache-dir CACHE_DIR]

The script will:
1. Locate the cached model files
2. Find modeling files that import LlamaFlashAttention2
3. Patch them with try-except blocks to handle the missing import gracefully
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import Optional, List


def find_huggingface_cache() -> Optional[Path]:
    """Find the Hugging Face cache directory."""
    # Check environment variable first
    cache_dir = os.environ.get('HF_HOME')
    if cache_dir:
        return Path(cache_dir)
    
    # Check default locations
    default_cache = Path.home() / '.cache' / 'huggingface'
    if default_cache.exists():
        return default_cache
    
    # Alternative location
    alt_cache = Path.home() / '.cache' / 'torch' / 'hub'
    if alt_cache.exists():
        return alt_cache
    
    return None


def find_model_files(model_name: str, cache_dir: Optional[Path] = None) -> List[Path]:
    """Find all Python files in the model cache directory."""
    if cache_dir is None:
        cache_dir = find_huggingface_cache()
    
    if cache_dir is None:
        print("Error: Could not find Hugging Face cache directory.")
        return []
    
    # Look for model files in the cache
    model_files = []
    
    # Search in hub directory
    hub_dir = cache_dir / 'hub'
    if hub_dir.exists():
        # Find directories matching the model name
        for model_dir in hub_dir.iterdir():
            if model_dir.is_dir() and model_name.replace('/', '--') in model_dir.name:
                # Find all Python files
                for py_file in model_dir.rglob('*.py'):
                    model_files.append(py_file)
    
    # Also check modules directory
    modules_dir = cache_dir / 'modules'
    if modules_dir.exists():
        for py_file in modules_dir.rglob('*.py'):
            model_files.append(py_file)
    
    return model_files


def check_file_needs_patching(file_path: Path) -> bool:
    """Check if a file needs patching for LlamaFlashAttention2 import."""
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Check if file imports LlamaFlashAttention2
        if 'LlamaFlashAttention2' in content:
            # Check if it's already patched
            if 'try:' in content and 'except ImportError:' in content and 'LlamaFlashAttention2' in content:
                # Might already be patched, check more carefully
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'from transformers.models.llama.modeling_llama import' in line and 'LlamaFlashAttention2' in line:
                        # Check if there's a try-except around it
                        if i > 0 and 'try:' not in '\n'.join(lines[max(0, i-5):i]):
                            return True
            else:
                return True
        
        return False
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False


def patch_file(file_path: Path, dry_run: bool = False) -> bool:
    """Patch a file to handle missing LlamaFlashAttention2 import."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # Pattern 1: Single line import
        pattern1 = r'from transformers\.models\.llama\.modeling_llama import ([^)]*LlamaFlashAttention2[^)]*)'
        
        def replace_single_import(match):
            imports = match.group(1)
            # Split imports and filter out LlamaFlashAttention2
            import_list = [imp.strip() for imp in imports.split(',')]
            other_imports = [imp for imp in import_list if 'LlamaFlashAttention2' not in imp]
            
            if other_imports:
                result = f'from transformers.models.llama.modeling_llama import {", ".join(other_imports)}\n'
            else:
                result = ''
            
            result += '''try:
    from transformers.models.llama.modeling_llama import LlamaFlashAttention2
except ImportError:
    LlamaFlashAttention2 = None  # Not available in this transformers version'''
            
            return result
        
        content = re.sub(pattern1, replace_single_import, content)
        
        # Pattern 2: Multi-line import with parentheses
        pattern2 = r'from transformers\.models\.llama\.modeling_llama import \(((?:[^)]|\n)*LlamaFlashAttention2(?:[^)]|\n)*)\)'
        
        def replace_multi_import(match):
            imports = match.group(1)
            # Split imports and filter out LlamaFlashAttention2
            # Handle both comma-separated and newline-separated imports
            import_list = [imp.strip() for imp in re.split(r'[,\n]+', imports) if imp.strip()]
            other_imports = [imp for imp in import_list if 'LlamaFlashAttention2' not in imp]
            
            if other_imports:
                joined_imports = ",\n    ".join(other_imports)
                result = f'from transformers.models.llama.modeling_llama import (\n    {joined_imports}\n)\n'
            else:
                result = ''
            
            result += '''try:
    from transformers.models.llama.modeling_llama import LlamaFlashAttention2
except ImportError:
    LlamaFlashAttention2 = None  # Not available in this transformers version'''
            
            return result
        
        content = re.sub(pattern2, replace_multi_import, content, flags=re.MULTILINE)
        
        if content != original_content:
            if dry_run:
                print(f"Would patch: {file_path}")
                return True
            else:
                # Create backup
                backup_path = file_path.with_suffix('.py.backup')
                if not backup_path.exists():
                    file_path.rename(backup_path)
                    print(f"Created backup: {backup_path}")
                
                # Write patched content
                file_path.write_text(content, encoding='utf-8')
                print(f"✓ Patched: {file_path}")
                return True
        
        return False
    
    except Exception as e:
        print(f"Error patching {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Fix LlamaFlashAttention2 import error in DeepSeek-OCR model files'
    )
    parser.add_argument(
        '--model-name',
        default='deepseek-ai/DeepSeek-OCR',
        help='Model name (default: deepseek-ai/DeepSeek-OCR)'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        help='Hugging Face cache directory (auto-detected if not specified)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be patched without making changes'
    )
    
    args = parser.parse_args()
    
    print(f"DeepSeek-OCR Flash Attention Import Fixer")
    print(f"=" * 50)
    print(f"Model: {args.model_name}")
    
    # Find cache directory
    cache_dir = args.cache_dir or find_huggingface_cache()
    if cache_dir is None:
        print("\nError: Could not find Hugging Face cache directory.")
        print("Please specify it with --cache-dir or set HF_HOME environment variable.")
        return 1
    
    print(f"Cache directory: {cache_dir}")
    
    # Find model files
    print(f"\nSearching for model files...")
    model_files = find_model_files(args.model_name, cache_dir)
    
    if not model_files:
        print(f"\nWarning: No model files found for {args.model_name}")
        print("The model may not be downloaded yet. Please run your script first to download the model,")
        print("then run this fix script again.")
        return 0
    
    print(f"Found {len(model_files)} Python files")
    
    # Check which files need patching
    files_to_patch = []
    for file_path in model_files:
        if check_file_needs_patching(file_path):
            files_to_patch.append(file_path)
    
    if not files_to_patch:
        print("\n✓ No files need patching. Your model is already compatible!")
        return 0
    
    print(f"\nFiles that need patching: {len(files_to_patch)}")
    for file_path in files_to_patch:
        print(f"  - {file_path.name}")
    
    if args.dry_run:
        print("\n[DRY RUN] No changes made. Run without --dry-run to apply patches.")
        return 0
    
    # Apply patches
    print(f"\nApplying patches...")
    patched_count = 0
    for file_path in files_to_patch:
        if patch_file(file_path, dry_run=args.dry_run):
            patched_count += 1
    
    print(f"\n{'=' * 50}")
    print(f"✓ Successfully patched {patched_count} file(s)")
    print(f"\nYou can now run your DeepSeek-OCR scripts without import errors!")
    print(f"Backups of original files were created with .backup extension.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
