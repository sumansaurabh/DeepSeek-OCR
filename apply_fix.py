#!/usr/bin/env python3
"""
Script to apply the fix for Issue #7: LlamaFlashAttention2 ImportError

This script downloads the model files, applies the patch, and provides
instructions for using the fixed model.

Usage:
    python apply_fix.py --model-path /path/to/save/model
"""

import argparse
import os
import sys
from pathlib import Path


def apply_patch_to_file(file_path: Path) -> bool:
    """
    Apply the fix directly to the modeling_deepseekv2.py file.

    Returns:
        bool: True if patch was applied successfully, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Check if already patched
        if 'Try to import LlamaFlashAttention2, fall back to None' in content:
            print(f"✓ File {file_path} is already patched!")
            return True

        # Original import block
        old_import = """from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2
)"""

        # New import block with error handling
        new_import = """# Try to import LlamaFlashAttention2, fall back to None if not available
# This handles transformers>=4.57.0 where LlamaFlashAttention2 was removed
try:
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaFlashAttention2
except ImportError:
    from transformers.models.llama.modeling_llama import LlamaAttention
    LlamaFlashAttention2 = None
"""

        # Apply first fix
        if old_import in content:
            content = content.replace(old_import, new_import)
            print("✓ Applied import fix")
        else:
            print("⚠ Warning: Could not find original import block")
            return False

        # Fix ATTENTION_CLASSES dictionary
        old_attention = '    "mha_flash_attention_2": LlamaFlashAttention2'
        new_attention = '    # Use LlamaFlashAttention2 if available, otherwise fall back to LlamaAttention\n    "mha_flash_attention_2": LlamaFlashAttention2 if LlamaFlashAttention2 is not None else LlamaAttention'

        if old_attention in content:
            content = content.replace(old_attention, new_attention)
            print("✓ Applied ATTENTION_CLASSES fix")
        else:
            print("⚠ Warning: Could not find ATTENTION_CLASSES entry")

        # Write the patched content back
        with open(file_path, 'w') as f:
            f.write(content)

        print(f"✓ Successfully patched {file_path}")
        return True

    except Exception as e:
        print(f"✗ Error applying patch: {e}")
        return False


def download_and_patch_model(model_path: str) -> bool:
    """
    Download the model and apply the patch.

    Args:
        model_path: Path where the model should be saved

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from huggingface_hub import snapshot_download

        print(f"Downloading model to {model_path}...")
        snapshot_download(
            repo_id="deepseek-ai/DeepSeek-OCR",
            local_dir=model_path,
            local_dir_use_symlinks=False
        )
        print("✓ Model downloaded successfully")

        # Apply patch
        modeling_file = Path(model_path) / "modeling_deepseekv2.py"
        if not modeling_file.exists():
            print(f"✗ Error: {modeling_file} not found")
            return False

        return apply_patch_to_file(modeling_file)

    except ImportError:
        print("✗ Error: huggingface_hub not installed. Install with: pip install huggingface-hub")
        return False
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return False


def print_usage_instructions(model_path: str):
    """Print instructions for using the patched model."""
    print("\n" + "="*60)
    print("FIX APPLIED SUCCESSFULLY!")
    print("="*60)
    print("\nYou can now use the model with the following code:")
    print("\n```python")
    print("from transformers import AutoModel, AutoTokenizer")
    print("import torch")
    print()
    print(f"model_name = '{model_path}'")
    print()
    print("tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)")
    print("model = AutoModel.from_pretrained(")
    print("    model_name,")
    print("    _attn_implementation='flash_attention_2',")
    print("    trust_remote_code=True,")
    print("    use_safetensors=True")
    print(")")
    print("model = model.eval().cuda().to(torch.bfloat16)")
    print()
    print("# Your inference code here...")
    print("```")
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Apply fix for DeepSeek-OCR Issue #7: LlamaFlashAttention2 ImportError"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path where the model should be downloaded and patched"
    )
    parser.add_argument(
        "--patch-existing",
        type=str,
        help="Path to existing model directory to patch (skips download)"
    )

    args = parser.parse_args()

    if args.patch_existing:
        # Patch existing model
        modeling_file = Path(args.patch_existing) / "modeling_deepseekv2.py"
        if not modeling_file.exists():
            print(f"✗ Error: {modeling_file} not found")
            sys.exit(1)

        if apply_patch_to_file(modeling_file):
            print_usage_instructions(args.patch_existing)
            sys.exit(0)
        else:
            sys.exit(1)

    elif args.model_path:
        # Download and patch
        if download_and_patch_model(args.model_path):
            print_usage_instructions(args.model_path)
            sys.exit(0)
        else:
            sys.exit(1)

    else:
        print("Error: Please specify either --model-path or --patch-existing")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
