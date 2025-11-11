"""
Safe wrapper for DeepSeek-OCR that automatically handles the LlamaFlashAttention2 import issue.

This script automatically applies the fix for GitHub Issue #7 before loading the model.
"""

from transformers import AutoModel, AutoTokenizer
import torch
import os
import sys
from pathlib import Path

# Add parent directory to path to import the fix script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def apply_flash_attention_fix():
    """Apply the flash attention import fix if needed."""
    try:
        # Try importing the fix script
        import fix_flash_attention_import as fixer
        
        print("Checking if model files need patching...")
        
        # Find cache directory
        cache_dir = fixer.find_huggingface_cache()
        if cache_dir is None:
            print("Warning: Could not find Hugging Face cache directory.")
            print("If you encounter import errors, please run fix_flash_attention_import.py manually.")
            return
        
        # Find model files
        model_name = 'deepseek-ai/DeepSeek-OCR'
        model_files = fixer.find_model_files(model_name, cache_dir)
        
        if not model_files:
            print("Model not yet downloaded. Will download on first use.")
            print("If you encounter import errors, please run: python fix_flash_attention_import.py")
            return
        
        # Check which files need patching
        files_to_patch = []
        for file_path in model_files:
            if fixer.check_file_needs_patching(file_path):
                files_to_patch.append(file_path)
        
        if not files_to_patch:
            print("✓ Model files are already compatible!")
            return
        
        print(f"Found {len(files_to_patch)} file(s) that need patching.")
        print("Applying patches...")
        
        # Apply patches
        patched_count = 0
        for file_path in files_to_patch:
            if fixer.patch_file(file_path, dry_run=False):
                patched_count += 1
        
        print(f"✓ Successfully patched {patched_count} file(s)")
        
    except ImportError:
        print("Warning: Could not import fix_flash_attention_import.py")
        print("If you encounter import errors, please run: python fix_flash_attention_import.py")
    except Exception as e:
        print(f"Warning: Error while applying fix: {e}")
        print("If you encounter import errors, please run: python fix_flash_attention_import.py")


def main():
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    model_name = 'deepseek-ai/DeepSeek-OCR'
    
    # Apply fix before loading model
    apply_flash_attention_fix()
    
    print("\nLoading tokenizer and model...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name, 
            _attn_implementation='flash_attention_2', 
            trust_remote_code=True, 
            use_safetensors=True
        )
        model = model.eval().cuda().to(torch.bfloat16)
        
        print("✓ Model loaded successfully!")
        
        # Example usage
        # prompt = "<image>\\nFree OCR. "
        prompt = "<image>\\n<|grounding|>Convert the document to markdown. "
        image_file = 'your_image.jpg'
        output_path = 'your/output/dir'
        
        print("\nModel is ready for inference!")
        print(f"Example usage:")
        print(f"  prompt: {prompt}")
        print(f"  image_file: {image_file}")
        print(f"  output_path: {output_path}")
        
        # Uncomment to run inference:
        # res = model.infer(
        #     tokenizer, 
        #     prompt=prompt, 
        #     image_file=image_file, 
        #     output_path=output_path, 
        #     base_size=1024, 
        #     image_size=640, 
        #     crop_mode=True, 
        #     save_results=True, 
        #     test_compress=True
        # )
        
    except ImportError as e:
        if 'LlamaFlashAttention2' in str(e):
            print("\n" + "="*60)
            print("ERROR: LlamaFlashAttention2 import failed!")
            print("="*60)
            print("\nThis error occurs when using incompatible transformers versions.")
            print("\nSolution 1 (Recommended): Use compatible transformers version")
            print("  pip install 'transformers>=4.46.3,<4.50.0'")
            print("\nSolution 2: Run the fix script manually")
            print("  python fix_flash_attention_import.py")
            print("\nSolution 3: Downgrade transformers")
            print("  pip install transformers==4.46.3")
            print("="*60)
            sys.exit(1)
        else:
            raise
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == '__main__':
    main()
