#!/usr/bin/env python3
"""
Test script to verify DeepSeek-OCR model loading
Tests compatibility with different transformers versions
"""

import sys
import os


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    missing_deps = []
    
    try:
        import transformers
        print(f"✓ transformers {transformers.__version__}")
    except ImportError:
        print("✗ transformers not installed")
        missing_deps.append("transformers")
    
    try:
        import torch
        print(f"✓ torch {torch.__version__}")
    except ImportError:
        print("✗ torch not installed")
        missing_deps.append("torch")
    
    try:
        import einops
        print(f"✓ einops installed")
    except ImportError:
        print("✗ einops not installed")
        missing_deps.append("einops")
    
    if missing_deps:
        print(f"\n✗ Missing dependencies: {', '.join(missing_deps)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    return True


def test_import_compatibility():
    """Test if LlamaFlashAttention2 import issue exists."""
    print("\nTesting LlamaFlashAttention2 import...")
    
    try:
        from transformers.models.llama.modeling_llama import LlamaFlashAttention2
        print("✓ LlamaFlashAttention2 is available")
        return True
    except ImportError as e:
        print(f"✗ LlamaFlashAttention2 import failed: {e}")
        print("  This is expected for transformers >= 4.51.0")
        return False


def test_model_loading(quick_test=True):
    """Test loading the DeepSeek-OCR model."""
    print("\nTesting model loading...")
    
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
        
        model_name = 'deepseek-ai/DeepSeek-OCR'
        
        print(f"Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        print("✓ Tokenizer loaded successfully")
        
        if quick_test:
            print("\nℹ Skipping full model loading (use --full-test to load the complete model)")
            print("  This is a quick test to verify tokenizer and basic compatibility")
            return True
        
        print(f"\nLoading model from {model_name}...")
        print("  This may take a while and requires GPU...")
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("⚠ Warning: CUDA not available. Model loading may fail or be very slow.")
            print("  Consider running this on a machine with GPU support.")
        
        model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation='eager',  # Use eager instead of flash_attention_2 for compatibility
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if torch.cuda.is_available() else 'cpu'
        )
        print("✓ Model loaded successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        if "LlamaFlashAttention2" in str(e):
            print("\n⚠ This is the known compatibility issue!")
            print("  Run: python fix_transformers_compatibility.py")
            print("  Or downgrade transformers: pip install 'transformers<4.51.0'")
        return False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False


def print_recommendations():
    """Print recommendations based on test results."""
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    try:
        import transformers
        version = transformers.__version__
        major, minor = map(int, version.split('.')[:2])
        
        if major > 4 or (major == 4 and minor >= 51):
            print("\n⚠ You're using transformers >= 4.51.0")
            print("\nOption 1 (Recommended): Use the compatibility patcher")
            print("  python fix_transformers_compatibility.py")
            print("\nOption 2: Downgrade transformers")
            print("  pip install 'transformers>=4.46.3,<4.51.0'")
            print("\nOption 3: Use eager attention instead of flash_attention_2")
            print("  Change _attn_implementation='flash_attention_2' to 'eager'")
        else:
            print("\n✓ Your transformers version should be compatible")
            print("  If you still encounter issues, try:")
            print("  1. Reinstalling dependencies: pip install -r requirements.txt")
            print("  2. Clearing HuggingFace cache: rm -rf ~/.cache/huggingface/")
    except:
        pass


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test DeepSeek-OCR model loading and compatibility"
    )
    parser.add_argument(
        "--full-test",
        action="store_true",
        help="Perform full model loading test (requires GPU and takes longer)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DeepSeek-OCR Model Loading Test")
    print("=" * 60)
    print()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Test import compatibility
    import_ok = test_import_compatibility()
    
    # Test model loading
    model_ok = test_model_loading(quick_test=not args.full_test)
    
    # Print recommendations
    print_recommendations()
    
    print("\n" + "=" * 60)
    if model_ok:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed. See recommendations above.")
    print("=" * 60)
    
    return 0 if model_ok else 1


if __name__ == "__main__":
    sys.exit(main())
