#!/usr/bin/env python3
"""
Example script demonstrating how to use DeepSeek-OCR with the compatibility workaround
for transformers >= 4.51.0

This script shows the recommended approach to handle the LlamaFlashAttention2 import issue.
"""

import os
import sys


def check_and_warn_transformers_version():
    """Check transformers version and warn if incompatible."""
    try:
        import transformers
        version = transformers.__version__
        major, minor = map(int, version.split('.')[:2])
        
        if major > 4 or (major == 4 and minor >= 51):
            print(f"⚠ Warning: Using transformers {version}")
            print("  This version may have compatibility issues.")
            print("  Consider running: python fix_transformers_compatibility.py")
            print()
            return False
        return True
    except ImportError:
        print("✗ transformers not installed")
        sys.exit(1)


def load_model_with_workaround():
    """Load DeepSeek-OCR model with compatibility workaround."""
    from transformers import AutoModel, AutoTokenizer
    import torch
    
    # Set environment variable for CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    model_name = 'deepseek-ai/DeepSeek-OCR'
    
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("✓ Tokenizer loaded successfully\n")
    
    print(f"Loading model from {model_name}...")
    print("  Note: Using 'eager' attention instead of 'flash_attention_2' for compatibility")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("⚠ Warning: CUDA not available. Model will load on CPU (very slow)")
        device_map = 'cpu'
        dtype = torch.float32
    else:
        device_map = 'auto'
        dtype = torch.bfloat16
    
    try:
        # Option 1: Use eager attention (most compatible)
        model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation='eager',  # Use eager instead of flash_attention_2
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=dtype,
            device_map=device_map
        )
        print("✓ Model loaded successfully with eager attention\n")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nTroubleshooting steps:")
        print("1. Run: python fix_transformers_compatibility.py")
        print("2. Or install compatible transformers: pip install 'transformers>=4.46.3,<4.51.0'")
        print("3. Check the README.md Troubleshooting section")
        sys.exit(1)
    
    return model, tokenizer


def run_inference_example(model, tokenizer):
    """Run a simple inference example."""
    import torch
    
    # Example prompts
    prompts = [
        "<image>\nFree OCR.",
        "<image>\n<|grounding|>Convert the document to markdown.",
        "<image>\n<|grounding|>OCR this image.",
    ]
    
    print("Example prompts:")
    for i, prompt in enumerate(prompts, 1):
        print(f"  {i}. {prompt}")
    
    print("\nTo run inference, you need to:")
    print("  1. Provide an image file path")
    print("  2. Set output directory")
    print("  3. Call model.infer() with appropriate parameters")
    
    print("\nExample usage:")
    print("""
    prompt = "<image>\\n<|grounding|>Convert the document to markdown."
    image_file = 'path/to/your/image.jpg'
    output_path = 'path/to/output/dir'
    
    res = model.infer(
        tokenizer, 
        prompt=prompt, 
        image_file=image_file, 
        output_path=output_path, 
        base_size=1024, 
        image_size=640, 
        crop_mode=True, 
        save_results=True, 
        test_compress=True
    )
    """)


def main():
    print("=" * 70)
    print("DeepSeek-OCR Example with Compatibility Workaround")
    print("=" * 70)
    print()
    
    # Check transformers version
    is_compatible = check_and_warn_transformers_version()
    
    if not is_compatible:
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Please fix compatibility issues first.")
            sys.exit(0)
        print()
    
    # Load model
    try:
        model, tokenizer = load_model_with_workaround()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    
    # Show example usage
    run_inference_example(model, tokenizer)
    
    print("\n" + "=" * 70)
    print("✓ Setup complete! You can now use the model for inference.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
