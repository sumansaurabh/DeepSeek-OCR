"""
Safe wrapper for running DeepSeek-OCR with compatibility checks.

This script addresses GitHub Issue #7:
ImportError: cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama.modeling_llama'

It performs version checks before attempting to load the model and provides
helpful error messages if compatibility issues are detected.

Usage:
    python run_dpsk_ocr_safe.py
"""

import sys
import os
import importlib.metadata
from packaging import version


def check_compatibility():
    """
    Check if the environment is compatible with DeepSeek-OCR.
    
    Returns:
        tuple: (is_compatible, error_messages)
    """
    errors = []
    
    # Check transformers version
    try:
        transformers_version = importlib.metadata.version("transformers")
        ver = version.parse(transformers_version)
        
        # DeepSeek-OCR requires transformers 4.46.x - 4.51.x
        # Versions 4.52+ removed LlamaFlashAttention2
        if ver >= version.parse("4.52.0"):
            errors.append(
                f"❌ transformers version {transformers_version} is too new.\n"
                f"   LlamaFlashAttention2 was removed in transformers 4.52+\n"
                f"   Required: transformers==4.46.3\n"
                f"   Fix: pip uninstall transformers -y && pip install transformers==4.46.3"
            )
        elif ver < version.parse("4.46.0"):
            errors.append(
                f"❌ transformers version {transformers_version} is too old.\n"
                f"   Required: transformers>=4.46.0,<4.52.0\n"
                f"   Fix: pip install transformers==4.46.3"
            )
    except importlib.metadata.PackageNotFoundError:
        errors.append(
            "❌ transformers is not installed.\n"
            "   Fix: pip install transformers==4.46.3"
        )
    
    # Check if LlamaFlashAttention2 can be imported
    try:
        from transformers.models.llama.modeling_llama import LlamaFlashAttention2
    except ImportError as e:
        errors.append(
            f"❌ Cannot import LlamaFlashAttention2 (GitHub Issue #7)\n"
            f"   Error: {str(e)}\n"
            f"   This usually means your transformers version is incompatible.\n"
            f"   Fix: pip uninstall transformers -y && pip install transformers==4.46.3"
        )
    
    # Check torch
    try:
        import torch
        torch_version = torch.__version__
        if version.parse(torch_version.split('+')[0]) < version.parse("2.0.0"):
            errors.append(
                f"❌ torch version {torch_version} is too old.\n"
                f"   Required: torch>=2.0.0\n"
                f"   Fix: pip install torch>=2.0.0"
            )
    except ImportError:
        errors.append(
            "❌ torch is not installed.\n"
            "   Fix: pip install torch>=2.0.0"
        )
    
    return len(errors) == 0, errors


def main():
    """Main function to run DeepSeek-OCR with safety checks."""
    print("=" * 70)
    print("DeepSeek-OCR Safe Runner")
    print("=" * 70)
    print()
    
    # Perform compatibility checks
    print("Checking environment compatibility...")
    is_compatible, errors = check_compatibility()
    
    if not is_compatible:
        print()
        print("❌ COMPATIBILITY ISSUES DETECTED:")
        print()
        for error in errors:
            print(error)
            print()
        print("=" * 70)
        print("Please fix the issues above before running DeepSeek-OCR.")
        print()
        print("For more detailed checks, run:")
        print("  python ../../check_compatibility.py")
        print("=" * 70)
        return 1
    
    print("✓ Environment is compatible!")
    print()
    print("=" * 70)
    print("Loading DeepSeek-OCR model...")
    print("=" * 70)
    print()
    
    # Import and run the actual model
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
        
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        
        model_name = 'deepseek-ai/DeepSeek-OCR'
        
        print(f"Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        print(f"Loading model from {model_name}...")
        print("(This may take a few minutes on first run...)")
        model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation='flash_attention_2',
            trust_remote_code=True,
            use_safetensors=True
        )
        model = model.eval().cuda().to(torch.bfloat16)
        
        print("✓ Model loaded successfully!")
        print()
        
        # Example usage
        # prompt = "<image>\\nFree OCR. "
        prompt = "<image>\\n<|grounding|>Convert the document to markdown. "
        image_file = 'your_image.jpg'
        output_path = 'your/output/dir'
        
        print("=" * 70)
        print("Model is ready for inference!")
        print("=" * 70)
        print()
        print("Example usage:")
        print(f"  prompt = '{prompt}'")
        print(f"  image_file = '{image_file}'")
        print(f"  output_path = '{output_path}'")
        print()
        print("To run inference, uncomment and modify the following lines:")
        print()
        print("  res = model.infer(")
        print("      tokenizer,")
        print("      prompt=prompt,")
        print("      image_file=image_file,")
        print("      output_path=output_path,")
        print("      base_size=1024,")
        print("      image_size=640,")
        print("      crop_mode=True,")
        print("      save_results=True,")
        print("      test_compress=True")
        print("  )")
        print()
        
        # Uncomment to run actual inference:
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
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ ERROR LOADING MODEL:")
        print("=" * 70)
        print(f"{type(e).__name__}: {str(e)}")
        print()
        print("If you see an ImportError related to LlamaFlashAttention2,")
        print("please ensure you have the correct transformers version:")
        print("  pip uninstall transformers -y")
        print("  pip install transformers==4.46.3")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
