"""
Example script demonstrating attention diagnostics usage

This script shows how to enable and use attention diagnostics to track:
- FlashAttention usage
- Attention fusion types
- Quantization information
- Backend implementations

Addresses GitHub Issue #3243: Myelin attention fusion and FlashAttention visibility
"""

import os
import torch

if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from config import (
    MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, CROP_MODE,
    ENABLE_ATTENTION_DIAGNOSTICS, ATTENTION_DIAGNOSTICS_VERBOSE,
    ATTENTION_DIAGNOSTICS_OUTPUT
)
from PIL import Image
from deepseek_ocr import DeepseekOCRForCausalLM
from vllm.model_executor.models.registry import ModelRegistry
from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor

# Import attention diagnostics
from attention_diagnostics import (
    enable_diagnostics, 
    disable_diagnostics, 
    get_collector
)


def main():
    """Main function demonstrating diagnostics usage"""
    
    print("="*60)
    print("DeepSeek-OCR with Attention Diagnostics")
    print("="*60)
    
    # Enable attention diagnostics
    if ENABLE_ATTENTION_DIAGNOSTICS:
        enable_diagnostics(verbose=ATTENTION_DIAGNOSTICS_VERBOSE)
        print("✓ Attention diagnostics enabled")
        if ATTENTION_DIAGNOSTICS_VERBOSE:
            print("✓ Verbose mode enabled")
    else:
        print("ℹ Attention diagnostics disabled (set ENABLE_ATTENTION_DIAGNOSTICS=True to enable)")
    
    print()
    
    # Register model
    ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
    
    # Initialize model
    print("Loading model...")
    llm = LLM(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=8192,
        swap_space=0,
        max_num_seqs=1,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
    )
    print("✓ Model loaded")
    print()
    
    # Prepare sample input
    if INPUT_PATH and os.path.exists(INPUT_PATH):
        image = Image.open(INPUT_PATH).convert('RGB')
        print(f"✓ Loaded image from {INPUT_PATH}")
    else:
        # Create a dummy image for testing
        print("ℹ No input image specified, using dummy image")
        image = Image.new('RGB', (640, 640), color='white')
    
    # Prepare input
    prompt = PROMPT
    cache_item = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": DeepseekOCRProcessor().tokenize_with_images(
                images=[image], 
                bos=True, 
                eos=True, 
                cropping=CROP_MODE
            )
        },
    }
    
    # Setup sampling parameters
    logits_processors = [
        NoRepeatNGramLogitsProcessor(
            ngram_size=30, 
            window_size=90, 
            whitelist_token_ids={128821, 128822}
        )
    ]
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,  # Reduced for testing
        logits_processors=logits_processors,
        skip_special_tokens=False,
    )
    
    # Run inference
    print("Running inference...")
    outputs = llm.generate([cache_item], sampling_params=sampling_params)
    print("✓ Inference complete")
    print()
    
    # Display results
    if outputs:
        result_text = outputs[0].outputs[0].text
        print("Generated text (first 200 chars):")
        print("-" * 60)
        print(result_text[:200] + "..." if len(result_text) > 200 else result_text)
        print("-" * 60)
        print()
    
    # Display diagnostics if enabled
    if ENABLE_ATTENTION_DIAGNOSTICS:
        collector = get_collector()
        
        print("\n" + "="*60)
        print("ATTENTION DIAGNOSTICS REPORT")
        print("="*60)
        
        # Print summary
        collector.print_summary()
        
        # Print detailed layer information
        if collector.diagnostics:
            print("\nDETAILED LAYER INFORMATION:")
            print("-"*60)
            for diag in collector.diagnostics[:5]:  # Show first 5 layers
                print(f"\n{diag}")
            
            if len(collector.diagnostics) > 5:
                print(f"\n... and {len(collector.diagnostics) - 5} more layers")
        
        # Export to file if path specified
        if ATTENTION_DIAGNOSTICS_OUTPUT:
            output_json = ATTENTION_DIAGNOSTICS_OUTPUT + ".json"
            output_txt = ATTENTION_DIAGNOSTICS_OUTPUT + ".txt"
            
            collector.export_to_file(output_json, format='json')
            collector.export_to_file(output_txt, format='txt')
            
            print(f"\n✓ Diagnostics exported to:")
            print(f"  - {output_json}")
            print(f"  - {output_txt}")
        
        print("\n" + "="*60)
        
        # Key findings
        summary = collector.get_summary()
        print("\nKEY FINDINGS:")
        print("-"*60)
        
        flash_count = summary.get('flash_attention_layers', 0)
        total_count = summary.get('total_layers', 0)
        
        if flash_count > 0:
            print(f"✓ FlashAttention is ENABLED in {flash_count}/{total_count} layers")
        else:
            print(f"ℹ FlashAttention is NOT enabled in any layers")
        
        backends = summary.get('backend_distribution', {})
        for backend, count in backends.items():
            print(f"  - {backend}: {count} layer(s)")
        
        fusions = summary.get('fusion_distribution', {})
        if fusions:
            print(f"\nFusion Types:")
            for fusion, count in fusions.items():
                print(f"  - {fusion}: {count} layer(s)")
        
        quants = summary.get('quantization_distribution', {})
        if quants:
            print(f"\nQuantization Types:")
            for quant, count in quants.items():
                print(f"  - {quant}: {count} layer(s)")
        
        print("="*60)
    
    print("\n✓ Script completed successfully")


if __name__ == "__main__":
    main()
