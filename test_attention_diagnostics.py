"""
Test script for attention diagnostics module

This script tests the attention diagnostics functionality without requiring
a full model run.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DeepSeek-OCR-master/DeepSeek-OCR-vllm'))

import torch
import torch.nn as nn
from attention_diagnostics import (
    AttentionDiagnosticsCollector,
    AttentionBackend,
    FusionType,
    QuantizationType,
    enable_diagnostics,
    disable_diagnostics,
    get_collector,
    collect_attention_diagnostics
)


class MockAttentionModule(nn.Module):
    """Mock attention module for testing"""
    def __init__(self, use_flash_attn=True, has_qkv_fusion=True):
        super().__init__()
        self.use_flash_attn = use_flash_attn
        self.num_heads = 8
        self.head_dim = 64
        
        if has_qkv_fusion:
            self.qkv_proj = nn.Linear(512, 512 * 3)
        else:
            self.q_proj = nn.Linear(512, 512)
            self.k_proj = nn.Linear(512, 512)
            self.v_proj = nn.Linear(512, 512)
        
        self.out_proj = nn.Linear(512, 512)


def test_basic_collection():
    """Test basic diagnostics collection"""
    print("Test 1: Basic Collection")
    print("-" * 60)
    
    collector = AttentionDiagnosticsCollector(enabled=True, verbose=False)
    module = MockAttentionModule(use_flash_attn=True, has_qkv_fusion=True)
    input_tensor = torch.randn(2, 128, 512)
    
    diag = collector.collect(
        module=module,
        layer_name="test_attention",
        use_flash_attn=True,
        num_heads=8,
        head_dim=64,
        input_tensor=input_tensor
    )
    
    assert diag is not None, "Diagnostics should be collected"
    assert diag.layer_name == "test_attention"
    assert diag.num_heads == 8
    assert diag.head_dim == 64
    assert diag.use_flash_attention == True
    assert diag.batch_size == 2
    assert diag.sequence_length == 128
    
    print(f"✓ Collected diagnostics for {diag.layer_name}")
    print(f"  Backend: {diag.backend.value}")
    print(f"  Fusion: {diag.fusion_type.value}")
    print(f"  Quantization: {diag.quantization.value}")
    print()


def test_backend_detection():
    """Test backend detection"""
    print("Test 2: Backend Detection")
    print("-" * 60)
    
    collector = AttentionDiagnosticsCollector(enabled=True, verbose=False)
    
    # Test with FlashAttention enabled
    module1 = MockAttentionModule(use_flash_attn=True)
    backend1 = collector.detect_backend(module1, use_flash_attn=True)
    print(f"✓ FlashAttention enabled: {backend1.value}")
    
    # Test with FlashAttention disabled
    module2 = MockAttentionModule(use_flash_attn=False)
    backend2 = collector.detect_backend(module2, use_flash_attn=False)
    print(f"✓ FlashAttention disabled: {backend2.value}")
    print()


def test_fusion_detection():
    """Test fusion type detection"""
    print("Test 3: Fusion Detection")
    print("-" * 60)
    
    collector = AttentionDiagnosticsCollector(enabled=True, verbose=False)
    
    # Test QKV fused
    module1 = MockAttentionModule(has_qkv_fusion=True)
    fusion1 = collector.detect_fusion(module1)
    print(f"✓ QKV fused module: {fusion1.value}")
    assert fusion1 == FusionType.QKV_FUSED
    
    # Test separate projections
    module2 = MockAttentionModule(has_qkv_fusion=False)
    fusion2 = collector.detect_fusion(module2)
    print(f"✓ Separate projections: {fusion2.value}")
    assert fusion2 == FusionType.NONE
    print()


def test_quantization_detection():
    """Test quantization detection"""
    print("Test 4: Quantization Detection")
    print("-" * 60)
    
    collector = AttentionDiagnosticsCollector(enabled=True, verbose=False)
    module = MockAttentionModule()
    
    # Test different dtypes
    tensor_fp32 = torch.randn(2, 128, 512, dtype=torch.float32)
    quant_fp32 = collector.detect_quantization(module, tensor_fp32)
    print(f"✓ FP32 tensor: {quant_fp32.value}")
    
    tensor_fp16 = torch.randn(2, 128, 512, dtype=torch.float16)
    quant_fp16 = collector.detect_quantization(module, tensor_fp16)
    print(f"✓ FP16 tensor: {quant_fp16.value}")
    assert quant_fp16 == QuantizationType.FP16
    
    tensor_bf16 = torch.randn(2, 128, 512, dtype=torch.bfloat16)
    quant_bf16 = collector.detect_quantization(module, tensor_bf16)
    print(f"✓ BF16 tensor: {quant_bf16.value}")
    assert quant_bf16 == QuantizationType.BF16
    print()


def test_summary_generation():
    """Test summary generation"""
    print("Test 5: Summary Generation")
    print("-" * 60)
    
    collector = AttentionDiagnosticsCollector(enabled=True, verbose=False)
    
    # Collect diagnostics for multiple layers
    for i in range(5):
        module = MockAttentionModule(use_flash_attn=(i % 2 == 0))
        input_tensor = torch.randn(2, 128, 512)
        
        collector.collect(
            module=module,
            layer_name=f"layer_{i}",
            use_flash_attn=(i % 2 == 0),
            num_heads=8,
            head_dim=64,
            input_tensor=input_tensor
        )
    
    summary = collector.get_summary()
    
    print(f"✓ Total layers: {summary['total_layers']}")
    print(f"✓ FlashAttention layers: {summary['flash_attention_layers']}")
    print(f"✓ Backend distribution: {summary['backend_distribution']}")
    
    assert summary['total_layers'] == 5
    assert summary['flash_attention_layers'] == 3  # Layers 0, 2, 4
    print()


def test_export():
    """Test export functionality"""
    print("Test 6: Export Functionality")
    print("-" * 60)
    
    collector = AttentionDiagnosticsCollector(enabled=True, verbose=False)
    
    # Collect some diagnostics
    module = MockAttentionModule()
    input_tensor = torch.randn(2, 128, 512)
    
    collector.collect(
        module=module,
        layer_name="export_test",
        use_flash_attn=True,
        num_heads=8,
        head_dim=64,
        input_tensor=input_tensor
    )
    
    # Export to files
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "test_report.json")
        txt_path = os.path.join(tmpdir, "test_report.txt")
        
        collector.export_to_file(json_path, format='json')
        collector.export_to_file(txt_path, format='txt')
        
        assert os.path.exists(json_path), "JSON file should exist"
        assert os.path.exists(txt_path), "Text file should exist"
        
        print(f"✓ JSON export successful: {json_path}")
        print(f"✓ Text export successful: {txt_path}")
        
        # Verify JSON content
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
            assert 'summary' in data
            assert 'layers' in data
            print(f"✓ JSON structure valid")
    print()


def test_global_collector():
    """Test global collector functionality"""
    print("Test 7: Global Collector")
    print("-" * 60)
    
    # Enable diagnostics
    enable_diagnostics(verbose=False)
    collector = get_collector()
    assert collector.enabled == True
    print("✓ Diagnostics enabled globally")
    
    # Collect using global function
    module = MockAttentionModule()
    input_tensor = torch.randn(2, 128, 512)
    
    diag = collect_attention_diagnostics(
        module=module,
        layer_name="global_test",
        use_flash_attn=True,
        num_heads=8,
        head_dim=64,
        input_tensor=input_tensor
    )
    
    assert diag is not None
    print(f"✓ Collected via global function: {diag.layer_name}")
    
    # Disable diagnostics
    disable_diagnostics()
    collector = get_collector()
    assert collector.enabled == False
    print("✓ Diagnostics disabled globally")
    print()


def test_diagnostics_dataclass():
    """Test AttentionDiagnostics dataclass"""
    print("Test 8: Diagnostics Dataclass")
    print("-" * 60)
    
    from attention_diagnostics import AttentionDiagnostics
    
    diag = AttentionDiagnostics(
        layer_name="test_layer",
        layer_id=0,
        backend=AttentionBackend.FLASH_ATTENTION_2,
        fusion_type=FusionType.QKV_FUSED,
        quantization=QuantizationType.BF16,
        use_flash_attention=True,
        num_heads=16,
        head_dim=64,
        sequence_length=256,
        batch_size=2,
        dtype="torch.bfloat16",
        device="cuda:0"
    )
    
    # Test to_dict
    diag_dict = diag.to_dict()
    assert diag_dict['layer_name'] == "test_layer"
    assert diag_dict['backend'] == "FlashAttention-2"
    print("✓ to_dict() works correctly")
    
    # Test to_json
    diag_json = diag.to_json()
    assert "test_layer" in diag_json
    assert "FlashAttention-2" in diag_json
    print("✓ to_json() works correctly")
    
    # Test __str__
    diag_str = str(diag)
    assert "test_layer" in diag_str
    assert "FlashAttention-2" in diag_str
    print("✓ __str__() works correctly")
    print()


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("ATTENTION DIAGNOSTICS TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        test_basic_collection,
        test_backend_detection,
        test_fusion_detection,
        test_quantization_detection,
        test_summary_generation,
        test_export,
        test_global_collector,
        test_diagnostics_dataclass,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()
    
    print("="*60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
