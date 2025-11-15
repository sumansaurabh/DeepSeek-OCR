"""
Simple test for attention diagnostics module structure
Tests without requiring PyTorch or other dependencies
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DeepSeek-OCR-master/DeepSeek-OCR-vllm'))

def test_imports():
    """Test that all modules can be imported"""
    print("Test 1: Module Imports")
    print("-" * 60)
    
    try:
        from attention_diagnostics import (
            AttentionBackend,
            FusionType,
            QuantizationType,
            AttentionDiagnostics,
            AttentionDiagnosticsCollector,
            enable_diagnostics,
            disable_diagnostics,
            get_collector,
            collect_attention_diagnostics
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_enums():
    """Test enum definitions"""
    print("\nTest 2: Enum Definitions")
    print("-" * 60)
    
    from attention_diagnostics import AttentionBackend, FusionType, QuantizationType
    
    # Test AttentionBackend
    backends = [
        AttentionBackend.FLASH_ATTENTION,
        AttentionBackend.FLASH_ATTENTION_2,
        AttentionBackend.SDPA,
        AttentionBackend.XFORMERS,
        AttentionBackend.NAIVE,
        AttentionBackend.MYELIN_FUSED,
        AttentionBackend.UNKNOWN
    ]
    print(f"✓ AttentionBackend has {len(backends)} values")
    
    # Test FusionType
    fusions = [
        FusionType.NONE,
        FusionType.QKV_FUSED,
        FusionType.FULL_FUSED,
        FusionType.MYELIN_OPTIMIZED
    ]
    print(f"✓ FusionType has {len(fusions)} values")
    
    # Test QuantizationType
    quants = [
        QuantizationType.NONE,
        QuantizationType.INT8,
        QuantizationType.FP8,
        QuantizationType.FP16,
        QuantizationType.BF16,
        QuantizationType.IMPLICIT
    ]
    print(f"✓ QuantizationType has {len(quants)} values")
    
    return True


def test_collector_creation():
    """Test collector instantiation"""
    print("\nTest 3: Collector Creation")
    print("-" * 60)
    
    from attention_diagnostics import AttentionDiagnosticsCollector
    
    # Test with defaults
    collector1 = AttentionDiagnosticsCollector()
    print(f"✓ Default collector created (enabled={collector1.enabled})")
    
    # Test with custom settings
    collector2 = AttentionDiagnosticsCollector(enabled=True, verbose=True)
    print(f"✓ Custom collector created (enabled={collector2.enabled}, verbose={collector2.verbose})")
    
    return True


def test_global_functions():
    """Test global functions"""
    print("\nTest 4: Global Functions")
    print("-" * 60)
    
    from attention_diagnostics import enable_diagnostics, disable_diagnostics, get_collector
    
    # Test enable
    enable_diagnostics(verbose=False)
    collector = get_collector()
    if collector.enabled:
        print("✓ enable_diagnostics() works")
    else:
        print("✗ enable_diagnostics() failed")
        return False
    
    # Test disable
    disable_diagnostics()
    collector = get_collector()
    if not collector.enabled:
        print("✓ disable_diagnostics() works")
    else:
        print("✗ disable_diagnostics() failed")
        return False
    
    return True


def test_file_structure():
    """Test that all required files exist"""
    print("\nTest 5: File Structure")
    print("-" * 60)
    
    base_path = os.path.join(os.path.dirname(__file__), 'DeepSeek-OCR-master/DeepSeek-OCR-vllm')
    
    files = [
        'attention_diagnostics.py',
        'run_with_diagnostics.py',
        'config.py',
        'deepencoder/clip_sdpa.py',
        'deepencoder/sam_vary_sdpa.py'
    ]
    
    all_exist = True
    for file in files:
        path = os.path.join(base_path, file)
        if os.path.exists(path):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            all_exist = False
    
    return all_exist


def test_documentation():
    """Test that documentation exists"""
    print("\nTest 6: Documentation")
    print("-" * 60)
    
    doc_path = os.path.join(os.path.dirname(__file__), 'ATTENTION_DIAGNOSTICS_README.md')
    
    if os.path.exists(doc_path):
        print(f"✓ Documentation exists: {doc_path}")
        
        # Check file size
        size = os.path.getsize(doc_path)
        print(f"✓ Documentation size: {size} bytes")
        
        return True
    else:
        print(f"✗ Documentation missing: {doc_path}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("ATTENTION DIAGNOSTICS SIMPLE TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        test_imports,
        test_enums,
        test_collector_creation,
        test_global_functions,
        test_file_structure,
        test_documentation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    if failed == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
