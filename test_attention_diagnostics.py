#!/usr/bin/env python3
"""
Test script for attention diagnostics feature (GitHub Issue #3243)

This script tests the attention diagnostics functionality without requiring
a full model inference run.
"""

import sys
import os

# Add the vllm directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DeepSeek-OCR-master', 'DeepSeek-OCR-vllm'))

def test_diagnostics_import():
    """Test that the diagnostics module can be imported."""
    print("Test 1: Import diagnostics module")
    try:
        from attention_diagnostics import get_diagnostics, print_attention_summary, log_attention_info
        print("  ✓ Successfully imported attention_diagnostics module")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import: {e}")
        return False

def test_diagnostics_initialization():
    """Test that diagnostics can be initialized."""
    print("\nTest 2: Initialize diagnostics")
    try:
        from attention_diagnostics import get_diagnostics
        diag = get_diagnostics()
        print("  ✓ Successfully initialized AttentionDiagnostics")
        
        # Check backend detection
        summary = diag.get_summary()
        print(f"  - FlashAttention available: {summary['backends']['flash_attention']['available']}")
        print(f"  - SDPA available: {summary['backends']['sdpa']['available']}")
        print(f"  - CUDA available: {summary['backends']['cuda']['available']}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to initialize: {e}")
        return False

def test_logging_functionality():
    """Test the logging functionality."""
    print("\nTest 3: Test logging functionality")
    try:
        from attention_diagnostics import log_attention_info, get_diagnostics
        
        # Reset stats
        diag = get_diagnostics()
        diag.reset_stats()
        
        # Log some attention calls
        log_attention_info('clip', use_flash=True, batch_size=2, seq_len=256, num_heads=16)
        log_attention_info('sam', use_flash=False, batch_size=2, seq_len=1024, num_heads=12)
        
        # Check stats
        summary = diag.get_summary()
        clip_calls = summary['attention_calls']['clip']
        sam_calls = summary['attention_calls']['sam']
        
        print(f"  ✓ Logged attention calls successfully")
        print(f"  - CLIP calls: {clip_calls}")
        print(f"  - SAM calls: {sam_calls}")
        
        if clip_calls == 1 and sam_calls == 1:
            print("  ✓ Statistics tracking working correctly")
            return True
        else:
            print("  ✗ Statistics tracking not working as expected")
            return False
    except Exception as e:
        print(f"  ✗ Failed logging test: {e}")
        return False

def test_summary_output():
    """Test the summary output."""
    print("\nTest 4: Test summary output")
    try:
        from attention_diagnostics import print_attention_summary
        print("  ✓ Generating summary output:")
        print_attention_summary()
        return True
    except Exception as e:
        print(f"  ✗ Failed to generate summary: {e}")
        return False

def test_config_integration():
    """Test config.py integration."""
    print("\nTest 5: Test config.py integration")
    try:
        from config import ENABLE_ATTENTION_DIAGNOSTICS, PRINT_ATTENTION_SUMMARY
        print(f"  ✓ Config loaded successfully")
        print(f"  - ENABLE_ATTENTION_DIAGNOSTICS: {ENABLE_ATTENTION_DIAGNOSTICS}")
        print(f"  - PRINT_ATTENTION_SUMMARY: {PRINT_ATTENTION_SUMMARY}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to load config: {e}")
        return False

def test_clip_attention_integration():
    """Test CLIP attention integration."""
    print("\nTest 6: Test CLIP attention integration")
    try:
        # Check if the file has been modified correctly
        clip_file = os.path.join(
            os.path.dirname(__file__), 
            'DeepSeek-OCR-master', 
            'DeepSeek-OCR-vllm', 
            'deepencoder', 
            'clip_sdpa.py'
        )
        
        with open(clip_file, 'r') as f:
            content = f.read()
            
        if 'from attention_diagnostics import log_attention_info' in content:
            print("  ✓ CLIP attention file has diagnostics import")
        else:
            print("  ✗ CLIP attention file missing diagnostics import")
            return False
            
        if 'log_attention_info' in content and 'attention_type=\'clip\'' in content:
            print("  ✓ CLIP attention file has logging calls")
        else:
            print("  ✗ CLIP attention file missing logging calls")
            return False
            
        return True
    except Exception as e:
        print(f"  ✗ Failed to check CLIP integration: {e}")
        return False

def test_sam_attention_integration():
    """Test SAM attention integration."""
    print("\nTest 7: Test SAM attention integration")
    try:
        # Check if the file has been modified correctly
        sam_file = os.path.join(
            os.path.dirname(__file__), 
            'DeepSeek-OCR-master', 
            'DeepSeek-OCR-vllm', 
            'deepencoder', 
            'sam_vary_sdpa.py'
        )
        
        with open(sam_file, 'r') as f:
            content = f.read()
            
        if 'from attention_diagnostics import log_attention_info' in content:
            print("  ✓ SAM attention file has diagnostics import")
        else:
            print("  ✗ SAM attention file missing diagnostics import")
            return False
            
        if 'log_attention_info' in content and 'attention_type=\'sam\'' in content:
            print("  ✓ SAM attention file has logging calls")
        else:
            print("  ✗ SAM attention file missing logging calls")
            return False
            
        return True
    except Exception as e:
        print(f"  ✗ Failed to check SAM integration: {e}")
        return False

def main():
    """Run all tests."""
    print("="*70)
    print("ATTENTION DIAGNOSTICS TEST SUITE (GitHub Issue #3243)")
    print("="*70)
    
    tests = [
        test_diagnostics_import,
        test_diagnostics_initialization,
        test_logging_functionality,
        test_summary_output,
        test_config_integration,
        test_clip_attention_integration,
        test_sam_attention_integration,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
