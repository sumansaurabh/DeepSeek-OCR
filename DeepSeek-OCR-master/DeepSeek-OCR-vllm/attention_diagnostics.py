"""
Attention Diagnostics Utility for DeepSeek-OCR

This module provides utilities to detect, log, and report which attention
implementation is being used during inference (FlashAttention vs SDPA).

Addresses GitHub Issue #3243: Provides transparency about attention fusion
and implementation details.
"""

import logging
import sys
from typing import Dict, Optional
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger("AttentionDiagnostics")


class AttentionDiagnostics:
    """
    Singleton class to track and report attention implementation details.
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AttentionDiagnostics, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.flash_attn_available = False
            self.flash_attn_version = None
            self.sdpa_available = False
            self.cuda_available = False
            self.attention_stats = {
                'flash_attn_calls': 0,
                'sdpa_calls': 0,
                'clip_attention_calls': 0,
                'sam_attention_calls': 0
            }
            self._check_attention_backends()
            AttentionDiagnostics._initialized = True
    
    def _check_attention_backends(self):
        """Check which attention backends are available."""
        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()
        
        # Check FlashAttention availability
        try:
            import flash_attn
            self.flash_attn_available = True
            self.flash_attn_version = getattr(flash_attn, '__version__', 'unknown')
            logger.info(f"✓ FlashAttention available (version: {self.flash_attn_version})")
        except ImportError:
            self.flash_attn_available = False
            logger.warning("✗ FlashAttention not available")
        
        # Check SDPA availability (PyTorch >= 2.0)
        try:
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                self.sdpa_available = True
                logger.info("✓ PyTorch SDPA (Scaled Dot Product Attention) available")
            else:
                self.sdpa_available = False
                logger.warning("✗ PyTorch SDPA not available (requires PyTorch >= 2.0)")
        except Exception as e:
            self.sdpa_available = False
            logger.warning(f"✗ Error checking SDPA availability: {e}")
        
        # Log CUDA info
        if self.cuda_available:
            logger.info(f"✓ CUDA available (device: {torch.cuda.get_device_name(0)})")
            logger.info(f"  CUDA version: {torch.version.cuda}")
            logger.info(f"  cuDNN version: {torch.backends.cudnn.version()}")
        else:
            logger.warning("✗ CUDA not available - attention performance will be limited")
    
    def log_attention_call(self, attention_type: str, implementation: str, 
                          batch_size: int = None, seq_len: int = None,
                          num_heads: int = None):
        """
        Log an attention operation call.
        
        Args:
            attention_type: Type of attention ('clip', 'sam', 'language')
            implementation: Implementation used ('flash_attn', 'sdpa', 'naive')
            batch_size: Batch size (optional)
            seq_len: Sequence length (optional)
            num_heads: Number of attention heads (optional)
        """
        # Update stats
        if implementation == 'flash_attn':
            self.attention_stats['flash_attn_calls'] += 1
        elif implementation == 'sdpa':
            self.attention_stats['sdpa_calls'] += 1
        
        if attention_type == 'clip':
            self.attention_stats['clip_attention_calls'] += 1
        elif attention_type == 'sam':
            self.attention_stats['sam_attention_calls'] += 1
        
        # Log details
        details = f"Attention: {attention_type.upper()} | Implementation: {implementation.upper()}"
        if batch_size is not None:
            details += f" | Batch: {batch_size}"
        if seq_len is not None:
            details += f" | SeqLen: {seq_len}"
        if num_heads is not None:
            details += f" | Heads: {num_heads}"
        
        logger.debug(details)
    
    def get_summary(self) -> Dict:
        """Get a summary of attention usage."""
        return {
            'backends': {
                'flash_attention': {
                    'available': self.flash_attn_available,
                    'version': self.flash_attn_version,
                    'calls': self.attention_stats['flash_attn_calls']
                },
                'sdpa': {
                    'available': self.sdpa_available,
                    'calls': self.attention_stats['sdpa_calls']
                },
                'cuda': {
                    'available': self.cuda_available,
                    'device': torch.cuda.get_device_name(0) if self.cuda_available else None
                }
            },
            'attention_calls': {
                'clip': self.attention_stats['clip_attention_calls'],
                'sam': self.attention_stats['sam_attention_calls'],
                'total': sum([
                    self.attention_stats['clip_attention_calls'],
                    self.attention_stats['sam_attention_calls']
                ])
            }
        }
    
    def print_summary(self):
        """Print a formatted summary of attention usage."""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("ATTENTION IMPLEMENTATION SUMMARY")
        print("="*70)
        
        print("\n📊 Backend Availability:")
        print(f"  FlashAttention: {'✓ Available' if summary['backends']['flash_attention']['available'] else '✗ Not Available'}")
        if summary['backends']['flash_attention']['available']:
            print(f"    Version: {summary['backends']['flash_attention']['version']}")
            print(f"    Calls: {summary['backends']['flash_attention']['calls']}")
        
        print(f"  PyTorch SDPA: {'✓ Available' if summary['backends']['sdpa']['available'] else '✗ Not Available'}")
        if summary['backends']['sdpa']['available']:
            print(f"    Calls: {summary['backends']['sdpa']['calls']}")
        
        print(f"  CUDA: {'✓ Available' if summary['backends']['cuda']['available'] else '✗ Not Available'}")
        if summary['backends']['cuda']['available']:
            print(f"    Device: {summary['backends']['cuda']['device']}")
        
        print("\n🔍 Attention Calls by Component:")
        print(f"  CLIP Encoder: {summary['attention_calls']['clip']}")
        print(f"  SAM Encoder: {summary['attention_calls']['sam']}")
        print(f"  Total: {summary['attention_calls']['total']}")
        
        # Recommendations
        print("\n💡 Recommendations:")
        if not summary['backends']['flash_attention']['available']:
            print("  ⚠ Install FlashAttention for optimal performance:")
            print("    pip install flash-attn==2.7.3 --no-build-isolation")
        elif summary['backends']['flash_attention']['calls'] == 0 and summary['attention_calls']['total'] > 0:
            print("  ⚠ FlashAttention is available but not being used.")
            print("    Check that use_flash_attn=True in model config.")
        else:
            print("  ✓ Attention implementation is optimized!")
        
        print("="*70 + "\n")
    
    def reset_stats(self):
        """Reset attention call statistics."""
        self.attention_stats = {
            'flash_attn_calls': 0,
            'sdpa_calls': 0,
            'clip_attention_calls': 0,
            'sam_attention_calls': 0
        }


# Global instance
_diagnostics = AttentionDiagnostics()


def get_diagnostics() -> AttentionDiagnostics:
    """Get the global AttentionDiagnostics instance."""
    return _diagnostics


def log_attention_info(attention_type: str, use_flash: bool, 
                       batch_size: int = None, seq_len: int = None,
                       num_heads: int = None):
    """
    Convenience function to log attention implementation info.
    
    Args:
        attention_type: Type of attention ('clip', 'sam', 'language')
        use_flash: Whether FlashAttention is being used
        batch_size: Batch size (optional)
        seq_len: Sequence length (optional)
        num_heads: Number of attention heads (optional)
    """
    implementation = 'flash_attn' if use_flash else 'sdpa'
    _diagnostics.log_attention_call(
        attention_type=attention_type,
        implementation=implementation,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads
    )


def print_attention_summary():
    """Print attention usage summary."""
    _diagnostics.print_summary()


if __name__ == "__main__":
    # Test the diagnostics
    diag = get_diagnostics()
    diag.print_summary()
