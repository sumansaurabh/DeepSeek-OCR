"""
Attention Diagnostics Module for DeepSeek-OCR

This module provides diagnostic capabilities to track and report attention implementation details,
including FlashAttention usage, fusion information, and quantization status.

Addresses GitHub Issue #3243: Myelin attention fusion and FlashAttention visibility
"""

import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import torch

logger = logging.getLogger(__name__)


class AttentionBackend(Enum):
    """Enumeration of attention backend implementations"""
    FLASH_ATTENTION = "FlashAttention"
    FLASH_ATTENTION_2 = "FlashAttention-2"
    SDPA = "ScaledDotProductAttention"
    XFORMERS = "xFormers"
    NAIVE = "Naive"
    MYELIN_FUSED = "Myelin-Fused"
    UNKNOWN = "Unknown"


class FusionType(Enum):
    """Types of attention fusion"""
    NONE = "None"
    QKV_FUSED = "QKV-Fused"
    FULL_FUSED = "Full-Fused"
    MYELIN_OPTIMIZED = "Myelin-Optimized"


class QuantizationType(Enum):
    """Types of quantization applied"""
    NONE = "None"
    INT8 = "INT8"
    FP8 = "FP8"
    FP16 = "FP16"
    BF16 = "BF16"
    IMPLICIT = "Implicit"


@dataclass
class AttentionDiagnostics:
    """Container for attention implementation diagnostics"""
    layer_name: str
    layer_id: int
    backend: AttentionBackend
    fusion_type: FusionType
    quantization: QuantizationType
    use_flash_attention: bool
    num_heads: int
    head_dim: int
    sequence_length: Optional[int] = None
    batch_size: Optional[int] = None
    dtype: Optional[str] = None
    device: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostics to dictionary"""
        result = asdict(self)
        result['backend'] = self.backend.value
        result['fusion_type'] = self.fusion_type.value
        result['quantization'] = self.quantization.value
        return result

    def to_json(self) -> str:
        """Convert diagnostics to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        """Human-readable string representation"""
        lines = [
            f"Attention Diagnostics for {self.layer_name} (Layer {self.layer_id}):",
            f"  Backend: {self.backend.value}",
            f"  Fusion Type: {self.fusion_type.value}",
            f"  Quantization: {self.quantization.value}",
            f"  FlashAttention Enabled: {self.use_flash_attention}",
            f"  Num Heads: {self.num_heads}",
            f"  Head Dimension: {self.head_dim}",
        ]
        
        if self.sequence_length:
            lines.append(f"  Sequence Length: {self.sequence_length}")
        if self.batch_size:
            lines.append(f"  Batch Size: {self.batch_size}")
        if self.dtype:
            lines.append(f"  Data Type: {self.dtype}")
        if self.device:
            lines.append(f"  Device: {self.device}")
        if self.additional_info:
            lines.append(f"  Additional Info: {self.additional_info}")
        
        return "\n".join(lines)


class AttentionDiagnosticsCollector:
    """Collects and manages attention diagnostics across layers"""
    
    def __init__(self, enabled: bool = True, verbose: bool = False):
        self.enabled = enabled
        self.verbose = verbose
        self.diagnostics: List[AttentionDiagnostics] = []
        self._layer_counter = 0
    
    def detect_backend(self, module: torch.nn.Module, use_flash_attn: bool) -> AttentionBackend:
        """Detect which attention backend is being used"""
        if use_flash_attn:
            try:
                import flash_attn
                version = getattr(flash_attn, '__version__', '1.x')
                if version.startswith('2'):
                    return AttentionBackend.FLASH_ATTENTION_2
                return AttentionBackend.FLASH_ATTENTION
            except ImportError:
                logger.warning("FlashAttention requested but not available")
                return AttentionBackend.SDPA
        
        # Check for SDPA usage
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            return AttentionBackend.SDPA
        
        return AttentionBackend.NAIVE
    
    def detect_fusion(self, module: torch.nn.Module) -> FusionType:
        """Detect the type of attention fusion being used"""
        # Check if QKV projection is fused
        if hasattr(module, 'qkv_proj'):
            return FusionType.QKV_FUSED
        
        # Check for separate Q, K, V projections
        if hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
            return FusionType.NONE
        
        # Check for Myelin optimization markers
        if hasattr(module, '_myelin_optimized') or hasattr(module, 'myelin_fused'):
            return FusionType.MYELIN_OPTIMIZED
        
        return FusionType.NONE
    
    def detect_quantization(self, module: torch.nn.Module, tensor: Optional[torch.Tensor] = None) -> QuantizationType:
        """Detect quantization type being used"""
        # Check module attributes for quantization config
        if hasattr(module, 'quantization_config'):
            config = module.quantization_config
            if hasattr(config, 'quant_type'):
                quant_type = config.quant_type.upper()
                if 'INT8' in quant_type:
                    return QuantizationType.INT8
                elif 'FP8' in quant_type:
                    return QuantizationType.FP8
        
        # Check tensor dtype if provided
        if tensor is not None:
            if tensor.dtype == torch.float16:
                return QuantizationType.FP16
            elif tensor.dtype == torch.bfloat16:
                return QuantizationType.BF16
            elif tensor.dtype == torch.int8:
                return QuantizationType.INT8
        
        # Check for implicit quantization markers
        if hasattr(module, '_implicit_quant') or hasattr(module, 'implicit_quantization'):
            return QuantizationType.IMPLICIT
        
        return QuantizationType.NONE
    
    def collect(
        self,
        module: torch.nn.Module,
        layer_name: str,
        use_flash_attn: bool,
        num_heads: int,
        head_dim: int,
        input_tensor: Optional[torch.Tensor] = None,
        **kwargs
    ) -> AttentionDiagnostics:
        """Collect diagnostics for an attention layer"""
        if not self.enabled:
            return None
        
        backend = self.detect_backend(module, use_flash_attn)
        fusion_type = self.detect_fusion(module)
        quantization = self.detect_quantization(module, input_tensor)
        
        # Extract runtime information from input tensor
        batch_size = None
        sequence_length = None
        dtype = None
        device = None
        
        if input_tensor is not None:
            if len(input_tensor.shape) >= 2:
                batch_size = input_tensor.shape[0]
                sequence_length = input_tensor.shape[1]
            dtype = str(input_tensor.dtype)
            device = str(input_tensor.device)
        
        diagnostics = AttentionDiagnostics(
            layer_name=layer_name,
            layer_id=self._layer_counter,
            backend=backend,
            fusion_type=fusion_type,
            quantization=quantization,
            use_flash_attention=use_flash_attn,
            num_heads=num_heads,
            head_dim=head_dim,
            sequence_length=sequence_length,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            additional_info=kwargs
        )
        
        self.diagnostics.append(diagnostics)
        self._layer_counter += 1
        
        if self.verbose:
            logger.info(f"\n{diagnostics}")
        
        return diagnostics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected diagnostics"""
        if not self.diagnostics:
            return {"message": "No diagnostics collected"}
        
        backend_counts = {}
        fusion_counts = {}
        quant_counts = {}
        flash_attn_count = 0
        
        for diag in self.diagnostics:
            # Count backends
            backend_counts[diag.backend.value] = backend_counts.get(diag.backend.value, 0) + 1
            
            # Count fusion types
            fusion_counts[diag.fusion_type.value] = fusion_counts.get(diag.fusion_type.value, 0) + 1
            
            # Count quantization types
            quant_counts[diag.quantization.value] = quant_counts.get(diag.quantization.value, 0) + 1
            
            # Count FlashAttention usage
            if diag.use_flash_attention:
                flash_attn_count += 1
        
        return {
            "total_layers": len(self.diagnostics),
            "flash_attention_layers": flash_attn_count,
            "backend_distribution": backend_counts,
            "fusion_distribution": fusion_counts,
            "quantization_distribution": quant_counts,
        }
    
    def print_summary(self):
        """Print a formatted summary of diagnostics"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("ATTENTION DIAGNOSTICS SUMMARY")
        print("="*60)
        print(f"Total Attention Layers: {summary.get('total_layers', 0)}")
        print(f"FlashAttention Enabled Layers: {summary.get('flash_attention_layers', 0)}")
        
        print("\nBackend Distribution:")
        for backend, count in summary.get('backend_distribution', {}).items():
            print(f"  {backend}: {count}")
        
        print("\nFusion Type Distribution:")
        for fusion, count in summary.get('fusion_distribution', {}).items():
            print(f"  {fusion}: {count}")
        
        print("\nQuantization Distribution:")
        for quant, count in summary.get('quantization_distribution', {}).items():
            print(f"  {quant}: {count}")
        
        print("="*60 + "\n")
    
    def export_to_file(self, filepath: str, format: str = 'json'):
        """Export diagnostics to a file"""
        if format == 'json':
            data = {
                'summary': self.get_summary(),
                'layers': [diag.to_dict() for diag in self.diagnostics]
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'txt':
            with open(filepath, 'w') as f:
                f.write("ATTENTION DIAGNOSTICS REPORT\n")
                f.write("="*60 + "\n\n")
                for diag in self.diagnostics:
                    f.write(str(diag) + "\n\n")
                f.write("\nSUMMARY\n")
                f.write("="*60 + "\n")
                f.write(json.dumps(self.get_summary(), indent=2))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Diagnostics exported to {filepath}")
    
    def reset(self):
        """Reset the collector"""
        self.diagnostics.clear()
        self._layer_counter = 0


# Global collector instance
_global_collector = AttentionDiagnosticsCollector(enabled=False)


def enable_diagnostics(verbose: bool = False):
    """Enable attention diagnostics globally"""
    global _global_collector
    _global_collector.enabled = True
    _global_collector.verbose = verbose
    logger.info("Attention diagnostics enabled")


def disable_diagnostics():
    """Disable attention diagnostics globally"""
    global _global_collector
    _global_collector.enabled = False
    logger.info("Attention diagnostics disabled")


def get_collector() -> AttentionDiagnosticsCollector:
    """Get the global diagnostics collector"""
    return _global_collector


def collect_attention_diagnostics(
    module: torch.nn.Module,
    layer_name: str,
    use_flash_attn: bool,
    num_heads: int,
    head_dim: int,
    input_tensor: Optional[torch.Tensor] = None,
    **kwargs
) -> Optional[AttentionDiagnostics]:
    """Convenience function to collect diagnostics using the global collector"""
    return _global_collector.collect(
        module=module,
        layer_name=layer_name,
        use_flash_attn=use_flash_attn,
        num_heads=num_heads,
        head_dim=head_dim,
        input_tensor=input_tensor,
        **kwargs
    )
