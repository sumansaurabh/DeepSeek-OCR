"""
Environment checker for DeepSeek-OCR.

This script validates that all required dependencies are installed with
compatible versions before running DeepSeek-OCR.

Usage:
    python check_environment.py
"""

import sys
import importlib.util
from typing import List, Tuple, Optional


class DependencyChecker:
    """Check and validate DeepSeek-OCR dependencies."""
    
    # Required packages with version constraints
    REQUIRED_PACKAGES = {
        'transformers': {
            'recommended': '4.46.3',
            'min': '4.40.0',
            'max': '4.46.9',
            'critical': True,
            'notes': 'Versions >=4.47 require compatibility_patch.py'
        },
        'tokenizers': {
            'recommended': '0.20.3',
            'min': '0.19.0',
            'max': None,
            'critical': True,
            'notes': None
        },
        'torch': {
            'recommended': '2.6.0',
            'min': '2.0.0',
            'max': None,
            'critical': True,
            'notes': 'CUDA 11.8+ recommended'
        },
        'einops': {
            'recommended': None,
            'min': None,
            'max': None,
            'critical': True,
            'notes': None
        },
        'PIL': {
            'package_name': 'Pillow',
            'recommended': None,
            'min': '8.0.0',
            'max': None,
            'critical': True,
            'notes': None
        },
        'numpy': {
            'recommended': None,
            'min': '1.20.0',
            'max': None,
            'critical': True,
            'notes': None
        },
        'fitz': {
            'package_name': 'PyMuPDF',
            'recommended': None,
            'min': None,
            'max': None,
            'critical': False,
            'notes': 'Required for PDF processing'
        },
        'img2pdf': {
            'recommended': None,
            'min': None,
            'max': None,
            'critical': False,
            'notes': 'Required for image to PDF conversion'
        },
        'easydict': {
            'recommended': None,
            'min': None,
            'max': None,
            'critical': False,
            'notes': None
        },
        'addict': {
            'recommended': None,
            'min': None,
            'max': None,
            'critical': False,
            'notes': None
        }
    }
    
    # Optional packages for enhanced functionality
    OPTIONAL_PACKAGES = {
        'flash_attn': {
            'recommended': '2.7.3',
            'min': '2.0.0',
            'max': None,
            'notes': 'Required for flash attention support'
        },
        'vllm': {
            'recommended': '0.8.5',
            'min': '0.8.0',
            'max': None,
            'notes': 'Required for vLLM inference'
        }
    }
    
    def __init__(self):
        self.results = []
        self.errors = []
        self.warnings = []
    
    def check_package(self, import_name: str, config: dict) -> Tuple[bool, Optional[str], str]:
        """
        Check if a package is installed and get its version.
        
        Args:
            import_name: Name used for importing the package
            config: Configuration dict with version constraints
            
        Returns:
            Tuple of (is_installed, version, status_message)
        """
        package_name = config.get('package_name', import_name)
        
        try:
            # Try to import the module
            module = __import__(import_name)
            
            # Get version
            version = None
            for attr in ['__version__', 'VERSION', 'version']:
                if hasattr(module, attr):
                    version = str(getattr(module, attr))
                    break
            
            if version is None:
                return (True, None, f"✓ {package_name} installed (version unknown)")
            
            # Check version constraints
            from packaging import version as pkg_version
            current_ver = pkg_version.parse(version)
            
            # Check recommended version
            if config.get('recommended'):
                recommended_ver = pkg_version.parse(config['recommended'])
                if current_ver == recommended_ver:
                    return (True, version, f"✓ {package_name} {version} (recommended)")
            
            # Check min version
            if config.get('min'):
                min_ver = pkg_version.parse(config['min'])
                if current_ver < min_ver:
                    msg = f"✗ {package_name} {version} (minimum: {config['min']})"
                    return (False, version, msg)
            
            # Check max version
            if config.get('max'):
                max_ver = pkg_version.parse(config['max'])
                if current_ver > max_ver:
                    msg = f"⚠ {package_name} {version} (maximum: {config['max']})"
                    if config.get('notes'):
                        msg += f"\n    Note: {config['notes']}"
                    return (True, version, msg)
            
            return (True, version, f"✓ {package_name} {version}")
            
        except ImportError:
            return (False, None, f"✗ {package_name} not installed")
        except Exception as e:
            return (False, None, f"✗ {package_name} error: {str(e)}")
    
    def check_cuda(self) -> Tuple[bool, str]:
        """Check CUDA availability."""
        try:
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                return (True, f"✓ CUDA {cuda_version} available ({device_count} device(s): {device_name})")
            else:
                return (False, "⚠ CUDA not available (CPU-only mode)")
        except ImportError:
            return (False, "✗ PyTorch not installed, cannot check CUDA")
        except Exception as e:
            return (False, f"✗ Error checking CUDA: {str(e)}")
    
    def run_checks(self) -> bool:
        """
        Run all dependency checks.
        
        Returns:
            bool: True if all critical dependencies are satisfied
        """
        print("=" * 70)
        print("DeepSeek-OCR Environment Check")
        print("=" * 70)
        
        all_critical_ok = True
        
        # Check required packages
        print("\n📦 Required Packages:")
        print("-" * 70)
        for import_name, config in self.REQUIRED_PACKAGES.items():
            is_ok, version, message = self.check_package(import_name, config)
            print(message)
            
            if config.get('notes'):
                print(f"    Note: {config['notes']}")
            
            if not is_ok and config.get('critical', True):
                all_critical_ok = False
                self.errors.append(message)
            elif not is_ok:
                self.warnings.append(message)
        
        # Check optional packages
        print("\n📦 Optional Packages:")
        print("-" * 70)
        for import_name, config in self.OPTIONAL_PACKAGES.items():
            is_ok, version, message = self.check_package(import_name, config)
            print(message)
            
            if config.get('notes'):
                print(f"    Note: {config['notes']}")
        
        # Check CUDA
        print("\n🖥️  Hardware:")
        print("-" * 70)
        cuda_ok, cuda_message = self.check_cuda()
        print(cuda_message)
        
        # Check transformers version specifically for compatibility
        print("\n🔍 Compatibility Check:")
        print("-" * 70)
        try:
            from compatibility_patch import check_compatibility
            is_compatible, version_str, message = check_compatibility()
            
            if is_compatible:
                print(f"✓ transformers {version_str}: {message}")
            else:
                print(f"⚠ transformers {version_str}: {message}")
                print("  → Run: python compatibility_patch.py")
                print("  → Or import compatibility_patch before loading the model")
        except ImportError:
            print("⚠ compatibility_patch.py not found")
            print("  → This script helps with transformers>=4.47 compatibility")
        
        # Summary
        print("\n" + "=" * 70)
        if all_critical_ok:
            print("✓ All critical dependencies satisfied!")
            if self.warnings:
                print(f"⚠ {len(self.warnings)} warning(s) - see above")
        else:
            print(f"✗ {len(self.errors)} critical error(s) found")
            print("\nTo install missing dependencies:")
            print("  pip install -r requirements.txt")
        print("=" * 70)
        
        return all_critical_ok


def main():
    """Main entry point."""
    checker = DependencyChecker()
    success = checker.run_checks()
    
    if not success:
        print("\n❌ Environment check failed!")
        print("Please install missing dependencies before running DeepSeek-OCR.")
        sys.exit(1)
    else:
        print("\n✅ Environment check passed!")
        print("You can now run DeepSeek-OCR.")
        sys.exit(0)


if __name__ == "__main__":
    main()
