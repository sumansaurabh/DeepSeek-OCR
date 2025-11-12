#!/usr/bin/env python3
"""
Verification script for DeepSeek-OCR transformers compatibility.
This script checks if the installed transformers version is compatible.
"""

import sys

def check_transformers_version():
    """Check if transformers version is compatible with DeepSeek-OCR."""
    try:
        import transformers
        version_str = transformers.__version__
        print(f"✓ Transformers installed: {version_str}")
        
        # Parse version
        version_parts = version_str.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1])
        
        # Check compatibility
        if major == 4 and 46 <= minor < 50:
            print("✅ Compatible transformers version detected!")
            print(f"   Version {version_str} is compatible with DeepSeek-OCR")
            return True
        elif major == 4 and minor >= 50:
            print("⚠️  WARNING: Incompatible transformers version!")
            print(f"   Version {version_str} will cause LlamaFlashAttention2 import errors")
            print("\n   Solution:")
            print("   pip uninstall transformers -y")
            print('   pip install "transformers>=4.46.3,<4.50.0"')
            print("\n   See TROUBLESHOOTING.md for more details.")
            return False
        elif major == 4 and minor < 46:
            print("⚠️  WARNING: Transformers version may be too old")
            print(f"   Version {version_str} is older than recommended 4.46.3")
            print("   Consider upgrading to 4.46.3 or newer (but <4.50.0)")
            return False
        else:
            print(f"⚠️  WARNING: Unexpected transformers version {version_str}")
            print("   Recommended: 4.46.3 <= version < 4.50.0")
            return False
            
    except ImportError:
        print("❌ Transformers not installed!")
        print("\n   Install with:")
        print('   pip install "transformers>=4.46.3,<4.50.0"')
        return False
    except Exception as e:
        print(f"❌ Error checking transformers version: {e}")
        return False

def check_other_dependencies():
    """Check other key dependencies."""
    dependencies = {
        'torch': 'PyTorch',
        'tokenizers': 'Tokenizers',
        'PIL': 'Pillow',
        'einops': 'Einops',
    }
    
    print("\nChecking other dependencies:")
    all_ok = True
    for module, name in dependencies.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {name}: {version}")
        except ImportError:
            print(f"  ✗ {name}: not installed")
            all_ok = False
    
    return all_ok

def main():
    """Main verification function."""
    print("=" * 60)
    print("DeepSeek-OCR Environment Verification")
    print("=" * 60)
    print()
    
    transformers_ok = check_transformers_version()
    deps_ok = check_other_dependencies()
    
    print()
    print("=" * 60)
    if transformers_ok and deps_ok:
        print("✅ Environment is ready for DeepSeek-OCR!")
        sys.exit(0)
    else:
        print("⚠️  Please fix the issues above before using DeepSeek-OCR")
        print("   See TROUBLESHOOTING.md for detailed solutions")
        sys.exit(1)

if __name__ == "__main__":
    main()
