#!/usr/bin/env python3
"""
RadarSimPublic Setup Script
Quick setup and verification for the radar simulation system
"""

import sys
import subprocess
import importlib
import os


def check_python_version():
    """Ensure Python 3.8+ is being used"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    required = {
        'numpy': '1.20.0',
        'scipy': '1.7.0',
        'matplotlib': '3.3.0',
        'yaml': '5.4.0'
    }
    
    missing = []
    for package, min_version in required.items():
        try:
            if package == 'yaml':
                mod = importlib.import_module('yaml')
            else:
                mod = importlib.import_module(package)
            
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {package}: {version}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}: not installed")
    
    return missing


def install_dependencies(missing):
    """Install missing dependencies"""
    if not missing:
        return True
        
    print("\n📦 Installing missing dependencies...")
    try:
        cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        subprocess.check_call(cmd)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("   Please run: pip install -r requirements.txt")
        return False


def verify_installation():
    """Run a simple test to verify the installation works"""
    print("\n🧪 Verifying installation...")
    
    try:
        # Test basic imports
        from src.radar import Radar, RadarParameters
        from src.target import Target, TargetType
        from src.signal import SignalProcessor
        print("✅ Core modules imported successfully")
        
        # Test basic functionality
        params = RadarParameters(
            frequency=10e9,
            power=1000,
            antenna_gain=30,
            pulse_width=1e-6,
            prf=1000,
            bandwidth=10e6,
            noise_figure=3.0,
            losses=2.0
        )
        radar = Radar(params)
        print("✅ Radar object created successfully")
        
        # Test SNR calculation
        snr = radar.snr(range_m=10000, rcs=1.0)
        print(f"✅ SNR calculation working: {snr:.1f} dB")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    """Main setup function"""
    print("=" * 60)
    print("RadarSimPublic Setup & Verification")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    print("\n📋 Checking dependencies...")
    missing = check_dependencies()
    
    # Install if needed
    if missing:
        if not install_dependencies(missing):
            sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ Setup complete! RadarSimPublic is ready to use.")
    print("\nNext steps:")
    print("1. Run a basic example: python examples/basic_simulation.py")
    print("2. Try the tutorial: python GETTING_STARTED/run_first_simulation.py")
    print("3. Explore configs: python run_scenario.py configs/scenarios/simple_tracking.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()