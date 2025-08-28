# Rust Acceleration for RadarSim

This document explains how to build and use the optional Rust acceleration module for improved performance.

## Prerequisites

1. **Install Rust** (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

2. **Install maturin** (Python-Rust build tool):
   ```bash
   pip install maturin
   ```

## Building the Rust Extension

### Option 1: Development Build (for testing)
```bash
# From the RadarSimPublic directory
maturin develop
```

This builds the extension in debug mode and installs it in your current Python environment.

### Option 2: Release Build (optimized)
```bash
# From the RadarSimPublic directory
maturin develop --release
```

This builds an optimized version for better performance.

### Option 3: Build Wheel (for distribution)
```bash
maturin build --release
# Install the wheel
pip install target/wheels/radar_core-*.whl
```

## Verifying Installation

After building, you can verify the Rust acceleration is available:

```python
import radar_core
print("Rust acceleration available!")
```

If successful, the warning "Rust acceleration not available" will no longer appear when running scenarios.

## What Gets Accelerated

The Rust module accelerates the following computationally intensive operations:

1. **Matched Filtering** (`matched_filter_rust`)
   - FFT-based correlation for pulse compression
   - Used in signal processing pipeline

2. **Range-Doppler Processing** (`range_doppler_map_rust`)
   - 2D FFT with optional windowing
   - Critical for radar signal analysis

3. **CFAR Detection** (`cfar_detect_rust`)
   - Constant False Alarm Rate detection
   - Fast sliding window processing

4. **Pulse Compression** (`pulse_compress_rust`)
   - Chirp matched filtering
   - Optimized FFT implementation

5. **Kalman Filter Prediction** (`kalman_predict_rust`)
   - Matrix operations for tracking
   - State and covariance prediction

## Performance Impact

Typical performance improvements with Rust acceleration:
- Matched filtering: 3-5x faster for large arrays
- Range-Doppler maps: 2-4x faster
- CFAR detection: 5-10x faster
- Kalman prediction: 2-3x faster for large state vectors

## Troubleshooting

### Error: "error: Microsoft Visual C++ 14.0 is required" (Windows)
Install Visual Studio Build Tools or Visual Studio Community with C++ development tools.

### Error: "error: linking with cc failed" (Linux)
Install build essentials:
```bash
sudo apt-get install build-essential  # Ubuntu/Debian
sudo yum groupinstall "Development Tools"  # RHEL/CentOS
```

### Error: "no default toolchain configured"
Run:
```bash
rustup default stable
```

### Python can't find radar_core after building
Make sure you're in the correct Python environment:
```bash
which python  # Should show your virtual env if using one
maturin develop  # Rebuild in correct environment
```

## Development Notes

The Rust source code is in `rust/src/lib.rs`. Key dependencies:
- `pyo3`: Python bindings
- `numpy`: NumPy array handling
- `ndarray`: Rust n-dimensional arrays
- `rustfft`: Fast Fourier Transform
- `num-complex`: Complex number support

To modify the Rust code:
1. Edit `rust/src/lib.rs`
2. Run `maturin develop` to rebuild
3. Test changes in Python

The Rust module is optional - the simulation will fall back to pure Python implementations if not available.