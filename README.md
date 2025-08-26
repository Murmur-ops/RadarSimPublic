# RadarSim: Advanced Radar Simulation with Machine Learning

## Overview

RadarSim is a comprehensive, physics-based radar simulation framework that combines signal processing and machine learning for real-time threat classification. Built entirely in Python with NumPy/SciPy, it provides a complete end-to-end radar system simulation without external dependencies.

## Key Features

### 🎯 **Core Capabilities**
- **Full IQ Data Generation**: Realistic in-phase/quadrature signal synthesis with receiver impairments
- **Machine Learning Inference**: Real-time threat classification using custom Transformer and CNN architectures
- **Advanced Jamming**: DRFM, noise, deception, and false target generation
- **Multi-Target Tracking**: IMM filters with multiple motion models
- **YAML Configuration**: Fully configurable scenarios without code changes
- **No External Dependencies**: Pure Python/NumPy implementation

### 🧠 **Machine Learning Pipeline**

#### Architecture
- **Transformer Network**: Temporal pattern analysis with multi-head attention (64-dim, 4 heads)
- **CNN Feature Extractor**: Spatial signature recognition with 1D convolutions
- **Ensemble Classifier**: Combined 192-dimensional feature vector processing
- **Real-time Performance**: <10ms inference latency

#### Threat Classification
- **CRITICAL**: Anti-ship missiles (PW ≤ 2μs, PRF > 500Hz)
- **HIGH**: Fighter aircraft (frequency agile, PW 5-20μs)  
- **MEDIUM**: Helicopters/patrol aircraft (blade modulation detection)
- **LOW**: Commercial/navigation (PW > 500μs, PRF < 30Hz)

### 📡 **Signal Processing**

#### IQ Generation Pipeline
```python
Transmit Waveform → Target Echoes → Propagation Effects → 
Clutter → Thermal Noise → Receiver Impairments → ADC Quantization
```

#### Receiver Effects Modeled
- I/Q amplitude and phase imbalance
- Phase noise (-80 dBc/Hz typical)
- DC offset compensation
- ADC quantization (configurable bit depth)

### 🎮 **Simulation Components**

1. **Radar Systems**
   - Pulse-Doppler processing
   - Range-Doppler map generation
   - CFAR detection
   - Beam scheduling and resource management

2. **Target Modeling**
   - Swerling RCS fluctuation models (0, 1, 2, 3, 4)
   - Complex motion patterns (weaving, pop-up, terminal guidance)
   - Micro-Doppler signatures

3. **Electronic Warfare**
   - Coherent jamming (DRFM-based)
   - Range/velocity gate pull-off
   - False target injection
   - Cooperative jamming networks

4. **Environmental Effects**
   - Atmospheric propagation
   - Sea/land clutter (K-distribution)
   - Weather effects (rain, humidity)
   - Multipath propagation

## Installation

```bash
# Clone the repository
git clone git@github.com:Murmur-ops/RadarSimPublic.git
cd RadarSimPublic

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (only NumPy, SciPy, Matplotlib required)
pip install numpy scipy matplotlib pyyaml
```

**Note**: The system includes optional support for MEEP electromagnetic simulations for enhanced RCS calculations, but **MEEP is not required**. The simulation runs perfectly with built-in RCS models based on publicly available data.

## Quick Start

### 1. Run a Basic Scenario

```bash
# Run a pre-configured naval engagement scenario
python run_scenario.py naval_engagement

# List available scenarios
python run_scenario.py --list

# Run with visualization
python run_scenario.py missile_defense --output results/
```

### 2. ML Threat Classification Demo

```bash
# Run the ML threat assessment demonstration
python ml_threat_priority.py

# Train a new classifier on synthetic data
python ml_training_pipeline.py
```

### 3. Generate IQ Data

```python
from src.iq_generator import IQDataGenerator, IQParameters
from src.radar import Radar
from src.waveforms import WaveformGenerator

# Initialize components
radar = Radar(frequency=9.5e9, power=1e6)
waveform_gen = WaveformGenerator(sample_rate=100e6)
iq_gen = IQDataGenerator(radar, waveform_gen)

# Generate IQ data for targets
iq_data = iq_gen.generate_cpi(
    targets=target_list,
    num_pulses=256,
    waveform_type=WaveformType.LFM
)

# Save IQ data
iq_gen.save_iq_data(iq_data, "output.iq", format="complex64")
```

## YAML Configuration System

The simulation is fully configurable through YAML files. Here's the structure:

### Configuration Sections

1. **Scenario**: Overall simulation parameters
2. **Radar**: System specifications and processing parameters
3. **Targets**: List of targets with kinematics and signatures
4. **Jammers**: Electronic warfare systems
5. **Environment**: Atmospheric and clutter conditions

### Parameter Flexibility

- **Frequency**: Any radar band (L, S, C, X, Ku, Ka)
- **Waveforms**: LFM, pulse, stepped frequency, Barker codes
- **Tracking**: Kalman, Extended Kalman, IMM filters
- **Detection**: CA-CFAR, OS-CFAR, adaptive thresholds

## Machine Learning Features

### PDW Feature Extraction (13 base features)
- Frequency statistics (mean, std)
- Pulse width characteristics  
- Amplitude patterns
- PRI (Pulse Repetition Interval) analysis
- Frequency agility detection
- Temporal density metrics

### Advanced Feature Extraction (19 total features)
- **Kinematic**: Velocity, acceleration, closing rate, turn rate
- **RCS**: Statistical moments, fluctuation rate
- **Doppler**: Spread, centroid, number of components
- **Spectral**: Entropy, flux, peak frequency, bandwidth

### Training Pipeline
1. **Synthetic Data Generation**: Based on publicly known radar characteristics
2. **Realistic RCS Models**: Built-in Swerling models and typical values
3. **Target Fluctuation**: Statistical RCS variation over time
4. **Real-time Inference**: Optimized for <10ms classification

## Performance Metrics

- **Detection Probability**: Per-target and aggregate
- **False Alarm Rate**: CFAR-controlled
- **Track Continuity**: Percentage of maintained tracks
- **Classification Accuracy**: ~85-90% on synthetic data
- **Inference Speed**: <10ms average (real-time capable)

## Architecture

```
RadarSim/
├── src/
│   ├── radar_simulation/     # Core radar physics
│   ├── tracking/             # Multi-target tracking
│   ├── classification/       # Feature extraction
│   ├── jamming/             # EW systems
│   └── iq_generator.py      # IQ data synthesis
├── configs/
│   └── scenarios/           # YAML configurations
├── examples/                # Example scripts
├── ml_threat_priority.py    # ML inference engine
├── ml_training_pipeline.py  # Training system
└── run_scenario.py          # Main simulation runner
```

## Example Use Cases

1. **Naval Defense**: Multi-missile engagement with ECM
2. **Air Defense**: Fighter intercept with jamming
3. **Surveillance**: Wide-area search with classification
4. **Training**: Generate labeled IQ data for ML models
5. **Algorithm Development**: Test tracking and detection algorithms

## Dependencies

Core requirements (minimal):
- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- PyYAML

Optional enhancements:
- H5py (for HDF5 IQ data format)
- MEEP (for electromagnetic RCS calculations - not required)

## Contributing

We welcome contributions! Areas of interest:
- Additional waveform types
- Enhanced clutter models
- New tracking algorithms
- Improved ML architectures
- Performance optimizations

## License

MIT License (see LICENSE file)

## Citation

If you use RadarSim in your research, please cite:
```bibtex
@software{radarsim2024,
  title={RadarSim: Advanced Radar Simulation with Machine Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/Murmur-ops/RadarSimPublic}
}
```

---

*RadarSim: Where physics meets machine learning in radar simulation*