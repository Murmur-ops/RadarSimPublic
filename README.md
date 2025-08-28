# RadarSim - Clean Repository

A comprehensive radar simulation framework for modeling detection, tracking, jamming, and networked radar systems. This is a cleaned and organized version of the repository for easier navigation.

## Key Features

### **Core Capabilities**
- **Full IQ Data Generation**: Realistic in-phase/quadrature signal synthesis with receiver impairments
- **Machine Learning Inference**: Real-time threat classification using custom Transformer and CNN architectures
- **Advanced Jamming**: DRFM, noise, deception, and false target generation
- **Multi-Target Tracking**: IMM filters with multiple motion models
- **YAML Configuration**: Fully configurable scenarios without code changes
- **No External Dependencies**: Pure Python/NumPy implementation

### **Machine Learning Pipeline**

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

### **Signal Processing**

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

### **Simulation Components**

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

## Repository Structure (Cleaned)

```
RadarSim/
├── src/                      # Core simulation modules
│   ├── radar.py             # Radar system implementation
│   ├── target.py            # Target modeling
│   ├── signal.py            # Signal processing
│   ├── tracking/            # Tracking algorithms (IMM, Kalman filters)
│   ├── jamming/             # Electronic warfare (DRFM, false targets)
│   ├── networked_radar/     # Multi-radar coordination
│   ├── classification/      # Target classification & signatures
│   └── resource_management/ # Beam scheduling & prioritization
│
├── configs/                  # Configuration files
│   ├── scenarios/           # Pre-built scenarios (YAML)
│   └── priorities/          # Priority configurations
│
├── demos/                    # Demonstration scripts
│   ├── basic_jamming_demo.py
│   ├── networked_radar_simple.py
│   └── missile_salvo_visualization.py
│
├── getting_started/          # Tutorial materials
│   ├── simple_demo.py       # Basic example script
│   ├── first_scenario.yaml  # Simple scenario config
│   └── STEP_BY_STEP_TUTORIAL.md
│
├── ml_models/               # Machine learning components
│   ├── ml_threat_priority.py
│   └── ml_training_pipeline.py
│
├── tests/                   # Unit and integration tests
│   └── test_*.py
│
└── docs/                    # Documentation
    ├── QUICK_START.md
    └── ML_INFERENCE_PIPELINE.md
```

### What Was Cleaned Up
- Removed 268MB venv directory from GETTING_STARTED
- Removed 96MB target directory (Rust build artifacts)
- Cleaned 301 __pycache__ directories with 2990 .pyc files
- Organized test files into dedicated tests/ directory
- Consolidated documentation into docs/ directory
- **Result: Reduced from 366MB to ~2MB**

## Installation

```bash
# Install dependencies (only NumPy, SciPy, Matplotlib required)
pip install -r requirements.txt
```

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
python -m ml_models.ml_threat_priority

# Train a new classifier on synthetic data
python -m ml_models.ml_training_pipeline

# Or use the simple example
python examples/simple_ml_demo.py
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
├── src/                     # Core simulation modules
│   ├── radar_simulation/    # Radar physics and propagation
│   ├── tracking/           # Multi-target tracking algorithms
│   ├── classification/    # Feature extraction and signatures
│   ├── jamming/           # Electronic warfare systems
│   ├── resource_management/ # Beam scheduling and prioritization
│   └── iq_generator.py    # IQ data synthesis
├── ml_models/              # Machine learning modules
│   ├── ml_threat_priority.py  # Inference engine
│   ├── ml_training_pipeline.py # Training system
│   └── ml_with_meep_rcs.py    # MEEP integration (optional)
├── configs/
│   └── scenarios/         # YAML scenario configurations
├── demos/                 # Demonstration scripts
│   └── basic_jamming_demo.py
├── examples/              # Simple example scripts
│   └── simple_ml_demo.py
└── run_scenario.py        # Main simulation runner
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

## References

This simulation framework builds upon foundational radar theory and modern signal processing techniques:

### Radar Fundamentals
- Richards, M. A., Scheer, J., Holm, W. A. (Eds.). (2010). *Principles of Modern Radar: Basic Principles*. SciTech Publishing.
- Skolnik, M. I. (2008). *Radar Handbook* (3rd ed.). McGraw-Hill.
- Levanon, N., & Mozeson, E. (2004). *Radar Signals*. Wiley-IEEE Press.

### Signal Processing & Detection
- Kay, S. M. (1998). *Fundamentals of Statistical Signal Processing: Detection Theory*. Prentice Hall.
- Mahafza, B. R. (2013). *Radar Systems Analysis and Design Using MATLAB* (3rd ed.). CRC Press.
- Farina, A., & Studer, F. A. (1985). *Radar Data Processing* (Vols. 1-2). Research Studies Press.

### Target Tracking
- Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation*. Wiley.
- Blackman, S., & Popoli, R. (1999). *Design and Analysis of Modern Tracking Systems*. Artech House.
- Blom, H. A. P., & Bar-Shalom, Y. (1988). The Interacting Multiple Model Algorithm for Systems with Markovian Switching Coefficients. *IEEE Transactions on Automatic Control*, 33(8), 780-783.

### Electronic Warfare & Jamming
- Schleher, D. C. (1999). *Electronic Warfare in the Information Age*. Artech House.
- Neri, F. (2018). *Introduction to Electronic Defense Systems* (3rd ed.). Artech House.
- Adamy, D. (2015). *EW 104: Electronic Warfare Against a New Generation of Threats*. Artech House.

### Machine Learning for Radar
- Ender, J., Leushacke, L., Brenner, A., & Wilden, H. (2011). Radar Techniques and Technologies. In *NATO Science and Technology Organization*.
- Chen, V. C., & Ling, H. (2002). *Time-Frequency Transforms for Radar Imaging and Signal Analysis*. Artech House.
- Haykin, S., & Deng, C. (1991). Classification of Radar Clutter Using Neural Networks. *IEEE Transactions on Neural Networks*, 2(6), 589-600.

### Radar Cross Section & Electromagnetic Theory
- Knott, E. F., Shaeffer, J. F., & Tuley, M. T. (2004). *Radar Cross Section* (2nd ed.). SciTech Publishing.
- Ulaby, F. T., & Long, D. G. (2014). *Microwave Radar and Radiometric Remote Sensing*. University of Michigan Press.

### Public Domain Radar Data
- Swerling, P. (1960). Probability of Detection for Fluctuating Targets. *IRE Transactions on Information Theory*, 6(2), 269-308.
- Various publicly available radar specifications from defense contractor publications and academic papers.

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