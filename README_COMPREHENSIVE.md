# RadarSim - Advanced Radar System Simulation Framework

A comprehensive, high-fidelity radar simulation framework with support for multiple radar types, advanced signal processing, electronic warfare, and networked operations.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git
- (Optional) Rust compiler for acceleration

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/RadarSimPublic.git
cd RadarSimPublic
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **(Optional) Install Rust acceleration:**
```bash
# Install Rust if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install maturin and build Rust extension
pip install maturin
maturin develop --release
```

### First Simulation

Run your first simulation in under 30 seconds:

```bash
# Run the getting started example
python GETTING_STARTED/run_first_simulation.py

# Or run a pre-configured scenario
python run_scenario.py simple_tracking.yaml
```

## 📚 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Getting Started Guide](#getting-started-guide)
- [Configuration](#configuration)
- [Examples](#examples)
- [Advanced Topics](#advanced-topics)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## ✨ Features

### Core Capabilities
- **Multi-Mode Radar Systems**: Pulse, CW, FMCW, pulse-Doppler
- **Advanced Signal Processing**: Matched filtering, pulse compression, CFAR detection
- **Target Tracking**: Kalman filters, IMM, JPDA, MHT algorithms
- **Electronic Warfare**: Jamming, DRFM, false targets, gate pull-off
- **Network-Centric Operations**: Distributed tracking, sensor fusion
- **IADS/SEAD**: Integrated air defense and suppression simulation

### Signal Processing
- Range-Doppler processing
- Beam forming and steering
- Clutter modeling and suppression
- STAP (Space-Time Adaptive Processing)
- Micro-Doppler signatures
- RCS modeling (Swerling targets)

### Tracking & Data Association
- Extended/Unscented Kalman Filters
- Interacting Multiple Model (IMM)
- Joint Probabilistic Data Association (JPDA)
- Multiple Hypothesis Tracking (MHT)
- Track initiation and management
- Sensor fusion with out-of-sequence measurements

### Electronic Warfare
- Noise jamming (barrage, spot)
- Deception jamming (DRFM-based)
- False target generation
- Range/velocity gate pull-off
- Coherent jamming techniques
- ECCM detection and mitigation

### Advanced Systems
- Networked radar with data fusion
- IADS with layered SAM systems
- SEAD operations with Wild Weasel
- Resource management and scheduling
- Cognitive radar adaptation

## 🏗️ Architecture

```
RadarSimPublic/
├── src/                      # Core simulation modules
│   ├── radar.py             # Main radar class
│   ├── target.py            # Target models
│   ├── signal.py            # Signal processing
│   ├── tracking/            # Tracking algorithms
│   ├── jamming/             # EW capabilities
│   ├── networked_radar/     # Network operations
│   ├── iads/                # Air defense systems
│   ├── sead/                # SEAD operations
│   └── classification/      # Target classification
├── configs/                  # Configuration files
│   ├── scenarios/           # Pre-built scenarios
│   └── templates/           # Config templates
├── demos/                    # Demonstration scripts
├── GETTING_STARTED/         # Quick start materials
├── rust/                    # Rust acceleration (optional)
└── results/                 # Output directory
```

## 📖 Getting Started Guide

### Step 1: Understanding the Basics

RadarSim uses YAML configuration files to define scenarios. A scenario includes:
- Radar systems and their parameters
- Targets and their trajectories
- Environmental conditions
- Simulation settings

### Step 2: Running Your First Simulation

```python
# GETTING_STARTED/run_first_simulation.py
import sys
sys.path.append('..')

from src.radar import Radar
from src.target import Target
import numpy as np
import matplotlib.pyplot as plt

# Create a simple radar
radar = Radar(
    position=np.array([0, 0, 0]),
    frequency=10e9,  # X-band
    power=1e3,        # 1 kW
    antenna_gain=30   # 30 dB
)

# Create a target
target = Target(
    position=np.array([10000, 5000, 3000]),  # 10km range
    velocity=np.array([-200, 0, 0]),         # 200 m/s closing
    rcs=10.0                                  # 10 m² RCS
)

# Simulate detection
snr = radar.calculate_snr(target)
print(f"Target SNR: {snr:.2f} dB")

if snr > radar.detection_threshold:
    print("Target detected!")
```

### Step 3: Using YAML Configurations

Create a scenario file `my_scenario.yaml`:

```yaml
scenario:
  name: "My First Scenario"
  duration: 60.0
  time_step: 0.1

radar_systems:
  - type: "pulse_doppler"
    position: [0, 0, 10]
    frequency: 9.4e9
    prf: 1000
    pulse_width: 1e-6
    antenna:
      gain: 35
      beamwidth: 2.0

targets:
  - name: "Aircraft-1"
    type: "aircraft"
    initial_position: [50000, 10000, 8000]
    velocity: [-250, 0, 0]
    rcs: 5.0
    trajectory: "straight"
```

Run it:
```bash
python run_scenario.py my_scenario.yaml
```

### Step 4: Visualizing Results

```python
from src.visualization import RadarDisplay

display = RadarDisplay()
display.plot_ppi(detections)  # Plan Position Indicator
display.plot_tracks(tracks)   # Track visualization
display.show()
```

## ⚙️ Configuration

### Radar Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| frequency | Operating frequency | 1-40 GHz |
| power | Transmit power | 1-1000 kW |
| prf | Pulse repetition frequency | 100-10000 Hz |
| pulse_width | Pulse duration | 0.1-100 μs |
| antenna_gain | Antenna gain | 20-40 dB |
| beamwidth | Antenna beamwidth | 1-10° |

### Target Models

- **Aircraft**: Commercial, fighter, bomber
- **Missiles**: Cruise, ballistic, hypersonic
- **Ships**: Corvette, frigate, carrier
- **Ground**: Vehicles, buildings, terrain

### Environment

- Atmospheric attenuation
- Ground clutter
- Sea clutter
- Weather effects
- Multipath propagation

## 💡 Examples

### 1. Air Defense Scenario
```bash
python run_scenario.py configs/scenarios/air_defense.yaml
```
Simulates layered air defense against multiple threats.

### 2. Electronic Warfare
```bash
python demos/growler_false_target_visualization.py
```
Demonstrates EA-18G Growler jamming and deception.

### 3. Networked Surveillance
```bash
python demos/networked_radar_demo.py
```
Shows distributed tracking with sensor fusion.

### 4. SEAD vs IADS
```bash
python demos/sead_vs_iads_demo.py
```
Wild Weasel SEAD operations against integrated air defense.

## 🔬 Advanced Topics

### Custom Signal Processing

```python
from src.signal import SignalProcessor

processor = SignalProcessor(sample_rate=10e6, bandwidth=1e6)

# Matched filtering
compressed = processor.matched_filter(received, reference)

# Range-Doppler map
rd_map = processor.range_doppler_processing(data_cube)

# CFAR detection
detections = processor.cfar_2d(rd_map, guard_cells=3, training_cells=10)
```

### Implementing Custom Jammers

```python
from src.jamming import Jammer

class CustomJammer(Jammer):
    def generate_signal(self, radar_signal):
        # Your jamming algorithm
        return jammed_signal
```

### Machine Learning Integration

```python
from ml_models.ml_threat_priority import ThreatClassifier

classifier = ThreatClassifier()
classifier.train(training_data)
threat_level = classifier.predict(target_features)
```

### Resource Management

```python
from src.resource_management import ResourceManager

manager = ResourceManager(radar)
manager.set_priority_function(custom_priority)
schedule = manager.optimize_beam_schedule(tracks)
```

## 📊 Performance Metrics

The framework tracks various performance metrics:

- **Detection**: Pd, Pfa, SNR
- **Tracking**: RMSE, track continuity, track purity
- **Resource**: Utilization, response time, coverage
- **Network**: Latency, fusion gain, track consistency

## 🛠️ API Reference

### Core Classes

#### Radar
```python
radar = Radar(position, frequency, power, **kwargs)
radar.detect(targets)
radar.track(detections)
```

#### Target
```python
target = Target(position, velocity, rcs)
target.update(dt)
target.get_signature()
```

#### Tracker
```python
tracker = KalmanTracker()
tracker.predict(dt)
tracker.update(measurement)
```

### Utilities

```python
from src.constants import c, k_b
from src.validators import validate_scenario
from src.environment import Atmosphere
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest src/tracking/tests/
python test_yaml_configs.py
python test_networked_basic.py
```

## 📈 Performance Optimization

1. **Use Rust acceleration** for 2-10x speedup
2. **Batch operations** when possible
3. **Adjust fidelity** based on requirements
4. **Use appropriate coordinate systems**
5. **Profile bottlenecks** with cProfile

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- New radar types (bistatic, passive)
- Additional jammers and ECCM
- Performance optimizations
- Documentation improvements
- Bug fixes and testing

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Signal processing algorithms from radar literature
- Tracking filters based on Bar-Shalom's texts
- EW techniques from Adamy's EW series
- Community contributors and testers

## 📧 Contact

For questions, issues, or collaboration:
- GitHub Issues: [Report bugs or request features](https://github.com/yourusername/RadarSimPublic/issues)
- Email: your.email@example.com

## 🗺️ Roadmap

### Upcoming Features
- [ ] Passive radar capabilities
- [ ] Cognitive radar adaptation
- [ ] GPU acceleration
- [ ] Real-time visualization
- [ ] Hardware-in-the-loop support
- [ ] Extended ML integration

---

**Happy Simulating!** 🚀

For detailed documentation, visit our [Wiki](https://github.com/yourusername/RadarSimPublic/wiki).