# RadarSimPublic - High-Fidelity Radar Simulation Platform

A comprehensive radar simulation framework for research, education, and system design. Pure Python implementation with NumPy acceleration.

## 🚀 Quick Start

### Installation (30 seconds)

```bash
# Clone repository
git clone https://github.com/Murmur-ops/RadarSimPublic.git
cd RadarSimPublic

# Run setup (installs dependencies automatically)
python setup.py
```

### First Simulation (1 minute)

```bash
# Run the quick start demo
python quick_start.py
```

This demonstrates:
- X-band radar detecting multiple targets
- SNR calculation with atmospheric effects
- Real-time visualization (PPI display)

## 📁 Repository Structure

```
RadarSimPublic/
├── src/                    # Core simulation modules
│   ├── radar.py           # Radar system modeling
│   ├── target.py          # Target dynamics & RCS
│   ├── signal.py          # Signal processing
│   ├── tracking/          # Kalman filters & tracking
│   ├── jamming/           # Electronic warfare
│   └── classification/    # ML-based classification
├── configs/               # YAML scenario configurations
├── examples/              # Example simulations
├── demos/                 # Advanced demonstrations
└── GETTING_STARTED/       # Tutorial materials
```

## 🎯 Core Features

### Radar Modeling
- **Multiple radar types**: Phased array, pulse-Doppler, FMCW
- **Frequency bands**: L, S, C, X, Ku (1-40 GHz)
- **Signal processing**: CFAR detection, matched filtering, Doppler processing
- **No cheating**: All detections from realistic signal processing

### Target Simulation
- **RCS models**: Swerling 0-4 fluctuation
- **Motion models**: CV, CA, CT, Singer, IMM
- **Target types**: Aircraft, missiles, drones, ships
- **Multi-target**: 100+ simultaneous targets

### Electronic Warfare
- **DRFM jamming**: Coherent false targets, gate pull-off
- **Noise jamming**: Barrage, spot, swept
- **ECCM techniques**: Frequency agility, burn-through

### Machine Learning
- **Real-time classification**: <10ms inference
- **Threat assessment**: Critical/High/Medium/Low
- **Neural architectures**: CNN + Transformer ensemble

## 💻 Basic Usage

```python
from src.radar import Radar, RadarParameters
from src.target import Target, TargetType, TargetMotion
import numpy as np

# Configure radar
params = RadarParameters(
    frequency=10e9,      # 10 GHz X-band
    power=5000,          # 5 kW
    antenna_gain=35,     # 35 dB
    prf=2000,           # 2 kHz PRF
    bandwidth=50e6      # 50 MHz
)
radar = Radar(params)

# Create target
target = Target(
    target_type=TargetType.AIRCRAFT,
    rcs=2.0,  # 2 m² RCS
    motion=TargetMotion(
        position=np.array([10000, 5000, 3000]),
        velocity=np.array([-200, 0, 0])
    )
)

# Calculate detection
snr = radar.snr(target.motion.range, target.rcs_mean)
print(f"SNR: {snr:.1f} dB")
```

## 📊 Example Scenarios

### 1. Basic Detection
```bash
python examples/basic_simulation.py
```
Simple radar detecting aircraft and drones.

### 2. Multi-Target Tracking
```bash
python demos/multi_target_tracking.py
```
Track 10+ targets with Kalman filtering.

### 3. Electronic Warfare
```bash
python demos/jamming_scenario.py
```
DRFM jammer creating false targets.

### 4. YAML Configuration
```bash
python run_scenario.py configs/scenarios/air_defense.yaml
```
Load complete scenarios from YAML files.

## 🔧 Configuration

Scenarios can be fully defined in YAML:

```yaml
radar:
  type: "phased_array"
  frequency: 10.0e9
  power: 5000
  
targets:
  - name: "Fighter-1"
    type: "aircraft"
    position: [20000, 10000, 5000]
    velocity: [-300, 0, 0]
    rcs: 2.0
```

## 📈 Performance

- **Update rate**: Up to 100 Hz
- **Target capacity**: 100+ simultaneous
- **Processing**: Pure Python/NumPy (no external dependencies)
- **ML inference**: <10ms per classification

## 📚 Documentation

- **Quick Start Guide**: [GETTING_STARTED/README.md](GETTING_STARTED/README.md)
- **API Reference**: Full docstrings in source code
- **Examples**: Commented examples in `examples/` directory
- **ML Pipeline**: [ML_INFERENCE_PIPELINE.md](ML_INFERENCE_PIPELINE.md)

## 🧪 Testing

```bash
# Verify installation
python setup.py

# Run minimal example
python quick_start.py

# Run all examples
python examples/run_all_examples.py
```

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code follows existing style
- Add tests for new features
- Update documentation
- Run `python setup.py` to verify

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

Built with NumPy, SciPy, and Matplotlib. Inspired by radar systems engineering principles from Skolnik, Richards, and Bar-Shalom.

---

**Getting Started?** Run `python quick_start.py` for an immediate demonstration!

**Questions?** Check the examples or open an issue on GitHub.