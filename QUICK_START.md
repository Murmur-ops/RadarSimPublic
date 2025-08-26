# RadarSim Quick Start Guide

## Installation (5 minutes)

```bash
# Clone the repository
git clone git@github.com:Murmur-ops/RadarSimPublic.git
cd RadarSimPublic

# Set up Python environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install minimal dependencies
pip install numpy scipy matplotlib pyyaml
```

## Run Your First Simulation (2 minutes)

### Option 1: Run Pre-built Scenario

```bash
# Run basic tracking scenario
python run_scenario.py basic_tracking

# View available scenarios
python run_scenario.py --list
```

### Option 2: ML Threat Classification Demo

```bash
# See machine learning in action
python ml_threat_priority.py
```

Expected output:
```
ML THREAT PRIORITY ASSESSMENT DEMONSTRATION
============================================================

1. Fire Control Radar (Missile Guidance)
  Classification: CRITICAL
  Confidence: 94.3%
  Inference time: 6.82ms

2. Search Radar (Surveillance)
  Classification: MEDIUM
  Confidence: 87.1%
  Inference time: 5.91ms
```

## Create Your First Custom Scenario

### Step 1: Create YAML Configuration

Create `my_scenario.yaml`:

```yaml
scenario:
  name: "My First Scenario"
  description: "Track a single aircraft"
  duration: 30.0
  time_step: 0.1

radar:
  type: "surveillance"
  parameters:
    frequency: 3.0e9  # S-band
    power: 5000
    antenna_gain: 35
    pulse_width: 1.0e-6
    prf: 1000
    bandwidth: 10.0e6
    noise_figure: 3
    losses: 2
    
  processing:
    range_resolution: 50
    max_range: 150000
    velocity_resolution: 1.0
    n_doppler_bins: 256
    detection_threshold: 12

targets:
  - name: "Aircraft-1"
    type: "commercial"
    initial_position:
      range: 100000  # 100km
      azimuth: 45
      elevation: 10
    velocity: -200  # 200 m/s inbound
    rcs: 100  # Large aircraft

environment:
  weather: "clear"
```

### Step 2: Run Your Scenario

```bash
python run_scenario.py my_scenario
```

## Generate IQ Data

```python
# simple_iq_generation.py
import numpy as np
from src.iq_generator import IQDataGenerator, IQParameters
from src.radar import Radar, RadarParameters
from src.waveforms import WaveformGenerator
from src.target import Target

# Setup radar
radar_params = RadarParameters(
    frequency=10e9,  # X-band
    power=1000,      # 1kW
    antenna_gain=30
)
radar = Radar(radar_params)

# Setup waveform generator
waveform_gen = WaveformGenerator(sample_rate=100e6)

# Create IQ generator
iq_gen = IQDataGenerator(radar, waveform_gen)

# Define a target
target = Target(
    range=10000,    # 10km
    velocity=-100,  # 100 m/s inbound
    rcs=5          # 5 m² RCS
)

# Generate IQ data
iq_data = iq_gen.generate_single_pulse(
    tx_waveform=waveform_gen.generate_lfm_chirp(),
    targets=[target]
)

print(f"Generated {len(iq_data)} IQ samples")
print(f"Signal power: {np.mean(np.abs(iq_data)**2):.2e} W")

# Save to file
iq_gen.save_iq_data(iq_data, "my_iq_data.dat", format="complex64")
```

## Understanding the Output

### Visualization Files

After running a scenario, check the `results/` directory:

- **Range vs Time Plot**: Shows target trajectories
- **Detection Probability**: Bar chart of detection rates
- **SNR vs Range**: Signal strength analysis
- **Metrics Summary**: Performance statistics

### Key Metrics Explained

- **Detection Probability (Pd)**: Percentage of time target was detected
- **False Alarm Rate (FAR)**: Ratio of false detections to total
- **Track Continuity**: Percentage of continuous tracking
- **Classification Accuracy**: ML model performance

## Common Scenarios

### 1. Missile Defense
```bash
python run_scenario.py missile_defense_simple
```
Simulates ship defending against sea-skimming missiles.

### 2. Jamming Effects
```bash
python run_scenario.py jamming_demo
```
Demonstrates electronic warfare and countermeasures.

### 3. ML Classification
```bash
python run_scenario.py ml_classification
```
Shows real-time threat classification across multiple target types.

## Troubleshooting

### Issue: "Module not found"
```bash
# Ensure you're in the venv
which python  # Should show venv path
# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "No scenarios found"
```bash
# Check config directory exists
ls configs/scenarios/
# Use example configs
cp example_configs/*.yaml configs/scenarios/
```

### Issue: "MEEP not found" warning
This is normal - MEEP is optional for enhanced RCS calculations. The simulation uses built-in RCS models.

## Next Steps

1. **Explore Examples**: Look at `example_configs/` for more scenarios
2. **Read ML Documentation**: See `ML_INFERENCE_PIPELINE.md`
3. **Modify Parameters**: Try changing radar frequency, power, or PRF
4. **Add Targets**: Create multi-target scenarios
5. **Enable Jamming**: Add jammers to test electronic warfare

## Getting Help

- Check the main README for detailed documentation
- Look at example scripts in the `examples/` directory
- Review the source code - it's well-commented
- File issues on GitHub for bugs or questions

## Performance Tips

- Start with shorter durations (30-60 seconds)
- Use fewer Doppler bins for faster processing
- Disable visualization with `--no-viz` for batch runs
- Reduce time_step for faster but less accurate simulation

---

You're now ready to simulate radar systems with machine learning! 🎯📡