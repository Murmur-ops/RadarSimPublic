# Step-by-Step RadarSim Tutorial

Welcome to RadarSim! This tutorial will walk you through everything from installation to running advanced simulations.

## Table of Contents
1. [Installation](#1-installation)
2. [First Simulation](#2-first-simulation)
3. [Understanding YAML Configurations](#3-understanding-yaml-configurations)
4. [Running Pre-built Scenarios](#4-running-pre-built-scenarios)
5. [Customizing Your Simulation](#5-customizing-your-simulation)
6. [Advanced Features](#6-advanced-features)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Installation

### Step 1.1: Prerequisites Check

First, verify you have the required software:

```bash
# Check Python version (need 3.8+)
python --version

# Check pip
pip --version

# Check git
git --version
```

### Step 1.2: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/RadarSimPublic.git

# Navigate to the directory
cd RadarSimPublic
```

### Step 1.3: Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 1.4: Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt
```

### Step 1.5: (Optional) Install Rust Acceleration

For better performance:

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install maturin
pip install maturin

# Build Rust extension
maturin develop --release
```

### Step 1.6: Verify Installation

```bash
# Test import
python -c "from src.radar import Radar; print('✓ RadarSim installed successfully!')"
```

---

## 2. First Simulation

### Step 2.1: Run the Getting Started Script

```bash
# Navigate to GETTING_STARTED directory
cd GETTING_STARTED

# Run your first simulation
python run_first_simulation.py
```

You should see output like:
```
============================================================
RadarSim - Getting Started Tutorial
============================================================

1. Setting up radar system...
   Radar at position: [0. 0. 10.]
   Operating frequency: 10.0 GHz
   Transmit power: 1000 W

2. Creating target...
   Target ID: Flight-001
   Initial position: [30. 15. 10.] km
   Velocity: 206.2 m/s
   RCS: 100.0 m²
...
```

### Step 2.2: Understanding the Output

The simulation will:
1. Create a radar at the origin
2. Generate an aircraft target
3. Simulate detection over 30 seconds
4. Track the target using a Kalman filter
5. Generate visualization plots

### Step 2.3: View Results

Check the `results/getting_started/` directory for:
- `first_simulation.png` - Visualization plots
- Detection and tracking data

---

## 3. Understanding YAML Configurations

### Step 3.1: Basic Structure

Open `first_scenario.yaml` to see the configuration structure:

```yaml
scenario:
  name: "Scenario Name"
  duration: 30.0      # Simulation time in seconds
  time_step: 0.1      # Update interval

radar_systems:        # Define one or more radars
  - name: "Radar-1"
    type: "pulse_doppler"
    # ... parameters

targets:              # Define targets
  - name: "Target-1"
    type: "aircraft"
    # ... parameters

environment:          # Environmental conditions
  atmosphere:
    temperature: 15.0
    # ... conditions
```

### Step 3.2: Key Parameters

**Radar Parameters:**
- `frequency`: Operating frequency (Hz)
- `power`: Transmit power (Watts)
- `prf`: Pulse Repetition Frequency
- `antenna_gain`: Antenna gain (dB)

**Target Parameters:**
- `position`: 3D position [x, y, z] (meters)
- `velocity`: 3D velocity [vx, vy, vz] (m/s)
- `rcs`: Radar Cross Section (m²)

### Step 3.3: Run the YAML Scenario

```bash
# From the RadarSimPublic root directory
cd ..
python run_scenario.py GETTING_STARTED/first_scenario.yaml
```

---

## 4. Running Pre-built Scenarios

### Step 4.1: List Available Scenarios

```bash
# List all scenarios
python run_scenario.py --list
```

### Step 4.2: Simple Tracking

```bash
# Basic single target tracking
python run_scenario.py simple_tracking.yaml
```

### Step 4.3: Air Defense

```bash
# Multi-layer air defense scenario
python run_scenario.py air_defense.yaml
```

### Step 4.4: Electronic Warfare

```bash
# Jamming and false targets
python run_scenario.py growler_false_target_deception.yaml
```

### Step 4.5: View Demonstrations

```bash
# Run visualization demos
python demos/missile_salvo_visualization.py
python demos/networked_radar_demo.py
python demos/sead_vs_iads_demo.py
```

---

## 5. Customizing Your Simulation

### Step 5.1: Create Custom Scenario

Create a new file `my_scenario.yaml`:

```yaml
scenario:
  name: "My Custom Scenario"
  duration: 60.0
  time_step: 0.1

radar_systems:
  - name: "CustomRadar"
    type: "pulse_doppler"
    position: [0, 0, 0]
    frequency: 9.4e9    # X-band
    power: 5000         # 5 kW
    antenna:
      gain: 40
      beamwidth: 1.5

targets:
  - name: "Fighter-1"
    type: "aircraft"
    subtype: "fighter"
    initial_position: [20000, 10000, 5000]
    velocity: [-300, 100, 0]
    rcs: 2.0
    
  - name: "Fighter-2"
    type: "aircraft"
    subtype: "fighter"
    initial_position: [-15000, 20000, 7000]
    velocity: [250, -150, 0]
    rcs: 2.0
```

Run it:
```bash
python run_scenario.py my_scenario.yaml
```

### Step 5.2: Programmatic Customization

Create `my_simulation.py`:

```python
import numpy as np
from src.radar import Radar
from src.target import Target
from src.jamming import NoiseJammer

# Create components
radar = Radar(
    position=np.array([0, 0, 0]),
    frequency=10e9,
    power=2000
)

target = Target(
    position=np.array([10000, 5000, 3000]),
    velocity=np.array([-200, 0, 0]),
    rcs=5.0
)

jammer = NoiseJammer(
    position=np.array([15000, 8000, 4000]),
    power=100,
    bandwidth=10e6
)

# Run simulation
for t in range(100):
    target.update(dt=0.1)
    snr = radar.calculate_snr(target)
    jnr = radar.calculate_jnr(jammer)
    
    if snr - jnr > radar.detection_threshold:
        print(f"Target detected at t={t*0.1:.1f}s")
```

---

## 6. Advanced Features

### Step 6.1: Networked Radar

```python
from src.networked_radar import NetworkedRadar, FusionCenter

# Create radar network
radars = [
    NetworkedRadar(position=np.array([0, 0, 0])),
    NetworkedRadar(position=np.array([10000, 0, 0])),
    NetworkedRadar(position=np.array([5000, 10000, 0]))
]

fusion_center = FusionCenter()
fusion_center.register_radars(radars)

# Fuse tracks
fused_track = fusion_center.fuse_tracks()
```

### Step 6.2: IADS Simulation

```python
from src.iads import IADSNetwork, SAMSite

# Create IADS
iads = IADSNetwork()

# Add SAM sites
iads.add_sam_site(SAMSite(
    position=np.array([0, 0, 0]),
    sam_type="SA-20",
    num_missiles=48
))

# Run engagement
iads.process_tracks(tracks)
iads.assign_weapons()
```

### Step 6.3: Machine Learning Integration

```python
from ml_models import ThreatClassifier

# Train classifier
classifier = ThreatClassifier()
classifier.train(training_data)

# Classify targets
for target in targets:
    features = extract_features(target)
    threat_level = classifier.predict(features)
```

---

## 7. Troubleshooting

### Common Issues and Solutions

#### Issue: "Module not found" error
```bash
# Make sure you're in the right directory
cd RadarSimPublic

# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall requirements
pip install -r requirements.txt
```

#### Issue: "Rust acceleration not available"
```bash
# Install Rust and rebuild
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin
maturin develop --release
```

#### Issue: Simulation runs slowly
- Enable Rust acceleration (see above)
- Reduce simulation fidelity in YAML:
  ```yaml
  simulation:
    time_step: 0.5  # Larger time step
    processing:
      reduced_fidelity: true
  ```

#### Issue: Out of memory
- Reduce number of targets
- Decrease simulation duration
- Disable data recording for some components

### Getting Help

1. Check the documentation in each module
2. Look at examples in `demos/` directory
3. Review test files in `src/tracking/tests/`
4. Submit issues on GitHub

---

## Next Steps

Now that you've completed the tutorial:

1. **Explore Scenarios**: Try all pre-built scenarios in `configs/scenarios/`
2. **Read Documentation**: Check module docstrings and README files
3. **Experiment**: Modify parameters and see effects
4. **Build Complex Scenarios**: Combine multiple features
5. **Contribute**: Add new features or improvements

## Quick Reference

### Running Simulations
```bash
# Run YAML scenario
python run_scenario.py <scenario.yaml>

# Run with visualization
python run_scenario.py <scenario.yaml> --visualize

# Run specific demo
python demos/<demo_name>.py
```

### Key Directories
- `src/` - Core simulation modules
- `configs/` - Configuration files
- `demos/` - Demonstration scripts
- `results/` - Output files
- `GETTING_STARTED/` - Tutorial materials

### Useful Commands
```bash
# List scenarios
python run_scenario.py --list

# Run tests
python -m pytest src/tracking/tests/

# Build Rust acceleration
maturin develop --release
```

---

**Congratulations!** You're now ready to use RadarSim for your radar simulation needs.

For advanced topics, check the README files in each module directory.