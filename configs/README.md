# RadarSim Configuration Guide

## Overview

RadarSim uses YAML configuration files to define complete simulation scenarios without modifying code. These configurations scale from simple single-radar setups to complex networked military engagements.

## 📁 Configuration Structure

```
configs/
├── universal_template.yaml    # Comprehensive template with ALL options
├── examples/                  # Ready-to-run examples
│   ├── single_aircraft.yaml  # Minimal: 1 radar, 1 target
│   └── networked_iads.yaml   # Complex: Multi-radar network with EW
└── scenarios/                 # Pre-built scenarios
    ├── air_defense.yaml
    ├── naval_engagement.yaml
    └── drone_swarm.yaml
```

## 🚀 Quick Start

### Simplest Case: Single Radar & Aircraft
```bash
python run_scenario.py configs/examples/single_aircraft.yaml
```

### Complex Case: Networked IADS
```bash
python run_scenario.py configs/examples/networked_iads.yaml
```

### Using the Universal Template
```bash
# Copy template and modify for your needs
cp configs/universal_template.yaml my_scenario.yaml
# Edit my_scenario.yaml
python run_scenario.py my_scenario.yaml
```

## 📋 Configuration Hierarchy

### 1. **Minimal Configuration** (single_aircraft.yaml)
- Single radar with basic parameters
- One or few targets
- Simple environment
- ~50 lines

### 2. **Standard Configuration** (most scenarios)
- Single/few radars
- Multiple targets with different types
- Tracking and detection
- Environmental effects
- ~200 lines

### 3. **Advanced Configuration** (networked_iads.yaml)
- Multiple networked radars
- Diverse threat scenarios
- Electronic warfare (jamming/ECCM)
- Resource management
- Weapon assignment
- ~500+ lines

## 🔧 Key Configuration Sections

### Core Sections (Required)

#### `scenario`
```yaml
scenario:
  name: "Scenario Name"
  duration: 60.0      # seconds
  time_step: 0.1      # seconds
```

#### `radar` (Single Radar)
```yaml
radar:
  type: "phased_array"
  parameters:
    frequency: 10.0e9  # Hz
    power: 100000      # Watts
    antenna_gain: 35   # dB
```

#### `targets`
```yaml
targets:
  - name: "Target-1"
    type: "aircraft"
    initial_position: [10000, 5000, 3000]  # meters
    velocity: [-200, 0, 0]  # m/s
    rcs:
      model: "swerling_1"
      mean_value: 2.0  # m²
```

### Advanced Sections (Optional)

#### `radars` (Multiple Radars)
```yaml
radars:
  - name: "Radar-1"
    type: "phased_array"
    position: [0, 0, 0]
    # ... parameters
  - name: "Radar-2"
    type: "mechanical"
    position: [10000, 0, 0]
    # ... parameters
```

#### `network` (Radar Networking)
```yaml
network:
  enabled: true
  fusion:
    algorithm: "covariance_intersection"
    update_rate: 10.0  # Hz
  data_links:
    latency: 0.01      # seconds
    bandwidth: 1.0e9   # bps
```

#### `jamming` (Electronic Warfare)
```yaml
jamming:
  enabled: true
  jammers:
    - name: "Jammer-1"
      type: "drfm"
      techniques:
        - type: "range_gate_pull_off"
        - type: "false_targets"
```

#### `eccm` (Counter-Countermeasures)
```yaml
eccm:
  enabled: true
  techniques:
    frequency_agility:
      enabled: true
      hop_rate: 100  # Hz
```

#### `resource_management`
```yaml
resource_management:
  enabled: true
  task_allocation:
    search: 0.3
    track: 0.5
    classification: 0.2
```

## 📊 Scaling Examples

### From Simple to Complex

#### Level 1: Basic Detection
```yaml
# 10 lines - Just detect something
radar:
  type: "basic"
  parameters:
    frequency: 10.0e9
    power: 50000
targets:
  - type: "aircraft"
    position: [10000, 0, 5000]
```

#### Level 2: Add Tracking
```yaml
# +20 lines - Track the target
radar:
  # ... previous config
  tracking:
    filter_type: "kalman"
    max_tracks: 10
```

#### Level 3: Add Environment
```yaml
# +30 lines - Real-world effects
environment:
  weather:
    type: "rain"
    rain_rate: 5.0
  clutter:
    enabled: true
    type: "ground"
```

#### Level 4: Add Jamming
```yaml
# +40 lines - Electronic warfare
jamming:
  enabled: true
  jammers:
    - type: "noise"
      power: 100
      bandwidth: 100.0e6
```

#### Level 5: Network Multiple Radars
```yaml
# +100 lines - Full IADS
radars:
  - name: "EWR-1"  # Early warning
  - name: "FCR-1"  # Fire control
  - name: "GFR-1"  # Gap filler
network:
  enabled: true
  fusion_center: [0, 0, 0]
```

## 🎯 Common Use Cases

### Air Traffic Control
- Use S-band radar (3 GHz)
- Large RCS targets (50-200 m²)
- No jamming
- Focus on tracking

### Military Air Defense
- Multiple radar bands
- Small RCS targets (0.1-5 m²)
- Heavy jamming environment
- Network fusion required

### Naval Scenario
- Sea clutter modeling
- Low altitude targets
- Multipath effects
- Ship-based radars

### Counter-UAS
- High-resolution X-band
- Very small targets (0.001-0.1 m²)
- Urban environment
- Short range focus

## 💡 Tips

1. **Start Simple**: Begin with `single_aircraft.yaml` and add complexity
2. **Use Templates**: Copy `universal_template.yaml` for all options
3. **Validate First**: Run with `--validate` flag to check configuration
4. **Monitor Performance**: Complex scenarios may need `time_step` adjustment
5. **Save Incrementally**: Use version control for configuration iterations

## 🔍 Validation

Check your configuration before running:
```bash
python run_scenario.py my_config.yaml --validate
```

This will verify:
- Required fields present
- Parameter ranges valid
- No conflicts between settings
- Performance estimates

## 📈 Performance Guidelines

| Scenario Complexity | Targets | Radars | Time Step | Update Rate | Real-time Factor |
|-------------------|---------|---------|-----------|-------------|------------------|
| Simple | 1-5 | 1 | 0.1s | 10 Hz | >10x |
| Standard | 5-20 | 1-3 | 0.05s | 20 Hz | 2-5x |
| Complex | 20-50 | 3-5 | 0.02s | 50 Hz | 0.5-2x |
| Extreme | 50+ | 5+ | 0.01s | 100 Hz | <1x |

## 🆘 Troubleshooting

### "Radar can't detect targets"
- Increase radar `power`
- Decrease target `range`
- Check `environment.weather` effects

### "Simulation runs slowly"
- Increase `time_step`
- Reduce number of targets
- Disable unused features

### "Tracks are lost"
- Adjust `tracking.filter_type`
- Increase `tracking.max_tracks`
- Check jamming levels

## 📚 Further Reading

- See `universal_template.yaml` for all available options with detailed comments
- Check `examples/` directory for working configurations
- Read source code docstrings for parameter details

---

**Pro Tip**: The `universal_template.yaml` file contains extensive comments explaining every parameter. Use it as your reference when building custom scenarios!