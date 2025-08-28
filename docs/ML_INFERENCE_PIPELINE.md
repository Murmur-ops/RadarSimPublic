# Machine Learning Inference Pipeline Documentation

## Overview

The RadarSim ML inference pipeline provides real-time threat classification using a custom-built ensemble of Transformer and CNN architectures. The system processes Pulse Descriptor Words (PDWs) to classify threats into five priority levels without requiring external ML frameworks.

## Pipeline Architecture

```
Raw Radar Returns
      ↓
PDW Generation (frequency, pulse width, PRI, amplitude)
      ↓
Feature Extraction (13 base + 19 advanced features)
      ↓
┌─────────────────┴─────────────────┐
↓                                   ↓
Transformer Network              CNN Network
(Temporal Analysis)           (Spatial Patterns)
↓                                   ↓
└─────────────────┬─────────────────┘
      ↓
Ensemble Classification (192-dim vector)
      ↓
Threat Priority Assignment
      ↓
Priority Queue Management
```

## Component Details

### 1. PDW Generation

Pulse Descriptor Words capture the essential characteristics of each radar pulse:

```python
@dataclass
class PDWSequence:
    timestamps: np.ndarray      # Time of arrival (seconds)
    frequencies: np.ndarray     # Carrier frequencies (Hz)
    pulse_widths: np.ndarray    # Pulse widths (seconds)
    amplitudes: np.ndarray      # Received amplitudes (dB)
    pri_values: np.ndarray      # Pulse repetition intervals
    aoa_values: np.ndarray      # Angle of arrival (degrees)
```

### 2. Feature Extraction

#### Base Features (13 dimensions)
- **Frequency domain**: mean, std deviation
- **Pulse characteristics**: width mean, std
- **Amplitude**: mean, std, dynamic range
- **PRI patterns**: mean, std, jitter detection
- **Agility metrics**: frequency hopping rate, max deviation
- **Temporal density**: pulses per second

#### Advanced Features (19 dimensions)
- **Kinematic** (5): velocity, acceleration, altitude, closing rate, turn rate
- **RCS** (5): mean, variance, max, min, fluctuation rate
- **Doppler** (5): spread, centroid, components, max shift, periodicity
- **Spectral** (4): entropy, flux, peak frequency, bandwidth

### 3. Neural Network Architecture

#### Transformer Block
```python
class TransformerBlock:
    - Input: 64-dimensional embeddings
    - Heads: 4 attention heads
    - Head dimension: 16
    - Output: Temporal feature vector
    
    Key operations:
    - Multi-head self-attention
    - Scaled dot-product attention
    - Temporal pooling
```

#### CNN Feature Extractor
```python
class CNNFeatureExtractor:
    - Input: 13-dimensional feature vector
    - Conv1: 32 filters, kernel size 3
    - Conv2: 64 filters, kernel size 3
    - Output: 128-dimensional spatial features
    
    Key operations:
    - 1D convolution
    - ReLU activation
    - Global average pooling
```

### 4. Classification Logic

The system implements naval-specific threat prioritization:

```python
# CRITICAL: Anti-ship missiles
if pulse_width <= 2e-6 and pulse_density > 500:
    threat_level = CRITICAL

# HIGH: Fighter aircraft
elif frequency_std > 50e6 or (5e-6 <= pulse_width <= 20e-6):
    threat_level = HIGH

# MEDIUM: Helicopters, patrol aircraft
elif 15e-6 <= pulse_width <= 30e-6 and amplitude_std > 2:
    threat_level = MEDIUM

# LOW: Commercial, navigation
elif pulse_width >= 500e-6 or pulse_density < 30:
    threat_level = LOW
```

### 5. Threat Priority Levels

| Priority | Description | Typical Characteristics |
|----------|-------------|------------------------|
| **CRITICAL** | Anti-ship missiles | PW ≤ 2μs, PRF > 500Hz, stable bearing |
| **HIGH** | Fighter/Attack aircraft | Frequency agile, PW 5-20μs, PRI stagger |
| **MEDIUM** | Helicopters, MPA | PW 15-30μs, blade modulation, stable frequency |
| **LOW** | Commercial/Navigation | PW > 500μs, PRF < 30Hz, rotating antenna |
| **UNKNOWN** | Unclassified | No matching signature pattern |

## Training Pipeline

### Synthetic Data Generation

The system generates realistic training data based on publicly known characteristics:

```python
class SyntheticDataGenerator:
    Threat signatures include:
    - Missiles: Harpoon, Exocet, Brahmos patterns
    - Aircraft: F-18, MiG-29K characteristics
    - Helicopters: Blade modulation effects
    - Commercial: Navigation radar patterns
```

### Training Process

1. **Data Generation**: Create balanced dataset (200+ samples per class)
2. **Feature Extraction**: Convert PDWs to feature vectors
3. **Network Training**: Backpropagation with cross-entropy loss
4. **Validation**: Test on held-out synthetic data
5. **Deployment**: <10ms inference target

## Performance Metrics

### Classification Accuracy
- Overall: ~85-90% on synthetic data
- Critical threats: >95% detection rate
- False alarm rate: <5%

### Inference Speed
```
Mean latency: 6-8ms
Max latency: <10ms
Min latency: 4ms
Real-time capable: ✓
```

### Resource Usage
- Memory: ~50MB for models
- CPU: Single-threaded Python
- No GPU required

## Usage Example

```python
# Initialize classifier
classifier = MLThreatClassifier()

# Create PDW sequence from radar data
pdw_sequence = PDWSequence(
    timestamps=detection_times,
    frequencies=carrier_freqs,
    pulse_widths=pulse_widths,
    amplitudes=signal_amplitudes,
    pri_values=pulse_intervals,
    aoa_values=angles
)

# Classify threat
priority, confidence, details = classifier.classify(pdw_sequence)

print(f"Threat: {priority.name}")
print(f"Confidence: {confidence:.2%}")
print(f"Inference time: {details['inference_time_ms']:.2f}ms")
```

## Integration with Tracking

The ML classifier integrates with the tracking system:

1. **Track Association**: Links classifications to track IDs
2. **Priority Queue**: Maintains sorted threat list
3. **Resource Allocation**: Guides radar beam scheduling
4. **Track Quality**: Influences track maintenance decisions

## Configuration via YAML

Enable ML classification in scenario files:

```yaml
radar:
  ml_classification:
    enabled: true
    model: "ensemble"
    inference_rate: 10  # Hz
    feature_extraction: "advanced"
    threat_priority_queue: true
```

## Key Advantages

1. **No External Dependencies**: Pure NumPy implementation
2. **Real-time Performance**: <10ms inference
3. **Interpretable**: Feature-based rules augment neural networks
4. **Configurable**: YAML-based configuration
5. **Extensible**: Easy to add new threat types

## Future Enhancements

- Graph neural networks for multi-target correlation
- Reinforcement learning for adaptive thresholds
- Transfer learning from real radar data
- Federated learning for distributed training
- Explainable AI visualizations