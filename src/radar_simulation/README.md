# Radar Simulation Validation Suite

This module provides a comprehensive validation framework for radar simulation systems, ensuring scientific accuracy, anti-cheating mechanisms, and proper integration with tracking systems.

## Overview

The validation suite consists of four main validators:

1. **AntiCheatValidator** - Ensures simulation integrity and prevents cheating
2. **PhysicsValidator** - Validates physics-based behaviors and constraints
3. **PerformanceValidator** - Tests performance characteristics and expected behaviors
4. **IntegrationValidator** - Validates compatibility and integration with tracking systems

## Quick Start

```python
from radar_simulation.validation import run_validation_suite

# Run validation with default parameters
report = run_validation_suite(verbose=True)
print(f"Overall Score: {report.overall_score:.2%}")
print(f"Tests Passed: {report.passed_tests}/{report.total_tests}")
```

## Validation Categories

### Anti-Cheating Tests

These tests ensure the simulation maintains integrity and doesn't have access to ground truth information during processing:

- **No Ground Truth Access** - Verifies simulator has no access to true target positions during processing
- **Signal Processing Derivation** - Ensures measurements are derived only from signal processing
- **Noise Randomness** - Checks that random noise is truly random and not correlated with targets
- **No Ground Truth Leakage** - Verifies no ground truth leaks through any processing path
- **Signal Processing Integrity** - Tests signal processing chain determinism and energy conservation

### Physics Validation Tests

These tests verify the simulation follows correct physics principles:

- **Range-SNR Relationship** - Verifies SNR decreases with range following R^4 law
- **Detection Probability Curves** - Tests that detection probability follows theoretical curves
- **False Alarm Rates** - Validates false alarm rates match CFAR design
- **Measurement Accuracy vs SNR** - Checks measurement accuracy degrades with lower SNR
- **Resolution Limits** - Verifies range and Doppler resolution limits
- **Doppler Shift Physics** - Tests Doppler shift calculations against physics

### Performance Validation Tests

These tests check that the simulation exhibits realistic performance characteristics:

- **Missed Detection Rates** - Tests that missed detections occur at expected rates
- **Measurement Error Statistics** - Verifies measurement errors have correct statistical properties
- **Blind Ranges and Velocities** - Checks that blind ranges and velocities exist as expected
- **Ambiguity Effects** - Validates that ambiguity effects are present and realistic
- **Clutter Effects** - Tests clutter suppression and multipath effects

### Integration Tests

These tests ensure compatibility with the broader radar tracking system:

- **Tracking Compatibility** - Tests compatibility with existing tracking system
- **Measurement Format** - Verifies measurement format matches expected structure
- **Timing and Synchronization** - Checks timing and synchronization requirements
- **Multi-Target Scenarios** - Tests performance with multiple targets
- **Performance Under Load** - Validates computational performance and stability

## Usage Examples

### Basic Validation

```python
from radar_simulation.validation import run_validation_suite

# Quick validation
report = run_validation_suite()
report.print_summary()
```

### Custom Configuration

```python
from radar import RadarParameters
from target import TargetGenerator
from radar_simulation.validation import run_validation_suite

# Custom radar parameters
radar_params = RadarParameters(
    frequency=10e9,    # 10 GHz
    power=100e3,       # 100 kW
    antenna_gain=35,   # 35 dB
    pulse_width=1e-6,  # 1 μs
    prf=1000,          # 1 kHz
    bandwidth=10e6,    # 10 MHz
    noise_figure=3,    # 3 dB
    losses=5           # 5 dB
)

# Custom targets
targets = TargetGenerator.create_random_scenario(5)

# Custom signal processor config
signal_config = {
    'sample_rate': 100e6,
    'bandwidth': 10e6
}

# Run validation
report = run_validation_suite(
    radar_params=radar_params,
    targets=targets,
    signal_processor_config=signal_config
)
```

### Individual Validators

```python
from radar_simulation.validation import (
    AntiCheatValidator, PhysicsValidator, 
    PerformanceValidator, IntegrationValidator
)
from radar import Radar, RadarParameters
from signal import SignalProcessor

# Setup
radar = Radar(RadarParameters(...))
signal_processor = SignalProcessor(100e6, 10e6)
targets = [...]

# Run individual validators
anti_cheat = AntiCheatValidator()
results = anti_cheat.run_validation(radar, targets, signal_processor)

for result in results:
    print(f"{result.test_name}: {'PASS' if result.passed else 'FAIL'}")
```

### Saving Reports

```python
from radar_simulation.validation import run_validation_suite, generate_validation_report

# Run validation
report = run_validation_suite()

# Generate comprehensive report with plots and save to file
generate_validation_report(
    report,
    filepath="validation_report.json",
    show_plots=True
)
```

## Pytest Integration

The validation suite includes pytest integration for automated testing:

```bash
# Run all validation tests
pytest tests/test_validation.py -v

# Run specific test categories
pytest tests/test_validation.py::TestAntiCheatValidator -v
pytest tests/test_validation.py::TestPhysicsValidator -v

# Run performance tests (marked as slow)
pytest tests/test_validation.py -m slow -v
```

## Scoring System

Each test returns a score from 0.0 to 1.0:
- **1.0** - Perfect performance
- **0.7-0.9** - Good performance with minor issues  
- **0.5-0.7** - Acceptable performance with some concerns
- **0.3-0.5** - Poor performance requiring attention
- **0.0-0.3** - Failing performance

The overall score is the average of all individual test scores.

## Validation Report

The `ValidationReport` class provides comprehensive results including:
- Overall score and test counts
- Individual test results with detailed messages
- Execution timing information
- Specific recommendations for improvement
- Summary statistics

Reports can be:
- Printed to console with formatted output
- Saved to JSON files for archival
- Used to generate plots and visualizations

## Expected Results

For a properly implemented radar simulation:
- **Overall Score**: Should be > 70%
- **Anti-Cheat Tests**: Should mostly pass (some may fail due to numerical precision)
- **Physics Tests**: Should largely pass with correct radar equation implementation
- **Performance Tests**: Should pass with realistic statistical behaviors
- **Integration Tests**: Should pass with proper measurement formatting

## Common Issues and Solutions

### Low Anti-Cheat Scores
- Review measurement generation to ensure no ground truth access
- Check that noise is truly random and independent
- Verify signal processing chain integrity

### Physics Test Failures
- Verify radar equation implementation follows R^4 law
- Check detection probability model matches theory
- Ensure CFAR implementation is correct

### Performance Issues
- Review measurement error models
- Check blind speed/range calculations
- Verify ambiguity handling

### Integration Failures
- Check measurement format compatibility
- Verify timing synchronization
- Test with tracking system interface

## API Reference

### Main Functions

- `run_validation_suite()` - Run complete validation with default or custom parameters
- `generate_validation_report()` - Generate and save comprehensive reports

### Validator Classes

- `AntiCheatValidator` - Anti-cheating validation tests
- `PhysicsValidator` - Physics compliance validation
- `PerformanceValidator` - Performance characteristic validation  
- `IntegrationValidator` - Integration compatibility validation

### Data Classes

- `ValidationResult` - Individual test result
- `ValidationReport` - Complete validation report
- `ValidationSuite` - Main orchestrator class

For detailed API documentation, see the docstrings in `validation.py`.