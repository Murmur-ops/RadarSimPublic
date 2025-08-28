# Tracking System Test Suite

This directory contains comprehensive tests for the RadarSim tracking system. The test suite covers all major components including Kalman filters, motion models, data association algorithms, IMM filters, and integrated tracking systems.

## Test Structure

### Core Test Modules

1. **`test_kalman_filters.py`** - Tests for all Kalman filter implementations
   - Standard Kalman Filter (KF)
   - Extended Kalman Filter (EKF)
   - Unscented Kalman Filter (UKF)
   - Utility functions and numerical stability

2. **`test_motion_models.py`** - Tests for motion models and coordinate transformations
   - Constant Velocity (CV) model
   - Constant Acceleration (CA) model
   - Coordinated Turn (CT) model
   - Singer acceleration model
   - Coordinate transformations (Cartesian ↔ Polar ↔ Spherical)
   - Model factory functions

3. **`test_association.py`** - Tests for data association algorithms
   - Global Nearest Neighbor (GNN)
   - Joint Probabilistic Data Association (JPDA)
   - Multiple Hypothesis Tracking (MHT)
   - Distance metrics and gating functions

4. **`test_imm_filter.py`** - Tests for Interacting Multiple Model filter
   - Model probability updates
   - State mixing and combination
   - Model switching detection
   - Performance metrics

5. **`test_integrated_trackers.py`** - Integration tests for complete tracking systems
   - IMM-JPDA integration
   - IMM-MHT integration
   - Multi-target scenarios
   - End-to-end tracking workflows

### Supporting Files

- **`conftest.py`** - Shared fixtures and test configuration
- **`__init__.py`** - Package initialization
- **`README.md`** - This documentation file

## Running Tests

### Prerequisites

Ensure you have pytest and required dependencies installed:

```bash
pip install pytest pytest-cov numpy scipy
```

### Basic Test Execution

Run all tests:
```bash
# From project root
pytest src/tracking/tests/

# With verbose output
pytest src/tracking/tests/ -v

# Run specific test file
pytest src/tracking/tests/test_kalman_filters.py -v
```

### Test Categories

Run tests by category using markers:

```bash
# Run only fast tests (exclude slow performance tests)
pytest src/tracking/tests/ -m "not slow"

# Run only integration tests
pytest src/tracking/tests/ -m "integration"

# Run only performance benchmarks
pytest src/tracking/tests/ -m "performance"
```

### Coverage Analysis

Generate coverage reports:

```bash
# Basic coverage
pytest src/tracking/tests/ --cov=src.tracking

# HTML coverage report
pytest src/tracking/tests/ --cov=src.tracking --cov-report=html

# Coverage with missing lines
pytest src/tracking/tests/ --cov=src.tracking --cov-report=term-missing
```

### Parallel Execution

Run tests in parallel for faster execution:

```bash
# Install pytest-xdist first
pip install pytest-xdist

# Run with multiple workers
pytest src/tracking/tests/ -n auto
```

## Test Categories and Scope

### Unit Tests

- **Kalman Filters**: Test individual filter components (predict, update, likelihood computation)
- **Motion Models**: Test state transition matrices, process noise, coordinate transforms
- **Association**: Test distance metrics, gating functions, assignment algorithms
- **IMM Components**: Test model mixing, probability updates, state combination

### Integration Tests

- **Complete Tracking Workflows**: End-to-end scenarios with real tracking data
- **Multi-Model Coordination**: IMM filter with multiple motion models
- **Multi-Target Tracking**: Scenarios with multiple targets and data association
- **Sensor Fusion**: Integration of multiple sensor inputs

### Performance Tests

- **Computational Benchmarks**: Timing of prediction and update cycles
- **Memory Usage**: Memory efficiency and leak detection
- **Numerical Stability**: Behavior with extreme inputs and edge cases
- **Scalability**: Performance with increasing numbers of targets/models

### Robustness Tests

- **Edge Cases**: Zero covariances, singular matrices, extreme measurements
- **Noise Handling**: Performance with various noise levels and outliers
- **Model Switching**: Behavior during rapid model transitions
- **Parameter Sensitivity**: Stability across parameter ranges

## Test Data and Fixtures

### Common Fixtures (from `conftest.py`)

- **States**: Sample 2D/3D state vectors for different motion models
- **Trajectories**: Generators for linear, accelerating, and turning motion
- **Measurements**: Sample measurement data with configurable noise
- **Functions**: Standard measurement functions for Cartesian and polar coordinates
- **Utilities**: Assertions for positive definiteness, symmetry, probability validation

### Trajectory Generators

```python
# Linear trajectory
true_pos, measurements = linear_trajectory_2d(
    num_points=20, dt=0.1, velocity=(1.0, 0.5), noise_std=0.05
)

# Accelerating trajectory  
true_pos, measurements = accelerating_trajectory_2d(
    num_points=20, dt=0.1, acceleration=(1.0, 0.5), noise_std=0.05
)

# Turning trajectory
true_pos, measurements = turning_trajectory_2d(
    num_points=30, dt=0.1, speed=5.0, turn_rate=0.1, noise_std=0.05
)
```

## Test Scenarios

### Scenario 1: Single Target Constant Velocity
Tests basic tracking of a target moving with constant velocity.
- **Models**: CV vs CA comparison
- **Validation**: State estimation accuracy, model probability evolution
- **Expected**: CV model should dominate

### Scenario 2: Single Target Maneuvering
Tests tracking of a target that switches from constant velocity to acceleration.
- **Models**: CV, CA, and CT models in IMM
- **Validation**: Model switching detection, estimation accuracy
- **Expected**: Appropriate model transitions

### Scenario 3: Multi-Target Tracking
Tests simultaneous tracking of multiple targets with different motion patterns.
- **Models**: Separate IMM filters per target
- **Association**: JPDA or MHT for measurement-to-track assignment
- **Validation**: Track purity, association accuracy

### Scenario 4: Cluttered Environment
Tests performance in high-clutter scenarios with false alarms.
- **Environment**: High false alarm rate, missed detections
- **Association**: Robust data association algorithms
- **Validation**: Track maintenance, false track suppression

### Scenario 5: Sensor Fusion
Tests integration of measurements from multiple sensors.
- **Sensors**: Different measurement types (Cartesian, polar)
- **Timing**: Out-of-sequence measurements
- **Validation**: Improved accuracy vs single sensor

## Performance Benchmarks

### Timing Benchmarks

Expected performance on modern hardware:
- **Kalman Filter Update**: < 0.1 ms per cycle
- **IMM Filter (3 models)**: < 0.5 ms per cycle  
- **JPDA (5 tracks, 10 detections)**: < 2 ms per cycle
- **MHT (100 hypotheses)**: < 10 ms per cycle

### Memory Usage

- **Single Track**: < 1 KB memory footprint
- **IMM Filter**: Memory proportional to number of models
- **Association Algorithms**: Memory grows with tracks × detections
- **History Tracking**: Bounded by configurable limits

### Accuracy Metrics

- **Position RMSE**: < 0.1 m for good SNR scenarios
- **Velocity RMSE**: < 0.05 m/s for good SNR scenarios
- **Track Purity**: > 95% for well-separated targets
- **Track Completeness**: > 90% for detectable targets

## Debugging Failed Tests

### Common Issues

1. **Numerical Instability**
   - Check for singular covariance matrices
   - Verify positive definiteness after updates
   - Examine process/measurement noise levels

2. **Association Failures**
   - Verify gating thresholds are appropriate
   - Check measurement noise assumptions
   - Examine track prediction accuracy

3. **Model Switching Issues**
   - Review transition probability matrices
   - Check model likelihood computations
   - Verify state dimension compatibility

4. **Performance Degradation**
   - Profile individual components
   - Check for memory leaks or unbounded growth
   - Verify algorithmic complexity assumptions

### Debug Output

Enable detailed logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run specific test with detailed output
pytest src/tracking/tests/test_kalman_filters.py::TestKalmanFilter::test_update_step -v -s
```

### Test Data Inspection

Fixtures provide reproducible test data. To inspect:

```python
def test_debug_trajectory(linear_trajectory_2d):
    true_pos, measurements = linear_trajectory_2d(num_points=10)
    
    import matplotlib.pyplot as plt
    true_pos = np.array(true_pos)
    measurements = np.array(measurements)
    
    plt.figure()
    plt.plot(true_pos[:, 0], true_pos[:, 1], 'b-', label='True')
    plt.plot(measurements[:, 0], measurements[:, 1], 'r.', label='Measured')
    plt.legend()
    plt.savefig('debug_trajectory.png')
```

## Contributing

### Adding New Tests

1. **Choose appropriate test module** based on component being tested
2. **Use existing fixtures** from `conftest.py` when possible
3. **Follow naming conventions**: `test_<functionality>_<condition>`
4. **Add markers** for test categories (slow, integration, performance)
5. **Include docstrings** explaining test purpose and validation criteria

### Test Design Principles

1. **Independence**: Tests should not depend on execution order
2. **Repeatability**: Use fixed random seeds for reproducible results
3. **Coverage**: Test both normal operation and edge cases
4. **Performance**: Mark slow tests appropriately
5. **Documentation**: Clear test names and docstrings

### Example Test Structure

```python
class TestNewComponent:
    """Test new tracking component."""
    
    @pytest.fixture
    def component_setup(self):
        """Set up component for testing."""
        # Setup code
        return component
    
    def test_normal_operation(self, component_setup):
        """Test normal operation with valid inputs."""
        # Test implementation
        pass
    
    def test_edge_case(self, component_setup):
        """Test behavior with edge case inputs."""
        # Edge case testing
        pass
    
    @pytest.mark.slow
    def test_performance(self, component_setup, performance_timer):
        """Test computational performance."""
        # Performance benchmark
        pass
```

This comprehensive test suite ensures the reliability, accuracy, and performance of the RadarSim tracking system across a wide range of scenarios and operating conditions.