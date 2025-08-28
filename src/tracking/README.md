# Motion Models for Radar Tracking

This module provides comprehensive motion models for radar target tracking applications. All models are designed to work with Extended Kalman Filters (EKF) and support multiple coordinate systems.

## Available Motion Models

### 1. Constant Velocity (CV) Model
- **Description**: Assumes target moves with constant velocity
- **State vectors**:
  - 2D: `[x, y, vx, vy]`
  - 3D: `[x, y, z, vx, vy, vz]`
- **Use case**: Targets with linear, non-maneuvering motion

### 2. Constant Acceleration (CA) Model  
- **Description**: Assumes target moves with constant acceleration
- **State vectors**:
  - 2D: `[x, y, vx, vy, ax, ay]`
  - 3D: `[x, y, z, vx, vy, vz, ax, ay, az]`
- **Use case**: Targets with predictable acceleration patterns

### 3. Coordinated Turn (CT) Model
- **Description**: Models coordinated turns with constant turn rate
- **State vectors**:
  - 2D with known turn rate: `[x, y, vx, vy]`
  - 2D with unknown turn rate: `[x, y, vx, vy, omega]`
  - 3D: `[x, y, z, vx, vy, vz, omega]`
- **Use case**: Aircraft and vehicles making coordinated turns

### 4. Singer Acceleration Model
- **Description**: Time-correlated acceleration with maneuver detection
- **State vectors**: Same as CA model
- **Features**: Adaptive process noise, maneuver detection
- **Use case**: Highly maneuvering targets with unpredictable motion

## Coordinate Systems

- **Cartesian 2D/3D**: Standard rectangular coordinates
- **Polar**: `[range, azimuth, range_rate, azimuth_rate]`
- **Spherical**: `[range, azimuth, elevation, range_rate, azimuth_rate, elevation_rate]`

## Usage Examples

### Basic Model Creation

```python
from tracking.motion_models import *
import numpy as np

# Create a 2D Constant Velocity model
params = ModelParameters(
    dt=0.1,  # Time step in seconds
    process_noise_std=1.0,
    coordinate_system=CoordinateSystem.CARTESIAN_2D
)
cv_model = ConstantVelocityModel(params)

# Initial state: [x, y, vx, vy]
state = np.array([0.0, 0.0, 20.0, 10.0])

# Predict next state
next_state = cv_model.predict_state(state)

# Get transition matrix and process noise covariance
F = cv_model.get_transition_matrix(state)
Q = cv_model.get_process_noise_covariance(state)
```

### Coordinated Turn Model with Unknown Turn Rate

```python
# CT model parameters
params = ModelParameters(
    dt=0.1,
    coordinate_system=CoordinateSystem.CARTESIAN_2D,
    additional_params={
        'turn_rate': None,  # Unknown turn rate (will be estimated)
        'turn_rate_noise': 0.1  # Turn rate process noise
    }
)
ct_model = CoordinatedTurnModel(params)

# State: [x, y, vx, vy, omega]
state = np.array([0.0, 0.0, 20.0, 0.0, 0.2])  # 0.2 rad/s turn rate
```

### Singer Model with Maneuver Detection

```python
# Singer model parameters
params = ModelParameters(
    dt=0.1,
    coordinate_system=CoordinateSystem.CARTESIAN_2D,
    additional_params={
        'alpha': 0.1,  # 1/tau (maneuver time constant)
        'max_acceleration': 10.0,  # Maximum expected acceleration
        'maneuver_threshold': 2.0  # Maneuver detection threshold
    }
)
singer_model = SingerAccelerationModel(params)

# Detect maneuvers using innovation sequence
innovation = np.array([2.5, 1.8])  # From Kalman filter
innovation_cov = np.eye(2)
maneuver_detected = singer_model.detect_maneuver(innovation, innovation_cov)
```

### Coordinate System Conversions

```python
# Convert Cartesian to polar coordinates
x, y, vx, vy = 1000.0, 500.0, 50.0, -20.0
range_val, azimuth, range_rate, azimuth_rate = cartesian_to_polar(x, y, vx, vy)

# Convert back to Cartesian
x2, y2, vx2, vy2 = polar_to_cartesian(range_val, azimuth, range_rate, azimuth_rate)

# 3D spherical conversions
x, y, z, vx, vy, vz = 1000.0, 500.0, 300.0, 50.0, -20.0, 10.0
r, az, el, rr, azr, elr = cartesian_to_spherical(x, y, z, vx, vy, vz)
```

### EKF Integration

```python
# Get measurement Jacobian for different sensor types
state = np.array([100.0, 50.0, 10.0, 5.0])

# For position-only sensors
H_cartesian = get_measurement_jacobian(state, "cartesian")

# For radar sensors (range/azimuth)
H_polar = get_measurement_jacobian(state, "polar")

# For 3D radar (range/azimuth/elevation)
state_3d = np.array([100.0, 50.0, 30.0, 10.0, 5.0, 2.0])
H_spherical = get_measurement_jacobian(state_3d, "spherical")
```

### Model Factory

```python
# Create models using factory function
cv_model = create_motion_model("cv", params)
ca_model = create_motion_model("ca", params)
ct_model = create_motion_model("ct", params)
singer_model = create_motion_model("singer", params)
```

## Mathematical Properties

All motion models provide:

1. **State Transition Matrix (F)**: Linear approximation of state evolution
2. **Process Noise Covariance (Q)**: Uncertainty in state prediction
3. **Jacobian Matrices**: For nonlinear models (EKF support)
4. **State Prediction**: One-step-ahead state prediction

### Matrix Properties
- F matrices have determinant = 1 (volume preserving for linear models)
- Q matrices are positive semidefinite
- All matrices are numerically stable

## Advanced Features

### Adaptive Process Noise
```python
# Estimate adaptive process noise based on innovation history
innovation_history = [np.array([1.2, 0.8]), np.array([0.5, 1.1]), ...]
adapted_noise = estimate_adaptive_process_noise(innovation_history)
```

### Turn Rate Computation
```python
# Compute turn rate from state vector
turn_rate = compute_turn_rate(state, CoordinateSystem.CARTESIAN_2D)
```

## Testing

Run the provided test scripts to verify functionality:

```bash
# Basic functionality test (no plotting)
python examples/basic_motion_models_test.py

# Comprehensive demo with visualizations
python examples/motion_models_demo.py
```

## Dependencies

- NumPy >= 1.24.0
- Python >= 3.8

Optional for visualization demos:
- Matplotlib >= 3.6.0

## Implementation Details

- All models inherit from the abstract `MotionModel` base class
- Supports both linear and nonlinear motion models
- Designed for real-time tracking applications
- Optimized for numerical stability and performance
- Comprehensive error handling and validation

## References

1. Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation*
2. Ristic, B., Arulampalam, S., & Gordon, N. (2004). *Beyond the Kalman Filter*
3. Singer, R. A. (1970). "Estimating Optimal Tracking Filter Performance for Manned Maneuvering Targets"