"""
Pytest configuration and shared fixtures for tracking tests.

This module provides common fixtures and configuration used across
all tracking system tests.

Author: RadarSim Project
"""

import pytest
import numpy as np
from typing import List, Tuple, Callable
import warnings

# Configure numpy to raise warnings as errors for better numerical stability testing
np.seterr(all='warn')

# Filter out specific warnings that are expected in testing
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")
warnings.filterwarnings("ignore", category=UserWarning, message="Singular innovation covariance matrix")


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_2d_state():
    """Sample 2D state vector [x, y, vx, vy]."""
    return np.array([1.0, 2.0, 0.5, 0.3])


@pytest.fixture
def sample_3d_state():
    """Sample 3D state vector [x, y, z, vx, vy, vz]."""
    return np.array([1.0, 2.0, 3.0, 0.5, 0.3, 0.1])


@pytest.fixture
def sample_2d_ca_state():
    """Sample 2D constant acceleration state [x, y, vx, vy, ax, ay]."""
    return np.array([1.0, 2.0, 0.5, 0.3, 0.1, 0.05])


@pytest.fixture
def sample_covariance_2d():
    """Sample 2D covariance matrix."""
    return np.array([[0.1, 0.01, 0.02, 0.005],
                     [0.01, 0.1, 0.005, 0.02],
                     [0.02, 0.005, 0.05, 0.01],
                     [0.005, 0.02, 0.01, 0.05]])


@pytest.fixture
def sample_measurements_2d():
    """Sample 2D measurements."""
    return [
        np.array([1.1, 2.05]),
        np.array([1.15, 2.08]),
        np.array([1.2, 2.1])
    ]


@pytest.fixture
def sample_measurement_noise():
    """Sample measurement noise covariance."""
    return np.eye(2) * 0.01


@pytest.fixture
def linear_trajectory_2d():
    """Generate linear trajectory in 2D."""
    def _generate_trajectory(num_points: int = 10, dt: float = 0.1, 
                           velocity: Tuple[float, float] = (1.0, 0.5),
                           noise_std: float = 0.05) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate linear trajectory with noise.
        
        Args:
            num_points: Number of trajectory points
            dt: Time step
            velocity: Velocity vector (vx, vy)
            noise_std: Standard deviation of measurement noise
            
        Returns:
            Tuple of (true_positions, noisy_measurements)
        """
        true_positions = []
        measurements = []
        
        for i in range(num_points):
            t = i * dt
            true_pos = np.array([velocity[0] * t, velocity[1] * t])
            noise = np.random.normal(0, noise_std, 2)
            measurement = true_pos + noise
            
            true_positions.append(true_pos)
            measurements.append(measurement)
        
        return true_positions, measurements
    
    return _generate_trajectory


@pytest.fixture
def accelerating_trajectory_2d():
    """Generate accelerating trajectory in 2D."""
    def _generate_trajectory(num_points: int = 10, dt: float = 0.1,
                           acceleration: Tuple[float, float] = (1.0, 0.5),
                           noise_std: float = 0.05) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate accelerating trajectory with noise.
        
        Args:
            num_points: Number of trajectory points
            dt: Time step
            acceleration: Acceleration vector (ax, ay)
            noise_std: Standard deviation of measurement noise
            
        Returns:
            Tuple of (true_positions, noisy_measurements)
        """
        true_positions = []
        measurements = []
        
        for i in range(num_points):
            t = i * dt
            # x = 0.5 * a * t^2 for motion starting from rest
            true_pos = np.array([0.5 * acceleration[0] * t**2, 
                               0.5 * acceleration[1] * t**2])
            noise = np.random.normal(0, noise_std, 2)
            measurement = true_pos + noise
            
            true_positions.append(true_pos)
            measurements.append(measurement)
        
        return true_positions, measurements
    
    return _generate_trajectory


@pytest.fixture
def turning_trajectory_2d():
    """Generate coordinated turn trajectory in 2D."""
    def _generate_trajectory(num_points: int = 20, dt: float = 0.1,
                           speed: float = 5.0, turn_rate: float = 0.1,
                           noise_std: float = 0.05) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate coordinated turn trajectory.
        
        Args:
            num_points: Number of trajectory points
            dt: Time step
            speed: Constant speed
            turn_rate: Turn rate in rad/s
            noise_std: Standard deviation of measurement noise
            
        Returns:
            Tuple of (true_positions, noisy_measurements)
        """
        true_positions = []
        measurements = []
        
        x, y = 0.0, 0.0
        heading = 0.0  # Initial heading
        
        for i in range(num_points):
            # Update position
            x += speed * np.cos(heading) * dt
            y += speed * np.sin(heading) * dt
            
            # Update heading
            heading += turn_rate * dt
            
            true_pos = np.array([x, y])
            noise = np.random.normal(0, noise_std, 2)
            measurement = true_pos + noise
            
            true_positions.append(true_pos)
            measurements.append(measurement)
        
        return true_positions, measurements
    
    return _generate_trajectory


@pytest.fixture
def measurement_functions_2d():
    """Standard 2D measurement functions."""
    def measurement_function(state: np.ndarray) -> np.ndarray:
        """Extract position from state."""
        return state[:2]
    
    def measurement_jacobian(state: np.ndarray) -> np.ndarray:
        """Jacobian for position measurements."""
        H = np.zeros((2, len(state)))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        return H
    
    return measurement_function, measurement_jacobian


@pytest.fixture
def polar_measurement_functions():
    """Polar coordinate measurement functions."""
    def measurement_function(state: np.ndarray) -> np.ndarray:
        """Convert Cartesian state to polar measurements [range, bearing]."""
        x, y = state[0], state[1]
        range_val = np.sqrt(x**2 + y**2)
        bearing = np.arctan2(y, x)
        return np.array([range_val, bearing])
    
    def measurement_jacobian(state: np.ndarray) -> np.ndarray:
        """Jacobian for polar measurements."""
        x, y = state[0], state[1]
        range_val = np.sqrt(x**2 + y**2)
        
        if range_val < 1e-6:
            return np.zeros((2, len(state)))
        
        H = np.zeros((2, len(state)))
        # Range derivatives
        H[0, 0] = x / range_val
        H[0, 1] = y / range_val
        # Bearing derivatives
        H[1, 0] = -y / (range_val**2)
        H[1, 1] = x / (range_val**2)
        
        return H
    
    return measurement_function, measurement_jacobian


@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed()
        
        def elapsed(self):
            if self.start_time is None:
                return 0.0
            end = self.end_time if self.end_time is not None else time.time()
            return end - self.start_time
    
    return Timer()


@pytest.fixture
def assert_positive_definite():
    """Utility to assert matrix is positive definite."""
    def _check_positive_definite(matrix: np.ndarray, tolerance: float = 1e-12):
        """Check if matrix is positive definite."""
        eigenvals = np.linalg.eigvals(matrix)
        assert np.all(eigenvals > tolerance), f"Matrix is not positive definite. Min eigenvalue: {np.min(eigenvals)}"
        return True
    
    return _check_positive_definite


@pytest.fixture
def assert_symmetric():
    """Utility to assert matrix is symmetric."""
    def _check_symmetric(matrix: np.ndarray, tolerance: float = 1e-12):
        """Check if matrix is symmetric."""
        assert np.allclose(matrix, matrix.T, atol=tolerance), "Matrix is not symmetric"
        return True
    
    return _check_symmetric


@pytest.fixture
def assert_probabilities_valid():
    """Utility to assert probabilities are valid."""
    def _check_probabilities(probs: np.ndarray, tolerance: float = 1e-10):
        """Check if probabilities are valid (non-negative, sum to 1)."""
        assert np.all(probs >= 0), f"Negative probabilities found: {probs}"
        assert abs(np.sum(probs) - 1.0) < tolerance, f"Probabilities don't sum to 1: sum = {np.sum(probs)}"
        return True
    
    return _check_probabilities


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add slow marker to tests that take a long time
    for item in items:
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        if "integration" in item.nodeid or "test_integrated" in item.nodeid:
            item.add_marker(pytest.mark.integration)