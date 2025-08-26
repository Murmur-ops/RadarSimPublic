"""
Comprehensive test suite for Kalman filter implementations.

This module provides unit tests for all Kalman filter variants including:
- Standard Kalman Filter (KF)
- Extended Kalman Filter (EKF) 
- Unscented Kalman Filter (UKF)

Tests cover initialization, predict/update cycles, numerical stability,
covariance matrix properties, and likelihood computations.

Author: RadarSim Project
"""

import pytest
import numpy as np
import numpy.testing as npt
from unittest.mock import Mock, patch
import warnings

from ..kalman_filters import (
    BaseKalmanFilter, KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter,
    initialize_constant_velocity_filter, initialize_constant_acceleration_filter,
    adaptive_noise_estimation, predict_multiple_steps, compute_nees, compute_nis
)


class TestBaseKalmanFilter:
    """Test the abstract base class functionality."""
    
    def test_cannot_instantiate_directly(self):
        """Test that BaseKalmanFilter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseKalmanFilter(4, 2)
    
    def test_compute_innovation(self):
        """Test innovation computation."""
        class MockFilter(BaseKalmanFilter):
            def predict(self, dt, u=None): pass
            def update(self, z): pass
        
        filter_obj = MockFilter(4, 2)
        z = np.array([1.0, 2.0])
        Hx = np.array([0.8, 1.9])
        
        filter_obj.compute_innovation(z, Hx)
        
        expected_innovation = np.array([0.2, 0.1])
        npt.assert_array_almost_equal(filter_obj.y, expected_innovation)
    
    def test_compute_log_likelihood_valid(self):
        """Test log-likelihood computation with valid covariance."""
        class MockFilter(BaseKalmanFilter):
            def predict(self, dt, u=None): pass
            def update(self, z): pass
        
        filter_obj = MockFilter(4, 2)
        filter_obj.y = np.array([0.1, 0.2])
        filter_obj.S = np.array([[1.0, 0.1], [0.1, 1.0]])
        
        log_likelihood = filter_obj.compute_log_likelihood()
        
        assert isinstance(log_likelihood, float)
        assert log_likelihood < 0  # Log-likelihood should be negative
        assert np.isfinite(log_likelihood)
    
    def test_compute_log_likelihood_singular(self):
        """Test log-likelihood computation with singular covariance."""
        class MockFilter(BaseKalmanFilter):
            def predict(self, dt, u=None): pass
            def update(self, z): pass
        
        filter_obj = MockFilter(4, 2)
        filter_obj.y = np.array([0.1, 0.2])
        filter_obj.S = np.array([[0.0, 0.0], [0.0, 0.0]])  # Singular matrix
        
        log_likelihood = filter_obj.compute_log_likelihood()
        
        assert log_likelihood == -np.inf
    
    def test_mahalanobis_distance(self):
        """Test Mahalanobis distance computation."""
        class MockFilter(BaseKalmanFilter):
            def predict(self, dt, u=None): pass
            def update(self, z): pass
        
        filter_obj = MockFilter(4, 2)
        filter_obj.S = np.eye(2)
        
        z = np.array([2.0, 2.0])
        Hx = np.array([0.0, 0.0])
        
        distance = filter_obj.mahalanobis(z, Hx)
        
        expected_distance = np.sqrt(8.0)  # sqrt(2^2 + 2^2)
        assert abs(distance - expected_distance) < 1e-10
    
    def test_measurement_validation_passes(self):
        """Test measurement validation with valid measurement."""
        class MockFilter(BaseKalmanFilter):
            def predict(self, dt, u=None): pass
            def update(self, z): pass
        
        filter_obj = MockFilter(4, 2)
        filter_obj.S = np.eye(2)
        
        z = np.array([1.0, 1.0])
        Hx = np.array([0.0, 0.0])
        
        is_valid = filter_obj.is_measurement_valid(z, Hx, gate_threshold=9.21)
        
        assert is_valid  # Distance = sqrt(2) ≈ 1.414 < 9.21
    
    def test_measurement_validation_fails(self):
        """Test measurement validation with invalid measurement."""
        class MockFilter(BaseKalmanFilter):
            def predict(self, dt, u=None): pass
            def update(self, z): pass
        
        filter_obj = MockFilter(4, 2)
        filter_obj.S = np.eye(2)
        
        z = np.array([10.0, 10.0])
        Hx = np.array([0.0, 0.0])
        
        is_valid = filter_obj.is_measurement_valid(z, Hx, gate_threshold=3.0)
        
        assert not is_valid  # Distance = sqrt(200) ≈ 14.14 > 3.0


class TestKalmanFilter:
    """Test the standard Kalman filter implementation."""
    
    @pytest.fixture
    def simple_kf(self):
        """Create a simple 2D constant velocity Kalman filter."""
        kf = KalmanFilter(4, 2)  # [x, y, vx, vy], [x, y]
        
        # Set up constant velocity model
        dt = 1.0
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        kf.Q = np.eye(4) * 0.1
        kf.R = np.eye(2) * 1.0
        kf.P = np.eye(4) * 10.0
        
        return kf
    
    def test_initialization(self):
        """Test Kalman filter initialization."""
        kf = KalmanFilter(4, 2, 1)
        
        assert kf.dim_x == 4
        assert kf.dim_z == 2
        assert kf.dim_u == 1
        assert kf.F.shape == (4, 4)
        assert kf.H.shape == (2, 4)
        assert kf.B.shape == (4, 1)
        assert kf.K.shape == (4, 2)
        
        # Check initial values
        npt.assert_array_equal(kf.x, np.zeros(4))
        npt.assert_array_equal(kf.F, np.eye(4))
        npt.assert_array_equal(kf.Q, np.eye(4))
        npt.assert_array_equal(kf.R, np.eye(2))
    
    def test_predict_without_control(self, simple_kf):
        """Test prediction step without control input."""
        # Set initial state
        simple_kf.x = np.array([1.0, 2.0, 0.5, 0.3])
        initial_P = simple_kf.P.copy()
        
        simple_kf.predict(1.0)
        
        # Check state prediction
        expected_x = np.array([1.5, 2.3, 0.5, 0.3])
        npt.assert_array_almost_equal(simple_kf.x, expected_x)
        
        # Check covariance prediction (should increase)
        assert np.all(np.diag(simple_kf.P) >= np.diag(initial_P))
        
        # Check that prior is stored
        npt.assert_array_equal(simple_kf.x_prior, np.array([1.0, 2.0, 0.5, 0.3]))
    
    def test_predict_with_control(self):
        """Test prediction step with control input."""
        kf = KalmanFilter(4, 2, 2)
        
        # Set up system matrices
        kf.F = np.eye(4)
        kf.B = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
        kf.Q = np.eye(4) * 0.1
        
        # Set initial state and control
        kf.x = np.array([1.0, 2.0, 0.0, 0.0])
        u = np.array([0.1, 0.2])
        
        kf.predict(1.0, u)
        
        expected_x = np.array([1.1, 2.2, 0.0, 0.0])
        npt.assert_array_almost_equal(kf.x, expected_x)
    
    def test_update_step(self, simple_kf):
        """Test update step with measurement."""
        # Set initial state and covariance
        simple_kf.x = np.array([1.0, 2.0, 0.5, 0.3])
        simple_kf.P = np.eye(4) * 5.0
        
        # Measurement
        z = np.array([1.1, 2.05])
        
        # Store initial state for comparison
        x_before = simple_kf.x.copy()
        P_before = simple_kf.P.copy()
        
        simple_kf.update(z)
        
        # State should move toward measurement
        assert abs(simple_kf.x[0] - z[0]) < abs(x_before[0] - z[0])
        assert abs(simple_kf.x[1] - z[1]) < abs(x_before[1] - z[1])
        
        # Covariance should decrease (information gain)
        assert np.all(np.diag(simple_kf.P) <= np.diag(P_before))
        
        # Check innovation computation
        expected_innovation = z - simple_kf.H @ x_before
        npt.assert_array_almost_equal(simple_kf.y, expected_innovation)
    
    def test_update_with_singular_covariance(self, simple_kf):
        """Test update with singular innovation covariance."""
        simple_kf.R = np.zeros((2, 2))  # Singular measurement noise
        simple_kf.P = np.zeros((4, 4))  # Singular state covariance
        
        z = np.array([1.0, 2.0])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            simple_kf.update(z)  # Should not crash
    
    def test_joseph_form_covariance_update(self, simple_kf):
        """Test that covariance update uses Joseph form for numerical stability."""
        simple_kf.x = np.array([1.0, 2.0, 0.5, 0.3])
        simple_kf.P = np.eye(4) * 5.0
        
        z = np.array([1.1, 2.05])
        simple_kf.update(z)
        
        # Check that covariance matrix is positive definite
        eigenvals = np.linalg.eigvals(simple_kf.P)
        assert np.all(eigenvals > 0)
        
        # Check that covariance matrix is symmetric
        npt.assert_array_almost_equal(simple_kf.P, simple_kf.P.T)
    
    def test_likelihood_computation(self, simple_kf):
        """Test likelihood computation during update."""
        simple_kf.x = np.array([1.0, 2.0, 0.5, 0.3])
        z = np.array([1.1, 2.05])
        
        simple_kf.update(z)
        
        assert hasattr(simple_kf, 'log_likelihood')
        assert isinstance(simple_kf.log_likelihood, float)
        assert simple_kf.log_likelihood < 0  # Log-likelihood should be negative
        assert np.isfinite(simple_kf.log_likelihood)


class TestExtendedKalmanFilter:
    """Test the Extended Kalman filter implementation."""
    
    @pytest.fixture
    def simple_ekf(self):
        """Create a simple Extended Kalman filter."""
        ekf = ExtendedKalmanFilter(4, 2, dt=1.0)
        
        # Set up simple nonlinear functions for testing
        def f_func(x, u, dt):
            # Simple nonlinear state transition
            x_new = x.copy()
            x_new[0] = x[0] + x[2] * dt
            x_new[1] = x[1] + x[3] * dt
            return x_new
        
        def F_jac(x, u, dt):
            # Jacobian of state transition
            return np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        
        def h_func(x):
            # Nonlinear measurement function (position only)
            return x[:2]
        
        def H_jac(x):
            # Jacobian of measurement function
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])
        
        ekf.set_state_transition(f_func, F_jac)
        ekf.set_measurement_function(h_func, H_jac)
        
        ekf.Q = np.eye(4) * 0.1
        ekf.R = np.eye(2) * 1.0
        ekf.P = np.eye(4) * 10.0
        
        return ekf
    
    def test_initialization(self):
        """Test EKF initialization."""
        ekf = ExtendedKalmanFilter(4, 2, dt=0.5)
        
        assert ekf.dim_x == 4
        assert ekf.dim_z == 2
        assert ekf.dt == 0.5
        assert ekf.f_func is None
        assert ekf.h_func is None
    
    def test_set_functions(self, simple_ekf):
        """Test setting nonlinear functions."""
        assert simple_ekf.f_func is not None
        assert simple_ekf.h_func is not None
        assert simple_ekf.F_jac is not None
        assert simple_ekf.H_jac is not None
    
    def test_predict_without_functions(self):
        """Test prediction fails without functions set."""
        ekf = ExtendedKalmanFilter(4, 2)
        
        with pytest.raises(ValueError, match="State transition function"):
            ekf.predict(1.0)
    
    def test_predict_step(self, simple_ekf):
        """Test EKF prediction step."""
        initial_state = np.array([1.0, 2.0, 0.5, 0.3])
        simple_ekf.x = initial_state.copy()
        initial_P = simple_ekf.P.copy()
        
        simple_ekf.predict(1.0)
        
        # Check state prediction
        expected_x = np.array([1.5, 2.3, 0.5, 0.3])
        npt.assert_array_almost_equal(simple_ekf.x, expected_x)
        
        # Check covariance increase
        assert np.all(np.diag(simple_ekf.P) >= np.diag(initial_P))
        
        # Check that prior is stored
        npt.assert_array_equal(simple_ekf.x_prior, initial_state)
    
    def test_update_without_functions(self):
        """Test update fails without functions set."""
        ekf = ExtendedKalmanFilter(4, 2)
        z = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="Measurement function"):
            ekf.update(z)
    
    def test_update_step(self, simple_ekf):
        """Test EKF update step."""
        simple_ekf.x = np.array([1.0, 2.0, 0.5, 0.3])
        simple_ekf.P = np.eye(4) * 5.0
        
        z = np.array([1.1, 2.05])
        
        x_before = simple_ekf.x.copy()
        P_before = simple_ekf.P.copy()
        
        simple_ekf.update(z)
        
        # State should move toward measurement
        assert abs(simple_ekf.x[0] - z[0]) < abs(x_before[0] - z[0])
        assert abs(simple_ekf.x[1] - z[1]) < abs(x_before[1] - z[1])
        
        # Covariance should decrease
        assert np.all(np.diag(simple_ekf.P) <= np.diag(P_before))
    
    def test_nonlinear_measurement_function(self):
        """Test EKF with truly nonlinear measurement function."""
        ekf = ExtendedKalmanFilter(4, 2)
        
        def f_func(x, u, dt):
            return x  # Identity for simplicity
        
        def F_jac(x, u, dt):
            return np.eye(4)
        
        def h_func(x):
            # Nonlinear measurement: polar coordinates
            return np.array([np.sqrt(x[0]**2 + x[1]**2), np.arctan2(x[1], x[0])])
        
        def H_jac(x):
            r = np.sqrt(x[0]**2 + x[1]**2)
            if r < 1e-6:
                return np.zeros((2, 4))
            return np.array([
                [x[0]/r, x[1]/r, 0, 0],
                [-x[1]/(r**2), x[0]/(r**2), 0, 0]
            ])
        
        ekf.set_state_transition(f_func, F_jac)
        ekf.set_measurement_function(h_func, H_jac)
        
        ekf.x = np.array([3.0, 4.0, 0.0, 0.0])
        ekf.P = np.eye(4) * 1.0
        ekf.Q = np.eye(4) * 0.01
        ekf.R = np.eye(2) * 0.1
        
        z = np.array([5.1, 0.9])  # range, bearing measurement
        ekf.update(z)
        
        # Should not crash and should update state
        assert np.all(np.isfinite(ekf.x))
        assert np.all(np.isfinite(ekf.P))


class TestUnscentedKalmanFilter:
    """Test the Unscented Kalman filter implementation."""
    
    @pytest.fixture
    def simple_ukf(self):
        """Create a simple UKF for testing."""
        def fx(x, dt):
            # Simple state transition
            x_new = x.copy()
            x_new[0] = x[0] + x[2] * dt
            x_new[1] = x[1] + x[3] * dt
            return x_new
        
        def hx(x):
            # Measurement function
            return x[:2]
        
        ukf = UnscentedKalmanFilter(4, 2, dt=1.0, hx=hx, fx=fx)
        ukf.Q = np.eye(4) * 0.1
        ukf.R = np.eye(2) * 1.0
        ukf.P = np.eye(4) * 10.0
        
        return ukf
    
    def test_initialization(self):
        """Test UKF initialization."""
        ukf = UnscentedKalmanFilter(4, 2, dt=0.5, alpha=0.001, beta=2.0, kappa=1.0)
        
        assert ukf.dim_x == 4
        assert ukf.dim_z == 2
        assert ukf.dt == 0.5
        assert ukf.alpha == 0.001
        assert ukf.beta == 2.0
        assert ukf.kappa == 1.0
        
        # Check sigma point dimensions
        expected_sigma_points = 2 * ukf.dim_aug + 1
        assert ukf.n_sigma == expected_sigma_points
        assert ukf.sigma_points_x.shape == (expected_sigma_points, 4)
        assert ukf.sigma_points_z.shape == (expected_sigma_points, 2)
    
    def test_weight_computation(self):
        """Test sigma point weight computation."""
        ukf = UnscentedKalmanFilter(4, 2, alpha=0.001, kappa=1.0)
        
        # Check weight properties
        assert len(ukf.Wm) == ukf.n_sigma
        assert len(ukf.Wc) == ukf.n_sigma
        assert abs(np.sum(ukf.Wm) - 1.0) < 1e-10
        assert abs(np.sum(ukf.Wc) - 1.0) < 1e-10
    
    def test_sigma_point_generation(self, simple_ukf):
        """Test sigma point generation."""
        x = np.array([1.0, 2.0, 0.5, 0.3])
        P = np.eye(4) * 2.0
        
        sigma_points = simple_ukf._generate_sigma_points(x, P)
        
        # Check dimensions
        assert sigma_points.shape[0] == simple_ukf.n_sigma
        
        # First sigma point should be the mean
        if simple_ukf.augment_state:
            npt.assert_array_almost_equal(sigma_points[0, :4], x)
        else:
            npt.assert_array_almost_equal(sigma_points[0], x)
    
    def test_unscented_transform(self, simple_ukf):
        """Test unscented transform."""
        # Create some test sigma points
        sigma_points = np.array([
            [1.0, 2.0],
            [1.1, 2.1],
            [0.9, 1.9],
            [1.05, 2.05],
            [0.95, 1.95]
        ])
        
        mean, cov = simple_ukf._unscented_transform(sigma_points)
        
        # Check dimensions
        assert len(mean) == 2
        assert cov.shape == (2, 2)
        
        # Mean should be close to average
        expected_mean = np.mean(sigma_points, axis=0)
        npt.assert_array_almost_equal(mean, expected_mean, decimal=1)
        
        # Covariance should be positive definite
        eigenvals = np.linalg.eigvals(cov)
        assert np.all(eigenvals >= 0)
    
    def test_predict_without_function(self):
        """Test prediction fails without state transition function."""
        ukf = UnscentedKalmanFilter(4, 2)
        
        with pytest.raises(ValueError, match="State transition function"):
            ukf.predict(1.0)
    
    def test_predict_step(self, simple_ukf):
        """Test UKF prediction step."""
        initial_state = np.array([1.0, 2.0, 0.5, 0.3])
        simple_ukf.x = initial_state.copy()
        initial_P = simple_ukf.P.copy()
        
        simple_ukf.predict(1.0)
        
        # Check that prediction occurred
        assert not np.array_equal(simple_ukf.x, initial_state)
        
        # Covariance should generally increase
        assert np.all(np.diag(simple_ukf.P) >= np.diag(initial_P) - 1e-10)
        
        # Check that prior is stored
        npt.assert_array_equal(simple_ukf.x_prior, initial_state)
    
    def test_update_without_function(self):
        """Test update fails without measurement function."""
        ukf = UnscentedKalmanFilter(4, 2, fx=lambda x, dt: x)
        z = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="Measurement function"):
            ukf.update(z)
    
    def test_update_step(self, simple_ukf):
        """Test UKF update step."""
        simple_ukf.x = np.array([1.0, 2.0, 0.5, 0.3])
        simple_ukf.P = np.eye(4) * 5.0
        
        z = np.array([1.1, 2.05])
        
        x_before = simple_ukf.x.copy()
        P_before = simple_ukf.P.copy()
        
        simple_ukf.update(z)
        
        # State should change toward measurement
        position_before = x_before[:2]
        position_after = simple_ukf.x[:2]
        
        # Distance to measurement should decrease
        dist_before = np.linalg.norm(position_before - z)
        dist_after = np.linalg.norm(position_after - z)
        assert dist_after <= dist_before
        
        # Covariance should generally decrease
        assert np.trace(simple_ukf.P) <= np.trace(P_before) + 1e-10
    
    def test_nonlinear_functions(self):
        """Test UKF with highly nonlinear functions."""
        def fx(x, dt):
            # Nonlinear state transition (coordinated turn)
            if len(x) >= 5:
                omega = x[4] if abs(x[4]) > 1e-6 else 1e-6
                s = np.sin(omega * dt)
                c = np.cos(omega * dt)
                
                x_new = x.copy()
                x_new[0] = x[0] + (x[2] * s - x[3] * (1 - c)) / omega
                x_new[1] = x[1] + (x[2] * (1 - c) + x[3] * s) / omega
                x_new[2] = x[2] * c - x[3] * s
                x_new[3] = x[2] * s + x[3] * c
                return x_new
            else:
                return x
        
        def hx(x):
            # Polar measurement
            return np.array([np.sqrt(x[0]**2 + x[1]**2), np.arctan2(x[1], x[0])])
        
        ukf = UnscentedKalmanFilter(5, 2, dt=0.1, hx=hx, fx=fx)
        ukf.x = np.array([10.0, 5.0, 2.0, 1.0, 0.1])
        ukf.P = np.eye(5)
        ukf.Q = np.eye(5) * 0.01
        ukf.R = np.eye(2) * 0.1
        
        # Test predict and update
        ukf.predict(0.1)
        z = np.array([11.2, 0.5])
        ukf.update(z)
        
        # Should not crash
        assert np.all(np.isfinite(ukf.x))
        assert np.all(np.isfinite(ukf.P))


class TestUtilityFunctions:
    """Test utility functions for Kalman filters."""
    
    def test_initialize_constant_velocity_filter(self):
        """Test CV filter initialization."""
        kf = initialize_constant_velocity_filter(dim=2, dt=0.5, 
                                                process_noise_std=2.0,
                                                measurement_noise_std=0.5)
        
        assert kf.dim_x == 4  # [x, y, vx, vy]
        assert kf.dim_z == 2  # [x, y]
        
        # Check state transition matrix
        expected_F = np.array([
            [1, 0, 0.5, 0],
            [0, 1, 0, 0.5],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        npt.assert_array_almost_equal(kf.F, expected_F)
        
        # Check measurement matrix
        expected_H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        npt.assert_array_almost_equal(kf.H, expected_H)
        
        # Check noise matrices
        assert np.all(np.diag(kf.R) == 0.25)  # 0.5^2
        assert kf.Q[0, 0] > 0  # Process noise should be positive
    
    def test_initialize_constant_acceleration_filter(self):
        """Test CA filter initialization."""
        kf = initialize_constant_acceleration_filter(dim=2, dt=0.5)
        
        assert kf.dim_x == 6  # [x, y, vx, vy, ax, ay]
        assert kf.dim_z == 2  # [x, y]
        
        # Check that acceleration terms are included in F
        assert kf.F[0, 4] > 0  # Position affected by acceleration
        assert kf.F[2, 4] > 0  # Velocity affected by acceleration
    
    def test_adaptive_noise_estimation(self):
        """Test adaptive noise estimation."""
        kf = KalmanFilter(4, 2)
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        kf.P = np.eye(4)
        kf.R = np.eye(2)
        
        # Create innovation history
        innovations = [np.array([0.1, 0.2]), np.array([0.15, 0.18]), 
                      np.array([0.08, 0.22]), np.array([0.12, 0.19])]
        
        R_before = kf.R.copy()
        adaptive_noise_estimation(kf, innovations, window_size=4)
        
        # R should be updated
        assert not np.array_equal(kf.R, R_before)
        
        # R should remain positive definite
        eigenvals = np.linalg.eigvals(kf.R)
        assert np.all(eigenvals > 0)
    
    def test_predict_multiple_steps(self):
        """Test multiple step prediction."""
        kf = initialize_constant_velocity_filter(dim=2, dt=1.0)
        kf.x = np.array([0.0, 0.0, 1.0, 1.0])  # Moving diagonally
        
        states, covariances = predict_multiple_steps(kf, n_steps=3, dt=1.0)
        
        assert states.shape == (3, 4)
        assert covariances.shape == (3, 4, 4)
        
        # States should show linear motion
        npt.assert_array_almost_equal(states[0], [1.0, 1.0, 1.0, 1.0])
        npt.assert_array_almost_equal(states[1], [2.0, 2.0, 1.0, 1.0])
        npt.assert_array_almost_equal(states[2], [3.0, 3.0, 1.0, 1.0])
        
        # Covariances should increase with time
        for i in range(1, 3):
            assert np.all(np.diag(covariances[i]) >= np.diag(covariances[i-1]))
        
        # Original state should be unchanged
        npt.assert_array_equal(kf.x, [0.0, 0.0, 1.0, 1.0])
    
    def test_compute_nees(self):
        """Test NEES computation."""
        true_state = np.array([1.0, 2.0, 0.5, 0.3])
        estimated_state = np.array([1.1, 2.05, 0.48, 0.32])
        covariance = np.eye(4) * 0.01
        
        nees = compute_nees(true_state, estimated_state, covariance)
        
        assert isinstance(nees, float)
        assert nees > 0
        assert np.isfinite(nees)
        
        # Test with singular covariance
        singular_cov = np.zeros((4, 4))
        nees_singular = compute_nees(true_state, estimated_state, singular_cov)
        assert nees_singular == np.inf
    
    def test_compute_nis(self):
        """Test NIS computation."""
        innovation = np.array([0.1, 0.2])
        innovation_cov = np.array([[0.01, 0.002], [0.002, 0.01]])
        
        nis = compute_nis(innovation, innovation_cov)
        
        assert isinstance(nis, float)
        assert nis > 0
        assert np.isfinite(nis)
        
        # Test with singular covariance
        singular_cov = np.zeros((2, 2))
        nis_singular = compute_nis(innovation, singular_cov)
        assert nis_singular == np.inf


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_covariance_positive_definiteness(self):
        """Test that covariance matrices remain positive definite."""
        kf = initialize_constant_velocity_filter(dim=2, dt=1.0)
        kf.x = np.array([0.0, 0.0, 1.0, 1.0])
        
        # Run multiple prediction and update cycles
        for i in range(10):
            kf.predict(1.0)
            
            # Check positive definiteness after prediction
            eigenvals = np.linalg.eigvals(kf.P)
            assert np.all(eigenvals > 0), f"Iteration {i}: P not positive definite after predict"
            
            # Add measurement
            z = np.array([float(i+1), float(i+1)]) + np.random.normal(0, 0.1, 2)
            kf.update(z)
            
            # Check positive definiteness after update
            eigenvals = np.linalg.eigvals(kf.P)
            assert np.all(eigenvals > 0), f"Iteration {i}: P not positive definite after update"
    
    def test_extreme_noise_values(self):
        """Test behavior with extreme noise values."""
        kf = KalmanFilter(4, 2)
        kf.F = np.eye(4)
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        
        # Very large process noise
        kf.Q = np.eye(4) * 1e6
        kf.R = np.eye(2) * 1e-6
        kf.P = np.eye(4)
        
        kf.predict(1.0)
        z = np.array([1.0, 2.0])
        kf.update(z)
        
        # Should not crash or produce invalid values
        assert np.all(np.isfinite(kf.x))
        assert np.all(np.isfinite(kf.P))
        
        # Very small process noise
        kf.Q = np.eye(4) * 1e-12
        kf.R = np.eye(2) * 1e6
        
        kf.predict(1.0)
        kf.update(z)
        
        assert np.all(np.isfinite(kf.x))
        assert np.all(np.isfinite(kf.P))
    
    def test_near_singular_matrices(self):
        """Test behavior with near-singular matrices."""
        kf = KalmanFilter(4, 2)
        kf.F = np.eye(4)
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        
        # Near-singular covariance
        kf.P = np.eye(4) * 1e-10
        kf.Q = np.eye(4) * 1e-12
        kf.R = np.eye(2) * 1e-10
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            kf.predict(1.0)
            z = np.array([1.0, 2.0])
            kf.update(z)
            
            # Should handle gracefully
            assert np.all(np.isfinite(kf.x))
    
    def test_large_time_steps(self):
        """Test behavior with large time steps."""
        kf = initialize_constant_velocity_filter(dim=2, dt=1000.0)  # Very large dt
        kf.x = np.array([0.0, 0.0, 1.0, 1.0])
        
        kf.predict(1000.0)
        
        # Should produce reasonable results
        assert np.all(np.isfinite(kf.x))
        assert np.all(np.isfinite(kf.P))
        
        # Position should have moved significantly
        assert abs(kf.x[0]) > 100
        assert abs(kf.x[1]) > 100


if __name__ == "__main__":
    pytest.main([__file__])