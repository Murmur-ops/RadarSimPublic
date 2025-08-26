"""
Comprehensive test suite for motion models.

Tests all motion model implementations including:
- Constant Velocity (CV)
- Constant Acceleration (CA) 
- Coordinated Turn (CT)
- Singer Acceleration
- Coordinate transformations
- Jacobian computations
- Model factory functions

Author: RadarSim Project
"""

import pytest
import numpy as np
import numpy.testing as npt
from unittest.mock import Mock, patch

from ..motion_models import (
    MotionModel, ModelParameters, CoordinateSystem,
    ConstantVelocityModel, ConstantAccelerationModel, 
    CoordinatedTurnModel, SingerAccelerationModel,
    cartesian_to_polar, polar_to_cartesian,
    cartesian_to_spherical, spherical_to_cartesian,
    compute_turn_rate, estimate_adaptive_process_noise,
    get_measurement_jacobian, create_motion_model
)


class TestModelParameters:
    """Test ModelParameters dataclass."""
    
    def test_initialization(self):
        """Test basic initialization."""
        params = ModelParameters(dt=0.1, process_noise_std=2.0)
        
        assert params.dt == 0.1
        assert params.process_noise_std == 2.0
        assert params.coordinate_system == CoordinateSystem.CARTESIAN_2D
        assert params.additional_params == {}
    
    def test_with_additional_params(self):
        """Test initialization with additional parameters."""
        additional = {"turn_rate": 0.1, "alpha": 0.05}
        params = ModelParameters(
            dt=0.1, 
            coordinate_system=CoordinateSystem.CARTESIAN_3D,
            additional_params=additional
        )
        
        assert params.coordinate_system == CoordinateSystem.CARTESIAN_3D
        assert params.additional_params["turn_rate"] == 0.1
        assert params.additional_params["alpha"] == 0.05


class TestConstantVelocityModel:
    """Test Constant Velocity motion model."""
    
    @pytest.fixture
    def cv_2d_params(self):
        """Create 2D CV model parameters."""
        return ModelParameters(
            dt=1.0, 
            process_noise_std=0.5,
            coordinate_system=CoordinateSystem.CARTESIAN_2D
        )
    
    @pytest.fixture
    def cv_3d_params(self):
        """Create 3D CV model parameters."""
        return ModelParameters(
            dt=1.0,
            process_noise_std=0.5, 
            coordinate_system=CoordinateSystem.CARTESIAN_3D
        )
    
    def test_2d_initialization(self, cv_2d_params):
        """Test 2D CV model initialization."""
        model = ConstantVelocityModel(cv_2d_params)
        
        assert model.state_dim == 4
        assert model.dt == 1.0
        assert model.coordinate_system == CoordinateSystem.CARTESIAN_2D
    
    def test_3d_initialization(self, cv_3d_params):
        """Test 3D CV model initialization."""
        model = ConstantVelocityModel(cv_3d_params)
        
        assert model.state_dim == 6
        assert model.coordinate_system == CoordinateSystem.CARTESIAN_3D
    
    def test_unsupported_coordinate_system(self):
        """Test error with unsupported coordinate system."""
        params = ModelParameters(dt=1.0, coordinate_system=CoordinateSystem.POLAR)
        
        with pytest.raises(ValueError, match="Unsupported coordinate system"):
            ConstantVelocityModel(params)
    
    def test_2d_transition_matrix(self, cv_2d_params):
        """Test 2D state transition matrix."""
        model = ConstantVelocityModel(cv_2d_params)
        state = np.array([1.0, 2.0, 0.5, 0.3])
        
        F = model.get_transition_matrix(state)
        
        expected_F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        npt.assert_array_equal(F, expected_F)
    
    def test_3d_transition_matrix(self, cv_3d_params):
        """Test 3D state transition matrix."""
        model = ConstantVelocityModel(cv_3d_params)
        state = np.array([1.0, 2.0, 3.0, 0.5, 0.3, 0.1])
        
        F = model.get_transition_matrix(state)
        
        expected_F = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        npt.assert_array_equal(F, expected_F)
    
    def test_2d_process_noise_covariance(self, cv_2d_params):
        """Test 2D process noise covariance."""
        model = ConstantVelocityModel(cv_2d_params)
        state = np.array([1.0, 2.0, 0.5, 0.3])
        
        Q = model.get_process_noise_covariance(state)
        
        assert Q.shape == (4, 4)
        assert np.all(np.diag(Q) > 0)  # Should be positive
        
        # Check discrete white noise acceleration structure
        sigma = 0.5
        dt = 1.0
        expected_variance_pos = (dt**4/4) * sigma**2
        expected_variance_vel = dt**2 * sigma**2
        
        assert abs(Q[0, 0] - expected_variance_pos) < 1e-10
        assert abs(Q[2, 2] - expected_variance_vel) < 1e-10
    
    def test_3d_process_noise_covariance(self, cv_3d_params):
        """Test 3D process noise covariance."""
        model = ConstantVelocityModel(cv_3d_params)
        state = np.array([1.0, 2.0, 3.0, 0.5, 0.3, 0.1])
        
        Q = model.get_process_noise_covariance(state)
        
        assert Q.shape == (6, 6)
        assert np.all(np.diag(Q) > 0)
        
        # Check symmetry
        npt.assert_array_almost_equal(Q, Q.T)
        
        # Check positive definiteness
        eigenvals = np.linalg.eigvals(Q)
        assert np.all(eigenvals > 0)
    
    def test_state_prediction(self, cv_2d_params):
        """Test state prediction."""
        model = ConstantVelocityModel(cv_2d_params)
        state = np.array([1.0, 2.0, 0.5, 0.3])
        
        predicted_state = model.predict_state(state)
        
        expected_state = np.array([1.5, 2.3, 0.5, 0.3])
        npt.assert_array_almost_equal(predicted_state, expected_state)
    
    def test_validate_state(self, cv_2d_params):
        """Test state validation."""
        model = ConstantVelocityModel(cv_2d_params)
        
        valid_state = np.array([1.0, 2.0, 0.5, 0.3])
        invalid_state = np.array([1.0, 2.0, 0.5])  # Wrong dimension
        
        assert model.validate_state(valid_state)
        assert not model.validate_state(invalid_state)


class TestConstantAccelerationModel:
    """Test Constant Acceleration motion model."""
    
    @pytest.fixture
    def ca_2d_params(self):
        """Create 2D CA model parameters."""
        return ModelParameters(
            dt=1.0,
            process_noise_std=0.5,
            coordinate_system=CoordinateSystem.CARTESIAN_2D
        )
    
    @pytest.fixture
    def ca_3d_params(self):
        """Create 3D CA model parameters."""
        return ModelParameters(
            dt=1.0,
            process_noise_std=0.5,
            coordinate_system=CoordinateSystem.CARTESIAN_3D
        )
    
    def test_2d_initialization(self, ca_2d_params):
        """Test 2D CA model initialization."""
        model = ConstantAccelerationModel(ca_2d_params)
        
        assert model.state_dim == 6  # [x, y, vx, vy, ax, ay]
        assert model.coordinate_system == CoordinateSystem.CARTESIAN_2D
    
    def test_3d_initialization(self, ca_3d_params):
        """Test 3D CA model initialization."""
        model = ConstantAccelerationModel(ca_3d_params)
        
        assert model.state_dim == 9  # [x, y, z, vx, vy, vz, ax, ay, az]
        assert model.coordinate_system == CoordinateSystem.CARTESIAN_3D
    
    def test_2d_transition_matrix(self, ca_2d_params):
        """Test 2D state transition matrix."""
        model = ConstantAccelerationModel(ca_2d_params)
        state = np.array([1.0, 2.0, 0.5, 0.3, 0.1, 0.05])
        
        F = model.get_transition_matrix(state)
        
        dt = 1.0
        expected_F = np.array([
            [1, 0, dt, 0, dt**2/2, 0],
            [0, 1, 0, dt, 0, dt**2/2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        npt.assert_array_equal(F, expected_F)
    
    def test_3d_transition_matrix(self, ca_3d_params):
        """Test 3D state transition matrix."""
        model = ConstantAccelerationModel(ca_3d_params)
        state = np.zeros(9)
        
        F = model.get_transition_matrix(state)
        
        assert F.shape == (9, 9)
        
        # Check position-velocity coupling
        dt = 1.0
        assert F[0, 3] == dt  # x position affected by vx
        assert F[1, 4] == dt  # y position affected by vy
        assert F[2, 5] == dt  # z position affected by vz
        
        # Check position-acceleration coupling
        assert F[0, 6] == dt**2/2  # x position affected by ax
        assert F[1, 7] == dt**2/2  # y position affected by ay
        assert F[2, 8] == dt**2/2  # z position affected by az
        
        # Check velocity-acceleration coupling
        assert F[3, 6] == dt  # vx affected by ax
        assert F[4, 7] == dt  # vy affected by ay
        assert F[5, 8] == dt  # vz affected by az
    
    def test_state_prediction(self, ca_2d_params):
        """Test state prediction with acceleration."""
        model = ConstantAccelerationModel(ca_2d_params)
        state = np.array([1.0, 2.0, 0.5, 0.3, 0.1, 0.05])
        
        predicted_state = model.predict_state(state)
        
        dt = 1.0
        expected_state = np.array([
            1.0 + 0.5 * dt + 0.1 * dt**2/2,    # x
            2.0 + 0.3 * dt + 0.05 * dt**2/2,   # y  
            0.5 + 0.1 * dt,                     # vx
            0.3 + 0.05 * dt,                    # vy
            0.1,                                # ax (unchanged)
            0.05                                # ay (unchanged)
        ])
        
        npt.assert_array_almost_equal(predicted_state, expected_state)
    
    def test_process_noise_structure(self, ca_2d_params):
        """Test process noise matrix structure."""
        model = ConstantAccelerationModel(ca_2d_params)
        state = np.zeros(6)
        
        Q = model.get_process_noise_covariance(state)
        
        assert Q.shape == (6, 6)
        
        # Check that Q is positive definite
        eigenvals = np.linalg.eigvals(Q)
        assert np.all(eigenvals > 0)
        
        # Check that Q is symmetric
        npt.assert_array_almost_equal(Q, Q.T)
        
        # Check discrete white noise jerk model structure
        assert Q[0, 2] > 0  # Position-velocity correlation
        assert Q[0, 4] > 0  # Position-acceleration correlation
        assert Q[2, 4] > 0  # Velocity-acceleration correlation


class TestCoordinatedTurnModel:
    """Test Coordinated Turn motion model."""
    
    @pytest.fixture
    def ct_2d_known_params(self):
        """Create 2D CT model with known turn rate."""
        return ModelParameters(
            dt=0.1,
            process_noise_std=0.1,
            coordinate_system=CoordinateSystem.CARTESIAN_2D,
            additional_params={"turn_rate": 0.1}  # Known turn rate
        )
    
    @pytest.fixture
    def ct_2d_unknown_params(self):
        """Create 2D CT model with unknown turn rate."""
        return ModelParameters(
            dt=0.1,
            process_noise_std=0.1,
            coordinate_system=CoordinateSystem.CARTESIAN_2D,
            additional_params={"turn_rate_noise": 0.01}
        )
    
    @pytest.fixture
    def ct_3d_params(self):
        """Create 3D CT model parameters."""
        return ModelParameters(
            dt=0.1,
            coordinate_system=CoordinateSystem.CARTESIAN_3D,
            additional_params={"turn_rate_noise": 0.01}
        )
    
    def test_known_turn_rate_initialization(self, ct_2d_known_params):
        """Test CT model with known turn rate."""
        model = CoordinatedTurnModel(ct_2d_known_params)
        
        assert model.state_dim == 4  # No turn rate in state
        assert model.known_turn_rate == 0.1
        assert model.turn_rate_noise == 0.1  # Default value
    
    def test_unknown_turn_rate_initialization(self, ct_2d_unknown_params):
        """Test CT model with unknown turn rate."""
        model = CoordinatedTurnModel(ct_2d_unknown_params)
        
        assert model.state_dim == 5  # Includes turn rate in state
        assert model.known_turn_rate is None
        assert model.turn_rate_noise == 0.01
    
    def test_3d_initialization(self, ct_3d_params):
        """Test 3D CT model."""
        model = CoordinatedTurnModel(ct_3d_params)
        
        assert model.state_dim == 7  # Always estimate turn rate in 3D
        assert model.coordinate_system == CoordinateSystem.CARTESIAN_3D
    
    def test_straight_line_motion(self, ct_2d_unknown_params):
        """Test transition matrix for straight line motion (omega ≈ 0)."""
        model = CoordinatedTurnModel(ct_2d_unknown_params)
        state = np.array([1.0, 2.0, 0.5, 0.3, 1e-8])  # Very small turn rate
        
        F = model.get_transition_matrix(state)
        
        # Should approximate constant velocity model
        dt = 0.1
        expected_F_cv = np.array([
            [1, 0, dt, 0, 0],
            [0, 1, 0, dt, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])
        
        npt.assert_array_almost_equal(F[:4, :4], expected_F_cv[:4, :4], decimal=6)
    
    def test_coordinated_turn_motion(self, ct_2d_unknown_params):
        """Test transition matrix for coordinated turn."""
        model = CoordinatedTurnModel(ct_2d_unknown_params)
        omega = 0.2
        state = np.array([1.0, 2.0, 0.5, 0.3, omega])
        
        F = model.get_transition_matrix(state)
        
        assert F.shape == (5, 5)
        
        # Check that F differs from straight line case
        dt = 0.1
        sin_omega_dt = np.sin(omega * dt)
        cos_omega_dt = np.cos(omega * dt)
        
        # Position updates should include turn effects
        assert abs(F[0, 2] - sin_omega_dt/omega) < 1e-10
        assert abs(F[0, 3] - (-(1-cos_omega_dt)/omega)) < 1e-10
        
        # Velocity updates should include rotation
        assert abs(F[2, 2] - cos_omega_dt) < 1e-10
        assert abs(F[2, 3] - (-sin_omega_dt)) < 1e-10
    
    def test_state_prediction_straight(self, ct_2d_known_params):
        """Test state prediction for straight line motion."""
        model = CoordinatedTurnModel(ct_2d_known_params)
        model.known_turn_rate = 1e-8  # Nearly zero
        
        state = np.array([1.0, 2.0, 0.5, 0.3])
        predicted_state = model.predict_state(state)
        
        dt = 0.1
        expected_state = np.array([1.05, 2.03, 0.5, 0.3])
        npt.assert_array_almost_equal(predicted_state, expected_state, decimal=6)
    
    def test_state_prediction_turn(self, ct_2d_unknown_params):
        """Test state prediction for coordinated turn."""
        model = CoordinatedTurnModel(ct_2d_unknown_params)
        omega = 0.5
        state = np.array([0.0, 0.0, 1.0, 0.0, omega])  # Moving in +x direction
        
        predicted_state = model.predict_state(state)
        
        dt = 0.1
        sin_omega_dt = np.sin(omega * dt)
        cos_omega_dt = np.cos(omega * dt)
        
        # Should turn in xy plane
        expected_x = (1.0 * sin_omega_dt) / omega
        expected_y = (1.0 * (1 - cos_omega_dt)) / omega
        expected_vx = 1.0 * cos_omega_dt
        expected_vy = 1.0 * sin_omega_dt
        
        assert abs(predicted_state[0] - expected_x) < 1e-10
        assert abs(predicted_state[1] - expected_y) < 1e-10
        assert abs(predicted_state[2] - expected_vx) < 1e-10
        assert abs(predicted_state[3] - expected_vy) < 1e-10
        assert predicted_state[4] == omega  # Turn rate unchanged
    
    def test_3d_coordinated_turn(self, ct_3d_params):
        """Test 3D coordinated turn (turn in horizontal plane)."""
        model = CoordinatedTurnModel(ct_3d_params)
        omega = 0.3
        state = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.2, omega])  # Moving in xy, constant vz
        
        predicted_state = model.predict_state(state)
        
        # Z component should be unaffected by horizontal turn
        dt = 0.1
        expected_z = 1.0 + 0.2 * dt
        expected_vz = 0.2
        
        assert abs(predicted_state[2] - expected_z) < 1e-10
        assert abs(predicted_state[5] - expected_vz) < 1e-10
        
        # XY motion should show turn effects
        assert predicted_state[0] != state[0] + state[3] * dt  # Not straight line
        assert predicted_state[1] != state[1] + state[4] * dt
    
    def test_process_noise_known_turn_rate(self, ct_2d_known_params):
        """Test process noise for known turn rate model."""
        model = CoordinatedTurnModel(ct_2d_known_params)
        state = np.array([1.0, 2.0, 0.5, 0.3])
        
        Q = model.get_process_noise_covariance(state)
        
        assert Q.shape == (4, 4)
        assert np.all(np.diag(Q) > 0)
        
        # Should not include turn rate noise
        eigenvals = np.linalg.eigvals(Q)
        assert np.all(eigenvals > 0)
    
    def test_process_noise_unknown_turn_rate(self, ct_2d_unknown_params):
        """Test process noise for unknown turn rate model."""
        model = CoordinatedTurnModel(ct_2d_unknown_params)
        state = np.array([1.0, 2.0, 0.5, 0.3, 0.1])
        
        Q = model.get_process_noise_covariance(state)
        
        assert Q.shape == (5, 5)
        assert Q[4, 4] > 0  # Turn rate noise should be positive
        
        # Check that turn rate noise is as specified
        dt = 0.1
        expected_turn_rate_noise = dt * model.turn_rate_noise**2
        assert abs(Q[4, 4] - expected_turn_rate_noise) < 1e-10


class TestSingerAccelerationModel:
    """Test Singer Acceleration motion model."""
    
    @pytest.fixture
    def singer_2d_params(self):
        """Create 2D Singer model parameters."""
        return ModelParameters(
            dt=0.1,
            process_noise_std=0.5,
            coordinate_system=CoordinateSystem.CARTESIAN_2D,
            additional_params={
                "alpha": 0.1,
                "max_acceleration": 5.0,
                "maneuver_threshold": 2.0
            }
        )
    
    def test_initialization(self, singer_2d_params):
        """Test Singer model initialization."""
        model = SingerAccelerationModel(singer_2d_params)
        
        assert model.state_dim == 6
        assert model.alpha == 0.1
        assert model.max_acceleration == 5.0
        assert model.maneuver_threshold == 2.0
        assert not model.maneuver_detected
    
    def test_transition_matrix(self, singer_2d_params):
        """Test Singer model transition matrix."""
        model = SingerAccelerationModel(singer_2d_params)
        state = np.zeros(6)
        
        F = model.get_transition_matrix(state)
        
        assert F.shape == (6, 6)
        
        # Check Singer model structure
        dt = 0.1
        alpha = 0.1
        exp_alpha_dt = np.exp(-alpha * dt)
        
        # Check acceleration decay
        assert abs(F[4, 4] - exp_alpha_dt) < 1e-10
        assert abs(F[5, 5] - exp_alpha_dt) < 1e-10
        
        # Check position-acceleration coupling
        c2 = (alpha * dt - 1 + exp_alpha_dt) / (alpha**2)
        assert abs(F[0, 4] - c2) < 1e-10
        assert abs(F[1, 5] - c2) < 1e-10
        
        # Check velocity-acceleration coupling
        c1 = (1 - exp_alpha_dt) / alpha
        assert abs(F[2, 4] - c1) < 1e-10
        assert abs(F[3, 5] - c1) < 1e-10
    
    def test_process_noise_covariance(self, singer_2d_params):
        """Test Singer model process noise."""
        model = SingerAccelerationModel(singer_2d_params)
        state = np.zeros(6)
        
        Q = model.get_process_noise_covariance(state)
        
        assert Q.shape == (6, 6)
        assert np.all(np.diag(Q) > 0)
        
        # Check that Q is symmetric and positive definite
        npt.assert_array_almost_equal(Q, Q.T)
        eigenvals = np.linalg.eigvals(Q)
        assert np.all(eigenvals > 0)
        
        # Check block structure for x and y axes
        # Q should have 3x3 blocks for each axis
        assert Q[0, 2] != 0  # Position-velocity correlation
        assert Q[0, 4] != 0  # Position-acceleration correlation
        assert Q[2, 4] != 0  # Velocity-acceleration correlation
    
    def test_maneuver_detection(self, singer_2d_params):
        """Test maneuver detection functionality."""
        model = SingerAccelerationModel(singer_2d_params)
        
        # Small innovation should not trigger maneuver
        innovation = np.array([0.1, 0.05])
        innovation_cov = np.eye(2) * 0.1
        
        is_maneuver = model.detect_maneuver(innovation, innovation_cov)
        assert not is_maneuver
        assert not model.maneuver_detected
        
        # Large innovation should trigger maneuver
        innovation = np.array([3.0, 2.5])
        is_maneuver = model.detect_maneuver(innovation, innovation_cov)
        assert is_maneuver
        assert model.maneuver_detected
    
    def test_maneuver_detection_singular_covariance(self, singer_2d_params):
        """Test maneuver detection with singular covariance."""
        model = SingerAccelerationModel(singer_2d_params)
        
        innovation = np.array([3.0, 2.5])
        singular_cov = np.zeros((2, 2))
        
        # Should use fallback method
        is_maneuver = model.detect_maneuver(innovation, singular_cov)
        assert isinstance(is_maneuver, bool)
    
    def test_adaptive_noise_during_maneuver(self, singer_2d_params):
        """Test that process noise increases during maneuvers."""
        model = SingerAccelerationModel(singer_2d_params)
        state = np.zeros(6)
        
        # Get baseline noise
        Q_normal = model.get_process_noise_covariance(state)
        
        # Trigger maneuver
        model.maneuver_detected = True
        Q_maneuver = model.get_process_noise_covariance(state)
        
        # Noise should be higher during maneuvers
        assert np.all(np.diag(Q_maneuver) >= np.diag(Q_normal))
        assert np.trace(Q_maneuver) > np.trace(Q_normal)
    
    def test_state_prediction(self, singer_2d_params):
        """Test Singer model state prediction."""
        model = SingerAccelerationModel(singer_2d_params)
        
        state = np.array([1.0, 2.0, 0.5, 0.3, 0.1, 0.05])
        predicted_state = model.predict_state(state)
        
        # Should use transition matrix
        F = model.get_transition_matrix(state)
        expected_state = F @ state
        
        npt.assert_array_almost_equal(predicted_state, expected_state)
        
        # Check that acceleration decays
        alpha = 0.1
        dt = 0.1
        expected_ax = 0.1 * np.exp(-alpha * dt)
        expected_ay = 0.05 * np.exp(-alpha * dt)
        
        assert abs(predicted_state[4] - expected_ax) < 1e-10
        assert abs(predicted_state[5] - expected_ay) < 1e-10


class TestCoordinateTransformations:
    """Test coordinate transformation functions."""
    
    def test_cartesian_to_polar(self):
        """Test Cartesian to polar conversion."""
        x, y = 3.0, 4.0
        vx, vy = 0.1, 0.2
        
        range_val, azimuth, range_rate, azimuth_rate = cartesian_to_polar(x, y, vx, vy)
        
        assert abs(range_val - 5.0) < 1e-10  # sqrt(3^2 + 4^2)
        assert abs(azimuth - np.arctan2(4.0, 3.0)) < 1e-10
        
        # Check range rate
        expected_range_rate = (x * vx + y * vy) / range_val
        assert abs(range_rate - expected_range_rate) < 1e-10
        
        # Check azimuth rate
        expected_azimuth_rate = (x * vy - y * vx) / (range_val**2)
        assert abs(azimuth_rate - expected_azimuth_rate) < 1e-10
    
    def test_polar_to_cartesian(self):
        """Test polar to Cartesian conversion."""
        range_val = 5.0
        azimuth = np.arctan2(4.0, 3.0)
        range_rate = 0.1
        azimuth_rate = 0.02
        
        x, y, vx, vy = polar_to_cartesian(range_val, azimuth, range_rate, azimuth_rate)
        
        assert abs(x - 3.0) < 1e-10
        assert abs(y - 4.0) < 1e-10
        
        # Check velocity conversion
        expected_vx = range_rate * np.cos(azimuth) - range_val * azimuth_rate * np.sin(azimuth)
        expected_vy = range_rate * np.sin(azimuth) + range_val * azimuth_rate * np.cos(azimuth)
        
        assert abs(vx - expected_vx) < 1e-10
        assert abs(vy - expected_vy) < 1e-10
    
    def test_cartesian_polar_roundtrip(self):
        """Test round-trip conversion between Cartesian and polar."""
        x_orig, y_orig = 3.0, 4.0
        vx_orig, vy_orig = 0.1, 0.2
        
        # Convert to polar and back
        range_val, azimuth, range_rate, azimuth_rate = cartesian_to_polar(
            x_orig, y_orig, vx_orig, vy_orig)
        x, y, vx, vy = polar_to_cartesian(range_val, azimuth, range_rate, azimuth_rate)
        
        npt.assert_array_almost_equal([x, y, vx, vy], [x_orig, y_orig, vx_orig, vy_orig])
    
    def test_cartesian_to_spherical(self):
        """Test Cartesian to spherical conversion."""
        x, y, z = 3.0, 4.0, 5.0
        vx, vy, vz = 0.1, 0.2, 0.15
        
        range_val, azimuth, elevation, range_rate, azimuth_rate, elevation_rate = \
            cartesian_to_spherical(x, y, z, vx, vy, vz)
        
        expected_range = np.sqrt(x**2 + y**2 + z**2)
        expected_azimuth = np.arctan2(y, x)
        expected_elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
        
        assert abs(range_val - expected_range) < 1e-10
        assert abs(azimuth - expected_azimuth) < 1e-10
        assert abs(elevation - expected_elevation) < 1e-10
    
    def test_spherical_to_cartesian(self):
        """Test spherical to Cartesian conversion."""
        range_val = np.sqrt(50)  # sqrt(3^2 + 4^2 + 5^2)
        azimuth = np.arctan2(4.0, 3.0)
        elevation = np.arctan2(5.0, 5.0)  # sqrt(3^2 + 4^2)
        range_rate = 0.1
        azimuth_rate = 0.02
        elevation_rate = 0.01
        
        x, y, z, vx, vy, vz = spherical_to_cartesian(
            range_val, azimuth, elevation, range_rate, azimuth_rate, elevation_rate)
        
        expected_x = range_val * np.cos(elevation) * np.cos(azimuth)
        expected_y = range_val * np.cos(elevation) * np.sin(azimuth)
        expected_z = range_val * np.sin(elevation)
        
        assert abs(x - expected_x) < 1e-10
        assert abs(y - expected_y) < 1e-10
        assert abs(z - expected_z) < 1e-10
    
    def test_cartesian_spherical_roundtrip(self):
        """Test round-trip conversion between Cartesian and spherical."""
        x_orig, y_orig, z_orig = 3.0, 4.0, 5.0
        vx_orig, vy_orig, vz_orig = 0.1, 0.2, 0.15
        
        # Convert to spherical and back
        range_val, azimuth, elevation, range_rate, azimuth_rate, elevation_rate = \
            cartesian_to_spherical(x_orig, y_orig, z_orig, vx_orig, vy_orig, vz_orig)
        x, y, z, vx, vy, vz = spherical_to_cartesian(
            range_val, azimuth, elevation, range_rate, azimuth_rate, elevation_rate)
        
        npt.assert_array_almost_equal(
            [x, y, z, vx, vy, vz], 
            [x_orig, y_orig, z_orig, vx_orig, vy_orig, vz_orig],
            decimal=10
        )
    
    def test_edge_cases(self):
        """Test edge cases in coordinate transformations."""
        # Zero position
        range_val, azimuth, range_rate, azimuth_rate = cartesian_to_polar(0.0, 0.0, 1.0, 1.0)
        assert range_val == 0.0
        assert range_rate == 0.0
        assert azimuth_rate == 0.0
        
        # Near-zero position in 3D
        range_val, azimuth, elevation, range_rate, azimuth_rate, elevation_rate = \
            cartesian_to_spherical(1e-12, 1e-12, 1.0, 0.0, 0.0, 1.0)
        assert abs(elevation - np.pi/2) < 1e-6  # Should be pointing up


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_compute_turn_rate_2d_cv(self):
        """Test turn rate computation for 2D CV state."""
        state = np.array([1.0, 2.0, 0.5, 0.3])  # No acceleration
        turn_rate = compute_turn_rate(state, CoordinateSystem.CARTESIAN_2D)
        
        assert turn_rate == 0.0  # No acceleration, no turn
    
    def test_compute_turn_rate_2d_ca(self):
        """Test turn rate computation for 2D CA state."""
        state = np.array([1.0, 2.0, 1.0, 0.0, 0.0, 1.0])  # Moving in x, accelerating in y
        turn_rate = compute_turn_rate(state, CoordinateSystem.CARTESIAN_2D)
        
        assert turn_rate > 0  # Should detect turn
        assert turn_rate == 1.0  # Centripetal acceleration / speed
    
    def test_compute_turn_rate_3d(self):
        """Test turn rate computation for 3D state."""
        state = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.5, 0.0, 1.0, 0.0])
        turn_rate = compute_turn_rate(state, CoordinateSystem.CARTESIAN_3D)
        
        assert turn_rate > 0  # Should detect horizontal turn
    
    def test_estimate_adaptive_process_noise(self):
        """Test adaptive process noise estimation."""
        # Small innovations should give low noise
        small_innovations = [np.array([0.1, 0.05]) for _ in range(5)]
        noise = estimate_adaptive_process_noise(small_innovations, base_noise=1.0)
        assert noise >= 1.0  # Should be at least base noise
        
        # Large innovations should give higher noise
        large_innovations = [np.array([2.0, 1.5]) for _ in range(5)]
        noise_large = estimate_adaptive_process_noise(large_innovations, base_noise=1.0)
        assert noise_large > noise
    
    def test_get_measurement_jacobian_cartesian(self):
        """Test measurement Jacobian for Cartesian measurements."""
        # 2D CV state
        state = np.array([1.0, 2.0, 0.5, 0.3])
        H = get_measurement_jacobian(state, "cartesian")
        
        expected_H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        npt.assert_array_equal(H, expected_H)
        
        # 3D CV state  
        state = np.array([1.0, 2.0, 3.0, 0.5, 0.3, 0.1])
        H = get_measurement_jacobian(state, "cartesian")
        
        expected_H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        npt.assert_array_equal(H, expected_H)
    
    def test_get_measurement_jacobian_polar(self):
        """Test measurement Jacobian for polar measurements."""
        state = np.array([3.0, 4.0, 0.5, 0.3])  # Position at (3,4)
        H = get_measurement_jacobian(state, "polar")
        
        range_val = 5.0
        expected_H = np.array([
            [3.0/range_val, 4.0/range_val, 0, 0],  # Range derivatives
            [-4.0/(range_val**2), 3.0/(range_val**2), 0, 0]  # Azimuth derivatives
        ])
        
        npt.assert_array_almost_equal(H, expected_H)
    
    def test_get_measurement_jacobian_spherical(self):
        """Test measurement Jacobian for spherical measurements."""
        state = np.array([3.0, 4.0, 5.0, 0.5, 0.3, 0.1])
        H = get_measurement_jacobian(state, "spherical")
        
        assert H.shape == (3, 6)
        
        # Check range derivatives (first row)
        range_val = np.sqrt(3**2 + 4**2 + 5**2)
        npt.assert_array_almost_equal(H[0, :3], [3.0/range_val, 4.0/range_val, 5.0/range_val])
    
    def test_get_measurement_jacobian_edge_cases(self):
        """Test measurement Jacobian edge cases."""
        # Near-zero position
        state = np.array([1e-8, 1e-8, 0.5, 0.3])
        H = get_measurement_jacobian(state, "polar")
        
        # Should not crash and should return zeros
        assert H.shape == (2, 4)
        npt.assert_array_almost_equal(H, np.zeros((2, 4)))


class TestModelFactory:
    """Test model factory function."""
    
    def test_create_cv_model(self):
        """Test creating CV model via factory."""
        params = ModelParameters(dt=1.0, coordinate_system=CoordinateSystem.CARTESIAN_2D)
        model = create_motion_model("cv", params)
        
        assert isinstance(model, ConstantVelocityModel)
        assert model.state_dim == 4
    
    def test_create_ca_model(self):
        """Test creating CA model via factory."""
        params = ModelParameters(dt=1.0, coordinate_system=CoordinateSystem.CARTESIAN_2D)
        model = create_motion_model("ca", params)
        
        assert isinstance(model, ConstantAccelerationModel)
        assert model.state_dim == 6
    
    def test_create_ct_model(self):
        """Test creating CT model via factory."""
        params = ModelParameters(
            dt=1.0, 
            coordinate_system=CoordinateSystem.CARTESIAN_2D,
            additional_params={"turn_rate": 0.1}
        )
        model = create_motion_model("ct", params)
        
        assert isinstance(model, CoordinatedTurnModel)
        assert model.known_turn_rate == 0.1
    
    def test_create_singer_model(self):
        """Test creating Singer model via factory."""
        params = ModelParameters(
            dt=1.0,
            coordinate_system=CoordinateSystem.CARTESIAN_2D,
            additional_params={"alpha": 0.1, "max_acceleration": 5.0}
        )
        model = create_motion_model("singer", params)
        
        assert isinstance(model, SingerAccelerationModel)
        assert model.alpha == 0.1
        assert model.max_acceleration == 5.0
    
    def test_unknown_model_type(self):
        """Test error with unknown model type."""
        params = ModelParameters(dt=1.0)
        
        with pytest.raises(ValueError, match="Unknown model type"):
            create_motion_model("unknown", params)
    
    def test_available_models(self):
        """Test that all expected models are available."""
        params = ModelParameters(dt=1.0)
        
        # Should not raise exceptions
        create_motion_model("cv", params)
        create_motion_model("ca", params)
        create_motion_model("ct", params)  
        create_motion_model("singer", params)


if __name__ == "__main__":
    pytest.main([__file__])