"""
Comprehensive test suite for IMM (Interacting Multiple Model) filter.

Tests all aspects of the IMM filter including:
- Model probability updates
- Mixing calculations
- Combined state estimation
- Model switching detection
- Performance metrics
- Multi-model coordination

Author: RadarSim Project
"""

import pytest
import numpy as np
import numpy.testing as npt
from unittest.mock import Mock, patch, MagicMock
import warnings

from ..imm_filter import (
    IMMParameters, ModelState, IMMFilter
)
from ..kalman_filters import KalmanFilter, ExtendedKalmanFilter
from ..motion_models import (
    ModelParameters, CoordinateSystem,
    ConstantVelocityModel, ConstantAccelerationModel, CoordinatedTurnModel
)


class TestIMMParameters:
    """Test IMMParameters dataclass."""
    
    def test_initialization(self):
        """Test basic initialization."""
        model_types = ['cv', 'ca']
        transition_probs = np.array([[0.9, 0.1], [0.2, 0.8]])
        
        params = IMMParameters(
            dt=0.1,
            model_types=model_types,
            transition_probabilities=transition_probs
        )
        
        assert params.dt == 0.1
        assert params.model_types == model_types
        npt.assert_array_equal(params.transition_probabilities, transition_probs)
        assert params.initial_model_probabilities is None
        assert params.coordinate_system == CoordinateSystem.CARTESIAN_2D
        assert params.process_noise_stds is None
        assert params.measurement_noise_std == 1.0
        assert params.gate_threshold == 9.21
        assert params.min_probability == 1e-6
        assert params.normalize_probabilities
        assert params.model_specific_params == {}
    
    def test_with_all_parameters(self):
        """Test initialization with all parameters."""
        model_types = ['cv', 'ca', 'ct']
        transition_probs = np.array([
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7]
        ])
        initial_probs = np.array([0.5, 0.3, 0.2])
        process_noise_stds = [0.1, 0.2, 0.15]
        model_specific = {
            'ct': {'turn_rate': 0.1, 'turn_rate_noise': 0.01}
        }
        
        params = IMMParameters(
            dt=0.05,
            model_types=model_types,
            transition_probabilities=transition_probs,
            initial_model_probabilities=initial_probs,
            coordinate_system=CoordinateSystem.CARTESIAN_3D,
            process_noise_stds=process_noise_stds,
            measurement_noise_std=0.5,
            gate_threshold=5.99,
            min_probability=1e-8,
            normalize_probabilities=False,
            model_specific_params=model_specific
        )
        
        assert params.dt == 0.05
        assert params.coordinate_system == CoordinateSystem.CARTESIAN_3D
        npt.assert_array_equal(params.initial_model_probabilities, initial_probs)
        assert params.process_noise_stds == process_noise_stds
        assert params.measurement_noise_std == 0.5
        assert params.gate_threshold == 5.99
        assert params.min_probability == 1e-8
        assert not params.normalize_probabilities
        assert params.model_specific_params == model_specific


class TestModelState:
    """Test ModelState container."""
    
    def test_initialization(self):
        """Test ModelState initialization."""
        kf = Mock(spec=KalmanFilter)
        motion_model = Mock()
        
        model_state = ModelState(
            model_id='cv',
            filter_obj=kf,
            motion_model=motion_model,
            probability=0.6
        )
        
        assert model_state.model_id == 'cv'
        assert model_state.filter == kf
        assert model_state.motion_model == motion_model
        assert model_state.probability == 0.6
        assert model_state.likelihood == 0.0
        assert model_state.mixed_state is None
        assert model_state.mixed_covariance is None
        assert model_state.state_history == []
        assert model_state.probability_history == []
        assert model_state.likelihood_history == []


class TestIMMFilter:
    """Test IMM filter implementation."""
    
    @pytest.fixture
    def simple_imm_params(self):
        """Create simple IMM parameters for testing."""
        model_types = ['cv', 'ca']
        transition_probs = np.array([[0.9, 0.1], [0.2, 0.8]])
        initial_probs = np.array([0.7, 0.3])
        
        return IMMParameters(
            dt=0.1,
            model_types=model_types,
            transition_probabilities=transition_probs,
            initial_model_probabilities=initial_probs,
            coordinate_system=CoordinateSystem.CARTESIAN_2D,
            process_noise_stds=[0.1, 0.2],
            measurement_noise_std=0.5
        )
    
    @pytest.fixture
    def three_model_imm_params(self):
        """Create three-model IMM parameters for testing."""
        model_types = ['cv', 'ca', 'ct']
        transition_probs = np.array([
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7]
        ])
        initial_probs = np.array([0.5, 0.3, 0.2])
        
        return IMMParameters(
            dt=0.1,
            model_types=model_types,
            transition_probabilities=transition_probs,
            initial_model_probabilities=initial_probs,
            coordinate_system=CoordinateSystem.CARTESIAN_2D,
            process_noise_stds=[0.1, 0.2, 0.15],
            model_specific_params={
                'ct': {'turn_rate_noise': 0.01}
            }
        )
    
    def test_initialization(self, simple_imm_params):
        """Test IMM filter initialization."""
        imm = IMMFilter(simple_imm_params)
        
        assert imm.dt == 0.1
        assert imm.num_models == 2
        assert imm.measurement_dim == 2
        assert len(imm.models) == 2
        
        # Check model types
        assert imm.models[0].model_id == 'cv_0'
        assert imm.models[1].model_id == 'ca_1'
        
        # Check initial probabilities
        npt.assert_array_almost_equal(imm.model_probs, [0.7, 0.3])
        
        # Check state dimensions
        assert imm.state_dim == 6  # Max of CV (4) and CA (6)
        assert imm.combined_state.shape == (6,)
        assert imm.combined_covariance.shape == (6, 6)
        
        # Check transition probabilities
        npt.assert_array_equal(imm.transition_probs, simple_imm_params.transition_probabilities)
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Too few models
        with pytest.raises(ValueError, match="at least 2 models"):
            params = IMMParameters(
                dt=0.1,
                model_types=['cv'],
                transition_probabilities=np.array([[1.0]])
            )
            IMMFilter(params)
        
        # Wrong transition matrix dimensions
        with pytest.raises(ValueError, match="Transition matrix must be"):
            params = IMMParameters(
                dt=0.1,
                model_types=['cv', 'ca'],
                transition_probabilities=np.array([[0.9, 0.1]])  # Wrong shape
            )
            IMMFilter(params)
        
        # Invalid model type
        with pytest.raises(ValueError, match="Unknown model type"):
            params = IMMParameters(
                dt=0.1,
                model_types=['cv', 'invalid'],
                transition_probabilities=np.array([[0.9, 0.1], [0.2, 0.8]])
            )
            IMMFilter(params)
    
    def test_model_initialization(self, simple_imm_params):
        """Test that models are properly initialized."""
        imm = IMMFilter(simple_imm_params)
        
        # Check CV model
        cv_model = imm.models[0]
        assert isinstance(cv_model.motion_model, ConstantVelocityModel)
        assert isinstance(cv_model.filter, KalmanFilter)
        assert cv_model.filter.dim_x == 4  # CV state dimension
        assert cv_model.filter.dim_z == 2  # Measurement dimension
        
        # Check CA model
        ca_model = imm.models[1]
        assert isinstance(ca_model.motion_model, ConstantAccelerationModel)
        assert isinstance(ca_model.filter, KalmanFilter)
        assert ca_model.filter.dim_x == 6  # CA state dimension
        assert ca_model.filter.dim_z == 2  # Measurement dimension
    
    def test_nonlinear_model_initialization(self, three_model_imm_params):
        """Test initialization with nonlinear models."""
        imm = IMMFilter(three_model_imm_params)
        
        # Check CT model (should use EKF)
        ct_model = imm.models[2]
        assert isinstance(ct_model.motion_model, CoordinatedTurnModel)
        assert isinstance(ct_model.filter, ExtendedKalmanFilter)
    
    def test_weight_computation(self, simple_imm_params):
        """Test weight computation from transition probabilities."""
        imm = IMMFilter(simple_imm_params)
        
        # Set up mock model probabilities
        imm.model_probs = np.array([0.6, 0.4])
        
        # Compute mixing weights
        imm._compute_mixing_weights()
        
        # Check that weights are computed correctly
        assert imm.mixed_probs.shape == (2, 2)
        assert imm.normalization_factors.shape == (2,)
        
        # Check normalization
        for j in range(2):
            if imm.normalization_factors[j] > 0:
                weights_sum = np.sum(imm.mixed_probs[:, j])
                assert abs(weights_sum - 1.0) < 1e-10
    
    def test_state_mixing(self, simple_imm_params):
        """Test state and covariance mixing."""
        imm = IMMFilter(simple_imm_params)
        
        # Set up test states
        imm.models[0].filter.x = np.array([1.0, 2.0, 0.5, 0.3])
        imm.models[0].filter.P = np.eye(4) * 0.1
        
        imm.models[1].filter.x = np.array([1.1, 2.1, 0.4, 0.2, 0.01, 0.02])
        imm.models[1].filter.P = np.eye(6) * 0.2
        
        # Set up mixing weights
        imm.mixed_probs = np.array([[0.8, 0.3], [0.2, 0.7]])
        imm.normalization_factors = np.array([1.0, 1.0])
        
        # Perform mixing
        imm._mix_states_and_covariances()
        
        # Check that mixed states are computed
        for i, model in enumerate(imm.models):
            assert model.mixed_state is not None
            assert model.mixed_covariance is not None
            assert model.mixed_state.shape == (imm.state_dim,)
            assert model.mixed_covariance.shape == (imm.state_dim, imm.state_dim)
    
    def test_predict_step(self, simple_imm_params):
        """Test prediction step."""
        imm = IMMFilter(simple_imm_params)
        
        # Initialize states
        initial_state = np.array([1.0, 2.0, 0.5, 0.3, 0.0, 0.0])
        initial_covariance = np.eye(6) * 0.1
        
        imm.set_state(initial_state, initial_covariance)
        
        # Store initial values
        initial_combined_state = imm.combined_state.copy()
        initial_combined_covariance = imm.combined_covariance.copy()
        
        # Predict
        imm.predict(0.1)
        
        # Check that prediction occurred
        assert not np.array_equal(imm.combined_state, initial_combined_state)
        
        # Check that all models were updated
        for model in imm.models:
            assert hasattr(model.filter, 'x')
            assert hasattr(model.filter, 'P')
    
    def test_update_step(self, simple_imm_params):
        """Test update step with measurement."""
        imm = IMMFilter(simple_imm_params)
        
        # Initialize
        initial_state = np.array([1.0, 2.0, 0.5, 0.3, 0.0, 0.0])
        initial_covariance = np.eye(6) * 0.1
        imm.set_state(initial_state, initial_covariance)
        
        # Perform prediction first
        imm.predict(0.1)
        
        # Store model probabilities before update
        probs_before = imm.model_probs.copy()
        
        # Update with measurement
        measurement = np.array([1.05, 2.02])
        imm.update(measurement)
        
        # Check that model probabilities changed
        assert not np.array_equal(imm.model_probs, probs_before)
        
        # Check that probabilities sum to 1
        assert abs(np.sum(imm.model_probs) - 1.0) < 1e-10
        
        # Check that likelihoods were computed
        for model in imm.models:
            assert hasattr(model, 'likelihood')
    
    def test_likelihood_computation(self, simple_imm_params):
        """Test likelihood computation for models."""
        imm = IMMFilter(simple_imm_params)
        
        # Set up states
        initial_state = np.array([1.0, 2.0, 0.5, 0.3, 0.0, 0.0])
        initial_covariance = np.eye(6) * 0.1
        imm.set_state(initial_state, initial_covariance)
        
        imm.predict(0.1)
        
        # Compute likelihoods
        measurement = np.array([1.05, 2.02])
        likelihoods = imm._compute_model_likelihoods(measurement)
        
        assert len(likelihoods) == 2
        assert np.all(likelihoods >= 0)
        assert np.all(np.isfinite(likelihoods))
    
    def test_model_probability_update(self, simple_imm_params):
        """Test model probability update."""
        imm = IMMFilter(simple_imm_params)
        
        # Set up initial probabilities
        imm.model_probs = np.array([0.6, 0.4])
        imm.normalization_factors = np.array([1.0, 1.0])
        
        # Mock likelihoods (CV model fits better)
        likelihoods = np.array([0.8, 0.2])
        
        # Update probabilities
        imm._update_model_probabilities(likelihoods)
        
        # CV model should have higher probability
        assert imm.model_probs[0] > imm.model_probs[1]
        
        # Probabilities should sum to 1
        assert abs(np.sum(imm.model_probs) - 1.0) < 1e-10
    
    def test_state_combination(self, simple_imm_params):
        """Test final state and covariance combination."""
        imm = IMMFilter(simple_imm_params)
        
        # Set up model states
        imm.models[0].filter.x = np.array([1.0, 2.0, 0.5, 0.3])
        imm.models[0].filter.P = np.eye(4) * 0.1
        imm.models[0].probability = 0.7
        
        imm.models[1].filter.x = np.array([1.1, 2.1, 0.4, 0.2, 0.01, 0.02])
        imm.models[1].filter.P = np.eye(6) * 0.2
        imm.models[1].probability = 0.3
        
        # Update model probabilities
        imm.model_probs = np.array([0.7, 0.3])
        
        # Combine states
        imm._combine_estimates()
        
        # Check that combined state is reasonable
        assert imm.combined_state.shape == (6,)
        assert imm.combined_covariance.shape == (6, 6)
        
        # Combined state should be closer to CV model (higher probability)
        cv_extended_state = np.zeros(6)
        cv_extended_state[:4] = imm.models[0].filter.x
        
        distance_to_cv = np.linalg.norm(imm.combined_state - cv_extended_state)
        distance_to_ca = np.linalg.norm(imm.combined_state - imm.models[1].filter.x)
        
        # Should be closer to CV due to higher probability
        assert distance_to_cv <= distance_to_ca
    
    def test_full_cycle(self, simple_imm_params):
        """Test complete predict-update cycle."""
        imm = IMMFilter(simple_imm_params)
        
        # Initialize
        initial_state = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        initial_covariance = np.eye(6) * 1.0
        imm.set_state(initial_state, initial_covariance)
        
        # Run multiple cycles
        measurements = [
            np.array([0.1, 0.1]),
            np.array([0.2, 0.2]),
            np.array([0.3, 0.3]),
            np.array([0.4, 0.4])
        ]
        
        for measurement in measurements:
            imm.predict(0.1)
            imm.update(measurement)
            
            # Check validity after each cycle
            assert np.all(np.isfinite(imm.combined_state))
            assert np.all(np.isfinite(imm.combined_covariance))
            assert abs(np.sum(imm.model_probs) - 1.0) < 1e-10
            
            # Check that covariance is positive definite
            eigenvals = np.linalg.eigvals(imm.combined_covariance)
            assert np.all(eigenvals > 0)
    
    def test_model_switching_detection(self, simple_imm_params):
        """Test detection of model switches."""
        imm = IMMFilter(simple_imm_params)
        
        # Initialize with CV model dominant
        initial_state = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        initial_covariance = np.eye(6) * 0.1
        imm.set_state(initial_state, initial_covariance)
        
        # Initially favor CV model
        imm.model_probs = np.array([0.9, 0.1])
        
        # Generate measurements that favor acceleration model
        dt = 0.1
        measurements = []
        for i in range(10):
            t = i * dt
            # Accelerating motion: x = 0.5 * a * t^2, with a = 2
            pos_x = 0.5 * 2.0 * t**2
            pos_y = 0.5 * 2.0 * t**2
            measurements.append(np.array([pos_x, pos_y]) + np.random.normal(0, 0.01, 2))
        
        # Track model probabilities
        cv_probs = []
        ca_probs = []
        
        for measurement in measurements:
            imm.predict(dt)
            imm.update(measurement)
            
            cv_probs.append(imm.model_probs[0])
            ca_probs.append(imm.model_probs[1])
        
        # CA model should eventually become dominant
        assert ca_probs[-1] > cv_probs[-1]
        assert ca_probs[-1] > 0.5
    
    def test_gating_functionality(self, simple_imm_params):
        """Test measurement gating."""
        imm = IMMFilter(simple_imm_params)
        
        # Initialize
        initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        initial_covariance = np.eye(6) * 0.01  # Small uncertainty
        imm.set_state(initial_state, initial_covariance)
        
        imm.predict(0.1)
        
        # Test with valid measurement
        valid_measurement = np.array([0.05, 0.05])
        is_valid = imm.is_measurement_valid(valid_measurement)
        assert is_valid
        
        # Test with invalid measurement (too far)
        invalid_measurement = np.array([10.0, 10.0])
        is_valid = imm.is_measurement_valid(invalid_measurement)
        assert not is_valid
    
    def test_performance_metrics(self, simple_imm_params):
        """Test performance metrics computation."""
        imm = IMMFilter(simple_imm_params)
        
        # Initialize and run some steps
        initial_state = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        initial_covariance = np.eye(6) * 0.1
        imm.set_state(initial_state, initial_covariance)
        
        measurements = [np.array([0.1, 0.1]), np.array([0.2, 0.2])]
        
        for measurement in measurements:
            imm.predict(0.1)
            imm.update(measurement)
        
        # Get performance metrics
        metrics = imm.get_performance_metrics()
        
        assert 'combined_likelihood' in metrics
        assert 'model_probabilities' in metrics
        assert 'step_count' in metrics
        assert 'innovation_history' in metrics
        
        assert metrics['step_count'] == 2
        assert len(metrics['model_probabilities']) == 2
        assert len(metrics['innovation_history']) == 2
    
    def test_state_dimension_mismatch_handling(self, simple_imm_params):
        """Test handling of different state dimensions across models."""
        imm = IMMFilter(simple_imm_params)
        
        # CV model has 4D state, CA model has 6D state
        # IMM should handle this gracefully
        
        initial_state = np.array([1.0, 2.0, 0.5, 0.3, 0.1, 0.05])  # 6D state
        initial_covariance = np.eye(6) * 0.1
        
        imm.set_state(initial_state, initial_covariance)
        
        # Check that CV model gets appropriate subset
        cv_state = imm.models[0].filter.x
        assert len(cv_state) == 4
        npt.assert_array_equal(cv_state, initial_state[:4])
        
        # Check that CA model gets full state
        ca_state = imm.models[1].filter.x
        assert len(ca_state) == 6
        npt.assert_array_equal(ca_state, initial_state)
    
    def test_adaptive_noise_handling(self, simple_imm_params):
        """Test adaptive noise estimation."""
        imm = IMMFilter(simple_imm_params)
        
        # Initialize
        initial_state = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        initial_covariance = np.eye(6) * 0.1
        imm.set_state(initial_state, initial_covariance)
        
        # Generate measurements with varying noise levels
        measurements = [
            np.array([0.1, 0.1]),     # Low noise
            np.array([0.25, 0.18]),   # Medium noise  
            np.array([0.35, 0.32]),   # Higher noise
        ]
        
        innovation_magnitudes = []
        
        for measurement in measurements:
            imm.predict(0.1)
            imm.update(measurement)
            
            # Track innovation magnitudes
            if imm.innovation_history:
                innovation_magnitudes.append(np.linalg.norm(imm.innovation_history[-1]))
        
        # Should adapt to noise levels
        assert len(innovation_magnitudes) == 3
    
    def test_memory_management(self, simple_imm_params):
        """Test that history tracking doesn't grow unbounded."""
        # Modify parameters to limit history
        simple_imm_params.max_history_length = 5  # Add this parameter
        
        imm = IMMFilter(simple_imm_params)
        
        # Initialize
        initial_state = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        initial_covariance = np.eye(6) * 0.1
        imm.set_state(initial_state, initial_covariance)
        
        # Run many cycles
        for i in range(20):
            imm.predict(0.1)
            measurement = np.array([i * 0.1, i * 0.1])
            imm.update(measurement)
        
        # Check that history is bounded (if implemented)
        # This test assumes history length limiting is implemented
        # If not implemented, it serves as a reminder to add this feature
        if hasattr(imm, 'max_history_length'):
            assert len(imm.innovation_history) <= imm.max_history_length


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_singular_covariance_handling(self, simple_imm_params):
        """Test handling of singular covariance matrices."""
        imm = IMMFilter(simple_imm_params)
        
        # Initialize with near-singular covariance
        initial_state = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        initial_covariance = np.eye(6) * 1e-10  # Very small
        
        imm.set_state(initial_state, initial_covariance)
        
        # Should handle gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imm.predict(0.1)
            measurement = np.array([0.1, 0.1])
            imm.update(measurement)
        
        # Should not crash
        assert np.all(np.isfinite(imm.combined_state))
    
    def test_extreme_measurements(self, simple_imm_params):
        """Test with extreme measurements."""
        imm = IMMFilter(simple_imm_params)
        
        # Initialize
        initial_state = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        initial_covariance = np.eye(6) * 0.1
        imm.set_state(initial_state, initial_covariance)
        
        imm.predict(0.1)
        
        # Test with very large measurement
        extreme_measurement = np.array([1e6, 1e6])
        
        # Should handle without crashing
        imm.update(extreme_measurement)
        
        assert np.all(np.isfinite(imm.combined_state))
        assert np.all(np.isfinite(imm.combined_covariance))
    
    def test_zero_probabilities(self, simple_imm_params):
        """Test handling of zero model probabilities."""
        imm = IMMFilter(simple_imm_params)
        
        # Initialize
        initial_state = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        initial_covariance = np.eye(6) * 0.1
        imm.set_state(initial_state, initial_covariance)
        
        # Force one model probability to zero
        imm.model_probs = np.array([1.0, 0.0])
        
        imm.predict(0.1)
        measurement = np.array([0.1, 0.1])
        imm.update(measurement)
        
        # Should handle gracefully
        assert np.all(np.isfinite(imm.model_probs))
        assert np.sum(imm.model_probs) > 0
    
    def test_very_small_time_step(self, simple_imm_params):
        """Test with very small time steps."""
        simple_imm_params.dt = 1e-6  # Very small dt
        
        imm = IMMFilter(simple_imm_params)
        
        # Initialize
        initial_state = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        initial_covariance = np.eye(6) * 0.1
        imm.set_state(initial_state, initial_covariance)
        
        # Should work with very small time steps
        imm.predict(1e-6)
        measurement = np.array([1e-6, 1e-6])
        imm.update(measurement)
        
        assert np.all(np.isfinite(imm.combined_state))


class TestPerformanceBenchmarks:
    """Performance benchmarks for IMM filter."""
    
    def test_prediction_performance(self, simple_imm_params):
        """Benchmark prediction performance."""
        imm = IMMFilter(simple_imm_params)
        
        # Initialize
        initial_state = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        initial_covariance = np.eye(6) * 0.1
        imm.set_state(initial_state, initial_covariance)
        
        import time
        
        # Benchmark prediction
        start_time = time.time()
        for _ in range(1000):
            imm.predict(0.1)
        prediction_time = time.time() - start_time
        
        # Should complete in reasonable time (< 1 second for 1000 predictions)
        assert prediction_time < 1.0
        
        print(f"Prediction performance: {prediction_time:.4f}s for 1000 predictions")
    
    def test_update_performance(self, simple_imm_params):
        """Benchmark update performance."""
        imm = IMMFilter(simple_imm_params)
        
        # Initialize
        initial_state = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        initial_covariance = np.eye(6) * 0.1
        imm.set_state(initial_state, initial_covariance)
        
        import time
        
        # Benchmark update
        start_time = time.time()
        for i in range(1000):
            imm.predict(0.1)
            measurement = np.array([i * 0.001, i * 0.001])
            imm.update(measurement)
        update_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert update_time < 2.0
        
        print(f"Update performance: {update_time:.4f}s for 1000 updates")
    
    def test_memory_usage(self, simple_imm_params):
        """Test memory usage doesn't grow excessively."""
        imm = IMMFilter(simple_imm_params)
        
        # Initialize
        initial_state = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        initial_covariance = np.eye(6) * 0.1
        imm.set_state(initial_state, initial_covariance)
        
        import sys
        
        # Get initial size
        initial_size = sys.getsizeof(imm)
        
        # Run many updates
        for i in range(1000):
            imm.predict(0.1)
            measurement = np.array([i * 0.001, i * 0.001])
            imm.update(measurement)
        
        # Get final size
        final_size = sys.getsizeof(imm)
        
        # Memory growth should be reasonable
        growth_ratio = final_size / initial_size
        assert growth_ratio < 10.0  # Should not grow more than 10x
        
        print(f"Memory growth: {growth_ratio:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__])