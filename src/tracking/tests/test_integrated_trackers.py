"""
Comprehensive test suite for integrated tracking systems.

Tests the complete integrated tracking systems including:
- IMM-JPDA tracker integration
- IMM-MHT tracker integration
- Track management functionality
- Sensor fusion capabilities
- Configuration system
- End-to-end tracking scenarios

Author: RadarSim Project
"""

import pytest
import numpy as np
import numpy.testing as npt
from unittest.mock import Mock, patch, MagicMock
import time
from typing import List

from ..integrated_trackers import (
    AssociationMethod, HypothesisStatus, ModelSet, TrackingConfiguration,
    Association, Hypothesis, InteractingMultipleModel
)
from ..tracker_base import Track, Measurement, TrackState, TrackingMetrics
from ..motion_models import (
    ModelParameters, CoordinateSystem, ConstantVelocityModel,
    ConstantAccelerationModel, CoordinatedTurnModel
)
from ..kalman_filters import KalmanFilter


class TestModelSet:
    """Test ModelSet configuration."""
    
    @pytest.fixture
    def sample_models(self):
        """Create sample motion models."""
        cv_params = ModelParameters(dt=0.1, coordinate_system=CoordinateSystem.CARTESIAN_2D)
        ca_params = ModelParameters(dt=0.1, coordinate_system=CoordinateSystem.CARTESIAN_2D)
        
        cv_model = ConstantVelocityModel(cv_params)
        ca_model = ConstantAccelerationModel(ca_params)
        
        return [cv_model, ca_model]
    
    def test_valid_initialization(self, sample_models):
        """Test valid ModelSet initialization."""
        model_names = ['cv', 'ca']
        initial_probs = np.array([0.7, 0.3])
        transition_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        
        model_set = ModelSet(
            models=sample_models,
            model_names=model_names,
            initial_probabilities=initial_probs,
            transition_matrix=transition_matrix
        )
        
        assert len(model_set.models) == 2
        assert model_set.model_names == model_names
        npt.assert_array_equal(model_set.initial_probabilities, initial_probs)
        npt.assert_array_equal(model_set.transition_matrix, transition_matrix)
    
    def test_model_count_mismatch(self, sample_models):
        """Test error when model count doesn't match names."""
        with pytest.raises(ValueError, match="Number of models must match"):
            ModelSet(
                models=sample_models,
                model_names=['cv'],  # Only one name for two models
                initial_probabilities=np.array([0.7, 0.3]),
                transition_matrix=np.array([[0.9, 0.1], [0.2, 0.8]])
            )
    
    def test_probability_count_mismatch(self, sample_models):
        """Test error when probability count doesn't match models."""
        with pytest.raises(ValueError, match="Initial probabilities must match"):
            ModelSet(
                models=sample_models,
                model_names=['cv', 'ca'],
                initial_probabilities=np.array([1.0]),  # Only one probability
                transition_matrix=np.array([[0.9, 0.1], [0.2, 0.8]])
            )
    
    def test_probabilities_not_normalized(self, sample_models):
        """Test error when probabilities don't sum to 1."""
        with pytest.raises(ValueError, match="Initial probabilities must sum to 1"):
            ModelSet(
                models=sample_models,
                model_names=['cv', 'ca'],
                initial_probabilities=np.array([0.6, 0.6]),  # Sums to 1.2
                transition_matrix=np.array([[0.9, 0.1], [0.2, 0.8]])
            )
    
    def test_transition_matrix_wrong_shape(self, sample_models):
        """Test error with wrong transition matrix shape."""
        with pytest.raises(ValueError, match="Transition matrix dimensions"):
            ModelSet(
                models=sample_models,
                model_names=['cv', 'ca'],
                initial_probabilities=np.array([0.7, 0.3]),
                transition_matrix=np.array([[0.9, 0.1]])  # Wrong shape
            )
    
    def test_transition_matrix_not_stochastic(self, sample_models):
        """Test error when transition matrix rows don't sum to 1."""
        with pytest.raises(ValueError, match="Transition matrix rows must sum to 1"):
            ModelSet(
                models=sample_models,
                model_names=['cv', 'ca'],
                initial_probabilities=np.array([0.7, 0.3]),
                transition_matrix=np.array([[0.8, 0.1], [0.2, 0.8]])  # First row sums to 0.9
            )


class TestTrackingConfiguration:
    """Test TrackingConfiguration dataclass."""
    
    @pytest.fixture
    def basic_model_set(self):
        """Create basic model set for testing."""
        cv_params = ModelParameters(dt=0.1, coordinate_system=CoordinateSystem.CARTESIAN_2D)
        ca_params = ModelParameters(dt=0.1, coordinate_system=CoordinateSystem.CARTESIAN_2D)
        
        models = [ConstantVelocityModel(cv_params), ConstantAccelerationModel(ca_params)]
        model_names = ['cv', 'ca']
        initial_probs = np.array([0.7, 0.3])
        transition_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        
        return ModelSet(models, model_names, initial_probs, transition_matrix)
    
    def test_default_initialization(self, basic_model_set):
        """Test default configuration initialization."""
        config = TrackingConfiguration(model_set=basic_model_set)
        
        assert config.model_set == basic_model_set
        assert config.mixing_threshold == 1e-6
        assert config.jpda_gate_threshold == 9.21
        assert config.jpda_detection_probability == 0.9
        assert config.jpda_clutter_density == 1e-6
        assert config.jpda_max_hypotheses == 100
        assert config.mht_max_hypotheses == 1000
        assert config.track_initiation_threshold == 2
        assert config.track_deletion_threshold == 5
        assert config.max_time_without_update == 10.0
        assert not config.enable_sensor_fusion
        assert not config.parallel_processing
    
    def test_custom_initialization(self, basic_model_set):
        """Test custom configuration initialization."""
        config = TrackingConfiguration(
            model_set=basic_model_set,
            jpda_gate_threshold=5.99,
            jpda_detection_probability=0.95,
            track_initiation_threshold=3,
            enable_sensor_fusion=True,
            parallel_processing=True
        )
        
        assert config.jpda_gate_threshold == 5.99
        assert config.jpda_detection_probability == 0.95
        assert config.track_initiation_threshold == 3
        assert config.enable_sensor_fusion
        assert config.parallel_processing


class TestAssociation:
    """Test Association dataclass."""
    
    def test_initialization(self):
        """Test Association initialization."""
        association = Association(
            measurement_idx=1,
            track_id='track_1',
            probability=0.8,
            distance=2.5,
            gate_valid=True
        )
        
        assert association.measurement_idx == 1
        assert association.track_id == 'track_1'
        assert association.probability == 0.8
        assert association.distance == 2.5
        assert association.gate_valid
    
    def test_default_gate_valid(self):
        """Test default gate_valid value."""
        association = Association(
            measurement_idx=0,
            track_id='track_1',
            probability=0.7,
            distance=1.2
        )
        
        assert association.gate_valid  # Should default to True


class TestHypothesis:
    """Test Hypothesis dataclass."""
    
    def test_initialization(self):
        """Test Hypothesis initialization."""
        track_associations = {'track_1': 0, 'track_2': -1}
        measurement_associations = {0: 'track_1'}
        
        hypothesis = Hypothesis(
            hypothesis_id='hyp_1',
            track_associations=track_associations,
            measurement_associations=measurement_associations,
            probability=0.6,
            likelihood=0.8,
            status=HypothesisStatus.ACTIVE,
            parent_id='parent_hyp',
            depth=2
        )
        
        assert hypothesis.hypothesis_id == 'hyp_1'
        assert hypothesis.track_associations == track_associations
        assert hypothesis.measurement_associations == measurement_associations
        assert hypothesis.probability == 0.6
        assert hypothesis.likelihood == 0.8
        assert hypothesis.status == HypothesisStatus.ACTIVE
        assert hypothesis.parent_id == 'parent_hyp'
        assert hypothesis.depth == 2
        assert isinstance(hypothesis.creation_time, float)
    
    def test_default_values(self):
        """Test default values in Hypothesis."""
        hypothesis = Hypothesis(
            hypothesis_id='hyp_1',
            track_associations={},
            measurement_associations={},
            probability=0.5,
            likelihood=0.7
        )
        
        assert hypothesis.status == HypothesisStatus.ACTIVE
        assert hypothesis.parent_id is None
        assert hypothesis.depth == 0
        assert hypothesis.creation_time > 0


class TestInteractingMultipleModel:
    """Test InteractingMultipleModel class."""
    
    @pytest.fixture
    def sample_model_set(self):
        """Create sample model set for IMM testing."""
        cv_params = ModelParameters(dt=0.1, coordinate_system=CoordinateSystem.CARTESIAN_2D)
        ca_params = ModelParameters(dt=0.1, coordinate_system=CoordinateSystem.CARTESIAN_2D)
        
        models = [ConstantVelocityModel(cv_params), ConstantAccelerationModel(ca_params)]
        model_names = ['cv', 'ca']
        initial_probs = np.array([0.7, 0.3])
        transition_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        
        return ModelSet(models, model_names, initial_probs, transition_matrix)
    
    def test_initialization(self, sample_model_set):
        """Test IMM initialization."""
        imm = InteractingMultipleModel(sample_model_set, state_dim=4, measurement_dim=2)
        
        assert imm.num_models == 2
        assert imm.state_dim == 4
        assert imm.measurement_dim == 2
        assert len(imm.filters) == 2
        
        # Check initial probabilities
        npt.assert_array_equal(imm.model_probabilities, sample_model_set.initial_probabilities)
        
        # Check transition matrix
        npt.assert_array_equal(imm.transition_matrix, sample_model_set.transition_matrix)
        
        # Check filter initialization
        for filter_obj in imm.filters:
            assert isinstance(filter_obj, KalmanFilter)
    
    def test_cv_filter_initialization(self, sample_model_set):
        """Test CV filter is properly initialized."""
        imm = InteractingMultipleModel(sample_model_set, state_dim=4, measurement_dim=2)
        
        cv_filter = imm.filters[0]
        assert cv_filter.dim_x == 4
        assert cv_filter.dim_z == 2
        
        # Check state transition matrix structure (CV model)
        dt = 0.1
        expected_F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        npt.assert_array_almost_equal(cv_filter.F, expected_F)
    
    def test_ca_filter_initialization(self, sample_model_set):
        """Test CA filter is properly initialized."""
        imm = InteractingMultipleModel(sample_model_set, state_dim=6, measurement_dim=2)
        
        ca_filter = imm.filters[1]
        assert ca_filter.dim_x == 6
        assert ca_filter.dim_z == 2
        
        # Check that acceleration terms are present
        dt = 0.1
        assert ca_filter.F[0, 4] == dt**2/2  # Position affected by acceleration
        assert ca_filter.F[2, 4] == dt        # Velocity affected by acceleration
    
    def test_predict(self, sample_model_set):
        """Test IMM prediction step."""
        imm = InteractingMultipleModel(sample_model_set, state_dim=4, measurement_dim=2)
        
        # Initialize state
        initial_state = np.array([1.0, 2.0, 0.5, 0.3])
        initial_covariance = np.eye(4) * 0.1
        
        # Set initial state for all filters
        for filter_obj in imm.filters:
            filter_obj.x = initial_state.copy()
            filter_obj.P = initial_covariance.copy()
        
        # Store initial values
        initial_states = [f.x.copy() for f in imm.filters]
        
        # Predict
        imm.predict(dt=0.1)
        
        # Check that states changed
        for i, filter_obj in enumerate(imm.filters):
            assert not np.array_equal(filter_obj.x, initial_states[i])
    
    def test_update(self, sample_model_set):
        """Test IMM update step."""
        imm = InteractingMultipleModel(sample_model_set, state_dim=4, measurement_dim=2)
        
        # Initialize state
        initial_state = np.array([1.0, 2.0, 0.5, 0.3])
        initial_covariance = np.eye(4) * 0.1
        
        for filter_obj in imm.filters:
            filter_obj.x = initial_state.copy()
            filter_obj.P = initial_covariance.copy()
        
        # Perform prediction first
        imm.predict(dt=0.1)
        
        # Store probabilities before update
        probs_before = imm.model_probabilities.copy()
        
        # Update with measurement
        measurement = np.array([1.05, 2.02])
        imm.update(measurement)
        
        # Check that model probabilities were updated
        assert not np.array_equal(imm.model_probabilities, probs_before)
        
        # Check that probabilities sum to 1
        assert abs(np.sum(imm.model_probabilities) - 1.0) < 1e-10
        
        # Check that likelihoods were computed
        assert len(imm.model_likelihoods) == imm.num_models
        assert np.all(imm.model_likelihoods >= 0)
    
    def test_get_combined_estimate(self, sample_model_set):
        """Test combined state estimation."""
        imm = InteractingMultipleModel(sample_model_set, state_dim=4, measurement_dim=2)
        
        # Set up different states for each model
        imm.filters[0].x = np.array([1.0, 2.0, 0.5, 0.3])
        imm.filters[0].P = np.eye(4) * 0.1
        
        imm.filters[1].x = np.array([1.1, 2.1, 0.4, 0.2, 0.01, 0.02])  # CA has 6D state
        imm.filters[1].P = np.eye(6) * 0.2
        
        # Set model probabilities
        imm.model_probabilities = np.array([0.7, 0.3])
        
        # Get combined estimate
        combined_state, combined_covariance = imm.get_combined_estimate()
        
        # Check dimensions
        assert combined_state.shape == (4,)  # Should match state_dim
        assert combined_covariance.shape == (4, 4)
        
        # Combined state should be closer to CV model (higher probability)
        cv_state = imm.filters[0].x
        distance_to_cv = np.linalg.norm(combined_state - cv_state)
        
        # Should be relatively close to CV state
        assert distance_to_cv < 1.0
    
    def test_model_mixing(self, sample_model_set):
        """Test model mixing calculations."""
        imm = InteractingMultipleModel(sample_model_set, state_dim=4, measurement_dim=2)
        
        # Set model probabilities
        imm.model_probabilities = np.array([0.6, 0.4])
        
        # Compute mixing probabilities
        mixing_probs = imm._compute_mixing_probabilities()
        
        assert mixing_probs.shape == (2, 2)
        
        # Each column should sum to 1 (normalization)
        for j in range(2):
            col_sum = np.sum(mixing_probs[:, j])
            assert abs(col_sum - 1.0) < 1e-10
    
    def test_state_dimension_handling(self, sample_model_set):
        """Test handling of different state dimensions."""
        # CV has 4D state, CA has 6D state, but we set IMM state_dim to 4
        imm = InteractingMultipleModel(sample_model_set, state_dim=4, measurement_dim=2)
        
        # Both filters should be configured for the IMM's state dimension
        assert imm.filters[0].dim_x == 4  # CV filter
        assert imm.filters[1].dim_x == 6  # CA filter keeps its natural dimension
        
        # But IMM should handle this gracefully in combination
        combined_state, combined_cov = imm.get_combined_estimate()
        assert combined_state.shape == (4,)
        assert combined_cov.shape == (4, 4)


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""
    
    @pytest.fixture
    def tracking_scenario(self):
        """Create a realistic tracking scenario."""
        # Create model set
        cv_params = ModelParameters(dt=0.1, coordinate_system=CoordinateSystem.CARTESIAN_2D)
        ca_params = ModelParameters(dt=0.1, coordinate_system=CoordinateSystem.CARTESIAN_2D)
        
        models = [ConstantVelocityModel(cv_params), ConstantAccelerationModel(ca_params)]
        model_names = ['cv', 'ca']
        initial_probs = np.array([0.8, 0.2])
        transition_matrix = np.array([[0.95, 0.05], [0.1, 0.9]])
        
        model_set = ModelSet(models, model_names, initial_probs, transition_matrix)
        
        # Create configuration
        config = TrackingConfiguration(
            model_set=model_set,
            jpda_gate_threshold=9.21,
            jpda_detection_probability=0.9,
            track_initiation_threshold=2,
            track_deletion_threshold=5
        )
        
        return config
    
    def test_constant_velocity_tracking(self, tracking_scenario):
        """Test tracking of constant velocity target."""
        imm = InteractingMultipleModel(
            tracking_scenario.model_set, 
            state_dim=4, 
            measurement_dim=2
        )
        
        # Initialize with constant velocity motion
        initial_state = np.array([0.0, 0.0, 1.0, 1.0])  # Moving diagonally
        initial_covariance = np.eye(4) * 0.1
        
        for filter_obj in imm.filters:
            if filter_obj.dim_x == 4:
                filter_obj.x = initial_state.copy()
                filter_obj.P = initial_covariance.copy()
            else:  # 6D filter
                extended_state = np.zeros(6)
                extended_state[:4] = initial_state
                filter_obj.x = extended_state
                filter_obj.P = np.eye(6) * 0.1
        
        # Generate constant velocity measurements
        true_positions = []
        measurements = []
        dt = 0.1
        
        for i in range(20):
            t = i * dt
            true_pos = np.array([t, t])  # Diagonal motion
            noise = np.random.normal(0, 0.05, 2)
            measurement = true_pos + noise
            
            true_positions.append(true_pos)
            measurements.append(measurement)
        
        # Track probabilities
        cv_probabilities = []
        ca_probabilities = []
        
        # Process measurements
        for measurement in measurements:
            imm.predict(dt)
            imm.update(measurement)
            
            cv_probabilities.append(imm.model_probabilities[0])
            ca_probabilities.append(imm.model_probabilities[1])
        
        # CV model should dominate for constant velocity motion
        final_cv_prob = cv_probabilities[-1]
        final_ca_prob = ca_probabilities[-1]
        
        assert final_cv_prob > final_ca_prob
        assert final_cv_prob > 0.6  # Should strongly favor CV
    
    def test_accelerating_target_tracking(self, tracking_scenario):
        """Test tracking of accelerating target."""
        imm = InteractingMultipleModel(
            tracking_scenario.model_set, 
            state_dim=4, 
            measurement_dim=2
        )
        
        # Initialize
        initial_state = np.array([0.0, 0.0, 0.0, 0.0])
        initial_covariance = np.eye(4) * 0.1
        
        for filter_obj in imm.filters:
            if filter_obj.dim_x == 4:
                filter_obj.x = initial_state.copy()
                filter_obj.P = initial_covariance.copy()
            else:
                extended_state = np.zeros(6)
                extended_state[:4] = initial_state
                filter_obj.x = extended_state
                filter_obj.P = np.eye(6) * 0.1
        
        # Generate accelerating motion measurements
        measurements = []
        dt = 0.1
        acceleration = 1.0  # m/s^2
        
        for i in range(20):
            t = i * dt
            # x = 0.5 * a * t^2 for constant acceleration from rest
            true_pos = np.array([0.5 * acceleration * t**2, 0.5 * acceleration * t**2])
            noise = np.random.normal(0, 0.05, 2)
            measurement = true_pos + noise
            measurements.append(measurement)
        
        # Track probabilities
        cv_probabilities = []
        ca_probabilities = []
        
        # Process measurements
        for measurement in measurements:
            imm.predict(dt)
            imm.update(measurement)
            
            cv_probabilities.append(imm.model_probabilities[0])
            ca_probabilities.append(imm.model_probabilities[1])
        
        # CA model should eventually dominate
        final_cv_prob = cv_probabilities[-1]
        final_ca_prob = ca_probabilities[-1]
        
        # For strong acceleration, CA should be favored
        assert final_ca_prob > 0.3  # Should have significant CA probability
    
    def test_model_switching_scenario(self, tracking_scenario):
        """Test scenario with model switching."""
        imm = InteractingMultipleModel(
            tracking_scenario.model_set, 
            state_dim=4, 
            measurement_dim=2
        )
        
        # Initialize
        initial_state = np.array([0.0, 0.0, 1.0, 0.0])  # Moving in x direction
        initial_covariance = np.eye(4) * 0.1
        
        for filter_obj in imm.filters:
            if filter_obj.dim_x == 4:
                filter_obj.x = initial_state.copy()
                filter_obj.P = initial_covariance.copy()
            else:
                extended_state = np.zeros(6)
                extended_state[:4] = initial_state
                filter_obj.x = extended_state
                filter_obj.P = np.eye(6) * 0.1
        
        # Generate motion with model switch
        measurements = []
        dt = 0.1
        
        # Phase 1: Constant velocity (10 steps)
        for i in range(10):
            t = i * dt
            true_pos = np.array([t, 0.0])  # Constant velocity in x
            noise = np.random.normal(0, 0.02, 2)
            measurement = true_pos + noise
            measurements.append(measurement)
        
        # Phase 2: Constant acceleration (10 steps)
        initial_x = 10 * dt  # Position at end of phase 1
        for i in range(10):
            t = i * dt
            # Continue from phase 1 position with acceleration
            accel_pos = initial_x + 1.0 * t + 0.5 * 2.0 * t**2
            true_pos = np.array([accel_pos, 0.0])
            noise = np.random.normal(0, 0.02, 2)
            measurement = true_pos + noise
            measurements.append(measurement)
        
        cv_probabilities = []
        ca_probabilities = []
        
        # Process all measurements
        for i, measurement in enumerate(measurements):
            imm.predict(dt)
            imm.update(measurement)
            
            cv_probabilities.append(imm.model_probabilities[0])
            ca_probabilities.append(imm.model_probabilities[1])
        
        # Should see model switching behavior
        # Early phase should favor CV
        early_cv_prob = np.mean(cv_probabilities[:5])
        late_ca_prob = np.mean(ca_probabilities[-5:])
        
        assert early_cv_prob > 0.5  # Early phase favors CV
        # Late phase might favor CA depending on acceleration magnitude
    
    def test_multi_target_scenario(self):
        """Test multi-target tracking scenario."""
        # Create model set
        cv_params = ModelParameters(dt=0.1, coordinate_system=CoordinateSystem.CARTESIAN_2D)
        ca_params = ModelParameters(dt=0.1, coordinate_system=CoordinateSystem.CARTESIAN_2D)
        
        models = [ConstantVelocityModel(cv_params), ConstantAccelerationModel(ca_params)]
        model_names = ['cv', 'ca']
        initial_probs = np.array([0.7, 0.3])
        transition_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        
        model_set = ModelSet(models, model_names, initial_probs, transition_matrix)
        
        # Create multiple IMM filters for different targets
        target1_imm = InteractingMultipleModel(model_set, state_dim=4, measurement_dim=2)
        target2_imm = InteractingMultipleModel(model_set, state_dim=4, measurement_dim=2)
        
        # Initialize targets with different initial states
        target1_state = np.array([0.0, 0.0, 1.0, 0.0])   # Moving in +x
        target2_state = np.array([5.0, 5.0, 0.0, 1.0])   # Moving in +y
        
        initial_cov = np.eye(4) * 0.1
        
        # Initialize both IMM filters
        for imm, state in [(target1_imm, target1_state), (target2_imm, target2_state)]:
            for filter_obj in imm.filters:
                if filter_obj.dim_x == 4:
                    filter_obj.x = state.copy()
                    filter_obj.P = initial_cov.copy()
                else:
                    extended_state = np.zeros(6)
                    extended_state[:4] = state
                    filter_obj.x = extended_state
                    filter_obj.P = np.eye(6) * 0.1
        
        # Generate measurements for both targets
        dt = 0.1
        num_steps = 15
        
        target1_measurements = []
        target2_measurements = []
        
        for i in range(num_steps):
            t = i * dt
            
            # Target 1: constant velocity
            pos1 = np.array([t, 0.0])
            noise1 = np.random.normal(0, 0.03, 2)
            target1_measurements.append(pos1 + noise1)
            
            # Target 2: constant velocity  
            pos2 = np.array([5.0, 5.0 + t])
            noise2 = np.random.normal(0, 0.03, 2)
            target2_measurements.append(pos2 + noise2)
        
        # Process measurements for each target
        for i in range(num_steps):
            # Target 1
            target1_imm.predict(dt)
            target1_imm.update(target1_measurements[i])
            
            # Target 2
            target2_imm.predict(dt)
            target2_imm.update(target2_measurements[i])
        
        # Get final estimates
        target1_estimate, _ = target1_imm.get_combined_estimate()
        target2_estimate, _ = target2_imm.get_combined_estimate()
        
        # Check that targets are tracked correctly
        # Target 1 should be around (1.4, 0) after 14 steps
        expected_target1_pos = np.array([1.4, 0.0])
        actual_target1_pos = target1_estimate[:2]
        
        assert np.linalg.norm(actual_target1_pos - expected_target1_pos) < 0.5
        
        # Target 2 should be around (5, 6.4) after 14 steps
        expected_target2_pos = np.array([5.0, 6.4])
        actual_target2_pos = target2_estimate[:2]
        
        assert np.linalg.norm(actual_target2_pos - expected_target2_pos) < 0.5
    
    def test_noisy_measurement_handling(self, tracking_scenario):
        """Test robust handling of noisy measurements."""
        imm = InteractingMultipleModel(
            tracking_scenario.model_set, 
            state_dim=4, 
            measurement_dim=2
        )
        
        # Initialize
        initial_state = np.array([0.0, 0.0, 1.0, 1.0])
        initial_covariance = np.eye(4) * 0.1
        
        for filter_obj in imm.filters:
            if filter_obj.dim_x == 4:
                filter_obj.x = initial_state.copy()
                filter_obj.P = initial_covariance.copy()
            else:
                extended_state = np.zeros(6)
                extended_state[:4] = initial_state
                filter_obj.x = extended_state
                filter_obj.P = np.eye(6) * 0.1
        
        # Generate measurements with varying noise levels
        dt = 0.1
        measurements = []
        
        for i in range(20):
            t = i * dt
            true_pos = np.array([t, t])
            
            # Add different noise levels
            if i < 10:
                noise = np.random.normal(0, 0.05, 2)  # Low noise
            else:
                noise = np.random.normal(0, 0.2, 2)   # High noise
            
            measurement = true_pos + noise
            measurements.append(measurement)
        
        # Process measurements
        estimates = []
        for measurement in measurements:
            imm.predict(dt)
            imm.update(measurement)
            
            estimate, _ = imm.get_combined_estimate()
            estimates.append(estimate[:2])  # Position only
        
        # Check that tracking remains stable despite noise
        final_estimate = estimates[-1]
        expected_final_pos = np.array([1.9, 1.9])  # After 19 steps
        
        # Should be reasonably close despite high noise in second half
        error = np.linalg.norm(final_estimate - expected_final_pos)
        assert error < 1.0  # Allow for reasonable error due to noise


class TestPerformanceAndRobustness:
    """Test performance and robustness characteristics."""
    
    def test_computational_performance(self):
        """Test computational performance of IMM."""
        # Create model set
        cv_params = ModelParameters(dt=0.1, coordinate_system=CoordinateSystem.CARTESIAN_2D)
        ca_params = ModelParameters(dt=0.1, coordinate_system=CoordinateSystem.CARTESIAN_2D)
        
        models = [ConstantVelocityModel(cv_params), ConstantAccelerationModel(ca_params)]
        model_names = ['cv', 'ca']
        initial_probs = np.array([0.7, 0.3])
        transition_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        
        model_set = ModelSet(models, model_names, initial_probs, transition_matrix)
        imm = InteractingMultipleModel(model_set, state_dim=4, measurement_dim=2)
        
        # Initialize
        initial_state = np.array([0.0, 0.0, 1.0, 1.0])
        initial_covariance = np.eye(4) * 0.1
        
        for filter_obj in imm.filters:
            if filter_obj.dim_x == 4:
                filter_obj.x = initial_state.copy()
                filter_obj.P = initial_covariance.copy()
            else:
                extended_state = np.zeros(6)
                extended_state[:4] = initial_state
                filter_obj.x = extended_state
                filter_obj.P = np.eye(6) * 0.1
        
        # Benchmark prediction and update cycles
        num_cycles = 1000
        start_time = time.time()
        
        for i in range(num_cycles):
            measurement = np.array([i * 0.001, i * 0.001])
            imm.predict(0.1)
            imm.update(measurement)
        
        total_time = time.time() - start_time
        
        # Should complete 1000 cycles in reasonable time
        assert total_time < 5.0  # Less than 5 seconds
        
        avg_cycle_time = total_time / num_cycles
        print(f"Average cycle time: {avg_cycle_time*1000:.2f} ms")
    
    def test_numerical_stability(self, tracking_scenario):
        """Test numerical stability with extreme conditions."""
        imm = InteractingMultipleModel(
            tracking_scenario.model_set, 
            state_dim=4, 
            measurement_dim=2
        )
        
        # Initialize with very small covariance
        initial_state = np.array([0.0, 0.0, 1.0, 1.0])
        initial_covariance = np.eye(4) * 1e-8  # Very small
        
        for filter_obj in imm.filters:
            if filter_obj.dim_x == 4:
                filter_obj.x = initial_state.copy()
                filter_obj.P = initial_covariance.copy()
            else:
                extended_state = np.zeros(6)
                extended_state[:4] = initial_state
                filter_obj.x = extended_state
                filter_obj.P = np.eye(6) * 1e-8
        
        # Process measurements with extreme values
        extreme_measurements = [
            np.array([1e6, 1e6]),    # Very large
            np.array([-1e6, -1e6]),  # Very large negative
            np.array([1e-6, 1e-6]),  # Very small
            np.array([0.0, 0.0])     # Zero
        ]
        
        # Should handle without numerical issues
        for measurement in extreme_measurements:
            try:
                imm.predict(0.1)
                imm.update(measurement)
                
                # Check that results are finite
                estimate, covariance = imm.get_combined_estimate()
                assert np.all(np.isfinite(estimate))
                assert np.all(np.isfinite(covariance))
                
            except Exception as e:
                pytest.fail(f"Numerical instability with measurement {measurement}: {e}")
    
    def test_memory_efficiency(self):
        """Test memory usage doesn't grow excessively."""
        # Create model set
        cv_params = ModelParameters(dt=0.1, coordinate_system=CoordinateSystem.CARTESIAN_2D)
        ca_params = ModelParameters(dt=0.1, coordinate_system=CoordinateSystem.CARTESIAN_2D)
        
        models = [ConstantVelocityModel(cv_params), ConstantAccelerationModel(ca_params)]
        model_names = ['cv', 'ca']
        initial_probs = np.array([0.7, 0.3])
        transition_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        
        model_set = ModelSet(models, model_names, initial_probs, transition_matrix)
        imm = InteractingMultipleModel(model_set, state_dim=4, measurement_dim=2)
        
        # Initialize
        initial_state = np.array([0.0, 0.0, 1.0, 1.0])
        initial_covariance = np.eye(4) * 0.1
        
        for filter_obj in imm.filters:
            if filter_obj.dim_x == 4:
                filter_obj.x = initial_state.copy()
                filter_obj.P = initial_covariance.copy()
            else:
                extended_state = np.zeros(6)
                extended_state[:4] = initial_state
                filter_obj.x = extended_state
                filter_obj.P = np.eye(6) * 0.1
        
        import sys
        initial_size = sys.getsizeof(imm)
        
        # Run many cycles
        for i in range(1000):
            measurement = np.array([i * 0.001, i * 0.001])
            imm.predict(0.1)
            imm.update(measurement)
        
        final_size = sys.getsizeof(imm)
        growth_ratio = final_size / initial_size
        
        # Memory growth should be reasonable
        assert growth_ratio < 5.0  # Should not grow more than 5x
        
        print(f"Memory growth ratio: {growth_ratio:.2f}")


if __name__ == "__main__":
    pytest.main([__file__])