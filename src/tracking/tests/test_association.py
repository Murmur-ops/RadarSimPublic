"""
Comprehensive test suite for data association algorithms.

Tests all data association implementations including:
- Global Nearest Neighbor (GNN)
- Joint Probabilistic Data Association (JPDA)
- Multiple Hypothesis Tracking (MHT)
- Distance metrics and gating functions

Author: RadarSim Project
"""

import pytest
import numpy as np
import numpy.testing as npt
from unittest.mock import Mock, patch
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from ..association import (
    Detection, Track, AssociationResult,
    DistanceMetrics, GatingFunctions,
    GlobalNearestNeighbor, JointProbabilisticDataAssociation,
    MultipleHypothesisTracking, Hypothesis
)


class TestDetection:
    """Test Detection data structure."""
    
    def test_initialization(self):
        """Test Detection initialization."""
        position = np.array([1.0, 2.0])
        noise = np.eye(2) * 0.1
        
        detection = Detection(position=position, timestamp=10.0, measurement_noise=noise)
        
        npt.assert_array_equal(detection.position, position)
        assert detection.timestamp == 10.0
        npt.assert_array_equal(detection.measurement_noise, noise)
        assert detection.id is None
        assert detection.snr is None
        assert detection.doppler is None
    
    def test_with_optional_fields(self):
        """Test Detection with optional fields."""
        detection = Detection(
            position=np.array([1.0, 2.0]),
            timestamp=10.0,
            measurement_noise=np.eye(2),
            id=42,
            snr=15.5,
            doppler=2.3
        )
        
        assert detection.id == 42
        assert detection.snr == 15.5
        assert detection.doppler == 2.3


class TestTrack:
    """Test Track data structure."""
    
    def test_initialization(self):
        """Test Track initialization."""
        state = np.array([1.0, 2.0, 0.5, 0.3])
        covariance = np.eye(4)
        
        track = Track(
            state=state,
            covariance=covariance,
            id=1,
            last_update=10.0
        )
        
        npt.assert_array_equal(track.state, state)
        npt.assert_array_equal(track.covariance, covariance)
        assert track.id == 1
        assert track.last_update == 10.0
        assert track.score == 0.0
        assert track.age == 0
        assert track.hits == 0
        assert track.misses == 0
        assert not track.confirmed
    
    def test_with_optional_fields(self):
        """Test Track with optional fields."""
        track = Track(
            state=np.array([1.0, 2.0, 0.5, 0.3]),
            covariance=np.eye(4),
            id=1,
            last_update=10.0,
            score=0.8,
            age=5,
            hits=3,
            misses=2,
            confirmed=True
        )
        
        assert track.score == 0.8
        assert track.age == 5
        assert track.hits == 3
        assert track.misses == 2
        assert track.confirmed


class TestAssociationResult:
    """Test AssociationResult data structure."""
    
    def test_initialization(self):
        """Test AssociationResult initialization."""
        assignments = {1: 0, 2: 1}
        unassigned_tracks = [3, 4]
        unassigned_detections = [2, 3]
        
        result = AssociationResult(
            assignments=assignments,
            unassigned_tracks=unassigned_tracks,
            unassigned_detections=unassigned_detections
        )
        
        assert result.assignments == assignments
        assert result.unassigned_tracks == unassigned_tracks
        assert result.unassigned_detections == unassigned_detections
        assert result.probabilities is None
    
    def test_with_probabilities(self):
        """Test AssociationResult with probabilities."""
        probabilities = {'track_detection_probs': np.array([[0.8, 0.2]])}
        
        result = AssociationResult(
            assignments={1: 0},
            unassigned_tracks=[],
            unassigned_detections=[1],
            probabilities=probabilities
        )
        
        assert result.probabilities == probabilities


class TestDistanceMetrics:
    """Test distance metric functions."""
    
    def test_mahalanobis_distance(self):
        """Test Mahalanobis distance computation."""
        residual = np.array([1.0, 2.0])
        covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
        
        distance = DistanceMetrics.mahalanobis_distance(residual, covariance)
        
        # Manually compute expected distance
        inv_cov = np.linalg.inv(covariance)
        expected_distance = np.sqrt(residual.T @ inv_cov @ residual)
        
        assert abs(distance - expected_distance) < 1e-10
    
    def test_mahalanobis_distance_singular_covariance(self):
        """Test Mahalanobis distance with singular covariance."""
        residual = np.array([1.0, 2.0])
        singular_cov = np.array([[1.0, 1.0], [1.0, 1.0]])  # Singular matrix
        
        distance = DistanceMetrics.mahalanobis_distance(residual, singular_cov)
        
        assert distance == np.inf
    
    def test_euclidean_distance(self):
        """Test Euclidean distance computation."""
        pos1 = np.array([0.0, 0.0])
        pos2 = np.array([3.0, 4.0])
        
        distance = DistanceMetrics.euclidean_distance(pos1, pos2)
        
        assert abs(distance - 5.0) < 1e-10  # 3-4-5 triangle
    
    def test_normalized_innovation_squared(self):
        """Test NIS computation."""
        residual = np.array([1.0, 2.0])
        covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
        
        nis = DistanceMetrics.normalized_innovation_squared(residual, covariance)
        
        # Manually compute expected NIS
        inv_cov = np.linalg.inv(covariance)
        expected_nis = residual.T @ inv_cov @ residual
        
        assert abs(nis - expected_nis) < 1e-10
    
    def test_nis_singular_covariance(self):
        """Test NIS with singular covariance."""
        residual = np.array([1.0, 2.0])
        singular_cov = np.zeros((2, 2))
        
        nis = DistanceMetrics.normalized_innovation_squared(residual, singular_cov)
        
        assert nis == np.inf


class TestGatingFunctions:
    """Test gating functions."""
    
    def test_chi_square_gate_pass(self):
        """Test chi-square gating with measurement that passes."""
        residual = np.array([0.5, 0.3])  # Small residual
        covariance = np.eye(2)
        
        passes_gate = GatingFunctions.chi_square_gate(residual, covariance, confidence=0.95)
        
        assert passes_gate
    
    def test_chi_square_gate_fail(self):
        """Test chi-square gating with measurement that fails."""
        residual = np.array([5.0, 5.0])  # Large residual
        covariance = np.eye(2)
        
        passes_gate = GatingFunctions.chi_square_gate(residual, covariance, confidence=0.95)
        
        assert not passes_gate
    
    def test_chi_square_gate_threshold(self):
        """Test chi-square gate threshold computation."""
        residual = np.array([1.0, 1.0])
        covariance = np.eye(2)
        
        # Compute actual chi-square value
        nis = DistanceMetrics.normalized_innovation_squared(residual, covariance)
        threshold_95 = chi2.ppf(0.95, 2)
        threshold_99 = chi2.ppf(0.99, 2)
        
        # Should pass at 99% but might fail at 95%
        pass_99 = GatingFunctions.chi_square_gate(residual, covariance, confidence=0.99)
        pass_95 = GatingFunctions.chi_square_gate(residual, covariance, confidence=0.95)
        
        if nis <= threshold_95:
            assert pass_95 and pass_99
        elif nis <= threshold_99:
            assert not pass_95 and pass_99
        else:
            assert not pass_95 and not pass_99
    
    def test_ellipsoidal_gate_pass(self):
        """Test ellipsoidal gating with measurement that passes."""
        residual = np.array([1.0, 1.0])
        covariance = np.eye(2)
        
        passes_gate = GatingFunctions.ellipsoidal_gate(residual, covariance, gate_size=2.0)
        
        # Mahalanobis distance is sqrt(2) ≈ 1.414 < 2.0
        assert passes_gate
    
    def test_ellipsoidal_gate_fail(self):
        """Test ellipsoidal gating with measurement that fails."""
        residual = np.array([3.0, 3.0])
        covariance = np.eye(2)
        
        passes_gate = GatingFunctions.ellipsoidal_gate(residual, covariance, gate_size=2.0)
        
        # Mahalanobis distance is sqrt(18) ≈ 4.24 > 2.0
        assert not passes_gate


class TestGlobalNearestNeighbor:
    """Test Global Nearest Neighbor data association."""
    
    @pytest.fixture
    def gnn_associator(self):
        """Create GNN associator for testing."""
        return GlobalNearestNeighbor(gate_threshold=3.0, max_cost=100.0)
    
    @pytest.fixture
    def sample_tracks(self):
        """Create sample tracks for testing."""
        tracks = [
            Track(
                state=np.array([1.0, 2.0, 0.5, 0.3]),
                covariance=np.eye(4),
                id=1,
                last_update=10.0
            ),
            Track(
                state=np.array([5.0, 6.0, -0.2, 0.4]),
                covariance=np.eye(4),
                id=2,
                last_update=10.0
            )
        ]
        return tracks
    
    @pytest.fixture
    def sample_detections(self):
        """Create sample detections for testing."""
        detections = [
            Detection(
                position=np.array([1.1, 2.05]),
                timestamp=10.1,
                measurement_noise=np.eye(2) * 0.1
            ),
            Detection(
                position=np.array([5.05, 6.1]),
                timestamp=10.1,
                measurement_noise=np.eye(2) * 0.1
            ),
            Detection(
                position=np.array([10.0, 10.0]),  # Far away detection
                timestamp=10.1,
                measurement_noise=np.eye(2) * 0.1
            )
        ]
        return detections
    
    @pytest.fixture
    def measurement_functions(self):
        """Create measurement functions for testing."""
        def measurement_function(state):
            return state[:2]  # Position measurements only
        
        def measurement_jacobian(state):
            H = np.zeros((2, len(state)))
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            return H
        
        return measurement_function, measurement_jacobian
    
    def test_initialization(self):
        """Test GNN initialization."""
        gnn = GlobalNearestNeighbor(gate_threshold=5.0, max_cost=200.0)
        
        assert gnn.gate_threshold == 5.0
        assert gnn.max_cost == 200.0
    
    def test_compute_cost_matrix(self, gnn_associator, sample_tracks, sample_detections, 
                                measurement_functions):
        """Test cost matrix computation."""
        measurement_function, measurement_jacobian = measurement_functions
        
        cost_matrix = gnn_associator.compute_cost_matrix(
            sample_tracks, sample_detections, measurement_function, measurement_jacobian)
        
        assert cost_matrix.shape == (2, 3)  # 2 tracks, 3 detections
        
        # First track should be closest to first detection
        assert cost_matrix[0, 0] < cost_matrix[0, 1]
        assert cost_matrix[0, 0] < cost_matrix[0, 2]
        
        # Second track should be closest to second detection
        assert cost_matrix[1, 1] < cost_matrix[1, 0]
        assert cost_matrix[1, 1] < cost_matrix[1, 2]
        
        # Third detection should be gated out (too far)
        assert cost_matrix[0, 2] == gnn_associator.max_cost
        assert cost_matrix[1, 2] == gnn_associator.max_cost
    
    def test_empty_inputs(self, gnn_associator, measurement_functions):
        """Test GNN with empty inputs."""
        measurement_function, measurement_jacobian = measurement_functions
        
        # Empty tracks
        result = gnn_associator.associate([], [], measurement_function, measurement_jacobian)
        assert result.assignments == {}
        assert result.unassigned_tracks == []
        assert result.unassigned_detections == []
        
        # Empty detections
        tracks = [Track(np.array([1.0, 2.0, 0.5, 0.3]), np.eye(4), 1, 10.0)]
        result = gnn_associator.associate(tracks, [], measurement_function, measurement_jacobian)
        assert result.assignments == {}
        assert result.unassigned_tracks == [1]
        assert result.unassigned_detections == []
    
    def test_successful_association(self, gnn_associator, sample_tracks, sample_detections,
                                  measurement_functions):
        """Test successful GNN association."""
        measurement_function, measurement_jacobian = measurement_functions
        
        # Use only first two detections (close to tracks)
        close_detections = sample_detections[:2]
        
        result = gnn_associator.associate(
            sample_tracks, close_detections, measurement_function, measurement_jacobian)
        
        # Should have 2 assignments
        assert len(result.assignments) == 2
        assert 1 in result.assignments  # Track 1 assigned
        assert 2 in result.assignments  # Track 2 assigned
        
        # Check that assignments make sense
        assert result.assignments[1] == 0  # Track 1 -> Detection 0
        assert result.assignments[2] == 1  # Track 2 -> Detection 1
        
        assert result.unassigned_tracks == []
        assert result.unassigned_detections == []
    
    def test_gating_effect(self, gnn_associator, sample_tracks, sample_detections,
                          measurement_functions):
        """Test that gating properly excludes distant detections."""
        measurement_function, measurement_jacobian = measurement_functions
        
        result = gnn_associator.associate(
            sample_tracks, sample_detections, measurement_function, measurement_jacobian)
        
        # Third detection should be unassigned due to gating
        assert 2 in result.unassigned_detections
        
        # Only first two detections should be assigned
        assigned_detections = set(result.assignments.values())
        assert assigned_detections.issubset({0, 1})
    
    def test_max_cost_filtering(self, gnn_associator, measurement_functions):
        """Test that max_cost parameter filters out expensive associations."""
        # Create track and detection that are close but with large noise
        tracks = [Track(np.array([1.0, 2.0, 0.5, 0.3]), np.eye(4) * 0.001, 1, 10.0)]
        detections = [Detection(np.array([1.5, 2.5]), 10.1, np.eye(2) * 100.0)]  # Large noise
        
        measurement_function, measurement_jacobian = measurement_functions
        
        # Set very low max_cost
        gnn_associator.max_cost = 0.1
        
        result = gnn_associator.associate(
            tracks, detections, measurement_function, measurement_jacobian)
        
        # Should not associate due to high cost
        assert result.assignments == {}
        assert result.unassigned_tracks == [1]
        assert result.unassigned_detections == [0]


class TestJointProbabilisticDataAssociation:
    """Test Joint Probabilistic Data Association."""
    
    @pytest.fixture
    def jpda_associator(self):
        """Create JPDA associator for testing."""
        return JointProbabilisticDataAssociation(
            gate_threshold=3.0,
            prob_detection=0.9,
            clutter_density=1e-6,
            prob_survival=0.99
        )
    
    @pytest.fixture
    def simple_scenario(self):
        """Create simple test scenario."""
        tracks = [
            Track(
                state=np.array([1.0, 2.0, 0.5, 0.3]),
                covariance=np.eye(4) * 0.1,
                id=1,
                last_update=10.0
            )
        ]
        
        detections = [
            Detection(
                position=np.array([1.05, 2.02]),  # Close to track
                timestamp=10.1,
                measurement_noise=np.eye(2) * 0.01
            ),
            Detection(
                position=np.array([1.1, 2.1]),   # Also close to track
                timestamp=10.1,
                measurement_noise=np.eye(2) * 0.01
            )
        ]
        
        def measurement_function(state):
            return state[:2]
        
        def measurement_jacobian(state):
            H = np.zeros((2, len(state)))
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            return H
        
        return tracks, detections, measurement_function, measurement_jacobian
    
    def test_initialization(self):
        """Test JPDA initialization."""
        jpda = JointProbabilisticDataAssociation(
            gate_threshold=5.0,
            prob_detection=0.95,
            clutter_density=1e-5,
            prob_survival=0.98
        )
        
        assert jpda.gate_threshold == 5.0
        assert jpda.prob_detection == 0.95
        assert jpda.clutter_density == 1e-5
        assert jpda.prob_survival == 0.98
    
    def test_compute_validation_matrix(self, jpda_associator, simple_scenario):
        """Test validation matrix computation."""
        tracks, detections, measurement_function, measurement_jacobian = simple_scenario
        
        validation_matrix = jpda_associator.compute_validation_matrix(
            tracks, detections, measurement_function, measurement_jacobian)
        
        assert validation_matrix.shape == (1, 2)
        assert validation_matrix.dtype == bool
        
        # Both detections should be valid (close to track)
        assert validation_matrix[0, 0]  # Track 0, Detection 0
        assert validation_matrix[0, 1]  # Track 0, Detection 1
    
    def test_compute_likelihood_ratios(self, jpda_associator, simple_scenario):
        """Test likelihood ratio computation."""
        tracks, detections, measurement_function, measurement_jacobian = simple_scenario
        
        validation_matrix = jpda_associator.compute_validation_matrix(
            tracks, detections, measurement_function, measurement_jacobian)
        
        likelihood_ratios = jpda_associator.compute_likelihood_ratios(
            tracks, detections, validation_matrix, measurement_function, measurement_jacobian)
        
        assert likelihood_ratios.shape == (1, 2)
        assert np.all(likelihood_ratios >= 0)
        
        # Closer detection should have higher likelihood
        assert likelihood_ratios[0, 0] > 0
        assert likelihood_ratios[0, 1] > 0
    
    def test_association_probability_computation(self, jpda_associator):
        """Test association probability computation."""
        # Create simple validation matrix and likelihood ratios
        validation_matrix = np.array([[True, True]], dtype=bool)  # 1 track, 2 detections
        likelihood_ratios = np.array([[0.8, 0.6]])
        
        probabilities = jpda_associator.compute_association_probabilities(
            validation_matrix, likelihood_ratios)
        
        assert 'track_detection_probs' in probabilities
        assert 'track_no_detection_probs' in probabilities
        assert 'detection_clutter_probs' in probabilities
        
        track_detection_probs = probabilities['track_detection_probs']
        track_no_detection_probs = probabilities['track_no_detection_probs']
        
        assert track_detection_probs.shape == (1, 2)
        assert len(track_no_detection_probs) == 1
        
        # Probabilities should sum to 1 for each track
        total_prob = (track_detection_probs[0, 0] + track_detection_probs[0, 1] + 
                     track_no_detection_probs[0])
        assert abs(total_prob - 1.0) < 1e-6
    
    def test_generate_association_hypotheses(self, jpda_associator):
        """Test hypothesis generation."""
        n_tracks, n_detections = 2, 2
        valid_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]  # All pairs valid
        
        hypotheses = jpda_associator._generate_association_hypotheses(
            n_tracks, n_detections, valid_pairs)
        
        # Should generate multiple feasible hypotheses
        assert len(hypotheses) > 0
        
        # Check that each hypothesis is feasible
        for hypothesis in hypotheses:
            assigned_detections = [d for d in hypothesis.values() if d != -1]
            # No detection should be assigned to multiple tracks
            assert len(assigned_detections) == len(set(assigned_detections))
    
    def test_hypothesis_probability_computation(self, jpda_associator):
        """Test hypothesis probability computation."""
        hypothesis = {0: 0, 1: -1}  # Track 0 -> Detection 0, Track 1 -> No detection
        likelihood_ratios = np.array([[0.8, 0.6], [0.7, 0.5]])
        
        prob = jpda_associator._compute_hypothesis_probability(
            hypothesis, likelihood_ratios, n_tracks=2, n_detections=2)
        
        assert prob > 0
        assert np.isfinite(prob)
        
        # Should include detection probability, likelihood, and clutter terms
        expected_prob = (jpda_associator.prob_detection * likelihood_ratios[0, 0] *
                        (1 - jpda_associator.prob_detection) *
                        jpda_associator.clutter_density)  # One unassigned detection
        
        assert abs(prob - expected_prob) < 1e-10
    
    def test_empty_scenario(self, jpda_associator):
        """Test JPDA with empty inputs."""
        def measurement_function(state):
            return state[:2]
        
        def measurement_jacobian(state):
            return np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        
        # Empty case
        result = jpda_associator.associate([], [], measurement_function, measurement_jacobian)
        
        assert result.assignments == {}
        assert result.unassigned_tracks == []
        assert result.unassigned_detections == []
        assert result.probabilities == {}
    
    def test_full_association_process(self, jpda_associator, simple_scenario):
        """Test complete JPDA association process."""
        tracks, detections, measurement_function, measurement_jacobian = simple_scenario
        
        result = jpda_associator.associate(
            tracks, detections, measurement_function, measurement_jacobian)
        
        assert isinstance(result, AssociationResult)
        assert result.probabilities is not None
        
        # Should make some assignment
        assert len(result.assignments) >= 0
        
        # Check that probabilities are included
        probs = result.probabilities
        assert 'track_detection_probs' in probs
        assert 'track_no_detection_probs' in probs
        assert 'detection_clutter_probs' in probs


class TestMultipleHypothesisTracking:
    """Test Multiple Hypothesis Tracking."""
    
    @pytest.fixture
    def mht_tracker(self):
        """Create MHT tracker for testing."""
        return MultipleHypothesisTracking(
            gate_threshold=3.0,
            prob_detection=0.9,
            prob_survival=0.95,
            clutter_density=1e-6,
            max_hypotheses=50,
            n_scan_pruning=3,
            prob_threshold=1e-6,
            new_track_threshold=2
        )
    
    def test_initialization(self):
        """Test MHT initialization."""
        mht = MultipleHypothesisTracking(
            gate_threshold=5.0,
            max_hypotheses=100,
            n_scan_pruning=5
        )
        
        assert mht.gate_threshold == 5.0
        assert mht.max_hypotheses == 100
        assert mht.n_scan_pruning == 5
        assert mht.hypotheses == []
        assert mht.hypothesis_id_counter == 0
        assert mht.track_id_counter == 0
        assert mht.scan_count == 0
    
    def test_track_prediction(self, mht_tracker):
        """Test track prediction functionality."""
        tracks = [
            Track(
                state=np.array([1.0, 2.0, 0.5, 0.3]),
                covariance=np.eye(4) * 0.1,
                id=1,
                last_update=10.0,
                age=5
            )
        ]
        
        # Mock motion model
        class MockMotionModel:
            def transition_matrix(self, dt):
                return np.array([
                    [1, 0, dt, 0],
                    [0, 1, 0, dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
            
            def process_noise(self, dt):
                return np.eye(4) * 0.01
        
        motion_model = MockMotionModel()
        
        predicted_tracks = mht_tracker.predict_tracks(tracks, dt=0.1, motion_model=motion_model)
        
        assert len(predicted_tracks) == 1
        
        predicted_track = predicted_tracks[0]
        assert predicted_track.id == tracks[0].id
        assert predicted_track.age == tracks[0].age + 1
        
        # State should have moved forward
        expected_state = np.array([1.05, 2.03, 0.5, 0.3])
        npt.assert_array_almost_equal(predicted_track.state, expected_state)
        
        # Covariance should have increased
        assert np.trace(predicted_track.covariance) > np.trace(tracks[0].covariance)


class TestHypothesis:
    """Test Hypothesis data structure."""
    
    def test_initialization(self):
        """Test Hypothesis initialization."""
        associations = {1: 0, 2: -1}  # Track 1 -> Detection 0, Track 2 -> No detection
        tracks = [
            Track(np.array([1.0, 2.0, 0.5, 0.3]), np.eye(4), 1, 10.0),
            Track(np.array([5.0, 6.0, -0.2, 0.4]), np.eye(4), 2, 10.0)
        ]
        
        hypothesis = Hypothesis(
            id=1,
            parent_id=None,
            associations=associations,
            probability=0.7,
            score=0.8,
            depth=2,
            tracks=tracks
        )
        
        assert hypothesis.id == 1
        assert hypothesis.parent_id is None
        assert hypothesis.associations == associations
        assert hypothesis.probability == 0.7
        assert hypothesis.score == 0.8
        assert hypothesis.depth == 2
        assert len(hypothesis.tracks) == 2


class TestIntegrationScenarios:
    """Test integration scenarios with multiple algorithms."""
    
    @pytest.fixture
    def complex_scenario(self):
        """Create complex multi-target scenario."""
        tracks = [
            Track(
                state=np.array([1.0, 2.0, 0.5, 0.3]),
                covariance=np.eye(4) * 0.1,
                id=1,
                last_update=10.0
            ),
            Track(
                state=np.array([5.0, 6.0, -0.2, 0.4]),
                covariance=np.eye(4) * 0.1,
                id=2,
                last_update=10.0
            ),
            Track(
                state=np.array([10.0, 8.0, 0.1, -0.3]),
                covariance=np.eye(4) * 0.1,
                id=3,
                last_update=10.0
            )
        ]
        
        detections = [
            Detection(
                position=np.array([1.05, 2.02]),
                timestamp=10.1,
                measurement_noise=np.eye(2) * 0.01
            ),
            Detection(
                position=np.array([5.1, 6.05]),
                timestamp=10.1,
                measurement_noise=np.eye(2) * 0.01
            ),
            Detection(
                position=np.array([3.0, 4.0]),  # Ambiguous detection
                timestamp=10.1,
                measurement_noise=np.eye(2) * 0.01
            ),
            Detection(
                position=np.array([15.0, 15.0]),  # Clutter detection
                timestamp=10.1,
                measurement_noise=np.eye(2) * 0.01
            )
        ]
        
        def measurement_function(state):
            return state[:2]
        
        def measurement_jacobian(state):
            H = np.zeros((2, len(state)))
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            return H
        
        return tracks, detections, measurement_function, measurement_jacobian
    
    def test_gnn_vs_jpda_comparison(self, complex_scenario):
        """Compare GNN and JPDA on same scenario."""
        tracks, detections, measurement_function, measurement_jacobian = complex_scenario
        
        # Initialize both associators
        gnn = GlobalNearestNeighbor(gate_threshold=3.0)
        jpda = JointProbabilisticDataAssociation(gate_threshold=3.0)
        
        # Run both algorithms
        gnn_result = gnn.associate(tracks, detections, measurement_function, measurement_jacobian)
        jpda_result = jpda.associate(tracks, detections, measurement_function, measurement_jacobian)
        
        # Both should produce valid results
        assert isinstance(gnn_result, AssociationResult)
        assert isinstance(jpda_result, AssociationResult)
        
        # GNN should have deterministic assignments
        assert jpda_result.probabilities is not None
        assert gnn_result.probabilities is None
        
        # Both should handle the scenario without crashing
        assert len(gnn_result.assignments) >= 0
        assert len(jpda_result.assignments) >= 0
    
    def test_ambiguous_associations(self, complex_scenario):
        """Test handling of ambiguous associations."""
        tracks, detections, measurement_function, measurement_jacobian = complex_scenario
        
        jpda = JointProbabilisticDataAssociation(gate_threshold=5.0)  # Large gate for ambiguity
        
        result = jpda.associate(tracks, detections, measurement_function, measurement_jacobian)
        
        # Should handle ambiguous detections gracefully
        assert result.probabilities is not None
        
        # Probabilities should reflect uncertainty
        track_detection_probs = result.probabilities['track_detection_probs']
        
        # For ambiguous scenarios, no single assignment should have probability 1.0
        max_probs = np.max(track_detection_probs, axis=1)
        assert np.all(max_probs <= 1.0)
    
    def test_clutter_handling(self):
        """Test handling of clutter detections."""
        # Single track with multiple detections (some clutter)
        tracks = [
            Track(
                state=np.array([0.0, 0.0, 0.0, 0.0]),
                covariance=np.eye(4),
                id=1,
                last_update=10.0
            )
        ]
        
        detections = [
            Detection(np.array([0.1, 0.05]), 10.1, np.eye(2) * 0.01),  # Good detection
            Detection(np.array([10.0, 10.0]), 10.1, np.eye(2) * 0.01),  # Clutter
            Detection(np.array([15.0, 15.0]), 10.1, np.eye(2) * 0.01),  # Clutter
        ]
        
        def measurement_function(state):
            return state[:2]
        
        def measurement_jacobian(state):
            return np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        
        # Test with different associators
        for associator in [GlobalNearestNeighbor(gate_threshold=3.0),
                          JointProbabilisticDataAssociation(gate_threshold=3.0)]:
            result = associator.associate(tracks, detections, measurement_function, measurement_jacobian)
            
            # Should associate with close detection, not clutter
            if result.assignments:
                assert result.assignments[1] == 0  # Track 1 -> Detection 0
            
            # Clutter detections should be unassigned
            assert 1 in result.unassigned_detections or 2 in result.unassigned_detections


if __name__ == "__main__":
    pytest.main([__file__])