"""
Integrated Tracking Systems for RadarSim

This module provides comprehensive integrated tracking systems that combine advanced algorithms:
- IMM-JPDA (Interacting Multiple Model - Joint Probabilistic Data Association)
- IMM-MHT (Interacting Multiple Model - Multiple Hypothesis Tracking)

These trackers handle multi-target tracking in cluttered environments with maneuvering targets,
providing robust performance through model mixing, sophisticated data association,
and comprehensive track management.

Author: RadarSim Development Team
"""

import numpy as np
import scipy.stats as stats
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from enum import Enum
import time
import uuid
import warnings
from collections import defaultdict, deque
import logging

# Import existing modules
from .kalman_filters import (
    BaseKalmanFilter, KalmanFilter, ExtendedKalmanFilter,
    initialize_constant_velocity_filter, initialize_constant_acceleration_filter
)
from .motion_models import (
    MotionModel, ModelParameters, CoordinateSystem,
    ConstantVelocityModel, ConstantAccelerationModel, CoordinatedTurnModel
)
from .tracker_base import BaseTracker, Track, Measurement, TrackState, TrackingMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssociationMethod(Enum):
    """Enumeration of data association methods."""
    JPDA = "jpda"
    MHT = "mht"
    NEAREST_NEIGHBOR = "nearest_neighbor"
    GLOBAL_NEAREST_NEIGHBOR = "global_nearest_neighbor"


class HypothesisStatus(Enum):
    """Status of tracking hypotheses."""
    ACTIVE = "active"
    PRUNED = "pruned"
    MERGED = "merged"


@dataclass
class ModelSet:
    """Configuration for multiple motion models in IMM."""
    models: List[MotionModel]
    model_names: List[str]
    initial_probabilities: np.ndarray
    transition_matrix: np.ndarray
    
    def __post_init__(self):
        """Validate model set configuration."""
        if len(self.models) != len(self.model_names):
            raise ValueError("Number of models must match number of names")
        if len(self.initial_probabilities) != len(self.models):
            raise ValueError("Initial probabilities must match number of models")
        if not np.allclose(np.sum(self.initial_probabilities), 1.0):
            raise ValueError("Initial probabilities must sum to 1.0")
        if self.transition_matrix.shape != (len(self.models), len(self.models)):
            raise ValueError("Transition matrix dimensions must match number of models")
        if not np.allclose(np.sum(self.transition_matrix, axis=1), 1.0):
            raise ValueError("Transition matrix rows must sum to 1.0")


@dataclass
class TrackingConfiguration:
    """Configuration parameters for integrated trackers."""
    
    # IMM parameters
    model_set: ModelSet
    mixing_threshold: float = 1e-6
    
    # JPDA parameters
    jpda_gate_threshold: float = 9.21  # Chi-squared 99% for 2D
    jpda_detection_probability: float = 0.9
    jpda_clutter_density: float = 1e-6
    jpda_max_hypotheses: int = 100
    
    # MHT parameters
    mht_gate_threshold: float = 9.21
    mht_max_hypotheses: int = 1000
    mht_max_depth: int = 10
    mht_pruning_threshold: float = 1e-6
    mht_merge_threshold: float = 0.1
    mht_confirmation_threshold: int = 3
    
    # Track management
    track_initiation_threshold: int = 2
    track_deletion_threshold: int = 5
    max_time_without_update: float = 10.0
    min_track_score: float = 0.1
    
    # Sensor fusion
    enable_sensor_fusion: bool = False
    out_of_sequence_handling: bool = False
    max_out_of_sequence_lag: float = 2.0
    
    # Performance optimization
    adaptive_clutter_estimation: bool = True
    parallel_processing: bool = False


@dataclass
class Association:
    """Data association between measurements and tracks."""
    measurement_idx: int
    track_id: str
    probability: float
    distance: float
    gate_valid: bool = True


@dataclass
class Hypothesis:
    """Tracking hypothesis for MHT."""
    hypothesis_id: str
    track_associations: Dict[str, int]  # track_id -> measurement_idx (-1 for no detection)
    measurement_associations: Dict[int, str]  # measurement_idx -> track_id
    probability: float
    likelihood: float
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    parent_id: Optional[str] = None
    creation_time: float = field(default_factory=time.time)
    depth: int = 0


class InteractingMultipleModel:
    """
    Interacting Multiple Model (IMM) filter for handling maneuvering targets.
    
    The IMM uses multiple motion models simultaneously and adaptively weights
    their contributions based on how well each model explains the measurements.
    """
    
    def __init__(self, model_set: ModelSet, state_dim: int, measurement_dim: int):
        """
        Initialize IMM filter.
        
        Args:
            model_set: Set of motion models and parameters
            state_dim: Dimension of state vector
            measurement_dim: Dimension of measurement vector
        """
        self.model_set = model_set
        self.num_models = len(model_set.models)
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # Initialize filters for each model
        self.filters = []
        for i, model in enumerate(model_set.models):
            if isinstance(model, ConstantVelocityModel):
                kf = initialize_constant_velocity_filter(
                    dim=2 if state_dim == 4 else 3,
                    process_noise_std=model.params.process_noise_std
                )
            elif isinstance(model, ConstantAccelerationModel):
                kf = initialize_constant_acceleration_filter(
                    dim=2 if state_dim == 6 else 3,
                    process_noise_std=model.params.process_noise_std
                )
            else:
                # Use Extended Kalman Filter for nonlinear models
                kf = ExtendedKalmanFilter(state_dim, measurement_dim)
                kf.set_state_transition(
                    lambda x, u, dt: model.predict_state(x),
                    lambda x, u, dt: model.get_jacobian(x)
                )
            
            self.filters.append(kf)
        
        # Model probabilities
        self.model_probabilities = model_set.initial_probabilities.copy()
        self.transition_matrix = model_set.transition_matrix.copy()
        
        # Mixing probabilities and states
        self.mixing_probabilities = np.zeros((self.num_models, self.num_models))
        self.mixed_states = [np.zeros(state_dim) for _ in range(self.num_models)]
        self.mixed_covariances = [np.eye(state_dim) for _ in range(self.num_models)]
        
        # Likelihood and innovation storage
        self.model_likelihoods = np.zeros(self.num_models)
        self.innovations = [np.zeros(measurement_dim) for _ in range(self.num_models)]
        self.innovation_covariances = [np.eye(measurement_dim) for _ in range(self.num_models)]
        
    def predict(self, dt: float) -> None:
        """
        Predict step for all models.
        
        Args:
            dt: Time step
        """
        # Step 1: Calculate mixing probabilities
        self._calculate_mixing_probabilities()
        
        # Step 2: Compute mixed initial conditions
        self._compute_mixed_initial_conditions()
        
        # Step 3: Predict with each model
        for i, filter_obj in enumerate(self.filters):
            filter_obj.x = self.mixed_states[i].copy()
            filter_obj.P = self.mixed_covariances[i].copy()
            filter_obj.predict(dt)
    
    def update(self, measurement: np.ndarray, measurement_covariance: np.ndarray) -> None:
        """
        Update step for all models and compute model probabilities.
        
        Args:
            measurement: Measurement vector
            measurement_covariance: Measurement covariance matrix
        """
        # Update measurement noise for all filters
        for filter_obj in self.filters:
            filter_obj.R = measurement_covariance
        
        # Step 4: Update each filter and compute likelihoods
        for i, filter_obj in enumerate(self.filters):
            # Store predicted state for likelihood calculation
            predicted_measurement = filter_obj.H @ filter_obj.x
            innovation = measurement - predicted_measurement
            innovation_cov = filter_obj.H @ filter_obj.P @ filter_obj.H.T + filter_obj.R
            
            # Store innovation information
            self.innovations[i] = innovation
            self.innovation_covariances[i] = innovation_cov
            
            # Compute likelihood
            self.model_likelihoods[i] = self._compute_likelihood(
                innovation, innovation_cov
            )
            
            # Update filter
            filter_obj.update(measurement)
        
        # Step 5: Update model probabilities
        self._update_model_probabilities()
    
    def get_combined_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get combined state estimate and covariance.
        
        Returns:
            Tuple of (combined_state, combined_covariance)
        """
        # Compute weighted average of states
        combined_state = np.zeros(self.state_dim)
        for i, filter_obj in enumerate(self.filters):
            combined_state += self.model_probabilities[i] * filter_obj.x
        
        # Compute combined covariance
        combined_cov = np.zeros((self.state_dim, self.state_dim))
        for i, filter_obj in enumerate(self.filters):
            diff = filter_obj.x - combined_state
            combined_cov += self.model_probabilities[i] * (
                filter_obj.P + np.outer(diff, diff)
            )
        
        return combined_state, combined_cov
    
    def get_model_probability(self, model_idx: int) -> float:
        """Get probability of specific model."""
        return self.model_probabilities[model_idx]
    
    def get_most_likely_model(self) -> int:
        """Get index of most likely model."""
        return np.argmax(self.model_probabilities)
    
    def _calculate_mixing_probabilities(self) -> None:
        """Calculate mixing probabilities for IMM."""
        c_bar = self.transition_matrix.T @ self.model_probabilities
        
        for i in range(self.num_models):
            for j in range(self.num_models):
                if c_bar[i] > 1e-10:
                    self.mixing_probabilities[j, i] = (
                        self.transition_matrix[j, i] * self.model_probabilities[j] / c_bar[i]
                    )
                else:
                    self.mixing_probabilities[j, i] = 1.0 / self.num_models
    
    def _compute_mixed_initial_conditions(self) -> None:
        """Compute mixed initial conditions for each model."""
        for i in range(self.num_models):
            # Mixed state
            self.mixed_states[i] = np.zeros(self.state_dim)
            for j in range(self.num_models):
                self.mixed_states[i] += (
                    self.mixing_probabilities[j, i] * self.filters[j].x
                )
            
            # Mixed covariance
            self.mixed_covariances[i] = np.zeros((self.state_dim, self.state_dim))
            for j in range(self.num_models):
                diff = self.filters[j].x - self.mixed_states[i]
                self.mixed_covariances[i] += self.mixing_probabilities[j, i] * (
                    self.filters[j].P + np.outer(diff, diff)
                )
    
    def _compute_likelihood(self, innovation: np.ndarray, 
                          innovation_cov: np.ndarray) -> float:
        """Compute measurement likelihood for a model."""
        try:
            det_cov = np.linalg.det(innovation_cov)
            if det_cov <= 0:
                return 1e-10
            
            inv_cov = np.linalg.inv(innovation_cov)
            likelihood = np.exp(-0.5 * innovation.T @ inv_cov @ innovation)
            likelihood /= np.sqrt((2 * np.pi) ** len(innovation) * det_cov)
            
            return max(likelihood, 1e-10)
        except np.linalg.LinAlgError:
            return 1e-10
    
    def _update_model_probabilities(self) -> None:
        """Update model probabilities based on likelihoods."""
        # Predicted model probabilities
        c_bar = self.transition_matrix.T @ self.model_probabilities
        
        # Updated probabilities
        numerator = self.model_likelihoods * c_bar
        normalizer = np.sum(numerator)
        
        if normalizer > 1e-10:
            self.model_probabilities = numerator / normalizer
        else:
            # Fallback to uniform distribution
            self.model_probabilities = np.ones(self.num_models) / self.num_models


class JPDATracker(BaseTracker):
    """
    Joint Probabilistic Data Association (JPDA) tracker with IMM filtering.
    
    JPDA handles measurement-to-track association uncertainties by considering
    all feasible associations probabilistically, making it suitable for
    tracking multiple targets in cluttered environments.
    """
    
    def __init__(self, config: TrackingConfiguration):
        """
        Initialize JPDA tracker.
        
        Args:
            config: Tracking configuration parameters
        """
        super().__init__(
            max_tracks=1000,
            gate_threshold=config.jpda_gate_threshold,
            track_confirmation_threshold=config.track_initiation_threshold,
            track_deletion_threshold=config.track_deletion_threshold
        )
        
        self.config = config
        self.model_set = config.model_set
        
        # IMM filters for each track
        self.imm_filters: Dict[str, InteractingMultipleModel] = {}
        
        # JPDA specific parameters
        self.detection_probability = config.jpda_detection_probability
        self.clutter_density = config.jpda_clutter_density
        self.max_hypotheses = config.jpda_max_hypotheses
        
        # Association matrices and probabilities
        self.association_matrix = None
        self.association_probabilities = None
        
        # Adaptive clutter estimation
        self.clutter_history = deque(maxlen=20)
        self.false_alarm_count = 0
        self.total_measurements = 0
        
    def predict(self, timestamp: float) -> None:
        """
        Predict all tracks to given timestamp.
        
        Args:
            timestamp: Target time for prediction
        """
        current_time = time.time()
        
        for track_id, track in self.tracks.items():
            if track.state_vector is not None:
                dt = timestamp - track.last_update_time
                
                # Predict using IMM filter
                if track_id in self.imm_filters:
                    self.imm_filters[track_id].predict(dt)
                    
                    # Update track with IMM estimate
                    combined_state, combined_cov = self.imm_filters[track_id].get_combined_estimate()
                    track.state_vector = combined_state
                    track.covariance = combined_cov
                
                # Update time since last update
                track.time_since_update = current_time - track.last_update_time
    
    def update(self, measurements: List[Measurement]) -> None:
        """
        Update tracks with new measurements using JPDA.
        
        Args:
            measurements: List of new measurements
        """
        if not measurements:
            # Handle missed detections
            self._handle_missed_detections()
            return
        
        # Update clutter statistics
        self.total_measurements += len(measurements)
        
        # Gate measurements for each track
        gated_measurements = self._gate_measurements_for_all_tracks(measurements)
        
        # Create association matrix
        self._create_association_matrix(measurements, gated_measurements)
        
        # Compute association probabilities
        self._compute_jpda_probabilities(measurements)
        
        # Update tracks with associated measurements
        self._update_tracks_with_jpda(measurements)
        
        # Initiate new tracks from unassociated measurements
        self._initiate_tracks_from_unassociated_measurements(measurements)
        
        # Update adaptive clutter estimation
        if self.config.adaptive_clutter_estimation:
            self._update_clutter_estimation(measurements)
        
        # Manage track states
        self.manage_tracks()
    
    def initiate_track(self, measurement: Measurement) -> Track:
        """
        Initiate a new track from a measurement.
        
        Args:
            measurement: Measurement to initiate track from
            
        Returns:
            New track object
        """
        track_id = str(uuid.uuid4())
        
        # Initialize state vector
        if measurement.velocity is not None:
            state_vector = np.concatenate([
                measurement.position[:2],
                measurement.velocity[:2]
            ])
        else:
            state_vector = np.concatenate([
                measurement.position[:2],
                np.zeros(2)
            ])
        
        # Initialize covariance
        initial_cov = np.diag([1.0, 1.0, 5.0, 5.0])
        if measurement.covariance is not None:
            pos_cov = measurement.covariance[:2, :2] if measurement.covariance.shape[0] >= 2 else np.eye(2)
            initial_cov[:2, :2] = pos_cov
        
        # Create track
        track = Track(
            track_id=track_id,
            initial_state=state_vector,
            initial_covariance=initial_cov
        )
        
        # Initialize IMM filter for this track
        self.imm_filters[track_id] = InteractingMultipleModel(
            self.model_set,
            state_dim=len(state_vector),
            measurement_dim=len(measurement.position[:2])
        )
        
        # Set initial state in IMM filters
        for filter_obj in self.imm_filters[track_id].filters:
            filter_obj.x = state_vector.copy()
            filter_obj.P = initial_cov.copy()
        
        # Update track with measurement
        track.update_state(state_vector, initial_cov, measurement, measurement.timestamp)
        
        return track
    
    def _gate_measurements_for_all_tracks(self, 
                                        measurements: List[Measurement]) -> Dict[str, List[Tuple[Measurement, float]]]:
        """
        Gate measurements for all tracks.
        
        Args:
            measurements: List of measurements
            
        Returns:
            Dictionary mapping track IDs to gated measurements
        """
        gated_measurements = {}
        
        for track_id, track in self.tracks.items():
            if track.state_vector is not None:
                gated = self.gate_measurements(track, measurements, measurements[0].timestamp)
                gated_measurements[track_id] = gated
        
        return gated_measurements
    
    def _create_association_matrix(self, measurements: List[Measurement],
                                 gated_measurements: Dict[str, List[Tuple[Measurement, float]]]) -> None:
        """
        Create association matrix for JPDA.
        
        Args:
            measurements: List of measurements
            gated_measurements: Gated measurements for each track
        """
        num_tracks = len(self.tracks)
        num_measurements = len(measurements)
        
        # Association matrix: tracks x measurements
        # 1 if association is feasible, 0 otherwise
        self.association_matrix = np.zeros((num_tracks, num_measurements), dtype=int)
        
        track_list = list(self.tracks.keys())
        
        for track_idx, track_id in enumerate(track_list):
            if track_id in gated_measurements:
                for measurement, distance in gated_measurements[track_id]:
                    measurement_idx = measurements.index(measurement)
                    self.association_matrix[track_idx, measurement_idx] = 1
    
    def _compute_jpda_probabilities(self, measurements: List[Measurement]) -> None:
        """
        Compute JPDA association probabilities.
        
        Args:
            measurements: List of measurements
        """
        num_tracks = len(self.tracks)
        num_measurements = len(measurements)
        
        if num_tracks == 0 or num_measurements == 0:
            self.association_probabilities = np.zeros((num_tracks, num_measurements))
            return
        
        # Initialize probability matrix
        self.association_probabilities = np.zeros((num_tracks, num_measurements))
        
        # Generate all feasible association hypotheses
        hypotheses = self._generate_association_hypotheses(num_tracks, num_measurements)
        
        # Limit number of hypotheses for computational efficiency
        if len(hypotheses) > self.max_hypotheses:
            hypotheses = hypotheses[:self.max_hypotheses]
        
        # Compute hypothesis probabilities
        hypothesis_probs = []
        track_list = list(self.tracks.keys())
        
        for hypothesis in hypotheses:
            prob = self._compute_hypothesis_probability(hypothesis, measurements, track_list)
            hypothesis_probs.append(prob)
        
        # Normalize hypothesis probabilities
        total_prob = sum(hypothesis_probs)
        if total_prob > 0:
            hypothesis_probs = [p / total_prob for p in hypothesis_probs]
        
        # Compute marginal association probabilities
        for hyp_idx, hypothesis in enumerate(hypotheses):
            for track_idx, measurement_idx in enumerate(hypothesis):
                if measurement_idx >= 0:  # Valid association
                    self.association_probabilities[track_idx, measurement_idx] += hypothesis_probs[hyp_idx]
    
    def _generate_association_hypotheses(self, num_tracks: int, num_measurements: int) -> List[List[int]]:
        """
        Generate all feasible association hypotheses.
        
        Args:
            num_tracks: Number of tracks
            num_measurements: Number of measurements
            
        Returns:
            List of association hypotheses
        """
        hypotheses = []
        
        # Each hypothesis is a list where hypothesis[i] = j means track i is associated with measurement j
        # -1 means no association (missed detection)
        
        def generate_recursive(track_idx: int, current_hypothesis: List[int], used_measurements: Set[int]):
            if track_idx == num_tracks:
                hypotheses.append(current_hypothesis.copy())
                return
            
            # Option 1: No association for this track (missed detection)
            if np.random.rand() < 1.0 - self.detection_probability:  # Simplified pruning
                current_hypothesis.append(-1)
                generate_recursive(track_idx + 1, current_hypothesis, used_measurements)
                current_hypothesis.pop()
            
            # Option 2: Associate with available measurements
            for meas_idx in range(num_measurements):
                if (meas_idx not in used_measurements and 
                    self.association_matrix[track_idx, meas_idx] == 1):
                    
                    current_hypothesis.append(meas_idx)
                    used_measurements.add(meas_idx)
                    generate_recursive(track_idx + 1, current_hypothesis, used_measurements)
                    used_measurements.remove(meas_idx)
                    current_hypothesis.pop()
        
        if num_tracks > 0:
            generate_recursive(0, [], set())
        
        return hypotheses
    
    def _compute_hypothesis_probability(self, hypothesis: List[int], 
                                      measurements: List[Measurement],
                                      track_list: List[str]) -> float:
        """
        Compute probability of an association hypothesis.
        
        Args:
            hypothesis: Association hypothesis
            measurements: List of measurements
            track_list: List of track IDs
            
        Returns:
            Hypothesis probability
        """
        prob = 1.0
        used_measurements = set()
        
        # Compute probability for each track-measurement association
        for track_idx, measurement_idx in enumerate(hypothesis):
            track_id = track_list[track_idx]
            track = self.tracks[track_id]
            
            if measurement_idx == -1:
                # Missed detection
                prob *= (1.0 - self.detection_probability)
            else:
                # Valid detection
                measurement = measurements[measurement_idx]
                used_measurements.add(measurement_idx)
                
                # Compute measurement likelihood
                if track_id in self.imm_filters:
                    imm = self.imm_filters[track_id]
                    
                    # Use combined likelihood from all models
                    likelihood = 0.0
                    for i, filter_obj in enumerate(imm.filters):
                        # Predicted measurement
                        H = filter_obj.H
                        predicted_meas = H @ filter_obj.x
                        innovation = measurement.position[:2] - predicted_meas
                        innovation_cov = H @ filter_obj.P @ H.T + measurement.covariance[:2, :2]
                        
                        model_likelihood = imm._compute_likelihood(innovation, innovation_cov)
                        likelihood += imm.model_probabilities[i] * model_likelihood
                    
                    prob *= self.detection_probability * likelihood
        
        # Account for false alarms (unassociated measurements)
        num_false_alarms = len(measurements) - len(used_measurements)
        if num_false_alarms > 0:
            prob *= (self.clutter_density ** num_false_alarms)
        
        return prob
    
    def _update_tracks_with_jpda(self, measurements: List[Measurement]) -> None:
        """
        Update tracks using JPDA association probabilities.
        
        Args:
            measurements: List of measurements
        """
        track_list = list(self.tracks.keys())
        
        for track_idx, track_id in enumerate(track_list):
            track = self.tracks[track_id]
            
            if track_id not in self.imm_filters:
                continue
            
            imm = self.imm_filters[track_id]
            
            # Compute weighted measurement update
            associated_measurements = []
            association_weights = []
            
            for meas_idx, measurement in enumerate(measurements):
                association_prob = self.association_probabilities[track_idx, meas_idx]
                if association_prob > 1e-6:
                    associated_measurements.append(measurement)
                    association_weights.append(association_prob)
            
            if associated_measurements:
                # Normalize weights
                total_weight = sum(association_weights)
                if total_weight > 0:
                    association_weights = [w / total_weight for w in association_weights]
                
                # Weighted update for each model in IMM
                for model_idx, filter_obj in enumerate(imm.filters):
                    # Store original state
                    original_state = filter_obj.x.copy()
                    original_cov = filter_obj.P.copy()
                    
                    # Reset to predicted state
                    filter_obj.x = original_state
                    filter_obj.P = original_cov
                    
                    # Weighted update
                    weighted_innovation = np.zeros(len(measurements[0].position[:2]))
                    weighted_cov = np.zeros((len(measurements[0].position[:2]), len(measurements[0].position[:2])))
                    
                    for measurement, weight in zip(associated_measurements, association_weights):
                        # Update with this measurement
                        temp_filter = filter_obj.__class__(filter_obj.dim_x, filter_obj.dim_z)
                        temp_filter.x = original_state.copy()
                        temp_filter.P = original_cov.copy()
                        temp_filter.F = filter_obj.F.copy()
                        temp_filter.H = filter_obj.H.copy()
                        temp_filter.R = measurement.covariance[:2, :2]
                        
                        temp_filter.update(measurement.position[:2])
                        
                        # Accumulate weighted contribution
                        weighted_innovation += weight * (temp_filter.x - original_state)
                        diff = temp_filter.x - original_state
                        weighted_cov += weight * (temp_filter.P + np.outer(diff, diff))
                    
                    # Apply weighted update
                    filter_obj.x = original_state + weighted_innovation
                    filter_obj.P = weighted_cov
                
                # Update track state with combined IMM estimate
                combined_state, combined_cov = imm.get_combined_estimate()
                track.update_state(
                    combined_state,
                    combined_cov,
                    associated_measurements[0],  # Use first measurement for timestamp
                    associated_measurements[0].timestamp
                )
                
                # Update track quality metrics
                avg_innovation = np.mean([
                    np.linalg.norm(m.position[:2] - combined_state[:2])
                    for m in associated_measurements
                ])
                track.update_quality_metrics(total_weight, avg_innovation)
            else:
                # No associated measurements - missed detection
                track.miss_count += 1
                track.age += 1
    
    def _initiate_tracks_from_unassociated_measurements(self, measurements: List[Measurement]) -> None:
        """
        Initiate new tracks from measurements not associated with existing tracks.
        
        Args:
            measurements: List of measurements
        """
        # Find unassociated measurements
        unassociated_indices = set(range(len(measurements)))
        
        for track_idx in range(len(self.tracks)):
            for meas_idx in range(len(measurements)):
                if self.association_probabilities[track_idx, meas_idx] > 0.5:
                    unassociated_indices.discard(meas_idx)
        
        # Initiate tracks from unassociated measurements
        for meas_idx in unassociated_indices:
            measurement = measurements[meas_idx]
            new_track = self.initiate_track(measurement)
            self.add_track(new_track)
            
            logger.info(f"Initiated new track {new_track.track_id} from unassociated measurement")
    
    def _handle_missed_detections(self) -> None:
        """Handle case when no measurements are received."""
        for track_id, track in self.tracks.items():
            track.miss_count += 1
            track.age += 1
            
            # Update time since last update
            current_time = time.time()
            track.time_since_update = current_time - track.last_update_time
    
    def _update_clutter_estimation(self, measurements: List[Measurement]) -> None:
        """
        Update adaptive clutter density estimation.
        
        Args:
            measurements: List of measurements
        """
        # Count false alarms (measurements not strongly associated with any track)
        false_alarms = 0
        
        for meas_idx in range(len(measurements)):
            max_association_prob = 0.0
            for track_idx in range(len(self.tracks)):
                max_association_prob = max(
                    max_association_prob,
                    self.association_probabilities[track_idx, meas_idx]
                )
            
            if max_association_prob < 0.3:  # Threshold for considering as false alarm
                false_alarms += 1
        
        self.false_alarm_count += false_alarms
        self.clutter_history.append(false_alarms)
        
        # Update clutter density using recent history
        if len(self.clutter_history) >= 5:
            avg_false_alarms = np.mean(list(self.clutter_history))
            # Assume surveillance volume (simplified)
            surveillance_volume = 1000.0  # m^2
            self.clutter_density = avg_false_alarms / surveillance_volume


class MHTTracker(BaseTracker):
    """
    Multiple Hypothesis Tracking (MHT) with IMM filtering.
    
    MHT maintains multiple hypotheses about target existence and track-measurement
    associations, providing optimal performance for multi-target tracking in
    dense clutter environments.
    """
    
    def __init__(self, config: TrackingConfiguration):
        """
        Initialize MHT tracker.
        
        Args:
            config: Tracking configuration parameters
        """
        super().__init__(
            max_tracks=1000,
            gate_threshold=config.mht_gate_threshold,
            track_confirmation_threshold=config.mht_confirmation_threshold,
            track_deletion_threshold=config.track_deletion_threshold
        )
        
        self.config = config
        self.model_set = config.model_set
        
        # MHT specific parameters
        self.max_hypotheses = config.mht_max_hypotheses
        self.max_depth = config.mht_max_depth
        self.pruning_threshold = config.mht_pruning_threshold
        self.merge_threshold = config.mht_merge_threshold
        
        # Hypothesis management
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.hypothesis_tree: Dict[str, List[str]] = {}  # parent -> children
        self.current_best_hypothesis: Optional[str] = None
        
        # IMM filters for each track in each hypothesis
        self.imm_filters: Dict[str, Dict[str, InteractingMultipleModel]] = {}
        
        # Track management
        self.global_track_counter = 0
        self.hypothesis_counter = 0
        
        # Initialize root hypothesis
        self._initialize_root_hypothesis()
    
    def predict(self, timestamp: float) -> None:
        """
        Predict all tracks in all hypotheses.
        
        Args:
            timestamp: Target time for prediction
        """
        for hypothesis_id, hypothesis in self.hypotheses.items():
            if hypothesis.status != HypothesisStatus.ACTIVE:
                continue
            
            # Predict all tracks in this hypothesis
            if hypothesis_id in self.imm_filters:
                for track_id, imm in self.imm_filters[hypothesis_id].items():
                    if track_id in self.tracks:
                        track = self.tracks[track_id]
                        dt = timestamp - track.last_update_time
                        
                        # Predict using IMM
                        imm.predict(dt)
                        
                        # Update track state
                        combined_state, combined_cov = imm.get_combined_estimate()
                        track.state_vector = combined_state
                        track.covariance = combined_cov
    
    def update(self, measurements: List[Measurement]) -> None:
        """
        Update tracks using MHT with new measurements.
        
        Args:
            measurements: List of new measurements
        """
        if not measurements:
            self._handle_missed_detections()
            return
        
        # Generate new hypotheses from current active hypotheses
        new_hypotheses = self._generate_new_hypotheses(measurements)
        
        # Add new hypotheses to the tree
        for hypothesis in new_hypotheses:
            self.hypotheses[hypothesis.hypothesis_id] = hypothesis
            
            # Initialize IMM filters for tracks in this hypothesis
            self._initialize_imm_filters_for_hypothesis(hypothesis)
        
        # Update all tracks in all hypotheses
        self._update_all_hypotheses(measurements)
        
        # Prune unlikely hypotheses
        self._prune_hypotheses()
        
        # Merge similar hypotheses
        self._merge_hypotheses()
        
        # Select best hypothesis
        self._select_best_hypothesis()
        
        # Update track states based on best hypothesis
        self._update_tracks_from_best_hypothesis()
        
        # Manage track states
        self.manage_tracks()
    
    def initiate_track(self, measurement: Measurement) -> Track:
        """
        Initiate a new track from a measurement.
        
        Args:
            measurement: Measurement to initiate track from
            
        Returns:
            New track object
        """
        track_id = f"track_{self.global_track_counter:06d}"
        self.global_track_counter += 1
        
        # Initialize state vector
        if measurement.velocity is not None:
            state_vector = np.concatenate([
                measurement.position[:2],
                measurement.velocity[:2]
            ])
        else:
            state_vector = np.concatenate([
                measurement.position[:2],
                np.zeros(2)
            ])
        
        # Initialize covariance
        initial_cov = np.diag([1.0, 1.0, 5.0, 5.0])
        if measurement.covariance is not None:
            pos_cov = measurement.covariance[:2, :2] if measurement.covariance.shape[0] >= 2 else np.eye(2)
            initial_cov[:2, :2] = pos_cov
        
        # Create track
        track = Track(
            track_id=track_id,
            initial_state=state_vector,
            initial_covariance=initial_cov
        )
        
        # Update track with measurement
        track.update_state(state_vector, initial_cov, measurement, measurement.timestamp)
        
        return track
    
    def _initialize_root_hypothesis(self) -> None:
        """Initialize the root hypothesis."""
        root_id = f"hyp_{self.hypothesis_counter:06d}"
        self.hypothesis_counter += 1
        
        root_hypothesis = Hypothesis(
            hypothesis_id=root_id,
            track_associations={},
            measurement_associations={},
            probability=1.0,
            likelihood=1.0,
            depth=0
        )
        
        self.hypotheses[root_id] = root_hypothesis
        self.hypothesis_tree[root_id] = []
        self.current_best_hypothesis = root_id
        self.imm_filters[root_id] = {}
    
    def _generate_new_hypotheses(self, measurements: List[Measurement]) -> List[Hypothesis]:
        """
        Generate new hypotheses from current active hypotheses and measurements.
        
        Args:
            measurements: List of new measurements
            
        Returns:
            List of new hypotheses
        """
        new_hypotheses = []
        
        for hypothesis_id, hypothesis in self.hypotheses.items():
            if (hypothesis.status != HypothesisStatus.ACTIVE or 
                hypothesis.depth >= self.max_depth):
                continue
            
            # Generate child hypotheses for this parent
            child_hypotheses = self._generate_child_hypotheses(hypothesis, measurements)
            new_hypotheses.extend(child_hypotheses)
            
            # Update hypothesis tree
            if hypothesis_id not in self.hypothesis_tree:
                self.hypothesis_tree[hypothesis_id] = []
            
            for child in child_hypotheses:
                self.hypothesis_tree[hypothesis_id].append(child.hypothesis_id)
                child.parent_id = hypothesis_id
        
        return new_hypotheses
    
    def _generate_child_hypotheses(self, parent_hypothesis: Hypothesis, 
                                 measurements: List[Measurement]) -> List[Hypothesis]:
        """
        Generate child hypotheses from a parent hypothesis.
        
        Args:
            parent_hypothesis: Parent hypothesis
            measurements: List of measurements
            
        Returns:
            List of child hypotheses
        """
        child_hypotheses = []
        
        # Get tracks associated with this hypothesis
        hypothesis_tracks = list(parent_hypothesis.track_associations.keys())
        
        # Generate association matrices for tracks and measurements
        if not hypothesis_tracks and not measurements:
            return []
        
        # Create gating matrix
        gating_matrix = self._create_gating_matrix(hypothesis_tracks, measurements, parent_hypothesis)
        
        # Generate all feasible association patterns
        association_patterns = self._generate_association_patterns(
            len(hypothesis_tracks), len(measurements), gating_matrix
        )
        
        # Create child hypothesis for each association pattern
        for pattern in association_patterns:
            child_id = f"hyp_{self.hypothesis_counter:06d}"
            self.hypothesis_counter += 1
            
            # Create track and measurement associations
            track_associations = {}
            measurement_associations = {}
            
            # Copy parent associations
            for track_id in hypothesis_tracks:
                track_associations[track_id] = parent_hypothesis.track_associations.get(track_id, -1)
            
            # Apply new associations from pattern
            for track_idx, meas_idx in enumerate(pattern):
                if track_idx < len(hypothesis_tracks) and meas_idx >= 0:
                    track_id = hypothesis_tracks[track_idx]
                    track_associations[track_id] = meas_idx
                    measurement_associations[meas_idx] = track_id
            
            # Handle new track initiations from unassociated measurements
            used_measurements = set(measurement_associations.keys())
            for meas_idx in range(len(measurements)):
                if meas_idx not in used_measurements:
                    # Create new track
                    new_track = self.initiate_track(measurements[meas_idx])
                    track_associations[new_track.track_id] = meas_idx
                    measurement_associations[meas_idx] = new_track.track_id
                    
                    # Add to global tracks
                    self.tracks[new_track.track_id] = new_track
            
            # Calculate hypothesis likelihood
            likelihood = self._calculate_hypothesis_likelihood(
                track_associations, measurement_associations, measurements, parent_hypothesis
            )
            
            # Create child hypothesis
            child_hypothesis = Hypothesis(
                hypothesis_id=child_id,
                track_associations=track_associations,
                measurement_associations=measurement_associations,
                probability=parent_hypothesis.probability * likelihood,
                likelihood=likelihood,
                parent_id=parent_hypothesis.hypothesis_id,
                depth=parent_hypothesis.depth + 1
            )
            
            child_hypotheses.append(child_hypothesis)
        
        return child_hypotheses
    
    def _create_gating_matrix(self, track_list: List[str], measurements: List[Measurement],
                            hypothesis: Hypothesis) -> np.ndarray:
        """
        Create gating matrix for tracks and measurements.
        
        Args:
            track_list: List of track IDs
            measurements: List of measurements
            hypothesis: Current hypothesis
            
        Returns:
            Gating matrix (tracks x measurements)
        """
        if not track_list or not measurements:
            return np.zeros((0, 0))
        
        gating_matrix = np.zeros((len(track_list), len(measurements)))
        
        for track_idx, track_id in enumerate(track_list):
            if track_id not in self.tracks:
                continue
            
            track = self.tracks[track_id]
            
            # Get IMM filter for this track in this hypothesis
            if (hypothesis.hypothesis_id in self.imm_filters and 
                track_id in self.imm_filters[hypothesis.hypothesis_id]):
                
                imm = self.imm_filters[hypothesis.hypothesis_id][track_id]
                combined_state, combined_cov = imm.get_combined_estimate()
                
                for meas_idx, measurement in enumerate(measurements):
                    # Compute Mahalanobis distance
                    predicted_meas = combined_state[:2]
                    innovation = measurement.position[:2] - predicted_meas
                    innovation_cov = combined_cov[:2, :2] + measurement.covariance[:2, :2]
                    
                    try:
                        inv_cov = np.linalg.inv(innovation_cov)
                        distance = innovation.T @ inv_cov @ innovation
                        
                        if distance <= self.gate_threshold:
                            gating_matrix[track_idx, meas_idx] = 1
                    except np.linalg.LinAlgError:
                        continue
        
        return gating_matrix
    
    def _generate_association_patterns(self, num_tracks: int, num_measurements: int,
                                     gating_matrix: np.ndarray) -> List[List[int]]:
        """
        Generate feasible association patterns.
        
        Args:
            num_tracks: Number of tracks
            num_measurements: Number of measurements
            gating_matrix: Gating constraints
            
        Returns:
            List of association patterns
        """
        patterns = []
        
        def generate_recursive(track_idx: int, current_pattern: List[int], 
                             used_measurements: Set[int]):
            if track_idx == num_tracks:
                patterns.append(current_pattern.copy())
                return
            
            # Option 1: No association (missed detection)
            current_pattern.append(-1)
            generate_recursive(track_idx + 1, current_pattern, used_measurements)
            current_pattern.pop()
            
            # Option 2: Associate with available measurements
            for meas_idx in range(num_measurements):
                if (meas_idx not in used_measurements and 
                    gating_matrix[track_idx, meas_idx] == 1):
                    
                    current_pattern.append(meas_idx)
                    used_measurements.add(meas_idx)
                    generate_recursive(track_idx + 1, current_pattern, used_measurements)
                    used_measurements.remove(meas_idx)
                    current_pattern.pop()
        
        if num_tracks > 0:
            generate_recursive(0, [], set())
        
        # Limit number of patterns for computational efficiency
        if len(patterns) > 100:
            patterns = patterns[:100]
        
        return patterns
    
    def _calculate_hypothesis_likelihood(self, track_associations: Dict[str, int],
                                       measurement_associations: Dict[int, str],
                                       measurements: List[Measurement],
                                       parent_hypothesis: Hypothesis) -> float:
        """
        Calculate likelihood of a hypothesis.
        
        Args:
            track_associations: Track to measurement associations
            measurement_associations: Measurement to track associations
            measurements: List of measurements
            parent_hypothesis: Parent hypothesis
            
        Returns:
            Hypothesis likelihood
        """
        likelihood = 1.0
        detection_prob = 0.9
        clutter_density = 1e-6
        
        # Likelihood from track-measurement associations
        for track_id, meas_idx in track_associations.items():
            if meas_idx == -1:
                # Missed detection
                likelihood *= (1.0 - detection_prob)
            else:
                # Valid detection
                if track_id in self.tracks and meas_idx < len(measurements):
                    track = self.tracks[track_id]
                    measurement = measurements[meas_idx]
                    
                    # Get measurement likelihood from IMM
                    if (parent_hypothesis.hypothesis_id in self.imm_filters and
                        track_id in self.imm_filters[parent_hypothesis.hypothesis_id]):
                        
                        imm = self.imm_filters[parent_hypothesis.hypothesis_id][track_id]
                        
                        # Combined likelihood from all models
                        meas_likelihood = 0.0
                        for i, filter_obj in enumerate(imm.filters):
                            predicted_meas = filter_obj.H @ filter_obj.x
                            innovation = measurement.position[:2] - predicted_meas
                            innovation_cov = (filter_obj.H @ filter_obj.P @ filter_obj.H.T + 
                                            measurement.covariance[:2, :2])
                            
                            model_likelihood = imm._compute_likelihood(innovation, innovation_cov)
                            meas_likelihood += imm.model_probabilities[i] * model_likelihood
                        
                        likelihood *= detection_prob * meas_likelihood
        
        # Account for false alarms
        num_false_alarms = len(measurements) - len(measurement_associations)
        if num_false_alarms > 0:
            likelihood *= (clutter_density ** num_false_alarms)
        
        return likelihood
    
    def _initialize_imm_filters_for_hypothesis(self, hypothesis: Hypothesis) -> None:
        """
        Initialize IMM filters for tracks in a hypothesis.
        
        Args:
            hypothesis: Hypothesis to initialize filters for
        """
        if hypothesis.hypothesis_id not in self.imm_filters:
            self.imm_filters[hypothesis.hypothesis_id] = {}
        
        for track_id in hypothesis.track_associations.keys():
            if track_id not in self.imm_filters[hypothesis.hypothesis_id]:
                if track_id in self.tracks:
                    track = self.tracks[track_id]
                    state_dim = len(track.state_vector) if track.state_vector is not None else 4
                    measurement_dim = 2
                    
                    # Create IMM filter
                    imm = InteractingMultipleModel(self.model_set, state_dim, measurement_dim)
                    
                    # Initialize with track state
                    if track.state_vector is not None:
                        for filter_obj in imm.filters:
                            filter_obj.x = track.state_vector.copy()
                            filter_obj.P = track.covariance.copy() if track.covariance is not None else np.eye(state_dim)
                    
                    self.imm_filters[hypothesis.hypothesis_id][track_id] = imm
    
    def _update_all_hypotheses(self, measurements: List[Measurement]) -> None:
        """
        Update all active hypotheses with measurements.
        
        Args:
            measurements: List of measurements
        """
        for hypothesis_id, hypothesis in self.hypotheses.items():
            if hypothesis.status != HypothesisStatus.ACTIVE:
                continue
            
            # Update tracks in this hypothesis
            if hypothesis_id in self.imm_filters:
                for track_id, meas_idx in hypothesis.track_associations.items():
                    if (track_id in self.imm_filters[hypothesis_id] and 
                        meas_idx >= 0 and meas_idx < len(measurements)):
                        
                        imm = self.imm_filters[hypothesis_id][track_id]
                        measurement = measurements[meas_idx]
                        
                        # Update IMM with measurement
                        imm.update(measurement.position[:2], measurement.covariance[:2, :2])
                        
                        # Update track state
                        if track_id in self.tracks:
                            combined_state, combined_cov = imm.get_combined_estimate()
                            self.tracks[track_id].update_state(
                                combined_state, combined_cov, measurement, measurement.timestamp
                            )
    
    def _prune_hypotheses(self) -> None:
        """Prune hypotheses with low probability."""
        to_prune = []
        
        for hypothesis_id, hypothesis in self.hypotheses.items():
            if (hypothesis.status == HypothesisStatus.ACTIVE and 
                hypothesis.probability < self.pruning_threshold):
                to_prune.append(hypothesis_id)
        
        for hypothesis_id in to_prune:
            self.hypotheses[hypothesis_id].status = HypothesisStatus.PRUNED
            
            # Remove IMM filters for pruned hypothesis
            if hypothesis_id in self.imm_filters:
                del self.imm_filters[hypothesis_id]
        
        logger.info(f"Pruned {len(to_prune)} hypotheses")
    
    def _merge_hypotheses(self) -> None:
        """Merge similar hypotheses."""
        # This is a simplified implementation
        # Full implementation would compare hypothesis similarity and merge appropriately
        pass
    
    def _select_best_hypothesis(self) -> None:
        """Select the hypothesis with highest probability."""
        best_hypothesis = None
        best_probability = -1.0
        
        for hypothesis_id, hypothesis in self.hypotheses.items():
            if (hypothesis.status == HypothesisStatus.ACTIVE and 
                hypothesis.probability > best_probability):
                best_probability = hypothesis.probability
                best_hypothesis = hypothesis_id
        
        self.current_best_hypothesis = best_hypothesis
    
    def _update_tracks_from_best_hypothesis(self) -> None:
        """Update track states based on the best hypothesis."""
        if self.current_best_hypothesis is None:
            return
        
        best_hypothesis = self.hypotheses[self.current_best_hypothesis]
        
        # Update track states with IMM estimates from best hypothesis
        if self.current_best_hypothesis in self.imm_filters:
            for track_id, imm in self.imm_filters[self.current_best_hypothesis].items():
                if track_id in self.tracks:
                    combined_state, combined_cov = imm.get_combined_estimate()
                    track = self.tracks[track_id]
                    track.state_vector = combined_state
                    track.covariance = combined_cov
    
    def _handle_missed_detections(self) -> None:
        """Handle case when no measurements are received."""
        # Propagate all hypotheses forward with missed detections
        for hypothesis_id, hypothesis in self.hypotheses.items():
            if hypothesis.status != HypothesisStatus.ACTIVE:
                continue
            
            # Update track miss counts
            for track_id in hypothesis.track_associations.keys():
                if track_id in self.tracks:
                    self.tracks[track_id].miss_count += 1
                    self.tracks[track_id].age += 1


# Utility functions and configuration helpers

def create_default_model_set(coordinate_system: CoordinateSystem = CoordinateSystem.CARTESIAN_2D,
                           dt: float = 1.0) -> ModelSet:
    """
    Create a default model set for IMM filtering.
    
    Args:
        coordinate_system: Coordinate system for models
        dt: Time step
        
    Returns:
        Default model set with CV, CA, and CT models
    """
    # Create motion models
    cv_params = ModelParameters(dt=dt, process_noise_std=1.0, coordinate_system=coordinate_system)
    ca_params = ModelParameters(dt=dt, process_noise_std=2.0, coordinate_system=coordinate_system)
    ct_params = ModelParameters(
        dt=dt, 
        process_noise_std=1.5, 
        coordinate_system=coordinate_system,
        additional_params={'turn_rate': None, 'turn_rate_noise': 0.1}
    )
    
    models = [
        ConstantVelocityModel(cv_params),
        ConstantAccelerationModel(ca_params),
        CoordinatedTurnModel(ct_params)
    ]
    
    model_names = ["CV", "CA", "CT"]
    
    # Initial model probabilities (favor CV)
    initial_probabilities = np.array([0.6, 0.3, 0.1])
    
    # Model transition matrix
    transition_matrix = np.array([
        [0.85, 0.10, 0.05],  # From CV
        [0.15, 0.80, 0.05],  # From CA
        [0.10, 0.10, 0.80]   # From CT
    ])
    
    return ModelSet(
        models=models,
        model_names=model_names,
        initial_probabilities=initial_probabilities,
        transition_matrix=transition_matrix
    )


def create_tracking_configuration(association_method: AssociationMethod = AssociationMethod.JPDA,
                                enable_sensor_fusion: bool = False,
                                coordinate_system: CoordinateSystem = CoordinateSystem.CARTESIAN_2D) -> TrackingConfiguration:
    """
    Create a tracking configuration with sensible defaults.
    
    Args:
        association_method: Data association method to optimize for
        enable_sensor_fusion: Whether to enable sensor fusion capabilities
        coordinate_system: Coordinate system for motion models
        
    Returns:
        Tracking configuration
    """
    model_set = create_default_model_set(coordinate_system)
    
    if association_method == AssociationMethod.JPDA:
        config = TrackingConfiguration(
            model_set=model_set,
            jpda_gate_threshold=9.21,
            jpda_detection_probability=0.9,
            jpda_clutter_density=1e-6,
            jpda_max_hypotheses=100,
            enable_sensor_fusion=enable_sensor_fusion
        )
    elif association_method == AssociationMethod.MHT:
        config = TrackingConfiguration(
            model_set=model_set,
            mht_gate_threshold=9.21,
            mht_max_hypotheses=1000,
            mht_max_depth=10,
            mht_pruning_threshold=1e-6,
            mht_merge_threshold=0.1,
            mht_confirmation_threshold=3,
            enable_sensor_fusion=enable_sensor_fusion
        )
    else:
        # Default configuration
        config = TrackingConfiguration(
            model_set=model_set,
            enable_sensor_fusion=enable_sensor_fusion
        )
    
    return config


class SensorFusionManager:
    """
    Manager for multi-sensor data fusion and out-of-sequence measurement handling.
    """
    
    def __init__(self, config: TrackingConfiguration):
        """
        Initialize sensor fusion manager.
        
        Args:
            config: Tracking configuration
        """
        self.config = config
        self.sensor_data_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_processing_time = 0.0
        
    def add_sensor_data(self, sensor_id: str, measurements: List[Measurement]) -> None:
        """
        Add sensor data to fusion buffer.
        
        Args:
            sensor_id: Unique sensor identifier
            measurements: List of measurements from sensor
        """
        for measurement in measurements:
            measurement.metadata['sensor_id'] = sensor_id
            self.sensor_data_buffer[sensor_id].append(measurement)
    
    def get_fused_measurements(self, current_time: float) -> List[Measurement]:
        """
        Get fused measurements for current time.
        
        Args:
            current_time: Current processing time
            
        Returns:
            List of fused measurements
        """
        all_measurements = []
        
        # Collect measurements from all sensors
        for sensor_id, buffer in self.sensor_data_buffer.items():
            for measurement in buffer:
                if (measurement.timestamp <= current_time and 
                    measurement.timestamp > self.last_processing_time):
                    all_measurements.append(measurement)
        
        # Sort by timestamp
        all_measurements.sort(key=lambda m: m.timestamp)
        
        # Handle out-of-sequence measurements if enabled
        if self.config.out_of_sequence_handling:
            all_measurements = self._handle_out_of_sequence(all_measurements, current_time)
        
        self.last_processing_time = current_time
        return all_measurements
    
    def _handle_out_of_sequence(self, measurements: List[Measurement], 
                               current_time: float) -> List[Measurement]:
        """
        Handle out-of-sequence measurements.
        
        Args:
            measurements: List of measurements
            current_time: Current time
            
        Returns:
            Processed measurements
        """
        # Filter out measurements that are too old
        max_lag = self.config.max_out_of_sequence_lag
        filtered_measurements = [
            m for m in measurements 
            if current_time - m.timestamp <= max_lag
        ]
        
        return filtered_measurements


class PerformanceMetrics:
    """
    Enhanced performance metrics for integrated tracking systems.
    """
    
    def __init__(self):
        """Initialize performance metrics."""
        self.ospa_history = []
        self.computational_time_history = []
        self.memory_usage_history = []
        self.track_purity_history = []
        self.track_completeness_history = []
        
    def compute_ospa_distance(self, estimated_tracks: Dict[str, np.ndarray],
                            ground_truth_tracks: Dict[str, np.ndarray],
                            cutoff_distance: float = 100.0,
                            order: int = 2) -> float:
        """
        Compute Optimal Sub-Pattern Assignment (OSPA) distance.
        
        Args:
            estimated_tracks: Dictionary of estimated track states
            ground_truth_tracks: Dictionary of ground truth track states
            cutoff_distance: OSPA cutoff distance
            order: OSPA order parameter
            
        Returns:
            OSPA distance
        """
        if not estimated_tracks and not ground_truth_tracks:
            return 0.0
        
        n_est = len(estimated_tracks)
        n_true = len(ground_truth_tracks)
        
        if n_est == 0:
            return cutoff_distance
        if n_true == 0:
            return cutoff_distance
        
        # Create distance matrix
        est_positions = np.array([state[:2] for state in estimated_tracks.values()])
        true_positions = np.array([state[:2] for state in ground_truth_tracks.values()])
        
        dist_matrix = cdist(est_positions, true_positions)
        
        # Apply cutoff
        dist_matrix = np.minimum(dist_matrix, cutoff_distance)
        
        # Solve assignment problem
        if n_est <= n_true:
            row_indices, col_indices = linear_sum_assignment(dist_matrix)
            assignment_cost = np.sum(dist_matrix[row_indices, col_indices])
            cardinality_penalty = (n_true - n_est) * cutoff_distance
        else:
            col_indices, row_indices = linear_sum_assignment(dist_matrix.T)
            assignment_cost = np.sum(dist_matrix[row_indices, col_indices])
            cardinality_penalty = (n_est - n_true) * cutoff_distance
        
        total_cost = assignment_cost + cardinality_penalty
        n_max = max(n_est, n_true)
        
        ospa_distance = (total_cost / n_max) ** (1.0 / order)
        
        self.ospa_history.append(ospa_distance)
        return ospa_distance
    
    def compute_track_metrics(self, estimated_tracks: Dict[str, List[np.ndarray]],
                            ground_truth_tracks: Dict[str, List[np.ndarray]],
                            distance_threshold: float = 5.0) -> Dict[str, float]:
        """
        Compute track-level performance metrics.
        
        Args:
            estimated_tracks: Dictionary of estimated track trajectories
            ground_truth_tracks: Dictionary of ground truth trajectories
            distance_threshold: Threshold for track association
            
        Returns:
            Dictionary of track metrics
        """
        # This is a simplified implementation
        # Full implementation would include sophisticated track association
        
        metrics = {
            'track_purity': 0.0,
            'track_completeness': 0.0,
            'false_track_rate': 0.0,
            'track_fragmentation': 0.0
        }
        
        # Store metrics
        self.track_purity_history.append(metrics['track_purity'])
        self.track_completeness_history.append(metrics['track_completeness'])
        
        return metrics
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all performance metrics."""
        summary = {}
        
        if self.ospa_history:
            summary['average_ospa'] = np.mean(self.ospa_history)
            summary['std_ospa'] = np.std(self.ospa_history)
        
        if self.computational_time_history:
            summary['average_computation_time'] = np.mean(self.computational_time_history)
        
        if self.track_purity_history:
            summary['average_track_purity'] = np.mean(self.track_purity_history)
        
        if self.track_completeness_history:
            summary['average_track_completeness'] = np.mean(self.track_completeness_history)
        
        return summary


# Example usage and demonstration
def create_sample_scenario() -> Tuple[JPDATracker, List[Measurement]]:
    """
    Create a sample tracking scenario for demonstration.
    
    Returns:
        Tuple of (tracker, sample_measurements)
    """
    # Create configuration
    config = create_tracking_configuration(AssociationMethod.JPDA)
    
    # Create tracker
    tracker = JPDATracker(config)
    
    # Generate sample measurements for two crossing targets
    measurements = []
    
    # Target 1: Moving right
    for i in range(20):
        t = i * 0.1
        x = 10 + 5 * t + 0.5 * np.random.randn()
        y = 20 + 1 * t + 0.3 * np.random.randn()
        
        measurement = Measurement(
            position=np.array([x, y, 0]),
            timestamp=t,
            covariance=np.diag([0.5, 0.5, 1.0])
        )
        measurements.append(measurement)
    
    # Target 2: Moving up and left
    for i in range(20):
        t = i * 0.1
        x = 50 - 3 * t + 0.5 * np.random.randn()
        y = 10 + 4 * t + 0.3 * np.random.randn()
        
        measurement = Measurement(
            position=np.array([x, y, 0]),
            timestamp=t,
            covariance=np.diag([0.5, 0.5, 1.0])
        )
        measurements.append(measurement)
    
    # Add some clutter measurements
    for i in range(10):
        t = np.random.uniform(0, 2.0)
        x = np.random.uniform(0, 60)
        y = np.random.uniform(0, 40)
        
        measurement = Measurement(
            position=np.array([x, y, 0]),
            timestamp=t,
            covariance=np.diag([1.0, 1.0, 1.0])
        )
        measurements.append(measurement)
    
    # Sort measurements by timestamp
    measurements.sort(key=lambda m: m.timestamp)
    
    return tracker, measurements


if __name__ == "__main__":
    # Demonstration of integrated tracking systems
    
    logger.info("Creating sample tracking scenario...")
    tracker, measurements = create_sample_scenario()
    
    # Process measurements in batches
    batch_size = 5
    total_time = 0.0
    
    for i in range(0, len(measurements), batch_size):
        batch = measurements[i:i+batch_size]
        
        if batch:
            batch_time = batch[-1].timestamp
            
            # Predict to batch time
            start_time = time.time()
            tracker.predict(batch_time)
            
            # Update with batch measurements
            tracker.update(batch)
            
            processing_time = time.time() - start_time
            total_time += processing_time
            
            # Print status
            track_count = tracker.get_track_count()
            logger.info(f"Time {batch_time:.1f}: {track_count['confirmed']} confirmed tracks, "
                       f"{track_count['tentative']} tentative tracks, "
                       f"Processing time: {processing_time*1000:.1f}ms")
    
    logger.info(f"Total processing time: {total_time*1000:.1f}ms")
    logger.info(f"Average time per measurement: {total_time/len(measurements)*1000:.1f}ms")
    
    # Get final track states
    final_states = tracker.get_track_states()
    logger.info(f"Final confirmed tracks: {len(final_states)}")
    
    for track_id, state in final_states.items():
        logger.info(f"Track {track_id}: position=({state[0]:.1f}, {state[1]:.1f}), "
                   f"velocity=({state[2]:.1f}, {state[3]:.1f})")