"""
Base classes and enums for radar tracking systems.

This module provides the foundational components for implementing radar tracking
algorithms, including track state management, base classes for tracks and trackers,
measurement handling, and performance metrics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import numpy.typing as npt
from collections import deque
import uuid
import time


class TrackState(Enum):
    """Enumeration for track states in multi-target tracking."""
    
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed" 
    DELETED = "deleted"


@dataclass
class Measurement:
    """
    Represents a radar measurement/observation.
    
    Attributes:
        position: Measured position vector [x, y, z] in meters
        velocity: Measured velocity vector [vx, vy, vz] in m/s (optional)
        timestamp: Time of measurement in seconds
        covariance: Measurement noise covariance matrix
        snr: Signal-to-noise ratio in dB (optional)
        range_rate: Radial velocity in m/s (optional)
        azimuth: Azimuth angle in radians (optional)
        elevation: Elevation angle in radians (optional)
        metadata: Additional measurement information
    """
    
    position: npt.NDArray[np.float64]
    timestamp: float
    covariance: npt.NDArray[np.float64]
    velocity: Optional[npt.NDArray[np.float64]] = None
    snr: Optional[float] = None
    range_rate: Optional[float] = None
    azimuth: Optional[float] = None
    elevation: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate measurement data after initialization."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)
        
        if self.velocity is not None:
            self.velocity = np.asarray(self.velocity, dtype=np.float64)
            
        # Validate dimensions
        if self.position.ndim != 1:
            raise ValueError("Position must be a 1D array")
        if self.covariance.ndim != 2 or self.covariance.shape[0] != self.covariance.shape[1]:
            raise ValueError("Covariance must be a square 2D array")
        if self.velocity is not None and self.velocity.shape != self.position.shape:
            raise ValueError("Velocity must have same shape as position")


class Track:
    """
    Represents a single target track with state history and quality metrics.
    
    This class maintains the complete state of a tracked target including
    its kinematic state, covariance, measurement history, and quality metrics.
    """
    
    def __init__(
        self,
        track_id: Optional[str] = None,
        initial_state: Optional[npt.NDArray[np.float64]] = None,
        initial_covariance: Optional[npt.NDArray[np.float64]] = None,
        max_history: int = 100
    ):
        """
        Initialize a new track.
        
        Args:
            track_id: Unique identifier for the track. If None, generates UUID.
            initial_state: Initial state vector [x, y, z, vx, vy, vz, ...]
            initial_covariance: Initial state covariance matrix
            max_history: Maximum number of states/measurements to store
        """
        self.track_id = track_id or str(uuid.uuid4())
        self.state = TrackState.TENTATIVE
        self.creation_time = time.time()
        self.last_update_time = self.creation_time
        self.max_history = max_history
        
        # State information
        self._state_vector: Optional[npt.NDArray[np.float64]] = None
        self._covariance: Optional[npt.NDArray[np.float64]] = None
        
        if initial_state is not None:
            self.state_vector = initial_state
        if initial_covariance is not None:
            self.covariance = initial_covariance
            
        # History storage using deques for efficient append/pop operations
        self.state_history: deque = deque(maxlen=max_history)
        self.covariance_history: deque = deque(maxlen=max_history)
        self.measurement_history: deque = deque(maxlen=max_history)
        self.timestamp_history: deque = deque(maxlen=max_history)
        
        # Quality metrics
        self.track_score: float = 0.0
        self.likelihood: float = 0.0
        self.hit_count: int = 0
        self.miss_count: int = 0
        self.age: int = 0  # Number of update cycles
        self.time_since_update: float = 0.0
        
        # Gating and association
        self.gate_size: float = 9.21  # Chi-squared threshold for 99% confidence (3 DOF)
        self.association_threshold: float = 0.1
        
    @property
    def state_vector(self) -> Optional[npt.NDArray[np.float64]]:
        """Get the current state vector."""
        return self._state_vector
    
    @state_vector.setter
    def state_vector(self, value: npt.NDArray[np.float64]) -> None:
        """Set the state vector with validation."""
        self._state_vector = np.asarray(value, dtype=np.float64)
        
    @property
    def covariance(self) -> Optional[npt.NDArray[np.float64]]:
        """Get the current covariance matrix."""
        return self._covariance
    
    @covariance.setter
    def covariance(self, value: npt.NDArray[np.float64]) -> None:
        """Set the covariance matrix with validation."""
        cov = np.asarray(value, dtype=np.float64)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("Covariance must be a square 2D array")
        self._covariance = cov
        
    @property
    def position(self) -> Optional[npt.NDArray[np.float64]]:
        """Extract position from state vector."""
        if self._state_vector is None:
            return None
        # Assume first 3 elements are position [x, y, z]
        return self._state_vector[:3]
    
    @property
    def velocity(self) -> Optional[npt.NDArray[np.float64]]:
        """Extract velocity from state vector."""
        if self._state_vector is None or len(self._state_vector) < 6:
            return None
        # Assume elements 3-5 are velocity [vx, vy, vz]
        return self._state_vector[3:6]
    
    def update_state(
        self,
        new_state: npt.NDArray[np.float64],
        new_covariance: npt.NDArray[np.float64],
        measurement: Optional[Measurement] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Update the track state and add to history.
        
        Args:
            new_state: New state vector
            new_covariance: New covariance matrix
            measurement: Associated measurement (optional)
            timestamp: Time of update (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Store current state in history before updating
        if self._state_vector is not None:
            self.state_history.append(self._state_vector.copy())
        if self._covariance is not None:
            self.covariance_history.append(self._covariance.copy())
        
        # Update current state
        self.state_vector = new_state
        self.covariance = new_covariance
        self.last_update_time = timestamp
        self.timestamp_history.append(timestamp)
        
        if measurement is not None:
            self.measurement_history.append(measurement)
            self.hit_count += 1
        else:
            self.miss_count += 1
            
        self.age += 1
        self.time_since_update = 0.0
        
    def predict_to_time(self, target_time: float) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Predict track state to a future time using constant velocity model.
        
        Args:
            target_time: Target time for prediction
            
        Returns:
            Tuple of (predicted_state, predicted_covariance)
            
        Raises:
            ValueError: If track has no current state
        """
        if self._state_vector is None or self._covariance is None:
            raise ValueError("Cannot predict: track has no current state")
            
        dt = target_time - self.last_update_time
        
        if dt <= 0:
            return self._state_vector.copy(), self._covariance.copy()
            
        # Simple constant velocity prediction
        state_dim = len(self._state_vector)
        F = np.eye(state_dim)
        
        # Assume state is [x, y, z, vx, vy, vz, ...] format
        if state_dim >= 6:
            F[0, 3] = dt  # x += vx * dt
            F[1, 4] = dt  # y += vy * dt
            F[2, 5] = dt  # z += vz * dt
            
        predicted_state = F @ self._state_vector
        
        # Process noise (simple model)
        Q = np.eye(state_dim) * 0.1 * dt**2
        predicted_covariance = F @ self._covariance @ F.T + Q
        
        return predicted_state, predicted_covariance
    
    def update_quality_metrics(self, likelihood: float, innovation: npt.NDArray[np.float64]) -> None:
        """
        Update track quality metrics.
        
        Args:
            likelihood: Measurement likelihood
            innovation: Innovation (measurement residual)
        """
        self.likelihood = likelihood
        
        # Update track score using exponential smoothing
        alpha = 0.1
        quality_score = np.exp(-0.5 * np.linalg.norm(innovation))
        self.track_score = alpha * quality_score + (1 - alpha) * self.track_score
        
    def is_confirmed(self) -> bool:
        """Check if track meets confirmation criteria."""
        return (self.hit_count >= 3 and 
                self.hit_count / max(self.age, 1) > 0.6 and
                self.track_score > 0.5)
    
    def should_delete(self, max_time_since_update: float = 5.0, min_track_score: float = 0.1) -> bool:
        """
        Check if track should be deleted based on quality criteria.
        
        Args:
            max_time_since_update: Maximum time without updates (seconds)
            min_track_score: Minimum acceptable track score
            
        Returns:
            True if track should be deleted
        """
        current_time = time.time()
        time_since_update = current_time - self.last_update_time
        
        return (time_since_update > max_time_since_update or
                self.track_score < min_track_score or
                (self.age > 10 and self.hit_count / self.age < 0.3))
    
    def get_state_at_time(self, timestamp: float) -> Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """
        Get interpolated state at a specific time.
        
        Args:
            timestamp: Desired timestamp
            
        Returns:
            Tuple of (state, covariance) at timestamp, or None if cannot interpolate
        """
        if not self.timestamp_history or not self.state_history:
            return None
            
        timestamps = list(self.timestamp_history)
        
        # Find bounding timestamps
        if timestamp <= timestamps[0]:
            return self.state_history[0].copy(), self.covariance_history[0].copy()
        elif timestamp >= timestamps[-1]:
            return self.predict_to_time(timestamp)
        
        # Linear interpolation between states
        for i in range(len(timestamps) - 1):
            if timestamps[i] <= timestamp <= timestamps[i + 1]:
                t1, t2 = timestamps[i], timestamps[i + 1]
                s1, s2 = self.state_history[i], self.state_history[i + 1]
                c1, c2 = self.covariance_history[i], self.covariance_history[i + 1]
                
                alpha = (timestamp - t1) / (t2 - t1)
                interpolated_state = (1 - alpha) * s1 + alpha * s2
                interpolated_cov = (1 - alpha) * c1 + alpha * c2
                
                return interpolated_state, interpolated_cov
                
        return None


class BaseTracker(ABC):
    """
    Abstract base class for radar tracking algorithms.
    
    This class defines the interface that all tracking algorithms must implement,
    including prediction, update, track management, and gating operations.
    """
    
    def __init__(
        self,
        max_tracks: int = 1000,
        gate_threshold: float = 9.21,
        track_confirmation_threshold: int = 3,
        track_deletion_threshold: int = 5
    ):
        """
        Initialize the base tracker.
        
        Args:
            max_tracks: Maximum number of simultaneous tracks
            gate_threshold: Chi-squared threshold for measurement gating
            track_confirmation_threshold: Number of hits needed for confirmation
            track_deletion_threshold: Number of misses before deletion
        """
        self.max_tracks = max_tracks
        self.gate_threshold = gate_threshold
        self.track_confirmation_threshold = track_confirmation_threshold
        self.track_deletion_threshold = track_deletion_threshold
        
        # Track storage
        self.tracks: Dict[str, Track] = {}
        self.confirmed_tracks: Dict[str, Track] = {}
        self.tentative_tracks: Dict[str, Track] = {}
        
        # Performance metrics
        self.tracking_metrics = TrackingMetrics()
        
    @abstractmethod
    def predict(self, timestamp: float) -> None:
        """
        Predict all tracks to the given timestamp.
        
        Args:
            timestamp: Target time for prediction
        """
        pass
    
    @abstractmethod
    def update(self, measurements: List[Measurement]) -> None:
        """
        Update tracks with new measurements.
        
        Args:
            measurements: List of new measurements
        """
        pass
    
    @abstractmethod
    def initiate_track(self, measurement: Measurement) -> Track:
        """
        Initiate a new track from a measurement.
        
        Args:
            measurement: Measurement to initiate track from
            
        Returns:
            New track object
        """
        pass
    
    def gate_measurements(
        self,
        track: Track,
        measurements: List[Measurement],
        timestamp: float
    ) -> List[Tuple[Measurement, float]]:
        """
        Gate measurements for a track using Mahalanobis distance.
        
        Args:
            track: Track to gate measurements for
            measurements: List of candidate measurements
            timestamp: Current timestamp
            
        Returns:
            List of (measurement, distance) pairs within gate
        """
        if track.state_vector is None or track.covariance is None:
            return []
            
        # Predict track to measurement time
        predicted_state, predicted_cov = track.predict_to_time(timestamp)
        predicted_position = predicted_state[:3]
        
        gated_measurements = []
        
        for measurement in measurements:
            # Calculate innovation
            innovation = measurement.position - predicted_position
            
            # Innovation covariance (prediction + measurement noise)
            H = np.eye(3, len(predicted_state))  # Observation matrix
            innovation_cov = H @ predicted_cov @ H.T + measurement.covariance
            
            # Mahalanobis distance
            try:
                inv_cov = np.linalg.inv(innovation_cov)
                distance = innovation.T @ inv_cov @ innovation
                
                if distance <= self.gate_threshold:
                    gated_measurements.append((measurement, distance))
                    
            except np.linalg.LinAlgError:
                # Singular covariance matrix, skip this measurement
                continue
                
        return sorted(gated_measurements, key=lambda x: x[1])
    
    def manage_tracks(self) -> None:
        """Manage track states (confirmation, deletion, pruning)."""
        tracks_to_delete = []
        
        for track_id, track in self.tracks.items():
            # Update time since last update
            current_time = time.time()
            track.time_since_update = current_time - track.last_update_time
            
            # Check for confirmation
            if track.state == TrackState.TENTATIVE and track.is_confirmed():
                track.state = TrackState.CONFIRMED
                self.confirmed_tracks[track_id] = track
                if track_id in self.tentative_tracks:
                    del self.tentative_tracks[track_id]
                    
            # Check for deletion
            elif track.should_delete():
                track.state = TrackState.DELETED
                tracks_to_delete.append(track_id)
                
        # Remove deleted tracks
        for track_id in tracks_to_delete:
            self._delete_track(track_id)
            
        # Prune excess tracks if needed
        self._prune_tracks()
    
    def _delete_track(self, track_id: str) -> None:
        """Remove a track from all storage dictionaries."""
        self.tracks.pop(track_id, None)
        self.confirmed_tracks.pop(track_id, None)
        self.tentative_tracks.pop(track_id, None)
        
    def _prune_tracks(self) -> None:
        """Remove excess tracks if over limit."""
        if len(self.tracks) <= self.max_tracks:
            return
            
        # Sort tracks by quality score (ascending) and remove worst
        sorted_tracks = sorted(
            self.tracks.items(),
            key=lambda x: x[1].track_score
        )
        
        num_to_remove = len(self.tracks) - self.max_tracks
        for i in range(num_to_remove):
            track_id = sorted_tracks[i][0]
            self._delete_track(track_id)
    
    def add_track(self, track: Track) -> None:
        """
        Add a new track to the tracker.
        
        Args:
            track: Track to add
        """
        self.tracks[track.track_id] = track
        if track.state == TrackState.TENTATIVE:
            self.tentative_tracks[track.track_id] = track
        elif track.state == TrackState.CONFIRMED:
            self.confirmed_tracks[track.track_id] = track
            
    def get_track_states(self, timestamp: Optional[float] = None) -> Dict[str, npt.NDArray[np.float64]]:
        """
        Get current states of all confirmed tracks.
        
        Args:
            timestamp: Time to predict states to (optional)
            
        Returns:
            Dictionary mapping track IDs to state vectors
        """
        states = {}
        target_time = timestamp or time.time()
        
        for track_id, track in self.confirmed_tracks.items():
            if track.state_vector is not None:
                try:
                    predicted_state, _ = track.predict_to_time(target_time)
                    states[track_id] = predicted_state
                except ValueError:
                    continue
                    
        return states
    
    def get_track_count(self) -> Dict[str, int]:
        """Get count of tracks by state."""
        return {
            'total': len(self.tracks),
            'confirmed': len(self.confirmed_tracks),
            'tentative': len(self.tentative_tracks),
            'deleted': len([t for t in self.tracks.values() if t.state == TrackState.DELETED])
        }


@dataclass
class TrackingMetrics:
    """
    Performance metrics for tracking systems.
    
    This class accumulates and computes various performance metrics
    for evaluating tracking algorithm performance.
    """
    
    # Detection metrics
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    
    # Track metrics
    total_tracks_initiated: int = 0
    total_tracks_confirmed: int = 0
    total_tracks_deleted: int = 0
    
    # Timing metrics
    average_track_lifetime: float = 0.0
    average_processing_time: float = 0.0
    
    # OSPA metrics storage
    ospa_distances: List[float] = field(default_factory=list)
    ospa_cardinality_errors: List[float] = field(default_factory=list)
    
    # Position error metrics
    position_errors: List[float] = field(default_factory=list)
    velocity_errors: List[float] = field(default_factory=list)
    
    def update_detection_metrics(
        self,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
        tn: int = 0
    ) -> None:
        """Update detection-level metrics."""
        self.true_positives += tp
        self.false_positives += fp
        self.false_negatives += fn
        self.true_negatives += tn
    
    def update_track_metrics(
        self,
        initiated: int = 0,
        confirmed: int = 0,
        deleted: int = 0
    ) -> None:
        """Update track-level metrics."""
        self.total_tracks_initiated += initiated
        self.total_tracks_confirmed += confirmed
        self.total_tracks_deleted += deleted
    
    def add_ospa_metrics(self, distance: float, cardinality_error: float) -> None:
        """Add OSPA (Optimal Sub-Pattern Assignment) metrics."""
        self.ospa_distances.append(distance)
        self.ospa_cardinality_errors.append(cardinality_error)
    
    def add_position_error(self, error: float) -> None:
        """Add position error measurement."""
        self.position_errors.append(error)
    
    def add_velocity_error(self, error: float) -> None:
        """Add velocity error measurement."""
        self.velocity_errors.append(error)
    
    def compute_detection_rate(self) -> float:
        """Compute detection rate (sensitivity/recall)."""
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else 0.0
    
    def compute_false_alarm_rate(self) -> float:
        """Compute false alarm rate."""
        denominator = self.false_positives + self.true_negatives
        return self.false_positives / denominator if denominator > 0 else 0.0
    
    def compute_precision(self) -> float:
        """Compute precision (positive predictive value)."""
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 0.0
    
    def compute_f1_score(self) -> float:
        """Compute F1 score (harmonic mean of precision and recall)."""
        precision = self.compute_precision()
        recall = self.compute_detection_rate()
        denominator = precision + recall
        return 2 * precision * recall / denominator if denominator > 0 else 0.0
    
    def compute_track_completion_rate(self) -> float:
        """Compute rate of tracks that were successfully confirmed."""
        return (self.total_tracks_confirmed / self.total_tracks_initiated 
                if self.total_tracks_initiated > 0 else 0.0)
    
    def compute_average_ospa_distance(self) -> float:
        """Compute average OSPA distance."""
        return np.mean(self.ospa_distances) if self.ospa_distances else 0.0
    
    def compute_average_position_error(self) -> float:
        """Compute average position error."""
        return np.mean(self.position_errors) if self.position_errors else 0.0
    
    def compute_average_velocity_error(self) -> float:
        """Compute average velocity error."""
        return np.mean(self.velocity_errors) if self.velocity_errors else 0.0
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all computed metrics."""
        return {
            'detection_rate': self.compute_detection_rate(),
            'false_alarm_rate': self.compute_false_alarm_rate(),
            'precision': self.compute_precision(),
            'f1_score': self.compute_f1_score(),
            'track_completion_rate': self.compute_track_completion_rate(),
            'average_ospa_distance': self.compute_average_ospa_distance(),
            'average_position_error': self.compute_average_position_error(),
            'average_velocity_error': self.compute_average_velocity_error(),
            'total_tracks': self.total_tracks_initiated,
            'confirmed_tracks': self.total_tracks_confirmed,
            'deleted_tracks': self.total_tracks_deleted
        }
    
    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.true_negatives = 0
        self.total_tracks_initiated = 0
        self.total_tracks_confirmed = 0
        self.total_tracks_deleted = 0
        self.average_track_lifetime = 0.0
        self.average_processing_time = 0.0
        self.ospa_distances.clear()
        self.ospa_cardinality_errors.clear()
        self.position_errors.clear()
        self.velocity_errors.clear()