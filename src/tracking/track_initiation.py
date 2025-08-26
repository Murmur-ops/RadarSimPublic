"""
Track Initiation Module for RadarSim

This module provides comprehensive track initiation algorithms for radar and sonar systems,
including M-out-of-N logic, Sequential Probability Ratio Test (SPRT), Track-Before-Detect (TBD),
and various initialization methods for different sensor types.

Author: RadarSim Development Team
"""

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union, Any
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Enumeration of supported sensor types."""
    RADAR = "radar"
    SONAR = "sonar"
    LIDAR = "lidar"
    INFRARED = "infrared"
    OPTICAL = "optical"


class TrackStatus(Enum):
    """Track status enumeration."""
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    TERMINATED = "terminated"


@dataclass
class Measurement:
    """Measurement data structure."""
    timestamp: float
    position: np.ndarray  # [x, y, z] or [range, azimuth, elevation]
    velocity: Optional[np.ndarray] = None
    acceleration: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None
    snr: Optional[float] = None
    sensor_id: Optional[str] = None
    measurement_id: Optional[str] = None


@dataclass
class Track:
    """Track data structure."""
    track_id: str
    status: TrackStatus
    measurements: List[Measurement]
    state_estimate: np.ndarray
    covariance_matrix: np.ndarray
    creation_time: float
    last_update_time: float
    quality_score: float
    m_out_of_n_count: int = 0
    sprt_log_likelihood: float = 0.0


class TrackInitiator:
    """
    Comprehensive track initiation system supporting multiple algorithms
    and sensor types.
    """
    
    def __init__(self, 
                 sensor_type: SensorType = SensorType.RADAR,
                 m_out_of_n_params: Dict[str, int] = None,
                 sprt_params: Dict[str, float] = None,
                 tbd_params: Dict[str, Any] = None,
                 gate_params: Dict[str, float] = None):
        """
        Initialize the track initiator.
        
        Args:
            sensor_type: Type of sensor (radar, sonar, etc.)
            m_out_of_n_params: Parameters for M-out-of-N logic
            sprt_params: Parameters for SPRT algorithm
            tbd_params: Parameters for Track-Before-Detect
            gate_params: Parameters for gating and clustering
        """
        self.sensor_type = sensor_type
        
        # Default M-out-of-N parameters
        self.m_out_of_n_params = m_out_of_n_params or {
            'M': 2,  # Number of detections required
            'N': 3,  # Out of last N scans
            'confirmation_threshold': 0.8
        }
        
        # Default SPRT parameters
        self.sprt_params = sprt_params or {
            'alpha': 0.05,  # False alarm probability
            'beta': 0.05,   # Miss detection probability
            'pd_target': 0.9,  # Target detection probability
            'pd_clutter': 0.1, # Clutter detection probability
            'upper_threshold': 2.94,  # ln((1-beta)/alpha)
            'lower_threshold': -2.94  # ln(beta/(1-alpha))
        }
        
        # Default TBD parameters
        self.tbd_params = tbd_params or {
            'num_particles': 1000,
            'dynamic_noise_std': 1.0,
            'measurement_noise_std': 0.5,
            'survival_probability': 0.99,
            'birth_probability': 0.1,
            'detection_threshold': 0.7
        }
        
        # Default gate parameters
        self.gate_params = gate_params or {
            'gate_size': 9.21,  # Chi-square 99% confidence for 2D
            'max_gate_distance': 50.0,
            'min_cluster_size': 2
        }
        
        self.active_tracks: List[Track] = []
        self.tentative_tracks: List[Track] = []
        self.track_counter = 0
    
    def process_measurements(self, measurements: List[Measurement]) -> List[Track]:
        """
        Process incoming measurements and update tracks.
        
        Args:
            measurements: List of measurements from sensors
            
        Returns:
            List of updated tracks
        """
        # Cluster measurements
        clustered_measurements = self._cluster_measurements(measurements)
        
        # Process each cluster
        for cluster in clustered_measurements:
            # Try to associate with existing tracks
            associated = self._associate_measurements(cluster)
            
            # Handle unassociated measurements
            unassociated = [m for m in cluster if not associated.get(m.measurement_id)]
            
            # Initialize new tracks from unassociated measurements
            if unassociated:
                self._initialize_new_tracks(unassociated)
        
        # Update track statuses
        self._update_track_statuses()
        
        # Prune poor quality tracks
        self._prune_tracks()
        
        return self.active_tracks + self.tentative_tracks
    
    def _cluster_measurements(self, measurements: List[Measurement]) -> List[List[Measurement]]:
        """
        Cluster measurements using gate-based clustering.
        
        Args:
            measurements: List of measurements
            
        Returns:
            List of measurement clusters
        """
        if not measurements:
            return []
        
        # Extract positions for clustering
        positions = np.array([m.position[:2] for m in measurements])  # Use x,y for 2D clustering
        
        # Compute pairwise distances
        distances = cdist(positions, positions)
        
        # Apply gate-based clustering
        clusters = []
        visited = set()
        
        for i, measurement in enumerate(measurements):
            if i in visited:
                continue
                
            cluster = [measurement]
            visited.add(i)
            
            # Find all measurements within gate distance
            for j, other_measurement in enumerate(measurements):
                if j != i and j not in visited:
                    if distances[i, j] <= self.gate_params['max_gate_distance']:
                        cluster.append(other_measurement)
                        visited.add(j)
            
            if len(cluster) >= self.gate_params['min_cluster_size']:
                clusters.append(cluster)
            else:
                # Handle single measurements
                clusters.append([measurement])
        
        return clusters
    
    def _associate_measurements(self, measurements: List[Measurement]) -> Dict[str, bool]:
        """
        Associate measurements with existing tracks.
        
        Args:
            measurements: List of measurements
            
        Returns:
            Dictionary indicating which measurements were associated
        """
        associated = {}
        
        for measurement in measurements:
            measurement_pos = measurement.position[:2]
            best_track = None
            min_distance = float('inf')
            
            # Check association with existing tracks
            for track in self.active_tracks + self.tentative_tracks:
                if not track.measurements:
                    continue
                    
                # Predict track position
                predicted_pos = self._predict_track_position(track)
                
                # Compute Mahalanobis distance
                distance = self._compute_mahalanobis_distance(
                    measurement_pos, predicted_pos, track.covariance_matrix[:2, :2]
                )
                
                # Check if within gate
                if distance <= self.gate_params['gate_size'] and distance < min_distance:
                    min_distance = distance
                    best_track = track
            
            # Associate measurement with best track
            if best_track is not None:
                self._update_track_with_measurement(best_track, measurement)
                associated[measurement.measurement_id] = True
            else:
                associated[measurement.measurement_id] = False
        
        return associated
    
    def _initialize_new_tracks(self, measurements: List[Measurement]) -> None:
        """
        Initialize new tracks from unassociated measurements.
        
        Args:
            measurements: List of unassociated measurements
        """
        for measurement in measurements:
            # Create new tentative track
            track_id = f"track_{self.track_counter:06d}"
            self.track_counter += 1
            
            # Initialize state estimate
            if len(measurement.position) >= 2:
                # Use two-point differencing if velocity is available
                if measurement.velocity is not None:
                    state_estimate = np.concatenate([
                        measurement.position[:2],
                        measurement.velocity[:2]
                    ])
                else:
                    state_estimate = np.concatenate([
                        measurement.position[:2],
                        np.zeros(2)  # Zero initial velocity
                    ])
            else:
                state_estimate = np.array([measurement.position[0], 0, 0, 0])
            
            # Initialize covariance matrix
            covariance_matrix = self._initialize_covariance_matrix(measurement)
            
            # Create track
            track = Track(
                track_id=track_id,
                status=TrackStatus.TENTATIVE,
                measurements=[measurement],
                state_estimate=state_estimate,
                covariance_matrix=covariance_matrix,
                creation_time=measurement.timestamp,
                last_update_time=measurement.timestamp,
                quality_score=self._compute_initial_quality_score(measurement),
                m_out_of_n_count=1,
                sprt_log_likelihood=0.0
            )
            
            self.tentative_tracks.append(track)
            logger.info(f"Initialized new tentative track: {track_id}")
    
    def _update_track_with_measurement(self, track: Track, measurement: Measurement) -> None:
        """
        Update an existing track with a new measurement.
        
        Args:
            track: Track to update
            measurement: New measurement
        """
        track.measurements.append(measurement)
        track.last_update_time = measurement.timestamp
        
        # Update M-out-of-N count
        track.m_out_of_n_count += 1
        
        # Update SPRT log-likelihood
        self._update_sprt_likelihood(track, measurement)
        
        # Update state estimate using Kalman filter-like update
        self._update_state_estimate(track, measurement)
        
        # Update quality score
        track.quality_score = self._compute_track_quality(track)
    
    def _update_track_statuses(self) -> None:
        """Update track statuses based on M-out-of-N and SPRT criteria."""
        
        # Check tentative tracks for confirmation
        confirmed_tracks = []
        for track in self.tentative_tracks[:]:
            if self._should_confirm_track_m_out_of_n(track) or self._should_confirm_track_sprt(track):
                track.status = TrackStatus.CONFIRMED
                self.tentative_tracks.remove(track)
                self.active_tracks.append(track)
                confirmed_tracks.append(track)
                logger.info(f"Confirmed track: {track.track_id}")
        
        # Check for track termination
        terminated_tracks = []
        for track_list in [self.active_tracks, self.tentative_tracks]:
            for track in track_list[:]:
                if self._should_terminate_track(track):
                    track.status = TrackStatus.TERMINATED
                    track_list.remove(track)
                    terminated_tracks.append(track)
                    logger.info(f"Terminated track: {track.track_id}")
    
    def _should_confirm_track_m_out_of_n(self, track: Track) -> bool:
        """
        Check if track should be confirmed using M-out-of-N logic.
        
        Args:
            track: Track to check
            
        Returns:
            True if track should be confirmed
        """
        if len(track.measurements) < self.m_out_of_n_params['N']:
            return False
        
        # Count detections in last N scans
        recent_measurements = track.measurements[-self.m_out_of_n_params['N']:]
        detection_count = len(recent_measurements)
        
        return detection_count >= self.m_out_of_n_params['M']
    
    def _should_confirm_track_sprt(self, track: Track) -> bool:
        """
        Check if track should be confirmed using SPRT.
        
        Args:
            track: Track to check
            
        Returns:
            True if track should be confirmed
        """
        return track.sprt_log_likelihood >= self.sprt_params['upper_threshold']
    
    def _should_terminate_track(self, track: Track) -> bool:
        """
        Check if track should be terminated.
        
        Args:
            track: Track to check
            
        Returns:
            True if track should be terminated
        """
        # Terminate based on SPRT
        if track.sprt_log_likelihood <= self.sprt_params['lower_threshold']:
            return True
        
        # Terminate based on quality score
        if track.quality_score < 0.3:
            return True
        
        # Terminate if no updates for too long
        time_since_update = track.measurements[-1].timestamp - track.last_update_time
        if time_since_update > 5.0:  # 5 time units without update
            return True
        
        return False
    
    def _update_sprt_likelihood(self, track: Track, measurement: Measurement) -> None:
        """
        Update SPRT log-likelihood ratio for a track.
        
        Args:
            track: Track to update
            measurement: New measurement
        """
        # Compute likelihood ratio
        pd_target = self.sprt_params['pd_target']
        pd_clutter = self.sprt_params['pd_clutter']
        
        # Simple likelihood based on SNR if available
        if measurement.snr is not None:
            # Higher SNR increases likelihood of target
            snr_factor = min(measurement.snr / 10.0, 1.0)
            likelihood_ratio = (pd_target * snr_factor) / pd_clutter
        else:
            # Use default likelihood ratio
            likelihood_ratio = pd_target / pd_clutter
        
        # Update log-likelihood
        track.sprt_log_likelihood += np.log(likelihood_ratio)
    
    def _predict_track_position(self, track: Track) -> np.ndarray:
        """
        Predict track position at current time.
        
        Args:
            track: Track to predict
            
        Returns:
            Predicted position
        """
        if len(track.measurements) < 2:
            return track.state_estimate[:2]
        
        # Simple constant velocity prediction
        last_measurement = track.measurements[-1]
        second_last_measurement = track.measurements[-2]
        
        dt = last_measurement.timestamp - second_last_measurement.timestamp
        velocity = (last_measurement.position[:2] - second_last_measurement.position[:2]) / dt
        
        # Predict forward by dt (assuming regular sampling)
        predicted_position = last_measurement.position[:2] + velocity * dt
        
        return predicted_position
    
    def _compute_mahalanobis_distance(self, 
                                    measurement_pos: np.ndarray, 
                                    predicted_pos: np.ndarray, 
                                    covariance: np.ndarray) -> float:
        """
        Compute Mahalanobis distance between measurement and prediction.
        
        Args:
            measurement_pos: Measurement position
            predicted_pos: Predicted position
            covariance: Covariance matrix
            
        Returns:
            Mahalanobis distance
        """
        try:
            diff = measurement_pos - predicted_pos
            inv_cov = np.linalg.inv(covariance + np.eye(len(covariance)) * 1e-6)
            distance = np.sqrt(diff.T @ inv_cov @ diff)
            return distance
        except np.linalg.LinAlgError:
            # Fallback to Euclidean distance
            return np.linalg.norm(measurement_pos - predicted_pos)
    
    def _initialize_covariance_matrix(self, measurement: Measurement) -> np.ndarray:
        """
        Initialize covariance matrix for new track.
        
        Args:
            measurement: Initial measurement
            
        Returns:
            Initial covariance matrix
        """
        if measurement.covariance is not None:
            # Use measurement covariance if available
            meas_cov = measurement.covariance
            if meas_cov.shape[0] >= 4:
                return meas_cov[:4, :4]
            else:
                # Expand to 4x4 for position and velocity
                cov = np.eye(4) * 10.0
                cov[:meas_cov.shape[0], :meas_cov.shape[1]] = meas_cov
                return cov
        else:
            # Default covariance based on sensor type
            if self.sensor_type == SensorType.RADAR:
                return np.diag([1.0, 1.0, 5.0, 5.0])  # [x, y, vx, vy]
            elif self.sensor_type == SensorType.SONAR:
                return np.diag([2.0, 2.0, 3.0, 3.0])
            else:
                return np.diag([0.5, 0.5, 2.0, 2.0])
    
    def _update_state_estimate(self, track: Track, measurement: Measurement) -> None:
        """
        Update track state estimate with new measurement.
        
        Args:
            track: Track to update
            measurement: New measurement
        """
        # Simple Kalman filter-like update
        # This is a simplified version - full implementation would use proper Kalman filtering
        
        if len(track.measurements) >= 2:
            # Use two-point differencing for velocity estimation
            current_pos = measurement.position[:2]
            prev_measurement = track.measurements[-2]
            prev_pos = prev_measurement.position[:2]
            dt = measurement.timestamp - prev_measurement.timestamp
            
            if dt > 0:
                velocity = (current_pos - prev_pos) / dt
                track.state_estimate[:2] = current_pos
                track.state_estimate[2:4] = velocity
        
        if len(track.measurements) >= 3:
            # Use three-point initialization for acceleration estimation
            acceleration = self._estimate_acceleration(track.measurements[-3:])
            if len(track.state_estimate) >= 6:
                track.state_estimate[4:6] = acceleration
    
    def _estimate_acceleration(self, measurements: List[Measurement]) -> np.ndarray:
        """
        Estimate acceleration using three-point differencing.
        
        Args:
            measurements: List of at least 3 measurements
            
        Returns:
            Estimated acceleration
        """
        if len(measurements) < 3:
            return np.zeros(2)
        
        # Extract positions and times
        positions = np.array([m.position[:2] for m in measurements[-3:]])
        times = np.array([m.timestamp for m in measurements[-3:]])
        
        # Compute velocities
        dt1 = times[1] - times[0]
        dt2 = times[2] - times[1]
        
        if dt1 <= 0 or dt2 <= 0:
            return np.zeros(2)
        
        v1 = (positions[1] - positions[0]) / dt1
        v2 = (positions[2] - positions[1]) / dt2
        
        # Compute acceleration
        acceleration = (v2 - v1) / ((dt1 + dt2) / 2)
        
        return acceleration
    
    def _compute_initial_quality_score(self, measurement: Measurement) -> float:
        """
        Compute initial quality score for a new track.
        
        Args:
            measurement: Initial measurement
            
        Returns:
            Quality score [0, 1]
        """
        score = 0.5  # Base score
        
        # Boost score based on SNR
        if measurement.snr is not None:
            snr_factor = min(measurement.snr / 20.0, 0.4)
            score += snr_factor
        
        # Sensor-specific adjustments
        if self.sensor_type == SensorType.RADAR:
            score += 0.1
        elif self.sensor_type == SensorType.SONAR:
            score += 0.05
        
        return min(score, 1.0)
    
    def _compute_track_quality(self, track: Track) -> float:
        """
        Compute quality score for an existing track.
        
        Args:
            track: Track to evaluate
            
        Returns:
            Quality score [0, 1]
        """
        base_score = 0.3
        
        # Factor in number of measurements
        measurement_factor = min(len(track.measurements) / 10.0, 0.3)
        base_score += measurement_factor
        
        # Factor in consistency of measurements
        if len(track.measurements) >= 3:
            consistency_score = self._compute_track_consistency(track)
            base_score += consistency_score * 0.3
        
        # Factor in SPRT likelihood
        sprt_factor = min(max(track.sprt_log_likelihood / 5.0, -0.2), 0.2)
        base_score += sprt_factor
        
        # Factor in average SNR if available
        snrs = [m.snr for m in track.measurements if m.snr is not None]
        if snrs:
            avg_snr = np.mean(snrs)
            snr_factor = min(avg_snr / 20.0, 0.2)
            base_score += snr_factor
        
        return min(max(base_score, 0.0), 1.0)
    
    def _compute_track_consistency(self, track: Track) -> float:
        """
        Compute track consistency score based on motion model.
        
        Args:
            track: Track to evaluate
            
        Returns:
            Consistency score [0, 1]
        """
        if len(track.measurements) < 3:
            return 0.5
        
        # Compute velocity variations
        velocities = []
        for i in range(1, len(track.measurements)):
            curr_pos = track.measurements[i].position[:2]
            prev_pos = track.measurements[i-1].position[:2]
            dt = track.measurements[i].timestamp - track.measurements[i-1].timestamp
            
            if dt > 0:
                velocity = np.linalg.norm((curr_pos - prev_pos) / dt)
                velocities.append(velocity)
        
        if not velocities:
            return 0.5
        
        # Compute coefficient of variation
        mean_velocity = np.mean(velocities)
        std_velocity = np.std(velocities)
        
        if mean_velocity > 0:
            cv = std_velocity / mean_velocity
            consistency = max(0, 1 - cv)  # Lower variation = higher consistency
        else:
            consistency = 0.5
        
        return consistency
    
    def _prune_tracks(self) -> None:
        """Remove tracks with poor quality scores."""
        min_quality = 0.2
        
        # Prune tentative tracks
        self.tentative_tracks = [
            track for track in self.tentative_tracks 
            if track.quality_score >= min_quality
        ]
        
        # Prune active tracks (more lenient threshold)
        self.active_tracks = [
            track for track in self.active_tracks 
            if track.quality_score >= min_quality * 0.5
        ]


class TrackBeforeDetect:
    """
    Track-Before-Detect (TBD) implementation for weak target detection.
    Includes both Viterbi-based and Particle Filter-based approaches.
    """
    
    def __init__(self, 
                 method: str = "particle_filter",
                 num_particles: int = 1000,
                 state_dim: int = 4,
                 measurement_dim: int = 2):
        """
        Initialize TBD processor.
        
        Args:
            method: TBD method ('viterbi' or 'particle_filter')
            num_particles: Number of particles for PF-TBD
            state_dim: Dimension of state vector
            measurement_dim: Dimension of measurement vector
        """
        self.method = method
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        if method == "particle_filter":
            self._initialize_particle_filter()
        elif method == "viterbi":
            self._initialize_viterbi()
    
    def _initialize_particle_filter(self) -> None:
        """Initialize particle filter for TBD."""
        # Initialize particles randomly
        self.particles = np.random.randn(self.num_particles, self.state_dim)
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Process and measurement noise
        self.Q = np.eye(self.state_dim) * 0.1  # Process noise
        self.R = np.eye(self.measurement_dim) * 1.0  # Measurement noise
    
    def _initialize_viterbi(self) -> None:
        """Initialize Viterbi-based TBD."""
        # State space discretization
        self.state_grid_size = 50
        self.state_bounds = [(-100, 100), (-100, 100), (-10, 10), (-10, 10)]  # [x, y, vx, vy]
        
        # Create state grid
        self.state_grid = self._create_state_grid()
        
        # Transition probabilities
        self.transition_probs = self._compute_transition_probabilities()
    
    def process_tbd(self, measurement_data: np.ndarray, timestamp: float) -> List[Track]:
        """
        Process TBD on measurement data.
        
        Args:
            measurement_data: Raw measurement data (e.g., range-Doppler map)
            timestamp: Measurement timestamp
            
        Returns:
            List of detected tracks
        """
        if self.method == "particle_filter":
            return self._process_pf_tbd(measurement_data, timestamp)
        elif self.method == "viterbi":
            return self._process_viterbi_tbd(measurement_data, timestamp)
        else:
            raise ValueError(f"Unknown TBD method: {self.method}")
    
    def _process_pf_tbd(self, measurement_data: np.ndarray, timestamp: float) -> List[Track]:
        """
        Process Particle Filter TBD.
        
        Args:
            measurement_data: Measurement data
            timestamp: Timestamp
            
        Returns:
            Detected tracks
        """
        # Predict particles
        self._predict_particles()
        
        # Update weights based on measurement likelihood
        self._update_particle_weights(measurement_data)
        
        # Resample particles
        self._resample_particles()
        
        # Extract tracks from particle clusters
        tracks = self._extract_tracks_from_particles(timestamp)
        
        return tracks
    
    def _process_viterbi_tbd(self, measurement_data: np.ndarray, timestamp: float) -> List[Track]:
        """
        Process Viterbi-based TBD.
        
        Args:
            measurement_data: Measurement data
            timestamp: Timestamp
            
        Returns:
            Detected tracks
        """
        # This is a simplified implementation
        # Full Viterbi TBD requires dynamic programming over state sequences
        
        # For now, return empty list - full implementation would require
        # extensive dynamic programming algorithms
        logger.warning("Viterbi TBD not fully implemented in this version")
        return []
    
    def _predict_particles(self) -> None:
        """Predict particle states using motion model."""
        dt = 1.0  # Assume unit time step
        
        # Simple constant velocity model
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Add process noise
        noise = np.random.multivariate_normal(
            np.zeros(self.state_dim), 
            self.Q, 
            self.num_particles
        )
        
        self.particles = (F @ self.particles.T).T + noise
    
    def _update_particle_weights(self, measurement_data: np.ndarray) -> None:
        """Update particle weights based on measurement likelihood."""
        # Extract measurement from data (simplified)
        # In practice, this would involve more sophisticated processing
        
        for i in range(self.num_particles):
            # Compute predicted measurement
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Measurement matrix
            predicted_meas = H @ self.particles[i]
            
            # Compute likelihood (simplified)
            # This should be based on actual measurement data processing
            likelihood = np.exp(-0.5 * np.sum(predicted_meas**2))
            
            self.weights[i] *= likelihood
        
        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)
    
    def _resample_particles(self) -> None:
        """Resample particles based on weights."""
        # Systematic resampling
        indices = np.random.choice(
            self.num_particles, 
            self.num_particles, 
            p=self.weights
        )
        
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def _extract_tracks_from_particles(self, timestamp: float) -> List[Track]:
        """Extract tracks from particle clusters."""
        # Simple clustering based on particle positions
        positions = self.particles[:, :2]
        
        # Use k-means-like clustering (simplified)
        # In practice, would use more sophisticated clustering
        
        tracks = []
        # This is a placeholder - real implementation would cluster particles
        # and create tracks from significant clusters
        
        return tracks
    
    def _create_state_grid(self) -> np.ndarray:
        """Create discretized state grid for Viterbi TBD."""
        # Create grid points for each state dimension
        grids = []
        for bounds in self.state_bounds:
            grid = np.linspace(bounds[0], bounds[1], self.state_grid_size)
            grids.append(grid)
        
        # Create meshgrid
        mesh = np.meshgrid(*grids, indexing='ij')
        
        # Flatten and stack to create state grid
        state_grid = np.stack([m.flatten() for m in mesh], axis=1)
        
        return state_grid
    
    def _compute_transition_probabilities(self) -> np.ndarray:
        """Compute state transition probabilities for Viterbi TBD."""
        num_states = len(self.state_grid)
        transition_probs = np.zeros((num_states, num_states))
        
        # Compute transition probabilities based on motion model
        # This is computationally intensive for large state spaces
        # Simplified implementation
        
        for i in range(num_states):
            for j in range(num_states):
                # Compute probability of transitioning from state i to state j
                # Based on motion model and process noise
                state_diff = self.state_grid[j] - self.state_grid[i]
                prob = np.exp(-0.5 * np.sum(state_diff**2))
                transition_probs[i, j] = prob
        
        # Normalize rows to make proper probability distributions
        row_sums = np.sum(transition_probs, axis=1, keepdims=True)
        transition_probs = transition_probs / (row_sums + 1e-10)
        
        return transition_probs


def create_default_track_initiator(sensor_type: SensorType = SensorType.RADAR) -> TrackInitiator:
    """
    Create a track initiator with default parameters for common use cases.
    
    Args:
        sensor_type: Type of sensor
        
    Returns:
        Configured TrackInitiator instance
    """
    # Sensor-specific parameter tuning
    if sensor_type == SensorType.RADAR:
        m_out_of_n = {'M': 2, 'N': 3, 'confirmation_threshold': 0.8}
        sprt = {
            'alpha': 0.01, 'beta': 0.05,
            'pd_target': 0.9, 'pd_clutter': 0.1,
            'upper_threshold': 4.6, 'lower_threshold': -2.9
        }
        gate = {'gate_size': 9.21, 'max_gate_distance': 30.0, 'min_cluster_size': 1}
        
    elif sensor_type == SensorType.SONAR:
        m_out_of_n = {'M': 3, 'N': 4, 'confirmation_threshold': 0.9}
        sprt = {
            'alpha': 0.05, 'beta': 0.1,
            'pd_target': 0.8, 'pd_clutter': 0.2,
            'upper_threshold': 2.9, 'lower_threshold': -2.3
        }
        gate = {'gate_size': 11.07, 'max_gate_distance': 50.0, 'min_cluster_size': 2}
        
    else:
        # Default parameters for other sensors
        m_out_of_n = {'M': 2, 'N': 3, 'confirmation_threshold': 0.8}
        sprt = {
            'alpha': 0.05, 'beta': 0.05,
            'pd_target': 0.85, 'pd_clutter': 0.15,
            'upper_threshold': 2.94, 'lower_threshold': -2.94
        }
        gate = {'gate_size': 9.21, 'max_gate_distance': 40.0, 'min_cluster_size': 1}
    
    return TrackInitiator(
        sensor_type=sensor_type,
        m_out_of_n_params=m_out_of_n,
        sprt_params=sprt,
        gate_params=gate
    )


# Example usage and testing functions
def create_sample_measurements(num_measurements: int = 10) -> List[Measurement]:
    """
    Create sample measurements for testing.
    
    Args:
        num_measurements: Number of measurements to create
        
    Returns:
        List of sample measurements
    """
    measurements = []
    
    for i in range(num_measurements):
        # Create a simple trajectory
        t = i * 0.1
        x = 10 + 5 * t + 0.1 * np.random.randn()
        y = 20 + 3 * t + 0.1 * np.random.randn()
        
        measurement = Measurement(
            timestamp=t,
            position=np.array([x, y, 0]),
            velocity=np.array([5, 3, 0]) + 0.5 * np.random.randn(3),
            snr=15 + 5 * np.random.randn(),
            measurement_id=f"meas_{i:03d}"
        )
        
        measurements.append(measurement)
    
    return measurements


if __name__ == "__main__":
    # Example usage
    
    # Create track initiator
    initiator = create_default_track_initiator(SensorType.RADAR)
    
    # Create sample measurements
    measurements = create_sample_measurements(20)
    
    # Process measurements
    tracks = initiator.process_measurements(measurements)
    
    print(f"Processed {len(measurements)} measurements")
    print(f"Created {len(tracks)} tracks")
    
    for track in tracks:
        print(f"Track {track.track_id}: {track.status.value}, "
              f"Quality: {track.quality_score:.3f}, "
              f"Measurements: {len(track.measurements)}")