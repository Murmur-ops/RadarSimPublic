"""
Data Association Algorithms for Multi-Target Tracking

This module implements comprehensive data association algorithms including:
- Global Nearest Neighbor (GNN)
- Joint Probabilistic Data Association (JPDA)
- Multiple Hypothesis Tracking (MHT)

Author: RadarSim Project
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Union
from dataclasses import dataclass
from collections import defaultdict
import itertools
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
import logging


@dataclass
class Detection:
    """Radar detection data structure."""
    position: np.ndarray  # [x, y, z] or [x, y]
    timestamp: float
    measurement_noise: np.ndarray  # Covariance matrix
    id: int = None
    snr: float = None
    doppler: float = None


@dataclass
class Track:
    """Track data structure."""
    state: np.ndarray  # State vector
    covariance: np.ndarray  # State covariance
    id: int
    last_update: float
    score: float = 0.0
    age: int = 0
    hits: int = 0
    misses: int = 0
    confirmed: bool = False


@dataclass
class AssociationResult:
    """Result of data association."""
    assignments: Dict[int, int]  # track_id -> detection_id
    unassigned_tracks: List[int]
    unassigned_detections: List[int]
    probabilities: Optional[Dict] = None  # For JPDA


class DistanceMetrics:
    """Collection of distance metrics for data association."""
    
    @staticmethod
    def mahalanobis_distance(residual: np.ndarray, covariance: np.ndarray) -> float:
        """
        Compute Mahalanobis distance.
        
        Args:
            residual: Measurement residual vector
            covariance: Innovation covariance matrix
            
        Returns:
            Mahalanobis distance
        """
        try:
            inv_cov = np.linalg.inv(covariance)
            distance = np.sqrt(residual.T @ inv_cov @ residual)
            return float(distance)
        except np.linalg.LinAlgError:
            return np.inf
    
    @staticmethod
    def euclidean_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two positions.
        
        Args:
            pos1: First position vector
            pos2: Second position vector
            
        Returns:
            Euclidean distance
        """
        return np.linalg.norm(pos1 - pos2)
    
    @staticmethod
    def normalized_innovation_squared(residual: np.ndarray, 
                                    covariance: np.ndarray) -> float:
        """
        Compute normalized innovation squared (NIS).
        
        Args:
            residual: Measurement residual vector
            covariance: Innovation covariance matrix
            
        Returns:
            Normalized innovation squared
        """
        try:
            inv_cov = np.linalg.inv(covariance)
            nis = residual.T @ inv_cov @ residual
            return float(nis)
        except np.linalg.LinAlgError:
            return np.inf


class GatingFunctions:
    """Gating functions for association validation."""
    
    @staticmethod
    def chi_square_gate(residual: np.ndarray, covariance: np.ndarray, 
                       confidence: float = 0.99) -> bool:
        """
        Chi-square gating using Mahalanobis distance.
        
        Args:
            residual: Measurement residual vector
            covariance: Innovation covariance matrix
            confidence: Confidence level (default 0.99)
            
        Returns:
            True if measurement passes gate
        """
        dof = len(residual)
        threshold = chi2.ppf(confidence, dof)
        nis = DistanceMetrics.normalized_innovation_squared(residual, covariance)
        return nis <= threshold
    
    @staticmethod
    def ellipsoidal_gate(residual: np.ndarray, covariance: np.ndarray,
                        gate_size: float = 3.0) -> bool:
        """
        Ellipsoidal gating using scaled covariance.
        
        Args:
            residual: Measurement residual vector
            covariance: Innovation covariance matrix
            gate_size: Gate size multiplier (default 3.0)
            
        Returns:
            True if measurement passes gate
        """
        distance = DistanceMetrics.mahalanobis_distance(residual, covariance)
        return distance <= gate_size


class GlobalNearestNeighbor:
    """
    Global Nearest Neighbor (GNN) data association algorithm.
    
    Uses the Hungarian algorithm to find optimal assignment that minimizes
    the total association cost.
    """
    
    def __init__(self, gate_threshold: float = 3.0, 
                 max_cost: float = 1e6):
        """
        Initialize GNN associator.
        
        Args:
            gate_threshold: Gating threshold for validation
            max_cost: Maximum allowed association cost
        """
        self.gate_threshold = gate_threshold
        self.max_cost = max_cost
        self.logger = logging.getLogger(__name__)
    
    def compute_cost_matrix(self, tracks: List[Track], 
                          detections: List[Detection],
                          measurement_function,
                          measurement_jacobian) -> np.ndarray:
        """
        Compute cost matrix using Mahalanobis distance.
        
        Args:
            tracks: List of active tracks
            detections: List of detections
            measurement_function: Function to predict measurements
            measurement_jacobian: Function to compute measurement Jacobian
            
        Returns:
            Cost matrix (n_tracks x n_detections)
        """
        n_tracks = len(tracks)
        n_detections = len(detections)
        
        if n_tracks == 0 or n_detections == 0:
            return np.array([]).reshape(0, max(n_detections, 1))
        
        cost_matrix = np.full((n_tracks, n_detections), self.max_cost)
        
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                # Predict measurement for track
                predicted_measurement = measurement_function(track.state)
                H = measurement_jacobian(track.state)
                
                # Compute innovation covariance
                S = H @ track.covariance @ H.T + detection.measurement_noise
                
                # Compute residual
                residual = detection.position - predicted_measurement
                
                # Apply gating
                if GatingFunctions.ellipsoidal_gate(residual, S, self.gate_threshold):
                    cost_matrix[i, j] = DistanceMetrics.mahalanobis_distance(
                        residual, S)
        
        return cost_matrix
    
    def associate(self, tracks: List[Track], detections: List[Detection],
                  measurement_function, measurement_jacobian) -> AssociationResult:
        """
        Perform GNN data association.
        
        Args:
            tracks: List of active tracks
            detections: List of detections
            measurement_function: Function to predict measurements
            measurement_jacobian: Function to compute measurement Jacobian
            
        Returns:
            Association result
        """
        if not tracks or not detections:
            return AssociationResult(
                assignments={},
                unassigned_tracks=[t.id for t in tracks],
                unassigned_detections=list(range(len(detections)))
            )
        
        # Compute cost matrix
        cost_matrix = self.compute_cost_matrix(
            tracks, detections, measurement_function, measurement_jacobian)
        
        if cost_matrix.size == 0:
            return AssociationResult(
                assignments={},
                unassigned_tracks=[t.id for t in tracks],
                unassigned_detections=list(range(len(detections)))
            )
        
        # Solve assignment problem using Hungarian algorithm
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        # Extract valid assignments
        assignments = {}
        assigned_tracks = set()
        assigned_detections = set()
        
        for t_idx, d_idx in zip(track_indices, detection_indices):
            if cost_matrix[t_idx, d_idx] < self.max_cost:
                assignments[tracks[t_idx].id] = d_idx
                assigned_tracks.add(t_idx)
                assigned_detections.add(d_idx)
        
        # Find unassigned tracks and detections
        unassigned_tracks = [tracks[i].id for i in range(len(tracks)) 
                           if i not in assigned_tracks]
        unassigned_detections = [i for i in range(len(detections)) 
                               if i not in assigned_detections]
        
        self.logger.info(f"GNN: {len(assignments)} associations, "
                        f"{len(unassigned_tracks)} unassigned tracks, "
                        f"{len(unassigned_detections)} unassigned detections")
        
        return AssociationResult(
            assignments=assignments,
            unassigned_tracks=unassigned_tracks,
            unassigned_detections=unassigned_detections
        )


class JointProbabilisticDataAssociation:
    """
    Joint Probabilistic Data Association (JPDA) algorithm.
    
    Computes association probabilities for all track-detection pairs
    and provides weighted updates for tracks.
    """
    
    def __init__(self, gate_threshold: float = 3.0,
                 prob_detection: float = 0.98,
                 clutter_density: float = 1e-6,
                 prob_survival: float = 0.99):
        """
        Initialize JPDA associator.
        
        Args:
            gate_threshold: Gating threshold for validation
            prob_detection: Probability of detection
            clutter_density: Clutter density (false alarms per unit volume)
            prob_survival: Track survival probability
        """
        self.gate_threshold = gate_threshold
        self.prob_detection = prob_detection
        self.clutter_density = clutter_density
        self.prob_survival = prob_survival
        self.logger = logging.getLogger(__name__)
    
    def compute_validation_matrix(self, tracks: List[Track],
                                detections: List[Detection],
                                measurement_function,
                                measurement_jacobian) -> np.ndarray:
        """
        Compute validation matrix indicating which detections are valid for each track.
        
        Args:
            tracks: List of active tracks
            detections: List of detections
            measurement_function: Function to predict measurements
            measurement_jacobian: Function to compute measurement Jacobian
            
        Returns:
            Validation matrix (n_tracks x n_detections)
        """
        n_tracks = len(tracks)
        n_detections = len(detections)
        
        if n_tracks == 0 or n_detections == 0:
            return np.zeros((n_tracks, n_detections), dtype=bool)
        
        validation_matrix = np.zeros((n_tracks, n_detections), dtype=bool)
        
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                # Predict measurement for track
                predicted_measurement = measurement_function(track.state)
                H = measurement_jacobian(track.state)
                
                # Compute innovation covariance
                S = H @ track.covariance @ H.T + detection.measurement_noise
                
                # Compute residual
                residual = detection.position - predicted_measurement
                
                # Apply gating
                validation_matrix[i, j] = GatingFunctions.ellipsoidal_gate(
                    residual, S, self.gate_threshold)
        
        return validation_matrix
    
    def compute_likelihood_ratios(self, tracks: List[Track],
                                detections: List[Detection],
                                validation_matrix: np.ndarray,
                                measurement_function,
                                measurement_jacobian) -> np.ndarray:
        """
        Compute likelihood ratios for valid track-detection pairs.
        
        Args:
            tracks: List of active tracks
            detections: List of detections
            validation_matrix: Validation matrix
            measurement_function: Function to predict measurements
            measurement_jacobian: Function to compute measurement Jacobian
            
        Returns:
            Likelihood ratio matrix
        """
        n_tracks, n_detections = validation_matrix.shape
        likelihood_ratios = np.zeros((n_tracks, n_detections))
        
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                if validation_matrix[i, j]:
                    # Predict measurement for track
                    predicted_measurement = measurement_function(track.state)
                    H = measurement_jacobian(track.state)
                    
                    # Compute innovation covariance
                    S = H @ track.covariance @ H.T + detection.measurement_noise
                    
                    # Compute residual
                    residual = detection.position - predicted_measurement
                    
                    # Compute likelihood ratio
                    det_S = np.linalg.det(2 * np.pi * S)
                    if det_S > 0:
                        exp_term = np.exp(-0.5 * DistanceMetrics.normalized_innovation_squared(
                            residual, S))
                        likelihood_ratios[i, j] = exp_term / np.sqrt(det_S)
        
        return likelihood_ratios
    
    def compute_association_probabilities(self, validation_matrix: np.ndarray,
                                        likelihood_ratios: np.ndarray) -> Dict:
        """
        Compute association probabilities using JPDA algorithm.
        
        Args:
            validation_matrix: Validation matrix
            likelihood_ratios: Likelihood ratio matrix
            
        Returns:
            Dictionary containing association probabilities
        """
        n_tracks, n_detections = validation_matrix.shape
        
        # Find all valid associations
        valid_pairs = [(i, j) for i in range(n_tracks) 
                      for j in range(n_detections) 
                      if validation_matrix[i, j]]
        
        if not valid_pairs:
            return {
                'track_detection_probs': np.zeros((n_tracks, n_detections)),
                'track_no_detection_probs': np.ones(n_tracks),
                'detection_clutter_probs': np.ones(n_detections)
            }
        
        # Generate all feasible association hypotheses
        hypotheses = self._generate_association_hypotheses(
            n_tracks, n_detections, valid_pairs)
        
        # Compute hypothesis probabilities
        hypothesis_probs = []
        for hypothesis in hypotheses:
            prob = self._compute_hypothesis_probability(
                hypothesis, likelihood_ratios, n_tracks, n_detections)
            hypothesis_probs.append(prob)
        
        # Normalize hypothesis probabilities
        total_prob = sum(hypothesis_probs)
        if total_prob > 0:
            hypothesis_probs = [p / total_prob for p in hypothesis_probs]
        
        # Compute marginal association probabilities
        track_detection_probs = np.zeros((n_tracks, n_detections))
        track_no_detection_probs = np.zeros(n_tracks)
        detection_clutter_probs = np.zeros(n_detections)
        
        for hypothesis, prob in zip(hypotheses, hypothesis_probs):
            for track_id, detection_id in hypothesis.items():
                if detection_id == -1:  # No detection
                    track_no_detection_probs[track_id] += prob
                else:
                    track_detection_probs[track_id, detection_id] += prob
            
            # Compute clutter probabilities
            assigned_detections = set(hypothesis.values())
            assigned_detections.discard(-1)  # Remove "no detection" marker
            
            for j in range(n_detections):
                if j not in assigned_detections:
                    detection_clutter_probs[j] += prob
        
        return {
            'track_detection_probs': track_detection_probs,
            'track_no_detection_probs': track_no_detection_probs,
            'detection_clutter_probs': detection_clutter_probs
        }
    
    def _generate_association_hypotheses(self, n_tracks: int, n_detections: int,
                                       valid_pairs: List[Tuple[int, int]]) -> List[Dict]:
        """
        Generate all feasible association hypotheses.
        
        Args:
            n_tracks: Number of tracks
            n_detections: Number of detections
            valid_pairs: List of valid (track_id, detection_id) pairs
            
        Returns:
            List of association hypotheses
        """
        hypotheses = []
        
        # Group valid pairs by track
        track_detections = defaultdict(list)
        for track_id, detection_id in valid_pairs:
            track_detections[track_id].append(detection_id)
        
        # Add "no detection" option for each track
        for track_id in range(n_tracks):
            track_detections[track_id].append(-1)  # -1 represents no detection
        
        # Generate all combinations
        track_options = []
        for track_id in range(n_tracks):
            track_options.append(track_detections[track_id])
        
        # Generate Cartesian product of all track options
        for combination in itertools.product(*track_options):
            hypothesis = {track_id: detection_id 
                         for track_id, detection_id in enumerate(combination)}
            
            # Check if hypothesis is feasible (no detection assigned to multiple tracks)
            assigned_detections = [d for d in hypothesis.values() if d != -1]
            if len(assigned_detections) == len(set(assigned_detections)):
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _compute_hypothesis_probability(self, hypothesis: Dict,
                                      likelihood_ratios: np.ndarray,
                                      n_tracks: int, n_detections: int) -> float:
        """
        Compute probability of a specific association hypothesis.
        
        Args:
            hypothesis: Association hypothesis
            likelihood_ratios: Likelihood ratio matrix
            n_tracks: Number of tracks
            n_detections: Number of detections
            
        Returns:
            Hypothesis probability
        """
        prob = 1.0
        
        # Track-detection associations
        for track_id, detection_id in hypothesis.items():
            if detection_id == -1:  # No detection
                prob *= (1 - self.prob_detection)
            else:
                prob *= self.prob_detection * likelihood_ratios[track_id, detection_id]
        
        # False alarm probability for unassigned detections
        assigned_detections = set(hypothesis.values())
        assigned_detections.discard(-1)
        
        n_false_alarms = n_detections - len(assigned_detections)
        prob *= (self.clutter_density ** n_false_alarms)
        
        return prob
    
    def associate(self, tracks: List[Track], detections: List[Detection],
                  measurement_function, measurement_jacobian) -> AssociationResult:
        """
        Perform JPDA data association.
        
        Args:
            tracks: List of active tracks
            detections: List of detections
            measurement_function: Function to predict measurements
            measurement_jacobian: Function to compute measurement Jacobian
            
        Returns:
            Association result with probabilities
        """
        if not tracks or not detections:
            return AssociationResult(
                assignments={},
                unassigned_tracks=[t.id for t in tracks],
                unassigned_detections=list(range(len(detections))),
                probabilities={}
            )
        
        # Compute validation matrix
        validation_matrix = self.compute_validation_matrix(
            tracks, detections, measurement_function, measurement_jacobian)
        
        # Compute likelihood ratios
        likelihood_ratios = self.compute_likelihood_ratios(
            tracks, detections, validation_matrix, 
            measurement_function, measurement_jacobian)
        
        # Compute association probabilities
        probabilities = self.compute_association_probabilities(
            validation_matrix, likelihood_ratios)
        
        # Make hard assignments based on maximum probability
        assignments = {}
        track_detection_probs = probabilities['track_detection_probs']
        
        for i, track in enumerate(tracks):
            max_prob = 0.0
            best_detection = -1
            
            for j in range(len(detections)):
                if track_detection_probs[i, j] > max_prob:
                    max_prob = track_detection_probs[i, j]
                    best_detection = j
            
            # Only assign if probability is above threshold
            if max_prob > 0.5:  # Simple threshold
                assignments[track.id] = best_detection
        
        # Find unassigned tracks and detections
        assigned_detections = set(assignments.values())
        unassigned_tracks = [track.id for track in tracks 
                           if track.id not in assignments]
        unassigned_detections = [i for i in range(len(detections))
                               if i not in assigned_detections]
        
        self.logger.info(f"JPDA: {len(assignments)} associations, "
                        f"{len(unassigned_tracks)} unassigned tracks, "
                        f"{len(unassigned_detections)} unassigned detections")
        
        return AssociationResult(
            assignments=assignments,
            unassigned_tracks=unassigned_tracks,
            unassigned_detections=unassigned_detections,
            probabilities=probabilities
        )


@dataclass
class Hypothesis:
    """MHT hypothesis data structure."""
    id: int
    parent_id: Optional[int]
    associations: Dict[int, int]  # track_id -> detection_id
    probability: float
    score: float
    depth: int
    tracks: List[Track]


class MultipleHypothesisTracking:
    """
    Multiple Hypothesis Tracking (MHT) algorithm.
    
    Maintains multiple association hypotheses and prunes based on
    probability and N-scan logic.
    """
    
    def __init__(self, gate_threshold: float = 3.0,
                 prob_detection: float = 0.98,
                 prob_survival: float = 0.99,
                 clutter_density: float = 1e-6,
                 max_hypotheses: int = 100,
                 n_scan_pruning: int = 5,
                 prob_threshold: float = 1e-6,
                 new_track_threshold: int = 3):
        """
        Initialize MHT tracker.
        
        Args:
            gate_threshold: Gating threshold for validation
            prob_detection: Probability of detection
            prob_survival: Track survival probability
            clutter_density: Clutter density
            max_hypotheses: Maximum number of hypotheses to maintain
            n_scan_pruning: Number of scans for N-scan pruning
            prob_threshold: Minimum probability threshold
            new_track_threshold: Minimum hits for track confirmation
        """
        self.gate_threshold = gate_threshold
        self.prob_detection = prob_detection
        self.prob_survival = prob_survival
        self.clutter_density = clutter_density
        self.max_hypotheses = max_hypotheses
        self.n_scan_pruning = n_scan_pruning
        self.prob_threshold = prob_threshold
        self.new_track_threshold = new_track_threshold
        
        self.hypotheses: List[Hypothesis] = []
        self.hypothesis_id_counter = 0
        self.track_id_counter = 0
        self.scan_count = 0
        
        self.logger = logging.getLogger(__name__)
    
    def predict_tracks(self, tracks: List[Track], dt: float,
                      motion_model) -> List[Track]:
        """
        Predict all tracks forward in time.
        
        Args:
            tracks: List of tracks to predict
            dt: Time step
            motion_model: Motion model for prediction
            
        Returns:
            List of predicted tracks
        """
        predicted_tracks = []
        
        for track in tracks:
            # Create copy of track
            predicted_track = Track(
                state=track.state.copy(),
                covariance=track.covariance.copy(),
                id=track.id,
                last_update=track.last_update,
                score=track.score,
                age=track.age + 1,
                hits=track.hits,
                misses=track.misses,
                confirmed=track.confirmed
            )
            
            # Predict state and covariance
            F = motion_model.transition_matrix(dt)
            Q = motion_model.process_noise(dt)
            
            predicted_track.state = F @ track.state
            predicted_track.covariance = F @ track.covariance @ F.T + Q
            
            predicted_tracks.append(predicted_track)
        
        return predicted_tracks
    
    def generate_hypotheses(self, parent_hypothesis: Hypothesis,
                          detections: List[Detection],
                          measurement_function,
                          measurement_jacobian) -> List[Hypothesis]:
        """
        Generate child hypotheses from parent hypothesis.
        
        Args:
            parent_hypothesis: Parent hypothesis
            detections: List of detections
            measurement_function: Function to predict measurements
            measurement_jacobian: Function to compute measurement Jacobian
            
        Returns:
            List of child hypotheses
        """
        child_hypotheses = []
        tracks = parent_hypothesis.tracks
        
        # Find valid associations for each track
        valid_associations = defaultdict(list)
        
        for i, track in enumerate(tracks):
            # Add "no detection" option
            valid_associations[i].append(-1)
            
            for j, detection in enumerate(detections):
                # Predict measurement for track
                predicted_measurement = measurement_function(track.state)
                H = measurement_jacobian(track.state)
                
                # Compute innovation covariance
                S = H @ track.covariance @ H.T + detection.measurement_noise
                
                # Compute residual
                residual = detection.position - predicted_measurement
                
                # Apply gating
                if GatingFunctions.ellipsoidal_gate(residual, S, self.gate_threshold):
                    valid_associations[i].append(j)
        
        # Generate all feasible combinations
        track_options = [valid_associations[i] for i in range(len(tracks))]
        
        for combination in itertools.product(*track_options):
            # Check feasibility (no detection assigned to multiple tracks)
            detection_assignments = [d for d in combination if d != -1]
            if len(detection_assignments) == len(set(detection_assignments)):
                
                # Create new hypothesis
                new_associations = {tracks[i].id: detection_id 
                                  for i, detection_id in enumerate(combination)}
                
                new_hypothesis = Hypothesis(
                    id=self.hypothesis_id_counter,
                    parent_id=parent_hypothesis.id,
                    associations=new_associations,
                    probability=0.0,  # Will be computed later
                    score=0.0,
                    depth=parent_hypothesis.depth + 1,
                    tracks=[]  # Will be updated later
                )
                
                self.hypothesis_id_counter += 1
                child_hypotheses.append(new_hypothesis)
        
        # Generate hypotheses for new track initiation
        unassigned_detections = set(range(len(detections)))
        
        # Remove detections already considered in associations
        for hypothesis in child_hypotheses:
            for detection_id in hypothesis.associations.values():
                if detection_id != -1:
                    unassigned_detections.discard(detection_id)
        
        # Create new tracks for unassigned detections
        for detection_id in unassigned_detections:
            # Create hypothesis with new track
            new_track = Track(
                state=np.concatenate([detections[detection_id].position, 
                                    np.zeros(len(detections[detection_id].position))]),
                covariance=np.eye(2 * len(detections[detection_id].position)) * 10,
                id=self.track_id_counter,
                last_update=detections[detection_id].timestamp,
                score=1.0,
                age=1,
                hits=1,
                misses=0,
                confirmed=False
            )
            
            new_hypothesis = Hypothesis(
                id=self.hypothesis_id_counter,
                parent_id=parent_hypothesis.id,
                associations={new_track.id: detection_id},
                probability=0.0,
                score=0.0,
                depth=parent_hypothesis.depth + 1,
                tracks=parent_hypothesis.tracks + [new_track]
            )
            
            self.track_id_counter += 1
            self.hypothesis_id_counter += 1
            child_hypotheses.append(new_hypothesis)
        
        return child_hypotheses
    
    def compute_hypothesis_probability(self, hypothesis: Hypothesis,
                                     detections: List[Detection],
                                     measurement_function,
                                     measurement_jacobian) -> float:
        """
        Compute probability of hypothesis.
        
        Args:
            hypothesis: Hypothesis to evaluate
            detections: List of detections
            measurement_function: Function to predict measurements
            measurement_jacobian: Function to compute measurement Jacobian
            
        Returns:
            Hypothesis probability
        """
        prob = 1.0
        
        # Track survival and detection probabilities
        for track in hypothesis.tracks:
            track_id = track.id
            
            if track_id in hypothesis.associations:
                detection_id = hypothesis.associations[track_id]
                
                if detection_id == -1:  # No detection
                    prob *= self.prob_survival * (1 - self.prob_detection)
                else:  # Detection
                    # Compute likelihood
                    detection = detections[detection_id]
                    predicted_measurement = measurement_function(track.state)
                    H = measurement_jacobian(track.state)
                    S = H @ track.covariance @ H.T + detection.measurement_noise
                    residual = detection.position - predicted_measurement
                    
                    # Likelihood computation
                    det_S = np.linalg.det(2 * np.pi * S)
                    if det_S > 0:
                        exp_term = np.exp(-0.5 * DistanceMetrics.normalized_innovation_squared(
                            residual, S))
                        likelihood = exp_term / np.sqrt(det_S)
                        prob *= self.prob_survival * self.prob_detection * likelihood
        
        # False alarm probability
        assigned_detections = set(hypothesis.associations.values())
        assigned_detections.discard(-1)
        n_false_alarms = len(detections) - len(assigned_detections)
        prob *= (self.clutter_density ** n_false_alarms)
        
        return prob
    
    def update_tracks(self, hypothesis: Hypothesis, detections: List[Detection],
                     measurement_function, measurement_jacobian) -> List[Track]:
        """
        Update tracks in hypothesis with associated detections.
        
        Args:
            hypothesis: Hypothesis containing associations
            detections: List of detections
            measurement_function: Function to predict measurements
            measurement_jacobian: Function to compute measurement Jacobian
            
        Returns:
            List of updated tracks
        """
        updated_tracks = []
        
        for track in hypothesis.tracks:
            updated_track = Track(
                state=track.state.copy(),
                covariance=track.covariance.copy(),
                id=track.id,
                last_update=track.last_update,
                score=track.score,
                age=track.age,
                hits=track.hits,
                misses=track.misses,
                confirmed=track.confirmed
            )
            
            if track.id in hypothesis.associations:
                detection_id = hypothesis.associations[track.id]
                
                if detection_id != -1:  # Associated detection
                    detection = detections[detection_id]
                    
                    # Kalman update
                    predicted_measurement = measurement_function(track.state)
                    H = measurement_jacobian(track.state)
                    S = H @ track.covariance @ H.T + detection.measurement_noise
                    K = track.covariance @ H.T @ np.linalg.inv(S)
                    
                    residual = detection.position - predicted_measurement
                    updated_track.state = track.state + K @ residual
                    updated_track.covariance = (np.eye(len(track.state)) - K @ H) @ track.covariance
                    updated_track.last_update = detection.timestamp
                    updated_track.hits += 1
                    updated_track.score += 1.0
                    
                    # Check for track confirmation
                    if updated_track.hits >= self.new_track_threshold:
                        updated_track.confirmed = True
                
                else:  # No detection (missed)
                    updated_track.misses += 1
                    updated_track.score -= 0.5
            
            updated_tracks.append(updated_track)
        
        return updated_tracks
    
    def prune_hypotheses(self):
        """Prune hypotheses based on probability and N-scan logic."""
        if not self.hypotheses:
            return
        
        # Sort hypotheses by probability
        self.hypotheses.sort(key=lambda h: h.probability, reverse=True)
        
        # Probability-based pruning
        self.hypotheses = [h for h in self.hypotheses 
                          if h.probability >= self.prob_threshold]
        
        # Limit number of hypotheses
        if len(self.hypotheses) > self.max_hypotheses:
            self.hypotheses = self.hypotheses[:self.max_hypotheses]
        
        # N-scan pruning
        if self.scan_count >= self.n_scan_pruning:
            # Keep only the best hypothesis from deep scans
            min_depth = min(h.depth for h in self.hypotheses)
            pruning_depth = min_depth + self.n_scan_pruning
            
            # Find hypotheses to prune
            to_prune = [h for h in self.hypotheses if h.depth <= pruning_depth]
            
            if to_prune:
                # Keep only the best hypothesis at pruning depth
                best_hypothesis = max(to_prune, key=lambda h: h.probability)
                self.hypotheses = [h for h in self.hypotheses 
                                 if h.depth > pruning_depth or h == best_hypothesis]
        
        # Renormalize probabilities
        total_prob = sum(h.probability for h in self.hypotheses)
        if total_prob > 0:
            for hypothesis in self.hypotheses:
                hypothesis.probability /= total_prob
        
        self.logger.info(f"MHT: Maintaining {len(self.hypotheses)} hypotheses after pruning")
    
    def get_best_tracks(self) -> List[Track]:
        """
        Get tracks from the best hypothesis.
        
        Returns:
            List of tracks from best hypothesis
        """
        if not self.hypotheses:
            return []
        
        best_hypothesis = max(self.hypotheses, key=lambda h: h.probability)
        return [track for track in best_hypothesis.tracks if track.confirmed]
    
    def associate(self, tracks: List[Track], detections: List[Detection],
                  measurement_function, measurement_jacobian,
                  motion_model, dt: float) -> AssociationResult:
        """
        Perform MHT data association.
        
        Args:
            tracks: List of current tracks
            detections: List of detections
            measurement_function: Function to predict measurements
            measurement_jacobian: Function to compute measurement Jacobian
            motion_model: Motion model for prediction
            dt: Time step
            
        Returns:
            Association result from best hypothesis
        """
        self.scan_count += 1
        
        # Initialize hypotheses if empty
        if not self.hypotheses:
            initial_hypothesis = Hypothesis(
                id=self.hypothesis_id_counter,
                parent_id=None,
                associations={},
                probability=1.0,
                score=0.0,
                depth=0,
                tracks=tracks
            )
            self.hypothesis_id_counter += 1
            self.hypotheses = [initial_hypothesis]
        
        # Predict tracks for all hypotheses
        for hypothesis in self.hypotheses:
            hypothesis.tracks = self.predict_tracks(
                hypothesis.tracks, dt, motion_model)
        
        # Generate new hypotheses
        new_hypotheses = []
        for hypothesis in self.hypotheses:
            child_hypotheses = self.generate_hypotheses(
                hypothesis, detections, measurement_function, measurement_jacobian)
            
            # Compute probabilities and update tracks
            for child in child_hypotheses:
                child.probability = (hypothesis.probability * 
                                   self.compute_hypothesis_probability(
                                       child, detections, measurement_function, 
                                       measurement_jacobian))
                child.tracks = self.update_tracks(
                    child, detections, measurement_function, measurement_jacobian)
            
            new_hypotheses.extend(child_hypotheses)
        
        self.hypotheses = new_hypotheses
        
        # Prune hypotheses
        self.prune_hypotheses()
        
        # Extract best association
        if not self.hypotheses:
            return AssociationResult(
                assignments={},
                unassigned_tracks=[t.id for t in tracks],
                unassigned_detections=list(range(len(detections)))
            )
        
        best_hypothesis = max(self.hypotheses, key=lambda h: h.probability)
        
        # Convert associations to result format
        assignments = {}
        for track_id, detection_id in best_hypothesis.associations.items():
            if detection_id != -1:
                assignments[track_id] = detection_id
        
        assigned_detections = set(assignments.values())
        unassigned_tracks = [t.id for t in tracks if t.id not in assignments]
        unassigned_detections = [i for i in range(len(detections))
                               if i not in assigned_detections]
        
        self.logger.info(f"MHT: {len(assignments)} associations from best hypothesis, "
                        f"maintaining {len(self.hypotheses)} hypotheses")
        
        return AssociationResult(
            assignments=assignments,
            unassigned_tracks=unassigned_tracks,
            unassigned_detections=unassigned_detections
        )


class AssociationManager:
    """
    Manager class for different data association algorithms.
    
    Provides a unified interface for switching between GNN, JPDA, and MHT.
    """
    
    def __init__(self, algorithm: str = 'gnn', **kwargs):
        """
        Initialize association manager.
        
        Args:
            algorithm: Algorithm type ('gnn', 'jpda', 'mht')
            **kwargs: Algorithm-specific parameters
        """
        self.algorithm = algorithm.lower()
        
        if self.algorithm == 'gnn':
            self.associator = GlobalNearestNeighbor(**kwargs)
        elif self.algorithm == 'jpda':
            self.associator = JointProbabilisticDataAssociation(**kwargs)
        elif self.algorithm == 'mht':
            self.associator = MultipleHypothesisTracking(**kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.logger = logging.getLogger(__name__)
    
    def associate(self, tracks: List[Track], detections: List[Detection],
                  measurement_function, measurement_jacobian,
                  motion_model=None, dt: float = None) -> AssociationResult:
        """
        Perform data association using selected algorithm.
        
        Args:
            tracks: List of tracks
            detections: List of detections
            measurement_function: Function to predict measurements
            measurement_jacobian: Function to compute measurement Jacobian
            motion_model: Motion model (required for MHT)
            dt: Time step (required for MHT)
            
        Returns:
            Association result
        """
        if self.algorithm == 'mht':
            if motion_model is None or dt is None:
                raise ValueError("MHT requires motion_model and dt parameters")
            return self.associator.associate(
                tracks, detections, measurement_function, 
                measurement_jacobian, motion_model, dt)
        else:
            return self.associator.associate(
                tracks, detections, measurement_function, measurement_jacobian)
    
    def get_tracks(self) -> List[Track]:
        """
        Get current tracks (only applicable for MHT).
        
        Returns:
            List of current tracks
        """
        if self.algorithm == 'mht':
            return self.associator.get_best_tracks()
        else:
            raise ValueError("get_tracks() only available for MHT algorithm")


def estimate_clutter_density(detections: List[Detection], 
                           surveillance_volume: float,
                           window_size: int = 10) -> float:
    """
    Estimate clutter density from detection history.
    
    Args:
        detections: List of detections
        surveillance_volume: Surveillance volume
        window_size: Window size for estimation
        
    Returns:
        Estimated clutter density
    """
    if len(detections) < window_size:
        return 1e-6  # Default low clutter density
    
    # Simple estimation based on recent detections
    recent_detections = detections[-window_size:]
    avg_detections_per_scan = len(recent_detections) / window_size
    
    # Assume some fraction are false alarms
    estimated_false_alarms = avg_detections_per_scan * 0.1  # 10% false alarm rate
    clutter_density = estimated_false_alarms / surveillance_volume
    
    return max(clutter_density, 1e-10)  # Avoid zero density


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Create sample tracks and detections
    sample_tracks = [
        Track(
            state=np.array([0, 0, 1, 1]),  # [x, y, vx, vy]
            covariance=np.eye(4),
            id=0,
            last_update=0.0
        ),
        Track(
            state=np.array([10, 5, -0.5, 0.5]),
            covariance=np.eye(4),
            id=1,
            last_update=0.0
        )
    ]
    
    sample_detections = [
        Detection(
            position=np.array([1.1, 0.9]),
            timestamp=1.0,
            measurement_noise=np.eye(2) * 0.1,
            id=0
        ),
        Detection(
            position=np.array([9.8, 5.2]),
            timestamp=1.0,
            measurement_noise=np.eye(2) * 0.1,
            id=1
        ),
        Detection(
            position=np.array([15, 15]),  # Potential false alarm
            timestamp=1.0,
            measurement_noise=np.eye(2) * 0.1,
            id=2
        )
    ]
    
    # Simple measurement function and Jacobian
    def measurement_function(state):
        return state[:2]  # Position only
    
    def measurement_jacobian(state):
        H = np.zeros((2, 4))
        H[:2, :2] = np.eye(2)
        return H
    
    # Test GNN
    print("Testing Global Nearest Neighbor:")
    gnn = GlobalNearestNeighbor()
    result = gnn.associate(sample_tracks, sample_detections, 
                          measurement_function, measurement_jacobian)
    print(f"Assignments: {result.assignments}")
    print(f"Unassigned tracks: {result.unassigned_tracks}")
    print(f"Unassigned detections: {result.unassigned_detections}")
    print()
    
    # Test JPDA
    print("Testing Joint Probabilistic Data Association:")
    jpda = JointProbabilisticDataAssociation()
    result = jpda.associate(sample_tracks, sample_detections,
                           measurement_function, measurement_jacobian)
    print(f"Assignments: {result.assignments}")
    print(f"Unassigned tracks: {result.unassigned_tracks}")
    print(f"Unassigned detections: {result.unassigned_detections}")
    print()
    
    # Test Association Manager
    print("Testing Association Manager:")
    manager = AssociationManager('gnn')
    result = manager.associate(sample_tracks, sample_detections,
                              measurement_function, measurement_jacobian)
    print(f"GNN via manager - Assignments: {result.assignments}")