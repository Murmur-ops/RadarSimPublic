"""
Distributed Tracking and Data Fusion for Networked Radar

Implements track-to-track fusion, association, and distributed tracking
algorithms for networked radar systems.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class NetworkTrack:
    """Extended track for network-wide tracking."""
    track_id: str
    local_id: str
    source_node: str
    state: np.ndarray
    covariance: np.ndarray
    timestamp: float
    quality: float = 1.0
    associated_nodes: Set[str] = field(default_factory=set)
    fusion_history: List[Dict] = field(default_factory=list)
    classification: Optional[str] = None
    priority: float = 0.5


class TrackAssociator:
    """
    Performs track-to-track association across multiple radar nodes.
    
    Uses statistical distance measures and the Hungarian algorithm
    for optimal assignment.
    """
    
    def __init__(self,
                 gate_threshold: float = 9.21,  # Chi-square 99% for 2D
                 max_time_diff: float = 1.0):
        """
        Initialize track associator.
        
        Args:
            gate_threshold: Statistical distance threshold for gating
            max_time_diff: Maximum time difference for association (seconds)
        """
        self.gate_threshold = gate_threshold
        self.max_time_diff = max_time_diff
        self.association_history: List[Dict] = []
    
    def mahalanobis_distance(self,
                            state1: np.ndarray,
                            cov1: np.ndarray,
                            state2: np.ndarray,
                            cov2: np.ndarray) -> float:
        """
        Calculate Mahalanobis distance between two tracks.
        
        Args:
            state1: First track state
            cov1: First track covariance
            state2: Second track state
            cov2: Second track covariance
            
        Returns:
            Mahalanobis distance
        """
        # State difference
        diff = state1 - state2
        
        # Combined covariance
        cov_sum = cov1 + cov2
        
        try:
            # Mahalanobis distance
            cov_inv = np.linalg.inv(cov_sum)
            distance = np.sqrt(diff.T @ cov_inv @ diff)
        except np.linalg.LinAlgError:
            # Fallback to Euclidean distance if covariance singular
            distance = np.linalg.norm(diff)
            logger.warning("Singular covariance in Mahalanobis distance, using Euclidean")
        
        return distance
    
    def calculate_association_matrix(self,
                                    tracks1: List[NetworkTrack],
                                    tracks2: List[NetworkTrack]) -> np.ndarray:
        """
        Calculate association cost matrix between two track sets.
        
        Args:
            tracks1: First set of tracks
            tracks2: Second set of tracks
            
        Returns:
            Cost matrix (n x m)
        """
        n = len(tracks1)
        m = len(tracks2)
        cost_matrix = np.full((n, m), np.inf)
        
        for i, track1 in enumerate(tracks1):
            for j, track2 in enumerate(tracks2):
                # Check time difference
                time_diff = abs(track1.timestamp - track2.timestamp)
                if time_diff > self.max_time_diff:
                    continue
                
                # Extract position components (assuming state = [x, y, vx, vy, ...])
                pos1 = track1.state[:2] if len(track1.state) >= 2 else track1.state
                pos2 = track2.state[:2] if len(track2.state) >= 2 else track2.state
                
                # Get position covariances
                cov1 = track1.covariance[:2, :2] if track1.covariance.shape[0] >= 2 else track1.covariance
                cov2 = track2.covariance[:2, :2] if track2.covariance.shape[0] >= 2 else track2.covariance
                
                # Calculate Mahalanobis distance
                distance = self.mahalanobis_distance(pos1, cov1, pos2, cov2)
                
                # Apply gating
                if distance < self.gate_threshold:
                    cost_matrix[i, j] = distance
        
        return cost_matrix
    
    def associate_tracks(self,
                        tracks1: List[NetworkTrack],
                        tracks2: List[NetworkTrack]) -> List[Tuple[int, int]]:
        """
        Associate tracks between two sets using Hungarian algorithm.
        
        Args:
            tracks1: First set of tracks
            tracks2: Second set of tracks
            
        Returns:
            List of associated track index pairs
        """
        if not tracks1 or not tracks2:
            return []
        
        # Calculate cost matrix
        cost_matrix = self.calculate_association_matrix(tracks1, tracks2)
        
        # Handle case where no valid associations exist
        if np.all(np.isinf(cost_matrix)):
            return []
        
        # Replace inf with large value for Hungarian algorithm
        max_cost = np.max(cost_matrix[~np.isinf(cost_matrix)]) if np.any(~np.isinf(cost_matrix)) else 1000
        cost_matrix[np.isinf(cost_matrix)] = max_cost * 10
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out invalid associations
        associations = []
        for i, j in zip(row_indices, col_indices):
            if cost_matrix[i, j] < self.gate_threshold:
                associations.append((i, j))
        
        # Log association results
        self.association_history.append({
            'num_tracks1': len(tracks1),
            'num_tracks2': len(tracks2),
            'num_associations': len(associations)
        })
        
        return associations


class DistributedTrackFusion:
    """
    Implements distributed track fusion algorithms.
    
    Supports multiple fusion methods including Covariance Intersection,
    Information Matrix Fusion, and Track-to-Track Fusion.
    """
    
    def __init__(self):
        """Initialize distributed track fusion."""
        self.fusion_history: List[Dict] = []
        self.track_database: Dict[str, NetworkTrack] = {}
        self.associator = TrackAssociator()
    
    def covariance_intersection(self,
                               tracks: List[NetworkTrack],
                               optimize_omega: bool = True) -> NetworkTrack:
        """
        Fuse multiple tracks using Covariance Intersection.
        
        CI provides consistent fusion when correlations are unknown.
        
        Args:
            tracks: List of tracks to fuse
            optimize_omega: Whether to optimize mixing weights
            
        Returns:
            Fused track
        """
        if not tracks:
            raise ValueError("No tracks to fuse")
        
        if len(tracks) == 1:
            return tracks[0]
        
        # Initialize with uniform weights
        weights = np.ones(len(tracks)) / len(tracks)
        
        if optimize_omega and len(tracks) == 2:
            # Optimize for two tracks
            weights[0] = self._optimal_omega_two_tracks(
                tracks[0].covariance, tracks[1].covariance
            )
            weights[1] = 1 - weights[0]
        elif optimize_omega:
            # Optimize for multiple tracks
            weights = self._optimal_omega_multi_tracks([t.covariance for t in tracks])
        
        # Perform fusion - ensure float64 dtype
        P_fused_inv = np.zeros_like(tracks[0].covariance, dtype=np.float64)
        x_fused_weighted = np.zeros_like(tracks[0].state, dtype=np.float64)
        
        for i, track in enumerate(tracks):
            P_inv = np.linalg.inv(track.covariance.astype(np.float64))
            P_fused_inv += weights[i] * P_inv
            x_fused_weighted += weights[i] * P_inv @ track.state.astype(np.float64)
        
        P_fused = np.linalg.inv(P_fused_inv)
        x_fused = P_fused @ x_fused_weighted
        
        # Create fused track
        fused_track = NetworkTrack(
            track_id=f"fused_{tracks[0].track_id}",
            local_id=tracks[0].local_id,
            source_node="fusion_center",
            state=x_fused,
            covariance=P_fused,
            timestamp=max(t.timestamp for t in tracks),
            quality=np.mean([t.quality for t in tracks]),
            associated_nodes=set().union(*[t.associated_nodes for t in tracks]) if tracks else set()
        )
        
        fused_track.fusion_history.append({
            'method': 'covariance_intersection',
            'source_tracks': [t.track_id for t in tracks],
            'weights': weights.tolist(),
            'timestamp': fused_track.timestamp
        })
        
        return fused_track
    
    def information_matrix_fusion(self, tracks: List[NetworkTrack]) -> NetworkTrack:
        """
        Fuse tracks using Information Matrix Fusion.
        
        Optimal when tracks are independent.
        
        Args:
            tracks: List of tracks to fuse
            
        Returns:
            Fused track
        """
        if not tracks:
            raise ValueError("No tracks to fuse")
        
        if len(tracks) == 1:
            return tracks[0]
        
        # Convert to information form - ensure float64 dtype
        Y_total = np.zeros_like(tracks[0].covariance, dtype=np.float64)
        y_total = np.zeros_like(tracks[0].state, dtype=np.float64)
        
        for track in tracks:
            Y = np.linalg.inv(track.covariance.astype(np.float64))  # Information matrix
            y = Y @ track.state.astype(np.float64)  # Information vector
            Y_total += Y
            y_total += y
        
        # Convert back to state space
        P_fused = np.linalg.inv(Y_total)
        x_fused = P_fused @ y_total
        
        # Create fused track
        fused_track = NetworkTrack(
            track_id=f"fused_{tracks[0].track_id}",
            local_id=tracks[0].local_id,
            source_node="fusion_center",
            state=x_fused,
            covariance=P_fused,
            timestamp=max(t.timestamp for t in tracks),
            quality=np.mean([t.quality for t in tracks]),
            associated_nodes=set().union(*[t.associated_nodes for t in tracks]) if tracks else set()
        )
        
        fused_track.fusion_history.append({
            'method': 'information_matrix_fusion',
            'source_tracks': [t.track_id for t in tracks],
            'timestamp': fused_track.timestamp
        })
        
        return fused_track
    
    def tracklet_fusion(self,
                       tracklets: List[NetworkTrack],
                       method: str = "covariance_intersection") -> NetworkTrack:
        """
        Fuse short track segments (tracklets) from different sensors.
        
        Args:
            tracklets: List of tracklets to fuse
            method: Fusion method to use
            
        Returns:
            Fused tracklet
        """
        if method == "covariance_intersection":
            return self.covariance_intersection(tracklets)
        elif method == "information_matrix":
            return self.information_matrix_fusion(tracklets)
        else:
            raise ValueError(f"Unknown fusion method: {method}")
    
    def _optimal_omega_two_tracks(self, P1: np.ndarray, P2: np.ndarray) -> float:
        """
        Find optimal omega for two-track Covariance Intersection.
        
        Args:
            P1: First covariance matrix
            P2: Second covariance matrix
            
        Returns:
            Optimal omega value
        """
        # Analytical solution for scalar case
        if P1.shape == (1, 1) and P2.shape == (1, 1):
            p1 = P1[0, 0]
            p2 = P2[0, 0]
            return p2 / (p1 + p2)
        
        # Numerical optimization for matrix case
        omegas = np.linspace(0.01, 0.99, 100)
        min_det = float('inf')
        best_omega = 0.5
        
        for omega in omegas:
            try:
                P_fused_inv = omega * np.linalg.inv(P1) + (1 - omega) * np.linalg.inv(P2)
                P_fused = np.linalg.inv(P_fused_inv)
                det = np.linalg.det(P_fused)
                
                if det < min_det:
                    min_det = det
                    best_omega = omega
            except np.linalg.LinAlgError:
                continue
        
        return best_omega
    
    def _optimal_omega_multi_tracks(self, covariances: List[np.ndarray]) -> np.ndarray:
        """
        Find optimal weights for multi-track Covariance Intersection.
        
        Uses fast approximation based on trace minimization.
        
        Args:
            covariances: List of covariance matrices
            
        Returns:
            Array of optimal weights
        """
        n = len(covariances)
        
        # Simple heuristic: weight inversely proportional to trace
        traces = np.array([np.trace(P) for P in covariances])
        weights = 1.0 / traces
        weights /= weights.sum()
        
        return weights
    
    def hierarchical_fusion(self,
                          local_tracks: Dict[str, List[NetworkTrack]],
                          hierarchy_levels: int = 2) -> List[NetworkTrack]:
        """
        Perform hierarchical track fusion.
        
        Fuses tracks in multiple levels for scalability.
        
        Args:
            local_tracks: Dictionary of node_id -> tracks
            hierarchy_levels: Number of hierarchy levels
            
        Returns:
            List of global fused tracks
        """
        if hierarchy_levels < 1:
            raise ValueError("Must have at least 1 hierarchy level")
        
        current_level_tracks = []
        
        # Level 1: Fuse tracks within each node
        for node_id, tracks in local_tracks.items():
            if not tracks:
                continue
            
            # Group tracks by proximity
            grouped = self._group_tracks_by_proximity(tracks)
            
            for group in grouped:
                if len(group) > 1:
                    fused = self.covariance_intersection(group)
                    current_level_tracks.append(fused)
                else:
                    current_level_tracks.append(group[0])
        
        # Additional levels of fusion
        for level in range(1, hierarchy_levels):
            if len(current_level_tracks) <= 1:
                break
            
            # Group and fuse at current level
            grouped = self._group_tracks_by_proximity(current_level_tracks)
            next_level_tracks = []
            
            for group in grouped:
                if len(group) > 1:
                    fused = self.covariance_intersection(group)
                    next_level_tracks.append(fused)
                else:
                    next_level_tracks.append(group[0])
            
            current_level_tracks = next_level_tracks
        
        return current_level_tracks
    
    def _group_tracks_by_proximity(self,
                                  tracks: List[NetworkTrack],
                                  threshold: float = 10.0) -> List[List[NetworkTrack]]:
        """
        Group tracks by spatial proximity.
        
        Args:
            tracks: List of tracks
            threshold: Distance threshold for grouping
            
        Returns:
            List of track groups
        """
        if not tracks:
            return []
        
        groups = []
        assigned = set()
        
        for i, track1 in enumerate(tracks):
            if i in assigned:
                continue
            
            group = [track1]
            assigned.add(i)
            
            for j, track2 in enumerate(tracks[i+1:], start=i+1):
                if j in assigned:
                    continue
                
                # Check distance between tracks
                distance = np.linalg.norm(track1.state[:2] - track2.state[:2])
                if distance < threshold:
                    group.append(track2)
                    assigned.add(j)
            
            groups.append(group)
        
        return groups


class ConsensusTracker:
    """
    Implements consensus-based distributed tracking.
    
    Nodes iteratively exchange information to reach consensus on track states.
    """
    
    def __init__(self,
                 node_id: str,
                 consensus_iterations: int = 10,
                 consensus_weight: float = 0.5):
        """
        Initialize consensus tracker.
        
        Args:
            node_id: Node identifier
            consensus_iterations: Number of consensus iterations
            consensus_weight: Weight for consensus updates
        """
        self.node_id = node_id
        self.consensus_iterations = consensus_iterations
        self.consensus_weight = consensus_weight
        self.local_tracks: Dict[str, NetworkTrack] = {}
        self.neighbor_tracks: Dict[str, Dict[str, NetworkTrack]] = {}
        self.consensus_history: List[Dict] = []
    
    def update_local_track(self, track: NetworkTrack):
        """Update local track estimate."""
        self.local_tracks[track.track_id] = track
    
    def receive_neighbor_track(self, neighbor_id: str, track: NetworkTrack):
        """Receive track from neighbor node."""
        if neighbor_id not in self.neighbor_tracks:
            self.neighbor_tracks[neighbor_id] = {}
        self.neighbor_tracks[neighbor_id][track.track_id] = track
    
    def consensus_iteration(self) -> Dict[str, NetworkTrack]:
        """
        Perform one consensus iteration.
        
        Returns:
            Updated local tracks
        """
        updated_tracks = {}
        
        for track_id, local_track in self.local_tracks.items():
            # Collect neighbor estimates for this track
            neighbor_estimates = []
            for neighbor_id, neighbor_tracks in self.neighbor_tracks.items():
                if track_id in neighbor_tracks:
                    neighbor_estimates.append(neighbor_tracks[track_id])
            
            if not neighbor_estimates:
                updated_tracks[track_id] = local_track
                continue
            
            # Consensus update
            consensus_state = local_track.state.copy()
            consensus_cov = local_track.covariance.copy()
            
            for neighbor_track in neighbor_estimates:
                # Weighted average for state
                consensus_state = (1 - self.consensus_weight) * consensus_state + \
                                self.consensus_weight * neighbor_track.state
                
                # Conservative covariance update
                consensus_cov = (1 - self.consensus_weight) * consensus_cov + \
                              self.consensus_weight * neighbor_track.covariance
            
            # Create updated track
            updated_track = NetworkTrack(
                track_id=track_id,
                local_id=local_track.local_id,
                source_node=self.node_id,
                state=consensus_state,
                covariance=consensus_cov,
                timestamp=local_track.timestamp,
                quality=local_track.quality
            )
            
            updated_tracks[track_id] = updated_track
        
        self.local_tracks = updated_tracks
        return updated_tracks
    
    def run_consensus(self) -> Dict[str, NetworkTrack]:
        """
        Run full consensus algorithm.
        
        Returns:
            Converged track estimates
        """
        for iteration in range(self.consensus_iterations):
            prev_tracks = {tid: t.state.copy() for tid, t in self.local_tracks.items()}
            
            # Perform consensus iteration
            self.consensus_iteration()
            
            # Check convergence
            max_change = 0.0
            for track_id in self.local_tracks:
                if track_id in prev_tracks:
                    change = np.linalg.norm(self.local_tracks[track_id].state - prev_tracks[track_id])
                    max_change = max(max_change, change)
            
            self.consensus_history.append({
                'iteration': iteration,
                'max_change': max_change,
                'num_tracks': len(self.local_tracks)
            })
            
            # Early termination if converged
            if max_change < 1e-6:
                logger.info(f"Consensus converged after {iteration + 1} iterations")
                break
        
        return self.local_tracks