"""
Data Fusion Center for Networked Radar Systems

Implements centralized, decentralized, and hybrid fusion architectures
for multi-radar networks.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

from .distributed_tracking import NetworkTrack, TrackAssociator, DistributedTrackFusion
from .communication import MessageRouter, NetworkProtocol, DataCompression

logger = logging.getLogger(__name__)


class FusionArchitecture(Enum):
    """Types of fusion architectures."""
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"


@dataclass
class FusionNode:
    """Represents a fusion node in the network."""
    node_id: str
    level: int  # Hierarchy level (0 = local, higher = more global)
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    coverage_area: Optional[np.ndarray] = None  # [min_x, min_y, max_x, max_y]
    processing_capacity: float = 1.0  # Relative processing capability


class DataFusionCenter:
    """
    Central data fusion center for networked radar.
    
    Manages track fusion, association, and distribution across the network.
    """
    
    def __init__(self,
                 center_id: str,
                 architecture: FusionArchitecture = FusionArchitecture.CENTRALIZED,
                 max_latency: float = 0.1):
        """
        Initialize data fusion center.
        
        Args:
            center_id: Fusion center identifier
            architecture: Fusion architecture type
            max_latency: Maximum acceptable latency (seconds)
        """
        self.center_id = center_id
        self.architecture = architecture
        self.max_latency = max_latency
        
        # Track management
        self.global_tracks: Dict[str, NetworkTrack] = {}
        self.local_tracks: Dict[str, Dict[str, NetworkTrack]] = {}  # node_id -> tracks
        self.track_id_mapping: Dict[str, str] = {}  # local_id -> global_id
        
        # Fusion components
        self.track_fusion = DistributedTrackFusion()
        self.track_associator = TrackAssociator()
        self.message_router = MessageRouter()
        
        # Network topology
        self.fusion_nodes: Dict[str, FusionNode] = {}
        self.node_connections: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics
        self.fusion_statistics = {
            'tracks_received': 0,
            'tracks_fused': 0,
            'tracks_distributed': 0,
            'fusion_cycles': 0,
            'average_latency': 0.0
        }
        
        logger.info(f"Initialized fusion center {center_id} with {architecture.value} architecture")
    
    def register_node(self, node: FusionNode):
        """Register a fusion node in the network."""
        self.fusion_nodes[node.node_id] = node
        if node.parent_id:
            self.node_connections[node.parent_id].add(node.node_id)
        for child_id in node.children_ids:
            self.node_connections[node.node_id].add(child_id)
    
    def receive_local_tracks(self, node_id: str, tracks: List[NetworkTrack]):
        """
        Receive local tracks from a radar node.
        
        Args:
            node_id: Source node ID
            tracks: List of local tracks
        """
        self.local_tracks[node_id] = {t.track_id: t for t in tracks}
        self.fusion_statistics['tracks_received'] += len(tracks)
        
        logger.debug(f"Received {len(tracks)} tracks from node {node_id}")
    
    def centralized_fusion(self) -> Dict[str, NetworkTrack]:
        """
        Perform centralized track fusion.
        
        All tracks sent to central node for processing.
        
        Returns:
            Dictionary of fused global tracks
        """
        all_local_tracks = []
        for node_tracks in self.local_tracks.values():
            all_local_tracks.extend(node_tracks.values())
        
        if not all_local_tracks:
            return {}
        
        # Group tracks by association
        track_groups = self._associate_all_tracks(all_local_tracks)
        
        # Fuse associated tracks
        fused_tracks = {}
        for group in track_groups:
            if len(group) > 1:
                # Multiple tracks - fuse them
                fused = self.track_fusion.covariance_intersection(group)
                fused_tracks[fused.track_id] = fused
                self.fusion_statistics['tracks_fused'] += 1
            else:
                # Single track - use as is
                fused_tracks[group[0].track_id] = group[0]
        
        self.global_tracks = fused_tracks
        self.fusion_statistics['fusion_cycles'] += 1
        
        return fused_tracks
    
    def decentralized_fusion(self) -> Dict[str, NetworkTrack]:
        """
        Perform decentralized track fusion.
        
        Each node performs local fusion with neighbors.
        
        Returns:
            Dictionary of fused tracks
        """
        fused_tracks = {}
        
        for node_id, local_tracks in self.local_tracks.items():
            # Get neighbor tracks
            neighbor_tracks = []
            for neighbor_id in self.node_connections.get(node_id, []):
                if neighbor_id in self.local_tracks:
                    neighbor_tracks.extend(self.local_tracks[neighbor_id].values())
            
            # Fuse local with neighbor tracks
            all_tracks = list(local_tracks.values()) + neighbor_tracks
            
            if all_tracks:
                # Associate and fuse
                track_groups = self._associate_all_tracks(all_tracks)
                
                for group in track_groups:
                    if len(group) > 1:
                        fused = self.track_fusion.covariance_intersection(group)
                        fused_tracks[fused.track_id] = fused
                        self.fusion_statistics['tracks_fused'] += 1
                    else:
                        fused_tracks[group[0].track_id] = group[0]
        
        self.global_tracks = fused_tracks
        self.fusion_statistics['fusion_cycles'] += 1
        
        return fused_tracks
    
    def hierarchical_fusion(self) -> Dict[str, NetworkTrack]:
        """
        Perform hierarchical track fusion.
        
        Tracks fused at multiple levels of hierarchy.
        
        Returns:
            Dictionary of fused global tracks
        """
        # Group nodes by hierarchy level
        levels = defaultdict(list)
        for node_id, node in self.fusion_nodes.items():
            levels[node.level].append(node_id)
        
        # Start from lowest level
        current_tracks = {}
        max_level = max(levels.keys()) if levels else 0
        
        for level in range(max_level + 1):
            level_tracks = []
            
            # Collect tracks at this level
            for node_id in levels[level]:
                if level == 0:
                    # Base level - use local tracks
                    if node_id in self.local_tracks:
                        level_tracks.extend(self.local_tracks[node_id].values())
                else:
                    # Higher levels - use fused tracks from children
                    node = self.fusion_nodes[node_id]
                    for child_id in node.children_ids:
                        if child_id in current_tracks:
                            level_tracks.append(current_tracks[child_id])
            
            # Fuse tracks at this level
            if level_tracks:
                track_groups = self._associate_all_tracks(level_tracks)
                
                for group in track_groups:
                    if len(group) > 1:
                        fused = self.track_fusion.covariance_intersection(group)
                        current_tracks[fused.track_id] = fused
                        self.fusion_statistics['tracks_fused'] += 1
                    else:
                        current_tracks[group[0].track_id] = group[0]
        
        self.global_tracks = current_tracks
        self.fusion_statistics['fusion_cycles'] += 1
        
        return current_tracks
    
    def hybrid_fusion(self,
                     centralized_nodes: Set[str],
                     decentralized_groups: List[Set[str]]) -> Dict[str, NetworkTrack]:
        """
        Perform hybrid fusion combining centralized and decentralized.
        
        Some nodes report to center, others fuse locally.
        
        Args:
            centralized_nodes: Nodes reporting to center
            decentralized_groups: Groups of nodes fusing locally
            
        Returns:
            Dictionary of fused tracks
        """
        fused_tracks = {}
        
        # Process centralized nodes
        central_tracks = []
        for node_id in centralized_nodes:
            if node_id in self.local_tracks:
                central_tracks.extend(self.local_tracks[node_id].values())
        
        if central_tracks:
            track_groups = self._associate_all_tracks(central_tracks)
            for group in track_groups:
                if len(group) > 1:
                    fused = self.track_fusion.covariance_intersection(group)
                    fused_tracks[fused.track_id] = fused
                else:
                    fused_tracks[group[0].track_id] = group[0]
        
        # Process decentralized groups
        for group_nodes in decentralized_groups:
            group_tracks = []
            for node_id in group_nodes:
                if node_id in self.local_tracks:
                    group_tracks.extend(self.local_tracks[node_id].values())
            
            if group_tracks:
                track_groups = self._associate_all_tracks(group_tracks)
                for tracks in track_groups:
                    if len(tracks) > 1:
                        fused = self.track_fusion.covariance_intersection(tracks)
                        fused_tracks[fused.track_id] = fused
                    else:
                        fused_tracks[tracks[0].track_id] = tracks[0]
        
        self.global_tracks = fused_tracks
        self.fusion_statistics['fusion_cycles'] += 1
        
        return fused_tracks
    
    def perform_fusion(self) -> Dict[str, NetworkTrack]:
        """
        Perform fusion based on configured architecture.
        
        Returns:
            Dictionary of fused tracks
        """
        if self.architecture == FusionArchitecture.CENTRALIZED:
            return self.centralized_fusion()
        elif self.architecture == FusionArchitecture.DECENTRALIZED:
            return self.decentralized_fusion()
        elif self.architecture == FusionArchitecture.HIERARCHICAL:
            return self.hierarchical_fusion()
        elif self.architecture == FusionArchitecture.HYBRID:
            # Default hybrid configuration
            centralized = set(list(self.local_tracks.keys())[:len(self.local_tracks)//2])
            decentralized = [set(list(self.local_tracks.keys())[len(self.local_tracks)//2:])]
            return self.hybrid_fusion(centralized, decentralized)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
    
    def distribute_global_tracks(self, compress: bool = True) -> int:
        """
        Distribute global tracks back to nodes.
        
        Args:
            compress: Whether to compress tracks for transmission
            
        Returns:
            Number of tracks distributed
        """
        count = 0
        
        for node_id in self.local_tracks:
            # Prepare tracks for distribution
            tracks_to_send = list(self.global_tracks.values())
            
            if compress:
                compressed_tracks = [DataCompression.compress_track({
                    'track_id': t.track_id,
                    'state': t.state,
                    'covariance': t.covariance,
                    'timestamp': t.timestamp,
                    'quality': t.quality
                }) for t in tracks_to_send]
                payload = compressed_tracks
            else:
                payload = tracks_to_send
            
            # Create distribution message
            message = NetworkProtocol.create_track_update_message(
                self.center_id, payload, compressed=compress
            )
            
            # Queue for distribution
            self.message_router.send_message(message, 0.0, priority=1)
            count += len(tracks_to_send)
        
        self.fusion_statistics['tracks_distributed'] = count
        return count
    
    def _associate_all_tracks(self, tracks: List[NetworkTrack]) -> List[List[NetworkTrack]]:
        """
        Associate all tracks into groups.
        
        Args:
            tracks: List of all tracks
            
        Returns:
            List of associated track groups
        """
        if not tracks:
            return []
        
        # Build association graph
        n = len(tracks)
        associated = np.zeros((n, n), dtype=bool)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check if tracks can be associated
                distance = self._calculate_track_distance(tracks[i], tracks[j])
                if distance < self.track_associator.gate_threshold:
                    associated[i, j] = True
                    associated[j, i] = True
        
        # Find connected components (track groups)
        groups = []
        visited = set()
        
        for i in range(n):
            if i in visited:
                continue
            
            # BFS to find connected component
            group = []
            queue = [i]
            
            while queue:
                idx = queue.pop(0)
                if idx in visited:
                    continue
                
                visited.add(idx)
                group.append(tracks[idx])
                
                # Add connected tracks
                for j in range(n):
                    if associated[idx, j] and j not in visited:
                        queue.append(j)
            
            groups.append(group)
        
        return groups
    
    def _calculate_track_distance(self, track1: NetworkTrack, track2: NetworkTrack) -> float:
        """
        Calculate statistical distance between tracks.
        
        Args:
            track1: First track
            track2: Second track
            
        Returns:
            Statistical distance
        """
        # Use position components for distance
        pos1 = track1.state[:2] if len(track1.state) >= 2 else track1.state
        pos2 = track2.state[:2] if len(track2.state) >= 2 else track2.state
        cov1 = track1.covariance[:2, :2] if track1.covariance.shape[0] >= 2 else track1.covariance
        cov2 = track2.covariance[:2, :2] if track2.covariance.shape[0] >= 2 else track2.covariance
        
        return self.track_associator.mahalanobis_distance(pos1, cov1, pos2, cov2)
    
    def get_fusion_metrics(self) -> Dict[str, Any]:
        """
        Get fusion performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = self.fusion_statistics.copy()
        
        # Add computed metrics
        if self.fusion_statistics['fusion_cycles'] > 0:
            metrics['tracks_per_cycle'] = (
                self.fusion_statistics['tracks_fused'] /
                self.fusion_statistics['fusion_cycles']
            )
        
        metrics['num_global_tracks'] = len(self.global_tracks)
        metrics['num_nodes'] = len(self.local_tracks)
        metrics['architecture'] = self.architecture.value
        
        # Track quality metrics
        if self.global_tracks:
            qualities = [t.quality for t in self.global_tracks.values()]
            metrics['average_track_quality'] = np.mean(qualities)
            metrics['min_track_quality'] = np.min(qualities)
            metrics['max_track_quality'] = np.max(qualities)
        
        return metrics
    
    def handle_track_conflict(self,
                             track1: NetworkTrack,
                             track2: NetworkTrack) -> NetworkTrack:
        """
        Handle conflicting track reports.
        
        Args:
            track1: First track
            track2: Second track
            
        Returns:
            Resolved track
        """
        # Resolve based on quality and timestamp
        if track1.quality > track2.quality:
            return track1
        elif track2.quality > track1.quality:
            return track2
        elif track1.timestamp > track2.timestamp:
            return track1
        else:
            return track2
    
    def optimize_fusion_schedule(self,
                                node_latencies: Dict[str, float],
                                processing_times: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize fusion schedule to minimize latency.
        
        Args:
            node_latencies: Communication latency for each node
            processing_times: Processing time for each node
            
        Returns:
            Optimal schedule (node_id -> fusion_time)
        """
        schedule = {}
        
        # Simple greedy scheduling - process lowest latency first
        sorted_nodes = sorted(node_latencies.keys(), key=lambda x: node_latencies[x])
        
        current_time = 0.0
        for node_id in sorted_nodes:
            # Account for communication and processing
            arrival_time = current_time + node_latencies[node_id]
            processing_time = processing_times.get(node_id, 0.01)
            
            schedule[node_id] = arrival_time
            current_time = arrival_time + processing_time
        
        return schedule