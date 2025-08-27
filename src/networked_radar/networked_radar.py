"""
Networked Radar System Implementation

This module implements distributed and networked radar capabilities including
bistatic, multistatic, and MIMO radar configurations with data fusion.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..radar import Radar, RadarParameters
from ..constants import SPEED_OF_LIGHT

logger = logging.getLogger(__name__)


class NetworkArchitecture(Enum):
    """Network architecture types for radar fusion."""
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
    HYBRID = "hybrid"
    HIERARCHICAL = "hierarchical"


class RadarNodeType(Enum):
    """Types of radar nodes in the network."""
    MONOSTATIC = "monostatic"  # Transmit and receive
    BISTATIC_TX = "bistatic_tx"  # Transmit only
    BISTATIC_RX = "bistatic_rx"  # Receive only
    MULTISTATIC = "multistatic"  # Part of multistatic network


@dataclass
class NetworkNode:
    """Represents a single node in the radar network."""
    node_id: str
    position: np.ndarray  # [x, y, z] in meters
    orientation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))  # [az, el, roll] in radians
    node_type: RadarNodeType = RadarNodeType.MONOSTATIC
    is_active: bool = True
    radar_params: Optional[RadarParameters] = None
    
    def __post_init__(self):
        """Ensure position and orientation are numpy arrays with float dtype."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.orientation = np.asarray(self.orientation, dtype=np.float64)


@dataclass
class CommunicationLink:
    """Communication link between radar nodes."""
    source_id: str
    target_id: str
    bandwidth: float  # bits/second
    latency: float  # seconds
    packet_loss_rate: float = 0.0  # 0 to 1
    is_bidirectional: bool = True
    max_message_size: int = 65536  # bytes
    
    def can_transmit(self, message_size: int, current_time: float) -> bool:
        """Check if link can transmit message."""
        if message_size > self.max_message_size:
            return False
        # Simulate packet loss
        if np.random.random() < self.packet_loss_rate:
            return False
        return True
    
    def transmission_time(self, message_size: int) -> float:
        """Calculate transmission time for message."""
        bits = message_size * 8
        return bits / self.bandwidth + self.latency


@dataclass
class NetworkMessage:
    """Message passed between radar nodes."""
    source_id: str
    target_id: str
    timestamp: float
    message_type: str
    payload: Any
    priority: int = 0
    size_bytes: int = 0
    
    def __post_init__(self):
        """Calculate message size if not provided."""
        if self.size_bytes == 0:
            # Rough estimate based on payload type
            if isinstance(self.payload, dict):
                self.size_bytes = len(str(self.payload)) * 2
            elif isinstance(self.payload, np.ndarray):
                self.size_bytes = self.payload.nbytes
            else:
                self.size_bytes = 1024  # Default size


class NetworkedRadar(Radar):
    """
    Extended radar class with networking capabilities.
    
    Supports distributed radar configurations including bistatic,
    multistatic, and MIMO radar systems with data fusion.
    """
    
    def __init__(self, 
                 params: RadarParameters,
                 node: NetworkNode,
                 network_architecture: NetworkArchitecture = NetworkArchitecture.CENTRALIZED):
        """
        Initialize networked radar node.
        
        Args:
            params: Radar parameters
            node: Network node configuration
            network_architecture: Type of network architecture
        """
        super().__init__(params)
        self.node = node
        self.network_architecture = network_architecture
        self.connected_nodes: Dict[str, NetworkNode] = {}
        self.communication_links: Dict[str, CommunicationLink] = {}
        self.message_queue: List[NetworkMessage] = []
        self.time_sync_error = 0.0  # Time synchronization error in seconds
        
        # Tracking and fusion state
        self.local_tracks: Dict[str, Any] = {}
        self.network_tracks: Dict[str, Any] = {}
        self.track_associations: Dict[str, List[str]] = {}
        
        logger.info(f"Initialized networked radar node {node.node_id} at position {node.position}")
    
    def add_connection(self, remote_node: NetworkNode, link: CommunicationLink):
        """
        Add connection to another radar node.
        
        Args:
            remote_node: Remote network node
            link: Communication link parameters
        """
        self.connected_nodes[remote_node.node_id] = remote_node
        self.communication_links[remote_node.node_id] = link
        logger.debug(f"Added connection from {self.node.node_id} to {remote_node.node_id}")
    
    def calculate_bistatic_range(self, 
                                 target_position: np.ndarray,
                                 transmitter_position: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate bistatic range for target.
        
        Args:
            target_position: Target position [x, y, z]
            transmitter_position: Transmitter position [x, y, z]
            
        Returns:
            Tuple of (bistatic_range, tx_range, rx_range)
        """
        # Range from transmitter to target
        tx_range = np.linalg.norm(target_position - transmitter_position)
        
        # Range from target to receiver (this node)
        rx_range = np.linalg.norm(target_position - self.node.position)
        
        # Total bistatic range
        bistatic_range = tx_range + rx_range
        
        return bistatic_range, tx_range, rx_range
    
    def calculate_bistatic_doppler(self,
                                  target_position: np.ndarray,
                                  target_velocity: np.ndarray,
                                  transmitter_position: np.ndarray,
                                  transmitter_velocity: Optional[np.ndarray] = None) -> float:
        """
        Calculate bistatic Doppler shift.
        
        Args:
            target_position: Target position [x, y, z]
            target_velocity: Target velocity [vx, vy, vz]
            transmitter_position: Transmitter position
            transmitter_velocity: Transmitter velocity (if moving)
            
        Returns:
            Bistatic Doppler frequency in Hz
        """
        if transmitter_velocity is None:
            transmitter_velocity = np.zeros(3)
        
        # Unit vectors from transmitter to target and target to receiver
        tx_to_target = target_position - transmitter_position
        tx_unit = tx_to_target / np.linalg.norm(tx_to_target)
        
        target_to_rx = self.node.position - target_position
        rx_unit = target_to_rx / np.linalg.norm(target_to_rx)
        
        # Relative velocities
        target_rel_velocity = target_velocity - transmitter_velocity
        
        # Doppler components
        tx_doppler = np.dot(target_rel_velocity, tx_unit)
        rx_doppler = np.dot(target_velocity, rx_unit)
        
        # Total bistatic Doppler
        wavelength = self.params.wavelength
        bistatic_doppler = (tx_doppler + rx_doppler) / wavelength
        
        return bistatic_doppler
    
    def send_message(self, target_id: str, message_type: str, payload: Any, priority: int = 0) -> bool:
        """
        Send message to another node.
        
        Args:
            target_id: Target node ID
            message_type: Type of message
            payload: Message payload
            priority: Message priority
            
        Returns:
            True if message queued successfully
        """
        if target_id not in self.communication_links:
            logger.warning(f"No link to node {target_id}")
            return False
        
        message = NetworkMessage(
            source_id=self.node.node_id,
            target_id=target_id,
            timestamp=self.current_time,  # Assumes current_time is tracked
            message_type=message_type,
            payload=payload,
            priority=priority
        )
        
        link = self.communication_links[target_id]
        if link.can_transmit(message.size_bytes, self.current_time):
            self.message_queue.append(message)
            return True
        
        return False
    
    def broadcast_tracks(self, tracks: Dict[str, Any]) -> int:
        """
        Broadcast local tracks to all connected nodes.
        
        Args:
            tracks: Local track dictionary
            
        Returns:
            Number of successful broadcasts
        """
        success_count = 0
        for node_id in self.connected_nodes:
            if self.send_message(node_id, "TRACK_UPDATE", tracks, priority=1):
                success_count += 1
        return success_count
    
    def fuse_tracks_covariance_intersection(self,
                                           track1: Dict[str, Any],
                                           track2: Dict[str, Any],
                                           omega: float = 0.5) -> Dict[str, Any]:
        """
        Fuse two tracks using Covariance Intersection.
        
        Covariance Intersection provides consistent fusion when
        cross-correlations between estimates are unknown.
        
        Args:
            track1: First track with 'state' and 'covariance'
            track2: Second track with 'state' and 'covariance'
            omega: Mixing parameter (0 to 1)
            
        Returns:
            Fused track
        """
        # Extract states and covariances
        x1 = np.asarray(track1['state'])
        P1 = np.asarray(track1['covariance'])
        x2 = np.asarray(track2['state'])
        P2 = np.asarray(track2['covariance'])
        
        # Covariance Intersection fusion
        P1_inv = np.linalg.inv(P1)
        P2_inv = np.linalg.inv(P2)
        
        # Fused covariance
        P_fused_inv = omega * P1_inv + (1 - omega) * P2_inv
        P_fused = np.linalg.inv(P_fused_inv)
        
        # Fused state
        x_fused = P_fused @ (omega * P1_inv @ x1 + (1 - omega) * P2_inv @ x2)
        
        # Create fused track
        fused_track = {
            'state': x_fused,
            'covariance': P_fused,
            'source_nodes': [track1.get('node_id', 'unknown'), track2.get('node_id', 'unknown')],
            'fusion_method': 'covariance_intersection',
            'omega': omega,
            'timestamp': max(track1.get('timestamp', 0), track2.get('timestamp', 0))
        }
        
        return fused_track
    
    def optimal_omega_ci(self, P1: np.ndarray, P2: np.ndarray) -> float:
        """
        Find optimal omega for Covariance Intersection by minimizing determinant.
        
        Args:
            P1: First covariance matrix
            P2: Second covariance matrix
            
        Returns:
            Optimal omega value
        """
        # Search for omega that minimizes determinant of fused covariance
        omegas = np.linspace(0.01, 0.99, 50)
        min_det = float('inf')
        best_omega = 0.5
        
        for omega in omegas:
            try:
                P1_inv = np.linalg.inv(P1)
                P2_inv = np.linalg.inv(P2)
                P_fused_inv = omega * P1_inv + (1 - omega) * P2_inv
                P_fused = np.linalg.inv(P_fused_inv)
                det = np.linalg.det(P_fused)
                
                if det < min_det:
                    min_det = det
                    best_omega = omega
            except np.linalg.LinAlgError:
                continue
        
        return best_omega
    
    def calculate_coverage_overlap(self, other_node: NetworkNode, max_range: float) -> float:
        """
        Calculate coverage overlap percentage with another node.
        
        Args:
            other_node: Other radar node
            max_range: Maximum detection range
            
        Returns:
            Overlap percentage (0 to 1)
        """
        # Distance between nodes
        distance = np.linalg.norm(self.node.position - other_node.position)
        
        if distance >= 2 * max_range:
            # No overlap
            return 0.0
        elif distance <= 0:
            # Same position, complete overlap
            return 1.0
        else:
            # Calculate overlap area using circle intersection formula
            # Simplified calculation for 2D
            r1 = r2 = max_range
            d = distance
            
            # Area of intersection of two circles
            if d <= abs(r1 - r2):
                # One circle inside the other
                return min(r1, r2)**2 / max(r1, r2)**2
            else:
                # Partial overlap
                part1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
                part2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
                part3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
                
                intersection_area = part1 + part2 - part3
                union_area = np.pi * r1**2 + np.pi * r2**2 - intersection_area
                
                return intersection_area / union_area
    
    def get_network_topology(self) -> Dict[str, Any]:
        """
        Get network topology information.
        
        Returns:
            Dictionary containing topology information
        """
        topology = {
            'node_id': self.node.node_id,
            'position': self.node.position.tolist(),
            'node_type': self.node.node_type.value,
            'connected_nodes': list(self.connected_nodes.keys()),
            'num_connections': len(self.connected_nodes),
            'architecture': self.network_architecture.value,
            'links': {}
        }
        
        for node_id, link in self.communication_links.items():
            topology['links'][node_id] = {
                'bandwidth': link.bandwidth,
                'latency': link.latency,
                'packet_loss': link.packet_loss_rate
            }
        
        return topology