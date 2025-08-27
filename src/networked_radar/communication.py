"""
Communication and Synchronization Module for Networked Radar

Implements communication protocols, time synchronization, and message handling
for distributed radar networks.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import heapq
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimeSyncMessage:
    """Time synchronization message for network-wide time alignment."""
    source_id: str
    timestamp: float
    sync_sequence: int
    clock_offset: float = 0.0
    clock_drift: float = 0.0
    quality: float = 1.0  # Quality/confidence of time source


class MessageRouter:
    """
    Routes messages between radar nodes in the network.
    
    Handles message queuing, prioritization, and delivery with
    realistic communication constraints.
    """
    
    def __init__(self, max_queue_size: int = 1000):
        """
        Initialize message router.
        
        Args:
            max_queue_size: Maximum number of messages in queue
        """
        self.max_queue_size = max_queue_size
        self.message_queue: List[Tuple[float, int, Any]] = []  # Priority queue (arrival_time, priority, message)
        self.message_counter = 0
        self.routing_table: Dict[str, List[str]] = {}  # Node ID to list of reachable nodes
        self.message_handlers: Dict[str, Callable] = {}
        self.statistics = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_dropped': 0,
            'total_latency': 0.0
        }
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register handler for specific message type."""
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for message type: {message_type}")
    
    def send_message(self, message: Any, arrival_time: float, priority: int = 0) -> bool:
        """
        Queue message for delivery.
        
        Args:
            message: Message to send
            arrival_time: When message arrives at destination
            priority: Message priority (higher = more important)
            
        Returns:
            True if message queued successfully
        """
        if len(self.message_queue) >= self.max_queue_size:
            # Drop lowest priority message if queue full
            if priority > self.message_queue[-1][1]:
                heapq.heappop(self.message_queue)
                self.statistics['messages_dropped'] += 1
            else:
                self.statistics['messages_dropped'] += 1
                return False
        
        # Use negative priority for max-heap behavior
        heapq.heappush(self.message_queue, (arrival_time, -priority, message))
        self.statistics['messages_sent'] += 1
        return True
    
    def process_messages(self, current_time: float) -> List[Any]:
        """
        Process messages that have arrived by current time.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of messages ready for processing
        """
        ready_messages = []
        
        while self.message_queue and self.message_queue[0][0] <= current_time:
            arrival_time, neg_priority, message = heapq.heappop(self.message_queue)
            ready_messages.append(message)
            self.statistics['messages_received'] += 1
            self.statistics['total_latency'] += (current_time - arrival_time)
            
            # Call registered handler if available
            if hasattr(message, 'message_type') and message.message_type in self.message_handlers:
                self.message_handlers[message.message_type](message)
        
        return ready_messages
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics."""
        stats = self.statistics.copy()
        if stats['messages_received'] > 0:
            stats['average_latency'] = stats['total_latency'] / stats['messages_received']
        else:
            stats['average_latency'] = 0.0
        stats['queue_size'] = len(self.message_queue)
        return stats


class TimeSync:
    """
    Time synchronization for distributed radar network.
    
    Implements simplified Network Time Protocol (NTP) like synchronization
    with clock offset and drift estimation.
    """
    
    def __init__(self, node_id: str, initial_offset: float = 0.0, drift_rate: float = 0.0):
        """
        Initialize time synchronization.
        
        Args:
            node_id: Node identifier
            initial_offset: Initial clock offset in seconds
            drift_rate: Clock drift rate in seconds/second
        """
        self.node_id = node_id
        self.clock_offset = initial_offset
        self.clock_drift = drift_rate
        self.last_sync_time = 0.0
        self.sync_history: deque = deque(maxlen=10)
        self.is_master = False
        self.sync_quality = 1.0
        
        # Kalman filter state for clock tracking
        self.kf_state = np.array([initial_offset, drift_rate])  # [offset, drift]
        self.kf_covariance = np.eye(2) * 0.01
        self.kf_process_noise = np.eye(2) * 1e-6
        self.kf_measurement_noise = 1e-4
    
    def get_synchronized_time(self, local_time: float) -> float:
        """
        Convert local time to synchronized network time.
        
        Args:
            local_time: Local clock time
            
        Returns:
            Synchronized network time
        """
        time_since_sync = local_time - self.last_sync_time
        return local_time + self.clock_offset + self.clock_drift * time_since_sync
    
    def estimate_offset(self, 
                       request_time: float,
                       server_receive_time: float,
                       server_transmit_time: float,
                       response_time: float) -> Tuple[float, float]:
        """
        Estimate clock offset using NTP-like algorithm.
        
        Args:
            request_time: Local time when request sent
            server_receive_time: Server time when request received
            server_transmit_time: Server time when response sent
            response_time: Local time when response received
            
        Returns:
            Tuple of (offset, round_trip_delay)
        """
        # Calculate round-trip delay
        round_trip_delay = (response_time - request_time) - (server_transmit_time - server_receive_time)
        
        # Calculate clock offset
        offset = ((server_receive_time - request_time) + (server_transmit_time - response_time)) / 2
        
        return offset, round_trip_delay
    
    def update_clock_kalman(self, measured_offset: float, measurement_time: float):
        """
        Update clock estimate using Kalman filter.
        
        Args:
            measured_offset: Measured clock offset
            measurement_time: Time of measurement
        """
        dt = measurement_time - self.last_sync_time
        if dt <= 0:
            return
        
        # Prediction step
        F = np.array([[1, dt], [0, 1]])  # State transition
        self.kf_state = F @ self.kf_state
        self.kf_covariance = F @ self.kf_covariance @ F.T + self.kf_process_noise
        
        # Update step
        H = np.array([1, 0])  # Measurement matrix (observe offset only)
        y = measured_offset - H @ self.kf_state  # Innovation
        S = H @ self.kf_covariance @ H.T + self.kf_measurement_noise  # Innovation covariance
        K = self.kf_covariance @ H.T / S  # Kalman gain
        
        self.kf_state = self.kf_state + K * y
        self.kf_covariance = (np.eye(2) - np.outer(K, H)) @ self.kf_covariance
        
        # Update clock parameters
        self.clock_offset = self.kf_state[0]
        self.clock_drift = self.kf_state[1]
        self.last_sync_time = measurement_time
        
        # Store in history
        self.sync_history.append({
            'time': measurement_time,
            'offset': measured_offset,
            'estimated_offset': self.clock_offset,
            'drift': self.clock_drift
        })
    
    def get_sync_quality(self) -> float:
        """
        Get synchronization quality metric.
        
        Returns:
            Quality score from 0 (poor) to 1 (excellent)
        """
        if len(self.sync_history) < 2:
            return 0.5
        
        # Calculate based on offset variance in recent history
        recent_offsets = [h['offset'] for h in self.sync_history]
        offset_std = np.std(recent_offsets)
        
        # Map standard deviation to quality score
        # std < 1us = excellent, std > 1ms = poor
        if offset_std < 1e-6:
            quality = 1.0
        elif offset_std > 1e-3:
            quality = 0.1
        else:
            quality = 1.0 - np.log10(offset_std / 1e-6) / 3
        
        self.sync_quality = np.clip(quality, 0.0, 1.0)
        return self.sync_quality


class DataCompression:
    """
    Data compression for efficient network communication.
    
    Implements track compression and decompression for bandwidth-limited links.
    """
    
    @staticmethod
    def compress_track(track: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress track data for transmission.
        
        Args:
            track: Full track dictionary
            
        Returns:
            Compressed track dictionary
        """
        compressed = {
            'id': track.get('track_id', 'unknown'),
            't': track.get('timestamp', 0.0),  # Shortened keys
            's': None,  # State
            'c': None,  # Covariance (compressed)
            'q': track.get('quality', 1.0)
        }
        
        # Compress state vector
        if 'state' in track:
            state = np.asarray(track['state'])
            # Quantize to reduce precision (16-bit floats)
            compressed['s'] = state.astype(np.float16).tolist()
        
        # Compress covariance matrix (store only diagonal + selected off-diagonals)
        if 'covariance' in track:
            cov = np.asarray(track['covariance'])
            # Store diagonal and first off-diagonal
            compressed['c'] = {
                'd': np.diag(cov).astype(np.float16).tolist(),  # Diagonal
                'o': np.diag(cov, k=1).astype(np.float16).tolist() if cov.shape[0] > 1 else []  # Off-diagonal
            }
        
        return compressed
    
    @staticmethod
    def decompress_track(compressed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decompress track data received from network.
        
        Args:
            compressed: Compressed track dictionary
            
        Returns:
            Full track dictionary
        """
        track = {
            'track_id': compressed.get('id', 'unknown'),
            'timestamp': compressed.get('t', 0.0),
            'quality': compressed.get('q', 1.0)
        }
        
        # Decompress state
        if 's' in compressed and compressed['s'] is not None:
            track['state'] = np.array(compressed['s'], dtype=np.float32)
        
        # Decompress covariance
        if 'c' in compressed and compressed['c'] is not None:
            diag = np.array(compressed['c']['d'], dtype=np.float32)
            n = len(diag)
            cov = np.diag(diag)
            
            # Restore off-diagonal elements if present
            if 'o' in compressed['c'] and len(compressed['c']['o']) > 0:
                off_diag = np.array(compressed['c']['o'], dtype=np.float32)
                for i in range(len(off_diag)):
                    cov[i, i+1] = off_diag[i]
                    cov[i+1, i] = off_diag[i]  # Symmetric
            
            track['covariance'] = cov
        
        return track
    
    @staticmethod
    def estimate_compression_ratio(track: Dict[str, Any]) -> float:
        """
        Estimate compression ratio for track data.
        
        Args:
            track: Track dictionary
            
        Returns:
            Compression ratio (0 to 1, lower is better)
        """
        original_size = 0
        compressed_size = 0
        
        if 'state' in track:
            state = np.asarray(track['state'])
            original_size += state.nbytes
            compressed_size += state.size * 2  # 16-bit floats
        
        if 'covariance' in track:
            cov = np.asarray(track['covariance'])
            original_size += cov.nbytes
            # Only diagonal + first off-diagonal stored
            compressed_size += (cov.shape[0] + cov.shape[0] - 1) * 2
        
        if original_size == 0:
            return 1.0
        
        return compressed_size / original_size


class NetworkProtocol:
    """
    Network protocol implementation for radar communication.
    
    Defines message types and protocol rules for distributed radar networks.
    """
    
    # Message types
    MSG_TRACK_UPDATE = "TRACK_UPDATE"
    MSG_TRACK_REQUEST = "TRACK_REQUEST"
    MSG_DETECTION_REPORT = "DETECTION_REPORT"
    MSG_RESOURCE_REQUEST = "RESOURCE_REQUEST"
    MSG_RESOURCE_ALLOCATION = "RESOURCE_ALLOCATION"
    MSG_TIME_SYNC = "TIME_SYNC"
    MSG_HEARTBEAT = "HEARTBEAT"
    MSG_FUSION_RESULT = "FUSION_RESULT"
    MSG_COMMAND = "COMMAND"
    MSG_STATUS = "STATUS"
    
    @staticmethod
    def create_track_update_message(node_id: str, tracks: List[Dict], compressed: bool = True) -> Dict[str, Any]:
        """Create track update message."""
        if compressed:
            tracks = [DataCompression.compress_track(t) for t in tracks]
        
        return {
            'message_type': NetworkProtocol.MSG_TRACK_UPDATE,
            'source_id': node_id,
            'timestamp': time.time(),
            'tracks': tracks,
            'compressed': compressed,
            'track_count': len(tracks)
        }
    
    @staticmethod
    def create_heartbeat_message(node_id: str, status: str = "ACTIVE") -> Dict[str, Any]:
        """Create heartbeat message for node liveness."""
        return {
            'message_type': NetworkProtocol.MSG_HEARTBEAT,
            'source_id': node_id,
            'timestamp': time.time(),
            'status': status,
            'uptime': 0  # Would track actual uptime
        }
    
    @staticmethod
    def validate_message(message: Dict[str, Any]) -> bool:
        """
        Validate message format.
        
        Args:
            message: Message dictionary
            
        Returns:
            True if valid
        """
        required_fields = ['message_type', 'source_id', 'timestamp']
        return all(field in message for field in required_fields)