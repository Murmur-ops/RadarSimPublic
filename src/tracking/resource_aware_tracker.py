"""
Resource-Aware Tracking System for RadarSim

This module provides a ResourceAwareTracker class that wraps existing IMM trackers
with intelligent resource management capabilities including:
- Track quality degradation modeling when updates are sparse  
- Predictive scheduling for high-priority tracks
- Coasting capability for low-priority tracks
- Interface with resource manager for beam requests
- Adaptive tracking based on available radar resources

The ResourceAwareTracker coordinates with the resource management system to
optimize tracking performance under resource constraints while maintaining
track continuity for critical targets.

Author: RadarSim Development Team
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import warnings

# Import existing RadarSim components
from .imm_filter import IMMFilter, IMMParameters
from .tracker_base import BaseTracker, Track, Measurement, TrackState, TrackingMetrics
from .integrated_trackers import InteractingMultipleModel, JPDATracker, TrackingConfiguration
from ..resource_management.resource_manager import ResourceManager, TargetInfo
from ..resource_management.priority_calculator import PriorityCalculator, ThreatLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrackQuality(Enum):
    """Track quality levels based on update frequency and uncertainty."""
    EXCELLENT = "excellent"    # Recent updates, low uncertainty
    GOOD = "good"             # Regular updates, moderate uncertainty  
    FAIR = "fair"             # Sparse updates, higher uncertainty
    POOR = "poor"             # Very sparse updates, high uncertainty
    COASTING = "coasting"     # No recent updates, predicting only


class ResourceRequest(Enum):
    """Types of resource requests from tracker."""
    URGENT_UPDATE = "urgent_update"        # High-priority immediate update
    SCHEDULED_UPDATE = "scheduled_update"  # Regular scheduled update
    CONFIRMATION = "confirmation"          # Confirm track existence
    CLASSIFICATION = "classification"      # Improve target classification
    COAST_CHECK = "coast_check"           # Check coasting track


@dataclass
class TrackResourceState:
    """
    Resource state information for a track.
    
    Attributes:
        track_id: Unique track identifier
        last_resource_time: Last time resources were allocated
        requested_update_interval: Desired update interval (seconds)
        minimum_update_interval: Minimum acceptable update interval
        priority_score: Current priority score
        quality_level: Current track quality assessment
        resource_requests: Pending resource requests
        coasting_start_time: When track started coasting (None if not coasting)
        prediction_horizon: How far ahead track can be predicted reliably
        uncertainty_growth_rate: Rate of uncertainty growth during coasting
        missed_opportunities: Count of missed resource allocation opportunities
    """
    track_id: str
    last_resource_time: float = 0.0
    requested_update_interval: float = 1.0
    minimum_update_interval: float = 5.0
    priority_score: float = 0.5
    quality_level: TrackQuality = TrackQuality.GOOD
    resource_requests: List[ResourceRequest] = field(default_factory=list)
    coasting_start_time: Optional[float] = None
    prediction_horizon: float = 10.0
    uncertainty_growth_rate: float = 0.1
    missed_opportunities: int = 0


@dataclass 
class QualityDegradationModel:
    """
    Model for track quality degradation over time without updates.
    
    Attributes:
        base_uncertainty_growth: Base rate of uncertainty growth per second
        velocity_uncertainty_factor: Additional uncertainty growth factor for velocity
        acceleration_uncertainty_factor: Additional uncertainty growth for acceleration
        maneuver_detection_threshold: Threshold for detecting potential maneuvers
        quality_degradation_rates: Rate of quality degradation for each level
        coasting_time_limits: Maximum coasting time for each quality level
    """
    base_uncertainty_growth: float = 0.5       # m²/s
    velocity_uncertainty_factor: float = 0.1   # (m/s)²/s
    acceleration_uncertainty_factor: float = 0.01  # (m/s²)²/s
    maneuver_detection_threshold: float = 2.0   # m/s²
    quality_degradation_rates: Dict[TrackQuality, float] = field(default_factory=lambda: {
        TrackQuality.EXCELLENT: 0.1,  # Excellent to Good in 10s
        TrackQuality.GOOD: 0.05,      # Good to Fair in 20s
        TrackQuality.FAIR: 0.02,      # Fair to Poor in 50s
        TrackQuality.POOR: 0.01       # Poor to Coasting in 100s
    })
    coasting_time_limits: Dict[TrackQuality, float] = field(default_factory=lambda: {
        TrackQuality.EXCELLENT: 30.0,
        TrackQuality.GOOD: 20.0,
        TrackQuality.FAIR: 10.0,
        TrackQuality.POOR: 5.0
    })


class ResourceAwareTracker:
    """
    Resource-aware tracking system that optimizes tracking performance under resource constraints.
    
    This tracker wraps existing IMM trackers with intelligent resource management,
    providing adaptive tracking capabilities, quality monitoring, and predictive
    scheduling to maintain optimal tracking performance with limited radar resources.
    """
    
    def __init__(self,
                 base_tracker: Optional[BaseTracker] = None,
                 resource_manager: Optional[ResourceManager] = None,
                 priority_calculator: Optional[PriorityCalculator] = None,
                 degradation_model: Optional[QualityDegradationModel] = None,
                 max_tracks: int = 200,
                 enable_predictive_scheduling: bool = True,
                 enable_adaptive_coasting: bool = True):
        """
        Initialize the resource-aware tracker.
        
        Args:
            base_tracker: Underlying tracking system (JPDA, MHT, etc.)
            resource_manager: Radar resource management system
            priority_calculator: Priority calculation system
            degradation_model: Track quality degradation model
            max_tracks: Maximum number of simultaneous tracks
            enable_predictive_scheduling: Enable predictive resource scheduling
            enable_adaptive_coasting: Enable adaptive track coasting
        """
        # Core components
        self.base_tracker = base_tracker or self._create_default_tracker()
        self.resource_manager = resource_manager
        self.priority_calculator = priority_calculator or PriorityCalculator()
        self.degradation_model = degradation_model or QualityDegradationModel()
        
        # Configuration
        self.max_tracks = max_tracks
        self.enable_predictive_scheduling = enable_predictive_scheduling
        self.enable_adaptive_coasting = enable_adaptive_coasting
        
        # Track resource state management
        self.track_resource_states: Dict[str, TrackResourceState] = {}
        self.resource_schedule: List[Tuple[float, str, ResourceRequest]] = []
        self.pending_resource_requests: Dict[str, List[ResourceRequest]] = defaultdict(list)
        
        # Quality monitoring
        self.quality_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.uncertainty_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # Performance metrics
        self.performance_metrics = {
            "total_resource_requests": 0,
            "granted_resource_requests": 0,
            "tracks_coasting": 0,
            "average_track_quality": 0.0,
            "resource_efficiency": 0.0,
            "prediction_accuracy": 0.0
        }
        
        # Simulation state
        self.current_time: float = 0.0
        self.last_resource_allocation_time: float = 0.0
        self.resource_allocation_interval: float = 0.1  # 100ms
        
        # Callbacks for resource management events
        self.resource_granted_callbacks: List[Callable[[str, ResourceRequest], None]] = []
        self.resource_denied_callbacks: List[Callable[[str, ResourceRequest], None]] = []
        self.quality_change_callbacks: List[Callable[[str, TrackQuality, TrackQuality], None]] = []
        self.coasting_start_callbacks: List[Callable[[str], None]] = []
        self.coasting_end_callbacks: List[Callable[[str], None]] = []
        
        logger.info("Initialized ResourceAwareTracker with predictive scheduling: %s, adaptive coasting: %s",
                   enable_predictive_scheduling, enable_adaptive_coasting)
    
    def predict(self, timestamp: float) -> None:
        """
        Perform prediction step with resource-aware adaptations.
        
        Args:
            timestamp: Current simulation time
        """
        self.current_time = timestamp
        
        # Update track quality assessments
        self._update_track_qualities(timestamp)
        
        # Perform predictive scheduling if enabled
        if self.enable_predictive_scheduling:
            self._perform_predictive_scheduling(timestamp)
        
        # Request resources for high-priority tracks
        self._request_resources_for_critical_tracks(timestamp)
        
        # Delegate to base tracker for actual prediction
        self.base_tracker.predict(timestamp)
        
        # Update coasting tracks
        self._update_coasting_tracks(timestamp)
    
    def update(self, measurements: List[Measurement]) -> None:
        """
        Update tracks with measurements, considering resource constraints.
        
        Args:
            measurements: List of new measurements
        """
        # Process resource allocations first
        allocated_resources = self._process_resource_allocations(self.current_time)
        
        # Filter measurements based on resource availability
        effective_measurements = self._filter_measurements_by_resources(
            measurements, allocated_resources
        )
        
        # Delegate to base tracker
        self.base_tracker.update(effective_measurements)
        
        # Update resource states for tracks that received updates
        self._update_resource_states_after_measurements(effective_measurements)
        
        # Manage track states with resource awareness
        self._manage_tracks_with_resource_awareness()
        
        # Update performance metrics
        self._update_performance_metrics()
    
    def request_resource_allocation(self, 
                                  track_id: str, 
                                  request_type: ResourceRequest,
                                  priority_boost: float = 0.0,
                                  deadline: Optional[float] = None) -> bool:
        """
        Request resource allocation for a specific track.
        
        Args:
            track_id: Track requiring resources
            request_type: Type of resource request
            priority_boost: Additional priority boost for this request
            deadline: Optional deadline for resource allocation
            
        Returns:
            True if request was submitted successfully
        """
        if track_id not in self.track_resource_states:
            logger.warning(f"Cannot request resources for unknown track {track_id}")
            return False
        
        # Add to pending requests
        self.pending_resource_requests[track_id].append(request_type)
        
        # Update priority if boost provided
        if priority_boost > 0:
            self.track_resource_states[track_id].priority_score += priority_boost
        
        # Schedule request if deadline provided
        if deadline is not None:
            self.resource_schedule.append((deadline, track_id, request_type))
            self.resource_schedule.sort()  # Keep sorted by time
        
        self.performance_metrics["total_resource_requests"] += 1
        
        logger.debug(f"Requested {request_type.value} for track {track_id}")
        return True
    
    def get_track_quality(self, track_id: str) -> Optional[TrackQuality]:
        """
        Get current quality level for a track.
        
        Args:
            track_id: Track identifier
            
        Returns:
            Current track quality level or None if track not found
        """
        resource_state = self.track_resource_states.get(track_id)
        return resource_state.quality_level if resource_state else None
    
    def get_coasting_tracks(self) -> List[str]:
        """
        Get list of tracks currently in coasting mode.
        
        Returns:
            List of track IDs that are coasting
        """
        coasting_tracks = []
        for track_id, state in self.track_resource_states.items():
            if state.quality_level == TrackQuality.COASTING:
                coasting_tracks.append(track_id)
        return coasting_tracks
    
    def force_track_update(self, track_id: str) -> bool:
        """
        Force an immediate resource allocation for a track.
        
        Args:
            track_id: Track to update immediately
            
        Returns:
            True if update was successful
        """
        if not self.resource_manager:
            logger.warning("No resource manager available for forced update")
            return False
        
        if track_id not in self.base_tracker.tracks:
            logger.warning(f"Track {track_id} not found for forced update")
            return False
        
        # Create high-priority target info
        track = self.base_tracker.tracks[track_id]
        target_info = self._create_target_info_from_track(track)
        
        # Request immediate resource allocation
        allocations = self.resource_manager.allocate_resources([target_info], self.current_time, 0.1)
        
        if allocations:
            # Simulate measurement for this track
            measurement = self._simulate_measurement_for_track(track)
            if measurement:
                self.base_tracker.update([measurement])
                self._update_resource_state_after_allocation(track_id)
                return True
        
        return False
    
    def get_resource_efficiency_metrics(self) -> Dict[str, float]:
        """
        Get resource efficiency metrics.
        
        Returns:
            Dictionary of efficiency metrics
        """
        if not self.track_resource_states:
            return {}
        
        # Calculate metrics
        total_tracks = len(self.track_resource_states)
        coasting_count = len(self.get_coasting_tracks())
        
        quality_scores = []
        for state in self.track_resource_states.values():
            quality_map = {
                TrackQuality.EXCELLENT: 1.0,
                TrackQuality.GOOD: 0.8,
                TrackQuality.FAIR: 0.6,
                TrackQuality.POOR: 0.4,
                TrackQuality.COASTING: 0.2
            }
            quality_scores.append(quality_map.get(state.quality_level, 0.5))
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        resource_efficiency = (
            self.performance_metrics["granted_resource_requests"] /
            max(self.performance_metrics["total_resource_requests"], 1)
        )
        
        return {
            "total_tracks": total_tracks,
            "coasting_tracks": coasting_count,
            "coasting_percentage": coasting_count / max(total_tracks, 1) * 100,
            "average_quality_score": avg_quality,
            "resource_efficiency": resource_efficiency,
            "total_resource_requests": self.performance_metrics["total_resource_requests"],
            "granted_requests": self.performance_metrics["granted_resource_requests"]
        }
    
    def add_resource_granted_callback(self, callback: Callable[[str, ResourceRequest], None]) -> None:
        """Add callback for resource granted events."""
        self.resource_granted_callbacks.append(callback)
    
    def add_resource_denied_callback(self, callback: Callable[[str, ResourceRequest], None]) -> None:
        """Add callback for resource denied events."""
        self.resource_denied_callbacks.append(callback)
    
    def add_quality_change_callback(self, callback: Callable[[str, TrackQuality, TrackQuality], None]) -> None:
        """Add callback for track quality changes."""
        self.quality_change_callbacks.append(callback)
    
    def add_coasting_start_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback for track coasting start events."""
        self.coasting_start_callbacks.append(callback)
    
    def add_coasting_end_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback for track coasting end events."""
        self.coasting_end_callbacks.append(callback)
    
    def _create_default_tracker(self) -> BaseTracker:
        """Create default JPDA tracker."""
        from .integrated_trackers import create_tracking_configuration, JPDATracker
        config = create_tracking_configuration()
        return JPDATracker(config)
    
    def _update_track_qualities(self, current_time: float) -> None:
        """Update quality assessments for all tracks."""
        for track_id, track in self.base_tracker.tracks.items():
            if track_id not in self.track_resource_states:
                # Initialize resource state for new track
                self.track_resource_states[track_id] = TrackResourceState(track_id=track_id)
            
            resource_state = self.track_resource_states[track_id]
            old_quality = resource_state.quality_level
            
            # Calculate new quality based on time since last update and uncertainty
            time_since_update = current_time - resource_state.last_resource_time
            new_quality = self._assess_track_quality(track, time_since_update)
            
            if new_quality != old_quality:
                resource_state.quality_level = new_quality
                
                # Log quality change
                logger.debug(f"Track {track_id} quality changed: {old_quality.value} -> {new_quality.value}")
                
                # Handle coasting transitions
                if new_quality == TrackQuality.COASTING and old_quality != TrackQuality.COASTING:
                    resource_state.coasting_start_time = current_time
                    for callback in self.coasting_start_callbacks:
                        callback(track_id)
                elif old_quality == TrackQuality.COASTING and new_quality != TrackQuality.COASTING:
                    resource_state.coasting_start_time = None
                    for callback in self.coasting_end_callbacks:
                        callback(track_id)
                
                # Notify callbacks
                for callback in self.quality_change_callbacks:
                    callback(track_id, old_quality, new_quality)
            
            # Update quality history
            self.quality_history[track_id].append((current_time, new_quality))
            
            # Update uncertainty history if track has covariance
            if track.covariance is not None:
                uncertainty = np.trace(track.covariance)
                self.uncertainty_history[track_id].append((current_time, uncertainty))
    
    def _assess_track_quality(self, track: Track, time_since_update: float) -> TrackQuality:
        """
        Assess track quality based on update frequency and uncertainty.
        
        Args:
            track: Track object
            time_since_update: Time since last resource allocation
            
        Returns:
            Assessed track quality level
        """
        # Base quality assessment on time since update
        if time_since_update <= 1.0:
            base_quality = TrackQuality.EXCELLENT
        elif time_since_update <= 3.0:
            base_quality = TrackQuality.GOOD
        elif time_since_update <= 8.0:
            base_quality = TrackQuality.FAIR
        elif time_since_update <= 15.0:
            base_quality = TrackQuality.POOR
        else:
            base_quality = TrackQuality.COASTING
        
        # Adjust based on track uncertainty if available
        if track.covariance is not None:
            uncertainty = np.trace(track.covariance)
            
            # High uncertainty degrades quality
            if uncertainty > 1000.0:
                if base_quality == TrackQuality.EXCELLENT:
                    base_quality = TrackQuality.GOOD
                elif base_quality == TrackQuality.GOOD:
                    base_quality = TrackQuality.FAIR
                elif base_quality == TrackQuality.FAIR:
                    base_quality = TrackQuality.POOR
            
            # Very high uncertainty forces coasting
            if uncertainty > 5000.0:
                base_quality = TrackQuality.COASTING
        
        # Consider track score for additional quality assessment
        if hasattr(track, 'track_score') and track.track_score < 0.3:
            # Poor track score degrades quality
            if base_quality == TrackQuality.EXCELLENT:
                base_quality = TrackQuality.GOOD
            elif base_quality == TrackQuality.GOOD:
                base_quality = TrackQuality.FAIR
        
        return base_quality
    
    def _perform_predictive_scheduling(self, current_time: float) -> None:
        """Perform predictive resource scheduling for high-priority tracks."""
        if not self.resource_manager:
            return
        
        # Find tracks that will need updates soon
        prediction_horizon = 5.0  # seconds
        tracks_needing_updates = []
        
        for track_id, resource_state in self.track_resource_states.items():
            time_since_update = current_time - resource_state.last_resource_time
            predicted_next_update = resource_state.requested_update_interval
            
            if (time_since_update + prediction_horizon >= predicted_next_update and
                resource_state.quality_level in [TrackQuality.EXCELLENT, TrackQuality.GOOD]):
                
                tracks_needing_updates.append(track_id)
        
        # Schedule predictive updates for high-priority tracks
        for track_id in tracks_needing_updates:
            if track_id in self.base_tracker.tracks:
                self.request_resource_allocation(
                    track_id, 
                    ResourceRequest.SCHEDULED_UPDATE,
                    deadline=current_time + prediction_horizon
                )
    
    def _request_resources_for_critical_tracks(self, current_time: float) -> None:
        """Request immediate resources for critical tracks."""
        critical_tracks = []
        
        for track_id, resource_state in self.track_resource_states.items():
            # Identify critical conditions
            time_since_update = current_time - resource_state.last_resource_time
            
            is_critical = (
                resource_state.quality_level == TrackQuality.POOR or
                time_since_update > resource_state.minimum_update_interval or
                resource_state.priority_score > 0.8
            )
            
            if is_critical and track_id in self.base_tracker.tracks:
                critical_tracks.append(track_id)
        
        # Request urgent updates for critical tracks
        for track_id in critical_tracks:
            self.request_resource_allocation(track_id, ResourceRequest.URGENT_UPDATE)
    
    def _process_resource_allocations(self, current_time: float) -> Dict[str, float]:
        """
        Process resource allocations with resource manager.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Dictionary mapping track IDs to allocated dwell times
        """
        if not self.resource_manager:
            return {}
        
        # Check if it's time for resource allocation
        if (current_time - self.last_resource_allocation_time < 
            self.resource_allocation_interval):
            return {}
        
        # Create target info for tracks needing resources
        targets_needing_resources = []
        for track_id, requests in self.pending_resource_requests.items():
            if requests and track_id in self.base_tracker.tracks:
                track = self.base_tracker.tracks[track_id]
                target_info = self._create_target_info_from_track(track)
                targets_needing_resources.append(target_info)
        
        if not targets_needing_resources:
            return {}
        
        # Request resource allocation
        try:
            allocations = self.resource_manager.allocate_resources(
                targets_needing_resources, current_time
            )
            
            allocated_resources = {}
            granted_tracks = set()
            
            for allocation in allocations:
                track_id = allocation.target_id
                allocated_resources[track_id] = allocation.dwell_time
                granted_tracks.add(track_id)
                
                # Update resource state
                if track_id in self.track_resource_states:
                    self.track_resource_states[track_id].last_resource_time = current_time
                    self.track_resource_states[track_id].missed_opportunities = 0
                
                # Clear granted requests
                if track_id in self.pending_resource_requests:
                    granted_requests = self.pending_resource_requests[track_id].copy()
                    self.pending_resource_requests[track_id].clear()
                    
                    # Notify callbacks
                    for request in granted_requests:
                        for callback in self.resource_granted_callbacks:
                            callback(track_id, request)
                        self.performance_metrics["granted_resource_requests"] += 1
            
            # Handle denied requests
            for track_id, requests in self.pending_resource_requests.items():
                if requests and track_id not in granted_tracks:
                    # Track missed opportunity
                    if track_id in self.track_resource_states:
                        self.track_resource_states[track_id].missed_opportunities += 1
                    
                    # Notify callbacks for denied requests
                    for request in requests:
                        for callback in self.resource_denied_callbacks:
                            callback(track_id, request)
            
            self.last_resource_allocation_time = current_time
            return allocated_resources
            
        except Exception as e:
            logger.error(f"Error in resource allocation: {e}")
            return {}
    
    def _filter_measurements_by_resources(self, 
                                        measurements: List[Measurement],
                                        allocated_resources: Dict[str, float]) -> List[Measurement]:
        """
        Filter measurements based on allocated resources.
        
        Args:
            measurements: Input measurements
            allocated_resources: Allocated resources by track ID
            
        Returns:
            Filtered measurements
        """
        if not allocated_resources:
            return measurements
        
        # For now, return all measurements
        # In a more sophisticated implementation, this would filter based on
        # which beam positions received resource allocations
        return measurements
    
    def _update_resource_states_after_measurements(self, measurements: List[Measurement]) -> None:
        """Update resource states for tracks that received measurements."""
        # This is a simplified implementation
        # In practice, would need to associate measurements with tracks
        for measurement in measurements:
            # Check if measurement has track association metadata
            if hasattr(measurement, 'metadata') and 'associated_track_id' in measurement.metadata:
                track_id = measurement.metadata['associated_track_id']
                if track_id in self.track_resource_states:
                    self.track_resource_states[track_id].last_resource_time = self.current_time
    
    def _update_resource_state_after_allocation(self, track_id: str) -> None:
        """Update resource state after successful allocation."""
        if track_id in self.track_resource_states:
            self.track_resource_states[track_id].last_resource_time = self.current_time
            self.track_resource_states[track_id].missed_opportunities = 0
    
    def _create_target_info_from_track(self, track: Track) -> TargetInfo:
        """Create TargetInfo from Track for resource manager."""
        position = track.position if track.position is not None else np.zeros(3)
        velocity = track.velocity if track.velocity is not None else np.zeros(3)
        
        # Estimate threat level based on track characteristics
        threat_level = ThreatLevel.UNKNOWN
        if hasattr(track, 'metadata') and 'threat_level' in track.metadata:
            threat_level = track.metadata['threat_level']
        
        # Calculate range rate
        range_to_target = np.linalg.norm(position)
        if range_to_target > 0:
            range_rate = np.dot(position, velocity) / range_to_target
        else:
            range_rate = 0.0
        
        return TargetInfo(
            target_id=track.track_id,
            position=position,
            velocity=velocity,
            classification=getattr(track, 'classification', 'unknown'),
            threat_level=threat_level,
            uncertainty_covariance=track.covariance if track.covariance is not None else np.eye(6),
            last_update_time=track.last_update_time,
            track_age=track.age,
            range_rate=range_rate,
            rcs_estimate=getattr(track, 'rcs_estimate', 1.0)
        )
    
    def _simulate_measurement_for_track(self, track: Track) -> Optional[Measurement]:
        """Simulate a measurement for a track (for forced updates)."""
        if track.position is None:
            return None
        
        # Add noise to position
        noise_std = 1.0
        measured_position = track.position + np.random.normal(0, noise_std, 3)
        
        return Measurement(
            position=measured_position,
            timestamp=self.current_time,
            covariance=np.eye(3) * noise_std**2,
            metadata={'simulated': True, 'associated_track_id': track.track_id}
        )
    
    def _update_coasting_tracks(self, current_time: float) -> None:
        """Update tracks that are in coasting mode."""
        if not self.enable_adaptive_coasting:
            return
        
        for track_id, resource_state in self.track_resource_states.items():
            if resource_state.quality_level == TrackQuality.COASTING:
                if track_id in self.base_tracker.tracks:
                    track = self.base_tracker.tracks[track_id]
                    
                    # Update uncertainty growth during coasting
                    if resource_state.coasting_start_time is not None:
                        coasting_duration = current_time - resource_state.coasting_start_time
                        
                        # Grow uncertainty based on coasting time
                        if track.covariance is not None:
                            uncertainty_growth = (
                                self.degradation_model.base_uncertainty_growth * coasting_duration
                            )
                            
                            # Add velocity-dependent uncertainty
                            if track.velocity is not None:
                                speed = np.linalg.norm(track.velocity)
                                velocity_uncertainty = (
                                    self.degradation_model.velocity_uncertainty_factor * 
                                    speed * coasting_duration
                                )
                                uncertainty_growth += velocity_uncertainty
                            
                            # Apply uncertainty growth
                            track.covariance += np.eye(track.covariance.shape[0]) * uncertainty_growth
    
    def _manage_tracks_with_resource_awareness(self) -> None:
        """Manage track states considering resource constraints."""
        # Delegate basic track management to base tracker
        self.base_tracker.manage_tracks()
        
        # Additional resource-aware management
        tracks_to_remove = []
        for track_id, resource_state in self.track_resource_states.items():
            if track_id not in self.base_tracker.tracks:
                tracks_to_remove.append(track_id)
            elif resource_state.quality_level == TrackQuality.COASTING:
                # Check if coasting track should be deleted
                if resource_state.coasting_start_time is not None:
                    coasting_duration = self.current_time - resource_state.coasting_start_time
                    max_coasting_time = self.degradation_model.coasting_time_limits.get(
                        TrackQuality.COASTING, 60.0
                    )
                    
                    if coasting_duration > max_coasting_time:
                        logger.info(f"Deleting track {track_id} after {coasting_duration:.1f}s of coasting")
                        tracks_to_remove.append(track_id)
        
        # Remove resource states for deleted tracks
        for track_id in tracks_to_remove:
            self.track_resource_states.pop(track_id, None)
            self.pending_resource_requests.pop(track_id, None)
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics."""
        if self.track_resource_states:
            coasting_count = len([
                state for state in self.track_resource_states.values()
                if state.quality_level == TrackQuality.COASTING
            ])
            self.performance_metrics["tracks_coasting"] = coasting_count
            
            # Calculate average quality
            quality_values = []
            quality_map = {
                TrackQuality.EXCELLENT: 1.0,
                TrackQuality.GOOD: 0.8,
                TrackQuality.FAIR: 0.6,
                TrackQuality.POOR: 0.4,
                TrackQuality.COASTING: 0.2
            }
            
            for state in self.track_resource_states.values():
                quality_values.append(quality_map.get(state.quality_level, 0.5))
            
            self.performance_metrics["average_track_quality"] = np.mean(quality_values)


# Utility functions for creating resource-aware trackers

def create_resource_aware_tracker(tracker_type: str = "jpda",
                                resource_manager: Optional[ResourceManager] = None,
                                enable_predictive_scheduling: bool = True,
                                enable_adaptive_coasting: bool = True) -> ResourceAwareTracker:
    """
    Create a resource-aware tracker with specified configuration.
    
    Args:
        tracker_type: Type of base tracker ("jpda", "mht")
        resource_manager: Resource management system
        enable_predictive_scheduling: Enable predictive resource scheduling
        enable_adaptive_coasting: Enable adaptive track coasting
        
    Returns:
        Configured resource-aware tracker
    """
    # Create base tracker
    if tracker_type.lower() == "jpda":
        from .integrated_trackers import create_tracking_configuration, JPDATracker
        config = create_tracking_configuration()
        base_tracker = JPDATracker(config)
    elif tracker_type.lower() == "mht":
        from .integrated_trackers import create_tracking_configuration, MHTTracker
        config = create_tracking_configuration()
        base_tracker = MHTTracker(config)
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")
    
    return ResourceAwareTracker(
        base_tracker=base_tracker,
        resource_manager=resource_manager,
        enable_predictive_scheduling=enable_predictive_scheduling,
        enable_adaptive_coasting=enable_adaptive_coasting
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Testing ResourceAwareTracker...")
    
    # Create resource-aware tracker
    tracker = create_resource_aware_tracker("jpda")
    
    # Add some callbacks for monitoring
    def on_quality_change(track_id: str, old_quality: TrackQuality, new_quality: TrackQuality):
        print(f"Track {track_id} quality: {old_quality.value} -> {new_quality.value}")
    
    def on_coasting_start(track_id: str):
        print(f"Track {track_id} started coasting")
    
    tracker.add_quality_change_callback(on_quality_change)
    tracker.add_coasting_start_callback(on_coasting_start)
    
    # Simulate some tracking scenarios
    print("Simulating resource-aware tracking...")
    
    current_time = 0.0
    dt = 0.1
    
    # Add some test measurements
    test_measurements = [
        Measurement(
            position=np.array([1000.0, 500.0, 100.0]),
            timestamp=current_time,
            covariance=np.eye(3) * 0.5,
            metadata={'target_type': 'test'}
        )
    ]
    
    for step in range(100):  # 10 seconds simulation
        current_time += dt
        
        # Predict and update
        tracker.predict(current_time)
        
        if step % 10 == 0:  # Measurements every second
            tracker.update(test_measurements)
        else:
            tracker.update([])  # No measurements
        
        # Log progress
        if step % 50 == 0:
            metrics = tracker.get_resource_efficiency_metrics()
            print(f"Time {current_time:.1f}s:")
            print(f"  Total tracks: {metrics.get('total_tracks', 0)}")
            print(f"  Coasting tracks: {metrics.get('coasting_tracks', 0)}")
            print(f"  Average quality: {metrics.get('average_quality_score', 0.0):.2f}")
    
    print("\nResourceAwareTracker test completed successfully!")