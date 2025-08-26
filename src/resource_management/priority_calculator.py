"""
Priority Calculator for Radar Resource Management

This module implements the PriorityCalculator class that computes priority scores
for radar targets based on threat assessment, track uncertainty, time since last
update, and special handling for missile classifications.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ThreatLevel(Enum):
    """Enumeration of threat levels for targets."""
    MISSILE = 5
    FIGHTER = 4
    BOMBER = 3
    TRANSPORT = 2
    CIVILIAN = 1
    UNKNOWN = 2


@dataclass
class TargetInfo:
    """Information about a target for priority calculation."""
    target_id: str
    position: np.ndarray
    velocity: np.ndarray
    classification: str
    threat_level: ThreatLevel
    uncertainty_covariance: np.ndarray
    last_update_time: float
    track_age: float
    range_rate: float
    rcs_estimate: float


class PriorityCalculator:
    """
    Calculates priority scores for radar targets based on multiple factors.
    
    The priority calculation considers:
    - Threat-based scoring using classification and behavior
    - Track uncertainty weighting based on covariance matrix
    - Time since last update factor
    - Special handling for missile classifications
    - Range and geometry factors
    """
    
    def __init__(self, 
                 max_time_without_update: float = 5.0,
                 missile_priority_multiplier: float = 3.0,
                 uncertainty_weight: float = 1.5,
                 time_weight: float = 2.0,
                 threat_weight: float = 3.0,
                 geometry_weight: float = 1.0):
        """
        Initialize the PriorityCalculator.
        
        Args:
            max_time_without_update: Maximum time before priority peaks (seconds)
            missile_priority_multiplier: Extra multiplier for missile targets
            uncertainty_weight: Weight for uncertainty factor
            time_weight: Weight for time since last update factor
            threat_weight: Weight for threat level factor
            geometry_weight: Weight for geometry/range factors
        """
        self.max_time_without_update = max_time_without_update
        self.missile_priority_multiplier = missile_priority_multiplier
        self.uncertainty_weight = uncertainty_weight
        self.time_weight = time_weight
        self.threat_weight = threat_weight
        self.geometry_weight = geometry_weight
        
        # Threat level base scores
        self.threat_scores = {
            ThreatLevel.MISSILE: 1.0,
            ThreatLevel.FIGHTER: 0.8,
            ThreatLevel.BOMBER: 0.6,
            ThreatLevel.TRANSPORT: 0.3,
            ThreatLevel.CIVILIAN: 0.1,
            ThreatLevel.UNKNOWN: 0.4
        }
    
    def calculate_priority(self, target: TargetInfo, current_time: float) -> float:
        """
        Calculate the priority score for a target.
        
        Args:
            target: Target information
            current_time: Current simulation time
            
        Returns:
            Priority score (higher values indicate higher priority)
        """
        # Calculate component scores
        threat_score = self._calculate_threat_score(target)
        uncertainty_score = self._calculate_uncertainty_score(target)
        time_score = self._calculate_time_score(target, current_time)
        geometry_score = self._calculate_geometry_score(target)
        
        # Weighted combination
        priority = (self.threat_weight * threat_score +
                   self.uncertainty_weight * uncertainty_score +
                   self.time_weight * time_score +
                   self.geometry_weight * geometry_score)
        
        # Special handling for missiles
        if target.threat_level == ThreatLevel.MISSILE:
            priority *= self.missile_priority_multiplier
            
        # Additional boost for fast-moving targets (potential missiles)
        speed = np.linalg.norm(target.velocity)
        if speed > 200.0:  # m/s, roughly Mach 0.6
            priority *= (1.0 + min(speed / 500.0, 2.0))
            
        return max(priority, 0.01)  # Ensure minimum priority
    
    def _calculate_threat_score(self, target: TargetInfo) -> float:
        """Calculate threat-based score."""
        base_score = self.threat_scores.get(target.threat_level, 0.4)
        
        # Enhance based on behavioral factors
        speed = np.linalg.norm(target.velocity)
        
        # High speed bonus
        if speed > 150.0:  # m/s
            base_score *= (1.0 + min(speed / 300.0, 1.5))
            
        # Approaching radar bonus (negative range rate)
        if target.range_rate < -50.0:  # Approaching at >50 m/s
            base_score *= (1.0 + min(abs(target.range_rate) / 200.0, 1.0))
            
        # Large RCS penalty for civilian/transport
        if target.threat_level in [ThreatLevel.CIVILIAN, ThreatLevel.TRANSPORT]:
            if target.rcs_estimate > 100.0:  # Large RCS suggests large aircraft
                base_score *= 0.5
                
        return base_score
    
    def _calculate_uncertainty_score(self, target: TargetInfo) -> float:
        """Calculate uncertainty-based score."""
        # Use trace of covariance matrix as uncertainty measure
        uncertainty = np.trace(target.uncertainty_covariance)
        
        # Normalize uncertainty (higher uncertainty = higher priority)
        # Use sigmoid function to bound the score
        normalized_uncertainty = 2.0 / (1.0 + np.exp(-uncertainty / 1000.0)) - 1.0
        
        return max(normalized_uncertainty, 0.1)
    
    def _calculate_time_score(self, target: TargetInfo, current_time: float) -> float:
        """Calculate time since last update score."""
        time_since_update = current_time - target.last_update_time
        
        # Exponential growth up to max_time_without_update
        if time_since_update <= 0:
            return 0.1
        
        # Sigmoid function for smooth priority increase
        normalized_time = time_since_update / self.max_time_without_update
        time_score = 2.0 / (1.0 + np.exp(-2.0 * normalized_time)) - 1.0
        
        return max(time_score, 0.1)
    
    def _calculate_geometry_score(self, target: TargetInfo) -> float:
        """Calculate geometry and range-based score."""
        range_to_target = np.linalg.norm(target.position)
        
        # Closer targets get higher priority (within reason)
        # Use inverse relationship with range, but not too steep
        if range_to_target < 1000.0:  # Very close targets
            range_score = 1.0
        elif range_to_target > 100000.0:  # Very far targets
            range_score = 0.2
        else:
            # Logarithmic decay with range
            range_score = 1.0 - 0.8 * np.log10(range_to_target / 1000.0) / 2.0
            
        return max(range_score, 0.1)
    
    def calculate_priorities_batch(self, 
                                 targets: List[TargetInfo], 
                                 current_time: float) -> Dict[str, float]:
        """
        Calculate priority scores for multiple targets.
        
        Args:
            targets: List of target information
            current_time: Current simulation time
            
        Returns:
            Dictionary mapping target IDs to priority scores
        """
        priorities = {}
        for target in targets:
            priorities[target.target_id] = self.calculate_priority(target, current_time)
            
        return priorities
    
    def get_top_priority_targets(self, 
                               targets: List[TargetInfo], 
                               current_time: float,
                               num_targets: int) -> List[str]:
        """
        Get the top priority target IDs.
        
        Args:
            targets: List of target information
            current_time: Current simulation time
            num_targets: Number of top priority targets to return
            
        Returns:
            List of target IDs sorted by priority (highest first)
        """
        priorities = self.calculate_priorities_batch(targets, current_time)
        
        # Sort by priority (descending)
        sorted_targets = sorted(priorities.items(), key=lambda x: x[1], reverse=True)
        
        return [target_id for target_id, _ in sorted_targets[:num_targets]]
    
    def update_weights(self, 
                      uncertainty_weight: Optional[float] = None,
                      time_weight: Optional[float] = None,
                      threat_weight: Optional[float] = None,
                      geometry_weight: Optional[float] = None):
        """
        Update priority calculation weights.
        
        Args:
            uncertainty_weight: New uncertainty weight
            time_weight: New time weight
            threat_weight: New threat weight
            geometry_weight: New geometry weight
        """
        if uncertainty_weight is not None:
            self.uncertainty_weight = uncertainty_weight
        if time_weight is not None:
            self.time_weight = time_weight
        if threat_weight is not None:
            self.threat_weight = threat_weight
        if geometry_weight is not None:
            self.geometry_weight = geometry_weight