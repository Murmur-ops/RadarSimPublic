"""
Resource Manager for Radar Systems

This module implements the ResourceManager class that coordinates radar resource
allocation across 120 beam positions with time budget management, priority-based
scheduling, and adaptive resource distribution.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import time

from .priority_calculator import PriorityCalculator, TargetInfo, ThreatLevel
from .beam_scheduler import BeamScheduler, BeamTask, BeamMode, BeamCapability


@dataclass
class BeamPosition:
    """Represents a radar beam position."""
    azimuth: float  # radians
    elevation: float  # radians
    beam_id: int
    is_available: bool = True
    last_used: float = 0.0
    
    def __post_init__(self):
        """Normalize angles after initialization."""
        self.azimuth = self._normalize_azimuth(self.azimuth)
        self.elevation = np.clip(self.elevation, -np.pi/2, np.pi/2)
    
    def _normalize_azimuth(self, azimuth: float) -> float:
        """Normalize azimuth to [0, 2π)."""
        return azimuth % (2 * np.pi)


@dataclass
class ResourceAllocation:
    """Represents resource allocation for a time window."""
    target_id: str
    beam_position: BeamPosition
    start_time: float
    dwell_time: float
    priority: float
    mode: BeamMode


@dataclass
class TimeBudget:
    """Time budget management for radar operations."""
    total_time: float = 1.0  # 1 second per full scan
    search_fraction: float = 0.4  # 40% for search
    track_fraction: float = 0.5   # 50% for tracking
    classification_fraction: float = 0.1  # 10% for classification
    
    def get_mode_budget(self, mode: BeamMode) -> float:
        """Get time budget for a specific mode."""
        if mode == BeamMode.SEARCH:
            return self.total_time * self.search_fraction
        elif mode == BeamMode.TRACK:
            return self.total_time * self.track_fraction
        elif mode == BeamMode.CLASSIFICATION:
            return self.total_time * self.classification_fraction
        else:  # CONFIRM
            return self.total_time * 0.05  # 5% for confirmation


class ResourceManager:
    """
    Manages radar resources across 120 beam positions with priority-based scheduling.
    
    The ResourceManager coordinates between the PriorityCalculator and BeamScheduler
    to efficiently allocate radar resources while respecting time budgets and
    operational constraints.
    """
    
    def __init__(self,
                 num_beam_positions: int = 120,
                 time_budget: TimeBudget = None,
                 priority_calculator: PriorityCalculator = None,
                 beam_scheduler: BeamScheduler = None):
        """
        Initialize the ResourceManager.
        
        Args:
            num_beam_positions: Number of available beam positions (default: 120)
            time_budget: Time budget allocation object
            priority_calculator: Priority calculation system
            beam_scheduler: Beam scheduling system
        """
        self.num_beam_positions = num_beam_positions
        self.time_budget = time_budget or TimeBudget()
        self.priority_calculator = priority_calculator or PriorityCalculator()
        self.beam_scheduler = beam_scheduler or BeamScheduler()
        
        # Initialize beam positions in a spherical grid
        self.beam_positions = self._initialize_beam_positions()
        
        # Resource allocation state
        self.current_allocations: List[ResourceAllocation] = []
        self.allocation_history: List[ResourceAllocation] = []
        self.time_usage_by_mode: Dict[BeamMode, float] = defaultdict(float)
        
        # Performance tracking
        self.scan_start_time = 0.0
        self.current_scan_time = 0.0
        self.completed_scans = 0
        
        # Target tracking
        self.active_targets: Dict[str, TargetInfo] = {}
        self.target_beam_assignments: Dict[str, int] = {}
    
    def _initialize_beam_positions(self) -> List[BeamPosition]:
        """
        Initialize 120 beam positions in a spherical grid pattern.
        
        Returns:
            List of BeamPosition objects covering the surveillance volume
        """
        positions = []
        
        # Create a roughly uniform distribution over a hemisphere
        # Using a spiral pattern for better coverage
        n = self.num_beam_positions
        
        for i in range(n):
            # Golden spiral distribution
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)
            
            theta = np.pi * (3 - np.sqrt(5)) * i  # Golden angle
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            # Convert to spherical coordinates
            azimuth = np.arctan2(z, x)
            if azimuth < 0:
                azimuth += 2 * np.pi
                
            elevation = np.arcsin(y)
            
            # Limit elevation to reasonable radar coverage
            elevation = np.clip(elevation, np.radians(-10), np.radians(80))
            
            positions.append(BeamPosition(
                azimuth=azimuth,
                elevation=elevation,
                beam_id=i
            ))
        
        return positions
    
    def allocate_resources(self, 
                         targets: List[TargetInfo], 
                         current_time: float,
                         scan_duration: float = None) -> List[ResourceAllocation]:
        """
        Allocate radar resources for the given targets.
        
        Args:
            targets: List of targets requiring radar resources
            current_time: Current simulation time
            scan_duration: Duration of this allocation cycle (default: time_budget.total_time)
            
        Returns:
            List of resource allocations for this cycle
        """
        if scan_duration is None:
            scan_duration = self.time_budget.total_time
        
        # Update internal target state
        self.active_targets = {target.target_id: target for target in targets}
        
        # Calculate priorities for all targets
        priorities = self.priority_calculator.calculate_priorities_batch(targets, current_time)
        
        # Group targets by required mode
        targets_by_mode = self._categorize_targets_by_mode(targets, current_time)
        
        # Allocate resources for each mode based on time budget
        allocations = []
        mode_start_time = current_time
        
        for mode in [BeamMode.TRACK, BeamMode.SEARCH, BeamMode.CLASSIFICATION, BeamMode.CONFIRM]:
            mode_targets = targets_by_mode.get(mode, [])
            if not mode_targets:
                continue
            
            mode_budget = self.time_budget.get_mode_budget(mode)
            mode_allocations = self._allocate_mode_resources(
                mode_targets, mode, priorities, mode_start_time, mode_budget
            )
            
            allocations.extend(mode_allocations)
            
            # Update time tracking
            mode_time_used = sum(alloc.dwell_time for alloc in mode_allocations)
            self.time_usage_by_mode[mode] += mode_time_used
        
        # Store allocations
        self.current_allocations = allocations
        self.allocation_history.extend(allocations)
        
        return allocations
    
    def _categorize_targets_by_mode(self, 
                                  targets: List[TargetInfo], 
                                  current_time: float) -> Dict[BeamMode, List[TargetInfo]]:
        """
        Categorize targets by required beam mode.
        
        Args:
            targets: List of targets
            current_time: Current simulation time
            
        Returns:
            Dictionary mapping beam modes to target lists
        """
        targets_by_mode = defaultdict(list)
        
        for target in targets:
            # Determine required mode based on target characteristics
            time_since_update = current_time - target.last_update_time
            
            if target.threat_level == ThreatLevel.MISSILE:
                # Missiles need frequent tracking
                targets_by_mode[BeamMode.TRACK].append(target)
            elif target.classification == "unknown" or target.classification == "":
                # Unknown targets need classification
                targets_by_mode[BeamMode.CLASSIFICATION].append(target)
            elif time_since_update > 2.0:
                # Targets not seen recently need confirmation
                targets_by_mode[BeamMode.CONFIRM].append(target)
            elif time_since_update > 0.5:
                # Regular tracking
                targets_by_mode[BeamMode.TRACK].append(target)
            else:
                # Recent targets can be in search mode
                targets_by_mode[BeamMode.SEARCH].append(target)
        
        return targets_by_mode
    
    def _allocate_mode_resources(self, 
                               targets: List[TargetInfo],
                               mode: BeamMode,
                               priorities: Dict[str, float],
                               start_time: float,
                               time_budget: float) -> List[ResourceAllocation]:
        """
        Allocate resources for targets in a specific mode.
        
        Args:
            targets: Targets requiring this mode
            mode: Beam operation mode
            priorities: Priority scores for all targets
            start_time: Start time for this mode allocation
            time_budget: Available time budget for this mode
            
        Returns:
            List of resource allocations
        """
        allocations = []
        
        # Sort targets by priority
        sorted_targets = sorted(targets, 
                              key=lambda t: priorities.get(t.target_id, 0.0), 
                              reverse=True)
        
        available_time = time_budget
        current_time = start_time
        
        for target in sorted_targets:
            if available_time <= 0:
                break
            
            # Find best beam position for this target
            beam_pos = self._find_best_beam_position(target)
            if beam_pos is None:
                continue
            
            # Calculate dwell time based on priority
            priority = priorities.get(target.target_id, 0.1)
            dwell_time = self.beam_scheduler._calculate_dwell_time(priority, mode)
            
            # Ensure we don't exceed time budget
            if dwell_time > available_time:
                dwell_time = available_time
            
            # Create allocation
            allocation = ResourceAllocation(
                target_id=target.target_id,
                beam_position=beam_pos,
                start_time=current_time,
                dwell_time=dwell_time,
                priority=priority,
                mode=mode
            )
            
            allocations.append(allocation)
            
            # Update tracking
            available_time -= dwell_time
            current_time += dwell_time
            beam_pos.last_used = current_time
            self.target_beam_assignments[target.target_id] = beam_pos.beam_id
        
        return allocations
    
    def _find_best_beam_position(self, target: TargetInfo) -> Optional[BeamPosition]:
        """
        Find the best beam position for a target.
        
        Args:
            target: Target information
            
        Returns:
            Best beam position or None if none available
        """
        target_pos = target.position
        target_azimuth = np.arctan2(target_pos[1], target_pos[0])
        if target_azimuth < 0:
            target_azimuth += 2 * np.pi
        
        target_range = np.linalg.norm(target_pos[:2])  # Horizontal range
        target_elevation = np.arctan2(target_pos[2], target_range)
        
        # Find closest available beam position
        best_beam = None
        min_distance = float('inf')
        
        for beam_pos in self.beam_positions:
            if not beam_pos.is_available:
                continue
            
            # Calculate angular distance
            az_diff = abs(beam_pos.azimuth - target_azimuth)
            az_diff = min(az_diff, 2*np.pi - az_diff)  # Wrap around
            
            el_diff = abs(beam_pos.elevation - target_elevation)
            
            distance = np.sqrt(az_diff**2 + el_diff**2)
            
            if distance < min_distance:
                min_distance = distance
                best_beam = beam_pos
        
        return best_beam
    
    def update_priorities(self, 
                         targets: List[TargetInfo], 
                         current_time: float) -> Dict[str, float]:
        """
        Update priority scores for all targets.
        
        Args:
            targets: List of targets
            current_time: Current simulation time
            
        Returns:
            Updated priority scores
        """
        return self.priority_calculator.calculate_priorities_batch(targets, current_time)
    
    def get_beam_schedule(self, 
                         targets: List[TargetInfo], 
                         current_time: float,
                         lookahead_time: float = 1.0) -> List[BeamTask]:
        """
        Get the beam schedule for the specified time window.
        
        Args:
            targets: List of targets
            current_time: Current simulation time
            lookahead_time: How far ahead to schedule (seconds)
            
        Returns:
            List of scheduled beam tasks
        """
        # Get resource allocations
        allocations = self.allocate_resources(targets, current_time, lookahead_time)
        
        # Convert allocations to beam tasks
        beam_tasks = []
        for alloc in allocations:
            task = BeamTask(
                target_id=alloc.target_id,
                beam_position=(alloc.beam_position.azimuth, alloc.beam_position.elevation),
                priority=alloc.priority,
                dwell_time=alloc.dwell_time,
                mode=alloc.mode,
                deadline=alloc.start_time + lookahead_time,
                created_time=current_time
            )
            beam_tasks.append(task)
        
        # Optimize task sequence
        optimized_tasks = self.beam_scheduler.optimize_beam_sequence(beam_tasks)
        
        return optimized_tasks
    
    def execute_scan_cycle(self, 
                          targets: List[TargetInfo], 
                          current_time: float) -> Dict[str, float]:
        """
        Execute a complete scan cycle.
        
        Args:
            targets: List of targets
            current_time: Current simulation time
            
        Returns:
            Dictionary mapping target IDs to completion times
        """
        # Get beam schedule
        beam_tasks = self.get_beam_schedule(targets, current_time)
        
        # Execute tasks through beam scheduler
        completion_times = self.beam_scheduler.execute_tasks(beam_tasks, current_time)
        
        # Update scan tracking
        self.scan_start_time = current_time
        self.current_scan_time = current_time
        self.completed_scans += 1
        
        return completion_times
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """
        Get resource utilization statistics.
        
        Returns:
            Dictionary of utilization metrics
        """
        total_budget = self.time_budget.total_time
        
        utilization = {}
        for mode, time_used in self.time_usage_by_mode.items():
            mode_budget = self.time_budget.get_mode_budget(mode)
            utilization[f"{mode.value}_utilization"] = time_used / max(mode_budget, 0.001)
        
        # Overall utilization
        total_used = sum(self.time_usage_by_mode.values())
        utilization["overall_utilization"] = total_used / max(total_budget, 0.001)
        
        # Beam position utilization
        beam_usage = defaultdict(float)
        for alloc in self.allocation_history:
            beam_usage[alloc.beam_position.beam_id] += alloc.dwell_time
        
        if beam_usage:
            utilization["avg_beam_utilization"] = np.mean(list(beam_usage.values()))
            utilization["max_beam_utilization"] = np.max(list(beam_usage.values()))
            utilization["min_beam_utilization"] = np.min(list(beam_usage.values()))
        
        return utilization
    
    def reset_scan_cycle(self):
        """Reset the scan cycle and clear temporary allocations."""
        self.current_allocations.clear()
        self.time_usage_by_mode.clear()
        
        # Mark all beam positions as available
        for beam_pos in self.beam_positions:
            beam_pos.is_available = True
    
    def get_beam_coverage_statistics(self) -> Dict[str, float]:
        """
        Get beam coverage statistics.
        
        Returns:
            Dictionary of coverage metrics
        """
        if not self.allocation_history:
            return {}
        
        # Calculate coverage metrics
        used_beams = set()
        total_allocations = len(self.allocation_history)
        
        for alloc in self.allocation_history:
            used_beams.add(alloc.beam_position.beam_id)
        
        coverage_fraction = len(used_beams) / self.num_beam_positions
        
        return {
            "total_beam_positions": self.num_beam_positions,
            "used_beam_positions": len(used_beams),
            "coverage_fraction": coverage_fraction,
            "total_allocations": total_allocations,
            "avg_allocations_per_beam": total_allocations / max(len(used_beams), 1)
        }