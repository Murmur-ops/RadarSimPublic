"""
Beam Scheduler for Radar Resource Management

This module implements the BeamScheduler class that manages beam positioning,
dwell time calculation, and adaptive revisit rates for radar resource allocation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import heapq
from collections import defaultdict


class BeamMode(Enum):
    """Beam operation modes."""
    SEARCH = "search"
    TRACK = "track"
    CONFIRM = "confirm"
    CLASSIFICATION = "classification"


@dataclass
class BeamTask:
    """Represents a beam task to be scheduled."""
    target_id: str
    beam_position: Tuple[float, float]  # (azimuth, elevation) in radians
    priority: float
    dwell_time: float
    mode: BeamMode
    deadline: float
    created_time: float
    
    def __lt__(self, other):
        """For priority queue ordering (higher priority first)."""
        return self.priority > other.priority


@dataclass
class BeamCapability:
    """Beam capability parameters."""
    min_dwell_time: float = 0.001  # 1 ms minimum
    max_dwell_time: float = 0.1    # 100 ms maximum
    beam_width: float = np.radians(3.0)  # 3 degree beam width
    scan_rate: float = np.radians(360.0)  # 360 deg/sec max scan rate
    settle_time: float = 0.002  # 2 ms beam settle time


class BeamScheduler:
    """
    Manages beam scheduling for phased array radar systems.
    
    Handles dwell time calculation based on priority, beam position assignment,
    and adaptive revisit rates for different target types and operational modes.
    """
    
    def __init__(self, 
                 beam_capability: BeamCapability = None,
                 max_concurrent_beams: int = 4,
                 search_revisit_interval: float = 2.0,
                 track_revisit_interval: float = 0.1):
        """
        Initialize the BeamScheduler.
        
        Args:
            beam_capability: Beam system capabilities
            max_concurrent_beams: Maximum number of concurrent beam tasks
            search_revisit_interval: Default search mode revisit interval (seconds)
            track_revisit_interval: Default track mode revisit interval (seconds)
        """
        self.beam_capability = beam_capability or BeamCapability()
        self.max_concurrent_beams = max_concurrent_beams
        self.search_revisit_interval = search_revisit_interval
        self.track_revisit_interval = track_revisit_interval
        
        # Scheduling state
        self.task_queue = []  # Priority queue of BeamTask objects
        self.active_tasks = {}  # Currently executing tasks
        self.completed_tasks = []  # History of completed tasks
        self.target_last_visit = {}  # Last visit time for each target
        self.target_revisit_intervals = {}  # Custom revisit intervals
        
        # Performance tracking
        self.total_beam_time = 0.0
        self.task_completion_times = []
        self.missed_deadlines = 0
    
    def add_beam_task(self, 
                     target_id: str,
                     beam_position: Tuple[float, float],
                     priority: float,
                     mode: BeamMode,
                     current_time: float,
                     custom_dwell_time: Optional[float] = None,
                     deadline_offset: float = 1.0) -> bool:
        """
        Add a new beam task to the scheduler.
        
        Args:
            target_id: Unique target identifier
            beam_position: (azimuth, elevation) in radians
            priority: Task priority (higher values = higher priority)
            mode: Beam operation mode
            current_time: Current simulation time
            custom_dwell_time: Override automatic dwell time calculation
            deadline_offset: Time from now when task expires (seconds)
            
        Returns:
            True if task was added successfully
        """
        # Calculate dwell time based on priority and mode
        if custom_dwell_time is not None:
            dwell_time = custom_dwell_time
        else:
            dwell_time = self._calculate_dwell_time(priority, mode)
        
        # Validate dwell time
        dwell_time = np.clip(dwell_time, 
                           self.beam_capability.min_dwell_time,
                           self.beam_capability.max_dwell_time)
        
        # Create task
        task = BeamTask(
            target_id=target_id,
            beam_position=beam_position,
            priority=priority,
            dwell_time=dwell_time,
            mode=mode,
            deadline=current_time + deadline_offset,
            created_time=current_time
        )
        
        # Add to queue
        heapq.heappush(self.task_queue, task)
        
        return True
    
    def _calculate_dwell_time(self, priority: float, mode: BeamMode) -> float:
        """
        Calculate dwell time based on priority and beam mode.
        
        Args:
            priority: Task priority
            mode: Beam operation mode
            
        Returns:
            Calculated dwell time in seconds
        """
        # Base dwell times for different modes
        base_times = {
            BeamMode.SEARCH: 0.005,        # 5 ms
            BeamMode.TRACK: 0.010,         # 10 ms
            BeamMode.CONFIRM: 0.020,       # 20 ms
            BeamMode.CLASSIFICATION: 0.050  # 50 ms
        }
        
        base_time = base_times.get(mode, 0.010)
        
        # Scale based on priority (higher priority gets more time)
        priority_factor = 1.0 + np.log10(max(priority, 0.1))
        
        return base_time * priority_factor
    
    def get_next_tasks(self, current_time: float, max_tasks: int = None) -> List[BeamTask]:
        """
        Get the next beam tasks to execute.
        
        Args:
            current_time: Current simulation time
            max_tasks: Maximum number of tasks to return (default: max_concurrent_beams)
            
        Returns:
            List of beam tasks to execute
        """
        if max_tasks is None:
            max_tasks = self.max_concurrent_beams
        
        next_tasks = []
        temp_queue = []
        
        # Extract tasks from queue, filtering expired ones
        while self.task_queue and len(next_tasks) < max_tasks:
            task = heapq.heappop(self.task_queue)
            
            if task.deadline < current_time:
                # Task has expired
                self.missed_deadlines += 1
                continue
            
            next_tasks.append(task)
        
        # Put any remaining tasks back in queue
        for task in temp_queue:
            heapq.heappush(self.task_queue, task)
        
        return next_tasks
    
    def execute_tasks(self, tasks: List[BeamTask], current_time: float) -> Dict[str, float]:
        """
        Execute beam tasks and return completion times.
        
        Args:
            tasks: List of beam tasks to execute
            current_time: Current simulation time
            
        Returns:
            Dictionary mapping target_id to task completion time
        """
        completion_times = {}
        
        for task in tasks:
            # Calculate beam steering time
            steering_time = self._calculate_steering_time(task.beam_position)
            
            # Total task time = steering + settle + dwell
            total_time = steering_time + self.beam_capability.settle_time + task.dwell_time
            completion_time = current_time + total_time
            
            # Update tracking
            completion_times[task.target_id] = completion_time
            self.target_last_visit[task.target_id] = completion_time
            self.completed_tasks.append(task)
            self.total_beam_time += total_time
            self.task_completion_times.append(total_time)
        
        return completion_times
    
    def _calculate_steering_time(self, beam_position: Tuple[float, float]) -> float:
        """
        Calculate time required to steer beam to position.
        
        Args:
            beam_position: (azimuth, elevation) in radians
            
        Returns:
            Steering time in seconds
        """
        # For simplicity, assume instantaneous steering for phased array
        # In reality, this would depend on beam steering mechanism
        return 0.0
    
    def update_adaptive_intervals(self, 
                                target_id: str, 
                                new_interval: float,
                                mode: BeamMode = None):
        """
        Update adaptive revisit interval for a target.
        
        Args:
            target_id: Target identifier
            new_interval: New revisit interval in seconds
            mode: Beam mode for this interval (optional)
        """
        if mode is not None:
            key = f"{target_id}_{mode.value}"
        else:
            key = target_id
            
        self.target_revisit_intervals[key] = new_interval
    
    def get_revisit_interval(self, target_id: str, mode: BeamMode) -> float:
        """
        Get revisit interval for a target and mode.
        
        Args:
            target_id: Target identifier
            mode: Beam operation mode
            
        Returns:
            Revisit interval in seconds
        """
        # Check for target-specific mode interval
        mode_key = f"{target_id}_{mode.value}"
        if mode_key in self.target_revisit_intervals:
            return self.target_revisit_intervals[mode_key]
        
        # Check for target-specific interval
        if target_id in self.target_revisit_intervals:
            return self.target_revisit_intervals[target_id]
        
        # Use default based on mode
        if mode == BeamMode.SEARCH:
            return self.search_revisit_interval
        elif mode == BeamMode.TRACK:
            return self.track_revisit_interval
        else:
            return (self.search_revisit_interval + self.track_revisit_interval) / 2
    
    def needs_revisit(self, target_id: str, mode: BeamMode, current_time: float) -> bool:
        """
        Check if a target needs to be revisited.
        
        Args:
            target_id: Target identifier
            mode: Beam operation mode
            current_time: Current simulation time
            
        Returns:
            True if target needs revisit
        """
        if target_id not in self.target_last_visit:
            return True
        
        last_visit = self.target_last_visit[target_id]
        interval = self.get_revisit_interval(target_id, mode)
        
        return (current_time - last_visit) >= interval
    
    def optimize_beam_sequence(self, tasks: List[BeamTask]) -> List[BeamTask]:
        """
        Optimize the sequence of beam tasks to minimize total time.
        
        Args:
            tasks: List of beam tasks to optimize
            
        Returns:
            Optimized sequence of beam tasks
        """
        if len(tasks) <= 1:
            return tasks
        
        # Simple nearest-neighbor optimization for beam positions
        optimized = [tasks[0]]
        remaining = tasks[1:]
        
        while remaining:
            current_pos = optimized[-1].beam_position
            
            # Find closest remaining task
            min_distance = float('inf')
            closest_idx = 0
            
            for i, task in enumerate(remaining):
                distance = self._angular_distance(current_pos, task.beam_position)
                if distance < min_distance:
                    min_distance = distance
                    closest_idx = i
            
            optimized.append(remaining.pop(closest_idx))
        
        return optimized
    
    def _angular_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Calculate angular distance between two beam positions.
        
        Args:
            pos1: First position (azimuth, elevation)
            pos2: Second position (azimuth, elevation)
            
        Returns:
            Angular distance in radians
        """
        az1, el1 = pos1
        az2, el2 = pos2
        
        # Convert to Cartesian coordinates on unit sphere
        x1 = np.cos(el1) * np.cos(az1)
        y1 = np.cos(el1) * np.sin(az1)
        z1 = np.sin(el1)
        
        x2 = np.cos(el2) * np.cos(az2)
        y2 = np.cos(el2) * np.sin(az2)
        z2 = np.sin(el2)
        
        # Dot product gives cosine of angle
        dot_product = x1*x2 + y1*y2 + z1*z2
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        return np.arccos(dot_product)
    
    def get_scheduler_statistics(self) -> Dict[str, float]:
        """
        Get scheduler performance statistics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.task_completion_times:
            return {}
        
        return {
            'total_beam_time': self.total_beam_time,
            'total_tasks_completed': len(self.completed_tasks),
            'average_task_time': np.mean(self.task_completion_times),
            'max_task_time': np.max(self.task_completion_times),
            'min_task_time': np.min(self.task_completion_times),
            'missed_deadlines': self.missed_deadlines,
            'tasks_in_queue': len(self.task_queue),
            'beam_utilization': self.total_beam_time / max(1.0, len(self.completed_tasks) * 0.1)
        }
    
    def clear_completed_tasks(self, older_than: float = None):
        """
        Clear completed task history.
        
        Args:
            older_than: Only clear tasks older than this time (optional)
        """
        if older_than is None:
            self.completed_tasks.clear()
            self.task_completion_times.clear()
        else:
            self.completed_tasks = [task for task in self.completed_tasks 
                                  if task.created_time > older_than]
            # Note: task_completion_times doesn't have timestamps, so clear all
            if len(self.completed_tasks) == 0:
                self.task_completion_times.clear()