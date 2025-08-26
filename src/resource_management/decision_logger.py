"""
Decision Logger for Radar Resource Management

This module implements comprehensive logging and analysis of resource allocation 
decisions including:
- All resource allocation decisions with timestamps
- Track priority changes over time
- Performance metrics calculation
- Export capabilities for analysis
- Real-time decision monitoring

The DecisionLogger provides detailed insight into resource management behavior,
enabling optimization and performance analysis of radar tracking systems.

Author: RadarSim Development Team
"""

import numpy as np
import json
import csv
import time
import uuid
import logging
import sqlite3
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timezone
import statistics
import pickle

# Import resource management components
from .resource_manager import ResourceAllocation, BeamPosition, BeamMode
from .priority_calculator import TargetInfo, ThreatLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of resource allocation decisions."""
    RESOURCE_GRANTED = "resource_granted"
    RESOURCE_DENIED = "resource_denied"
    PRIORITY_UPDATE = "priority_update"
    TARGET_ADDED = "target_added"
    TARGET_REMOVED = "target_removed"
    BEAM_ALLOCATION = "beam_allocation"
    MODE_CHANGE = "mode_change"
    SCHEDULE_UPDATE = "schedule_update"


class LogLevel(Enum):
    """Logging levels for different types of decisions."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DecisionRecord:
    """
    Record of a single resource allocation decision.
    
    Attributes:
        decision_id: Unique identifier for this decision
        timestamp: Time when decision was made
        decision_type: Type of decision (granted, denied, etc.)
        target_id: Target affected by the decision
        beam_id: Beam position involved (if applicable)
        priority_score: Priority score at time of decision
        dwell_time: Allocated dwell time (if applicable)
        beam_mode: Beam operation mode
        reasoning: Explanation of decision reasoning
        context: Additional context information
        metrics: Performance metrics at time of decision
        log_level: Severity level of this decision
    """
    decision_id: str
    timestamp: float
    decision_type: DecisionType
    target_id: Optional[str] = None
    beam_id: Optional[int] = None
    priority_score: Optional[float] = None
    dwell_time: Optional[float] = None
    beam_mode: Optional[BeamMode] = None
    reasoning: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    log_level: LogLevel = LogLevel.INFO


@dataclass
class PriorityChangeRecord:
    """
    Record of priority changes for a target over time.
    
    Attributes:
        target_id: Target identifier
        timestamp: Time of priority change
        old_priority: Previous priority score
        new_priority: New priority score
        change_reason: Reason for priority change
        contributing_factors: Factors that influenced the change
    """
    target_id: str
    timestamp: float
    old_priority: float
    new_priority: float
    change_reason: str = ""
    contributing_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """
    Snapshot of system performance metrics at a point in time.
    
    Attributes:
        timestamp: Time of snapshot
        total_targets: Number of active targets
        total_allocations: Number of active resource allocations
        average_priority: Average priority across all targets
        resource_utilization: Fraction of resources in use
        beam_utilization: Per-beam utilization statistics
        decision_latency: Average time to make allocation decisions
        missed_opportunities: Number of missed allocation opportunities
        quality_metrics: Track quality metrics
    """
    timestamp: float
    total_targets: int = 0
    total_allocations: int = 0
    average_priority: float = 0.0
    resource_utilization: float = 0.0
    beam_utilization: Dict[int, float] = field(default_factory=dict)
    decision_latency: float = 0.0
    missed_opportunities: int = 0
    quality_metrics: Dict[str, float] = field(default_factory=dict)


class DecisionLogger:
    """
    Comprehensive decision logger for radar resource management.
    
    This class logs all resource allocation decisions, tracks priority changes,
    calculates performance metrics, and provides export capabilities for
    detailed analysis of resource management behavior.
    """
    
    def __init__(self, 
                 log_file: Optional[str] = None,
                 database_file: Optional[str] = None,
                 max_memory_records: int = 10000,
                 enable_real_time_analysis: bool = True,
                 performance_snapshot_interval: float = 1.0):
        """
        Initialize the decision logger.
        
        Args:
            log_file: Optional file path for text logging
            database_file: Optional SQLite database file for structured logging
            max_memory_records: Maximum number of records to keep in memory
            enable_real_time_analysis: Enable real-time performance analysis
            performance_snapshot_interval: Interval for performance snapshots (seconds)
        """
        # Configuration
        self.log_file = log_file
        self.database_file = database_file
        self.max_memory_records = max_memory_records
        self.enable_real_time_analysis = enable_real_time_analysis
        self.performance_snapshot_interval = performance_snapshot_interval
        
        # Data storage
        self.decision_records: deque = deque(maxlen=max_memory_records)
        self.priority_changes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.performance_snapshots: deque = deque(maxlen=1000)
        
        # Real-time tracking
        self.current_priorities: Dict[str, float] = {}
        self.current_allocations: Dict[str, ResourceAllocation] = {}
        self.active_targets: Set[str] = set()
        self.beam_usage_tracking: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Performance metrics
        self.decision_count_by_type: Dict[DecisionType, int] = defaultdict(int)
        self.decision_times: deque = deque(maxlen=1000)
        self.last_performance_snapshot: float = 0.0
        
        # Database connection
        self.db_connection: Optional[sqlite3.Connection] = None
        if database_file:
            self._initialize_database()
        
        # File logging
        self.file_logger: Optional[logging.Logger] = None
        if log_file:
            self._initialize_file_logging()
        
        logger.info("DecisionLogger initialized with %s records in memory", max_memory_records)
    
    def log_decision(self,
                    decision_type: DecisionType,
                    target_id: Optional[str] = None,
                    beam_id: Optional[int] = None,
                    priority_score: Optional[float] = None,
                    dwell_time: Optional[float] = None,
                    beam_mode: Optional[BeamMode] = None,
                    reasoning: str = "",
                    context: Optional[Dict[str, Any]] = None,
                    metrics: Optional[Dict[str, float]] = None,
                    log_level: LogLevel = LogLevel.INFO) -> str:
        """
        Log a resource allocation decision.
        
        Args:
            decision_type: Type of decision being logged
            target_id: Target affected by decision
            beam_id: Beam position involved
            priority_score: Priority score at time of decision
            dwell_time: Allocated dwell time
            beam_mode: Beam operation mode
            reasoning: Explanation of decision reasoning
            context: Additional context information
            metrics: Performance metrics at time of decision
            log_level: Severity level of this decision
            
        Returns:
            Unique decision ID for this record
        """
        decision_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Create decision record
        record = DecisionRecord(
            decision_id=decision_id,
            timestamp=timestamp,
            decision_type=decision_type,
            target_id=target_id,
            beam_id=beam_id,
            priority_score=priority_score,
            dwell_time=dwell_time,
            beam_mode=beam_mode,
            reasoning=reasoning,
            context=context or {},
            metrics=metrics or {},
            log_level=log_level
        )
        
        # Store in memory
        self.decision_records.append(record)
        
        # Update tracking
        self.decision_count_by_type[decision_type] += 1
        
        # Log to file if configured
        if self.file_logger:
            self._log_to_file(record)
        
        # Log to database if configured
        if self.db_connection:
            self._log_to_database(record)
        
        # Update real-time tracking
        if target_id:
            if decision_type == DecisionType.TARGET_ADDED:
                self.active_targets.add(target_id)
            elif decision_type == DecisionType.TARGET_REMOVED:
                self.active_targets.discard(target_id)
            
            if priority_score is not None:
                old_priority = self.current_priorities.get(target_id, 0.0)
                if abs(old_priority - priority_score) > 0.01:  # Significant change
                    self.log_priority_change(target_id, old_priority, priority_score, reasoning)
                self.current_priorities[target_id] = priority_score
        
        # Update beam usage tracking
        if beam_id is not None and dwell_time is not None:
            self.beam_usage_tracking[beam_id].append((timestamp, dwell_time))
        
        # Perform real-time analysis if enabled
        if self.enable_real_time_analysis:
            self._update_real_time_metrics(timestamp)
        
        # Debug logging
        logger.debug(f"Logged {decision_type.value} decision: {decision_id}")
        
        return decision_id
    
    def log_priority_change(self,
                          target_id: str,
                          old_priority: float,
                          new_priority: float,
                          change_reason: str = "",
                          contributing_factors: Optional[Dict[str, float]] = None) -> None:
        """
        Log a priority change for a target.
        
        Args:
            target_id: Target identifier
            old_priority: Previous priority score
            new_priority: New priority score
            change_reason: Reason for priority change
            contributing_factors: Factors that influenced the change
        """
        timestamp = time.time()
        
        change_record = PriorityChangeRecord(
            target_id=target_id,
            timestamp=timestamp,
            old_priority=old_priority,
            new_priority=new_priority,
            change_reason=change_reason,
            contributing_factors=contributing_factors or {}
        )
        
        self.priority_changes[target_id].append(change_record)
        
        # Log as decision for consistency
        self.log_decision(
            DecisionType.PRIORITY_UPDATE,
            target_id=target_id,
            priority_score=new_priority,
            reasoning=f"Priority change: {old_priority:.3f} -> {new_priority:.3f} ({change_reason})",
            context={
                "old_priority": old_priority,
                "new_priority": new_priority,
                "change_magnitude": abs(new_priority - old_priority),
                "contributing_factors": contributing_factors or {}
            }
        )
    
    def log_resource_allocation(self, allocation: ResourceAllocation, granted: bool, reasoning: str = "") -> None:
        """
        Log a resource allocation decision.
        
        Args:
            allocation: Resource allocation object
            granted: Whether the allocation was granted or denied
            reasoning: Explanation for the decision
        """
        decision_type = DecisionType.RESOURCE_GRANTED if granted else DecisionType.RESOURCE_DENIED
        
        self.log_decision(
            decision_type=decision_type,
            target_id=allocation.target_id,
            beam_id=allocation.beam_position.beam_id,
            priority_score=allocation.priority,
            dwell_time=allocation.dwell_time if granted else None,
            beam_mode=allocation.mode,
            reasoning=reasoning,
            context={
                "start_time": allocation.start_time,
                "beam_azimuth": allocation.beam_position.azimuth,
                "beam_elevation": allocation.beam_position.elevation,
                "requested_dwell_time": allocation.dwell_time
            }
        )
        
        # Track current allocations
        if granted:
            self.current_allocations[allocation.target_id] = allocation
        
    def take_performance_snapshot(self, additional_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Take a snapshot of current system performance.
        
        Args:
            additional_metrics: Additional metrics to include in snapshot
        """
        timestamp = time.time()
        
        # Calculate beam utilization
        beam_utilization = {}
        current_time = timestamp
        for beam_id, usage_history in self.beam_usage_tracking.items():
            # Calculate utilization over last 10 seconds
            recent_usage = [
                dwell_time for ts, dwell_time in usage_history
                if current_time - ts <= 10.0
            ]
            beam_utilization[beam_id] = sum(recent_usage) / 10.0  # Fraction of time used
        
        # Calculate decision latency
        recent_decision_times = [
            dt for dt in self.decision_times
            if dt is not None
        ]
        avg_decision_latency = np.mean(recent_decision_times) if recent_decision_times else 0.0
        
        # Create performance snapshot
        snapshot = PerformanceSnapshot(
            timestamp=timestamp,
            total_targets=len(self.active_targets),
            total_allocations=len(self.current_allocations),
            average_priority=np.mean(list(self.current_priorities.values())) if self.current_priorities else 0.0,
            resource_utilization=np.mean(list(beam_utilization.values())) if beam_utilization else 0.0,
            beam_utilization=beam_utilization,
            decision_latency=avg_decision_latency,
            missed_opportunities=self._count_recent_missed_opportunities(),
            quality_metrics=additional_metrics or {}
        )
        
        self.performance_snapshots.append(snapshot)
        self.last_performance_snapshot = timestamp
        
        logger.debug(f"Performance snapshot: {len(self.active_targets)} targets, "
                    f"{snapshot.resource_utilization:.2f} resource utilization")
    
    def get_decision_statistics(self, 
                              time_window: Optional[float] = None) -> Dict[str, Any]:
        """
        Get statistics about logged decisions.
        
        Args:
            time_window: Time window to analyze (seconds from now), None for all time
            
        Returns:
            Dictionary of decision statistics
        """
        current_time = time.time()
        cutoff_time = current_time - time_window if time_window else 0.0
        
        # Filter records by time window
        filtered_records = [
            record for record in self.decision_records
            if record.timestamp >= cutoff_time
        ]
        
        if not filtered_records:
            return {"error": "No records in specified time window"}
        
        # Count decisions by type
        decisions_by_type = defaultdict(int)
        decisions_by_level = defaultdict(int)
        target_decisions = defaultdict(int)
        
        for record in filtered_records:
            decisions_by_type[record.decision_type.value] += 1
            decisions_by_level[record.log_level.value] += 1
            if record.target_id:
                target_decisions[record.target_id] += 1
        
        # Calculate success rates
        granted = decisions_by_type.get(DecisionType.RESOURCE_GRANTED.value, 0)
        denied = decisions_by_type.get(DecisionType.RESOURCE_DENIED.value, 0)
        success_rate = granted / max(granted + denied, 1)
        
        # Priority statistics
        priority_scores = [
            record.priority_score for record in filtered_records
            if record.priority_score is not None
        ]
        
        return {
            "total_decisions": len(filtered_records),
            "time_window": time_window,
            "decisions_by_type": dict(decisions_by_type),
            "decisions_by_level": dict(decisions_by_level),
            "success_rate": success_rate,
            "unique_targets": len(target_decisions),
            "priority_statistics": {
                "mean": np.mean(priority_scores) if priority_scores else 0.0,
                "std": np.std(priority_scores) if priority_scores else 0.0,
                "min": np.min(priority_scores) if priority_scores else 0.0,
                "max": np.max(priority_scores) if priority_scores else 0.0
            },
            "most_active_targets": dict(sorted(target_decisions.items(), 
                                             key=lambda x: x[1], reverse=True)[:10])
        }
    
    def get_priority_trends(self, target_id: str) -> Dict[str, Any]:
        """
        Get priority change trends for a specific target.
        
        Args:
            target_id: Target to analyze
            
        Returns:
            Dictionary of priority trend information
        """
        if target_id not in self.priority_changes:
            return {"error": f"No priority changes recorded for target {target_id}"}
        
        changes = list(self.priority_changes[target_id])
        if not changes:
            return {"error": f"No priority changes recorded for target {target_id}"}
        
        # Extract time series data
        timestamps = [change.timestamp for change in changes]
        priorities = [change.new_priority for change in changes]
        
        # Calculate trends
        if len(priorities) > 1:
            # Linear trend
            time_deltas = np.array(timestamps) - timestamps[0]
            slope, intercept = np.polyfit(time_deltas, priorities, 1)
            
            # Volatility
            priority_changes_magnitudes = [
                abs(changes[i].new_priority - changes[i].old_priority)
                for i in range(len(changes))
            ]
            
        else:
            slope = 0.0
            priority_changes_magnitudes = [0.0]
        
        return {
            "target_id": target_id,
            "total_changes": len(changes),
            "current_priority": priorities[-1] if priorities else 0.0,
            "min_priority": min(priorities) if priorities else 0.0,
            "max_priority": max(priorities) if priorities else 0.0,
            "average_priority": np.mean(priorities) if priorities else 0.0,
            "priority_trend_slope": slope,
            "priority_volatility": np.mean(priority_changes_magnitudes),
            "time_span": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0,
            "recent_changes": [asdict(change) for change in changes[-5:]]  # Last 5 changes
        }
    
    def get_performance_metrics(self, time_window: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            time_window: Time window to analyze (seconds), None for all time
            
        Returns:
            Dictionary of performance metrics
        """
        current_time = time.time()
        cutoff_time = current_time - time_window if time_window else 0.0
        
        # Filter snapshots by time window
        filtered_snapshots = [
            snapshot for snapshot in self.performance_snapshots
            if snapshot.timestamp >= cutoff_time
        ]
        
        if not filtered_snapshots:
            return {"error": "No performance snapshots in specified time window"}
        
        # Resource utilization metrics
        utilization_values = [s.resource_utilization for s in filtered_snapshots]
        target_counts = [s.total_targets for s in filtered_snapshots]
        allocation_counts = [s.total_allocations for s in filtered_snapshots]
        
        # Decision latency metrics
        latency_values = [s.decision_latency for s in filtered_snapshots if s.decision_latency > 0]
        
        # Missed opportunities
        missed_opportunities = [s.missed_opportunities for s in filtered_snapshots]
        
        return {
            "time_window": time_window,
            "snapshot_count": len(filtered_snapshots),
            "resource_utilization": {
                "mean": np.mean(utilization_values),
                "std": np.std(utilization_values),
                "min": np.min(utilization_values),
                "max": np.max(utilization_values)
            },
            "target_statistics": {
                "mean_count": np.mean(target_counts),
                "max_count": np.max(target_counts),
                "min_count": np.min(target_counts)
            },
            "allocation_statistics": {
                "mean_count": np.mean(allocation_counts),
                "max_count": np.max(allocation_counts),
                "allocation_efficiency": np.mean(allocation_counts) / max(np.mean(target_counts), 1)
            },
            "decision_latency": {
                "mean": np.mean(latency_values) if latency_values else 0.0,
                "std": np.std(latency_values) if latency_values else 0.0,
                "95th_percentile": np.percentile(latency_values, 95) if latency_values else 0.0
            },
            "missed_opportunities": {
                "total": sum(missed_opportunities),
                "rate": np.mean(missed_opportunities)
            }
        }
    
    def export_to_csv(self, filename: str, include_snapshots: bool = True) -> None:
        """
        Export logged data to CSV format.
        
        Args:
            filename: Output filename
            include_snapshots: Whether to include performance snapshots
        """
        try:
            with open(filename, 'w', newline='') as csvfile:
                # Export decision records
                if self.decision_records:
                    fieldnames = list(asdict(self.decision_records[0]).keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for record in self.decision_records:
                        row = asdict(record)
                        # Convert complex types to strings
                        if row['decision_type']:
                            row['decision_type'] = row['decision_type'].value
                        if row['beam_mode']:
                            row['beam_mode'] = row['beam_mode'].value
                        if row['log_level']:
                            row['log_level'] = row['log_level'].value
                        row['context'] = json.dumps(row['context'])
                        row['metrics'] = json.dumps(row['metrics'])
                        writer.writerow(row)
            
            # Export performance snapshots if requested
            if include_snapshots and self.performance_snapshots:
                snapshot_filename = filename.replace('.csv', '_snapshots.csv')
                with open(snapshot_filename, 'w', newline='') as csvfile:
                    fieldnames = list(asdict(self.performance_snapshots[0]).keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for snapshot in self.performance_snapshots:
                        row = asdict(snapshot)
                        row['beam_utilization'] = json.dumps(row['beam_utilization'])
                        row['quality_metrics'] = json.dumps(row['quality_metrics'])
                        writer.writerow(row)
            
            logger.info(f"Exported decision log to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            raise
    
    def export_to_json(self, filename: str) -> None:
        """
        Export logged data to JSON format.
        
        Args:
            filename: Output filename
        """
        try:
            export_data = {
                "metadata": {
                    "export_time": time.time(),
                    "total_decisions": len(self.decision_records),
                    "total_snapshots": len(self.performance_snapshots),
                    "active_targets": len(self.active_targets)
                },
                "decision_records": [],
                "performance_snapshots": [],
                "priority_changes": {},
                "statistics": self.get_decision_statistics()
            }
            
            # Convert decision records
            for record in self.decision_records:
                record_dict = asdict(record)
                record_dict['decision_type'] = record.decision_type.value
                if record.beam_mode:
                    record_dict['beam_mode'] = record.beam_mode.value
                record_dict['log_level'] = record.log_level.value
                export_data["decision_records"].append(record_dict)
            
            # Convert performance snapshots
            for snapshot in self.performance_snapshots:
                export_data["performance_snapshots"].append(asdict(snapshot))
            
            # Convert priority changes
            for target_id, changes in self.priority_changes.items():
                export_data["priority_changes"][target_id] = [
                    asdict(change) for change in changes
                ]
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported decision log to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            raise
    
    def clear_logs(self, keep_recent: Optional[float] = None) -> None:
        """
        Clear logged data, optionally keeping recent records.
        
        Args:
            keep_recent: Time window to keep (seconds), None clears everything
        """
        if keep_recent is None:
            # Clear everything
            self.decision_records.clear()
            self.priority_changes.clear()
            self.performance_snapshots.clear()
            self.decision_count_by_type.clear()
            self.decision_times.clear()
        else:
            # Keep recent records
            current_time = time.time()
            cutoff_time = current_time - keep_recent
            
            # Filter decision records
            recent_decisions = [
                record for record in self.decision_records
                if record.timestamp >= cutoff_time
            ]
            self.decision_records.clear()
            self.decision_records.extend(recent_decisions)
            
            # Filter performance snapshots
            recent_snapshots = [
                snapshot for snapshot in self.performance_snapshots
                if snapshot.timestamp >= cutoff_time
            ]
            self.performance_snapshots.clear()
            self.performance_snapshots.extend(recent_snapshots)
            
            # Filter priority changes
            for target_id in list(self.priority_changes.keys()):
                recent_changes = [
                    change for change in self.priority_changes[target_id]
                    if change.timestamp >= cutoff_time
                ]
                if recent_changes:
                    self.priority_changes[target_id] = deque(recent_changes, maxlen=1000)
                else:
                    del self.priority_changes[target_id]
        
        logger.info(f"Cleared logs, keeping recent {keep_recent}s" if keep_recent else "Cleared all logs")
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database for structured logging."""
        try:
            self.db_connection = sqlite3.connect(self.database_file)
            cursor = self.db_connection.cursor()
            
            # Create decision records table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS decision_records (
                    decision_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    decision_type TEXT,
                    target_id TEXT,
                    beam_id INTEGER,
                    priority_score REAL,
                    dwell_time REAL,
                    beam_mode TEXT,
                    reasoning TEXT,
                    context TEXT,
                    metrics TEXT,
                    log_level TEXT
                )
            ''')
            
            # Create performance snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_snapshots (
                    timestamp REAL PRIMARY KEY,
                    total_targets INTEGER,
                    total_allocations INTEGER,
                    average_priority REAL,
                    resource_utilization REAL,
                    beam_utilization TEXT,
                    decision_latency REAL,
                    missed_opportunities INTEGER,
                    quality_metrics TEXT
                )
            ''')
            
            self.db_connection.commit()
            logger.info(f"Initialized database: {self.database_file}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.db_connection = None
    
    def _initialize_file_logging(self) -> None:
        """Initialize file logging."""
        try:
            self.file_logger = logging.getLogger(f"decision_logger_{id(self)}")
            self.file_logger.setLevel(logging.DEBUG)
            
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.file_logger.addHandler(handler)
            
            logger.info(f"Initialized file logging: {self.log_file}")
            
        except Exception as e:
            logger.error(f"Failed to initialize file logging: {e}")
            self.file_logger = None
    
    def _log_to_file(self, record: DecisionRecord) -> None:
        """Log record to file."""
        if self.file_logger:
            log_message = (
                f"[{record.decision_type.value}] "
                f"Target: {record.target_id or 'N/A'}, "
                f"Beam: {record.beam_id or 'N/A'}, "
                f"Priority: {record.priority_score or 'N/A'}, "
                f"Dwell: {record.dwell_time or 'N/A'}s, "
                f"Reason: {record.reasoning}"
            )
            
            log_func = getattr(self.file_logger, record.log_level.value, self.file_logger.info)
            log_func(log_message)
    
    def _log_to_database(self, record: DecisionRecord) -> None:
        """Log record to database."""
        if self.db_connection:
            try:
                cursor = self.db_connection.cursor()
                cursor.execute('''
                    INSERT INTO decision_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.decision_id,
                    record.timestamp,
                    record.decision_type.value,
                    record.target_id,
                    record.beam_id,
                    record.priority_score,
                    record.dwell_time,
                    record.beam_mode.value if record.beam_mode else None,
                    record.reasoning,
                    json.dumps(record.context),
                    json.dumps(record.metrics),
                    record.log_level.value
                ))
                self.db_connection.commit()
            except Exception as e:
                logger.error(f"Failed to log to database: {e}")
    
    def _update_real_time_metrics(self, timestamp: float) -> None:
        """Update real-time performance metrics."""
        # Take performance snapshot if interval has passed
        if timestamp - self.last_performance_snapshot >= self.performance_snapshot_interval:
            self.take_performance_snapshot()
    
    def _count_recent_missed_opportunities(self, time_window: float = 10.0) -> int:
        """Count missed opportunities in recent time window."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        missed_count = 0
        for record in self.decision_records:
            if (record.timestamp >= cutoff_time and 
                record.decision_type == DecisionType.RESOURCE_DENIED):
                missed_count += 1
        
        return missed_count


# Utility functions for creating and configuring decision loggers

def create_decision_logger(log_directory: str = "logs",
                         enable_database: bool = True,
                         enable_file_logging: bool = True) -> DecisionLogger:
    """
    Create a configured decision logger.
    
    Args:
        log_directory: Directory for log files
        enable_database: Enable SQLite database logging
        enable_file_logging: Enable text file logging
        
    Returns:
        Configured DecisionLogger instance
    """
    import os
    
    # Create log directory if it doesn't exist
    os.makedirs(log_directory, exist_ok=True)
    
    # Generate timestamped filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = None
    if enable_file_logging:
        log_file = os.path.join(log_directory, f"decision_log_{timestamp}.log")
    
    database_file = None
    if enable_database:
        database_file = os.path.join(log_directory, f"decision_log_{timestamp}.db")
    
    return DecisionLogger(
        log_file=log_file,
        database_file=database_file,
        enable_real_time_analysis=True
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Testing DecisionLogger...")
    
    # Create logger
    logger_instance = create_decision_logger()
    
    # Log some example decisions
    target_ids = ["target_001", "target_002", "target_003"]
    
    for i in range(20):
        target_id = np.random.choice(target_ids)
        priority = np.random.uniform(0.1, 1.0)
        
        # Random decision type
        if np.random.random() < 0.7:  # 70% grant rate
            decision_type = DecisionType.RESOURCE_GRANTED
            dwell_time = np.random.uniform(0.1, 0.5)
        else:
            decision_type = DecisionType.RESOURCE_DENIED
            dwell_time = None
        
        logger_instance.log_decision(
            decision_type=decision_type,
            target_id=target_id,
            beam_id=np.random.randint(0, 120),
            priority_score=priority,
            dwell_time=dwell_time,
            beam_mode=BeamMode.TRACK,
            reasoning=f"Automated test decision {i}",
            context={"test_iteration": i}
        )
        
        # Simulate some time passing
        time.sleep(0.01)
    
    # Take performance snapshot
    logger_instance.take_performance_snapshot({
        "test_metric": 0.85,
        "efficiency": 0.92
    })
    
    # Get statistics
    stats = logger_instance.get_decision_statistics(time_window=10.0)
    print("\nDecision Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Export data
    try:
        logger_instance.export_to_json("test_decision_log.json")
        print("\nExported decision log to test_decision_log.json")
    except Exception as e:
        print(f"Export failed: {e}")
    
    print("\nDecisionLogger test completed successfully!")