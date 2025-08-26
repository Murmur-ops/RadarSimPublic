"""
Dynamic Scenario Management System for RadarSim

This module provides comprehensive dynamic scenario management capabilities including:
- Runtime target injection based on time triggers
- Event queue system for scheduled events
- YAML configuration support for scenario events
- Automatic threat reassessment on new target detection
- Integration with existing radar simulation components

The DynamicScenarioManager allows for complex, evolving scenarios where targets
appear and disappear over time, enabling realistic testing of radar tracking
systems under dynamic conditions.

Author: RadarSim Development Team
"""

import numpy as np
import yaml
import time
import uuid
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import heapq
from abc import ABC, abstractmethod

# Import existing RadarSim components
from ..target import Target
from ..environment import Environment
from ..tracking.tracker_base import Measurement
from ..classification.threat_assessment import ThreatAssessment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Enumeration of event types in dynamic scenarios."""
    TARGET_INJECTION = "target_injection"
    TARGET_REMOVAL = "target_removal"
    PARAMETER_CHANGE = "parameter_change"
    MANEUVER_START = "maneuver_start"
    MANEUVER_END = "maneuver_end"
    THREAT_UPDATE = "threat_update"
    ENVIRONMENT_CHANGE = "environment_change"
    CUSTOM_EVENT = "custom_event"


class EventStatus(Enum):
    """Status of scenario events."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ScenarioEvent:
    """
    Represents a single event in the dynamic scenario.
    
    Attributes:
        event_id: Unique identifier for the event
        event_type: Type of event (from EventType enum)
        trigger_time: Time when event should be triggered (seconds)
        parameters: Event-specific parameters
        priority: Event priority (higher numbers = higher priority)
        repeating: Whether this is a repeating event
        repeat_interval: Interval for repeating events (seconds)
        condition_func: Optional condition function that must return True for event to trigger
        callback_func: Optional callback function to execute when event triggers
        status: Current status of the event
        metadata: Additional event metadata
    """
    event_id: str
    event_type: EventType
    trigger_time: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    repeating: bool = False
    repeat_interval: float = 0.0
    condition_func: Optional[Callable[[], bool]] = None
    callback_func: Optional[Callable[['ScenarioEvent'], Any]] = None
    status: EventStatus = EventStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other: 'ScenarioEvent') -> bool:
        """Compare events for priority queue ordering."""
        if self.trigger_time != other.trigger_time:
            return self.trigger_time < other.trigger_time
        return self.priority > other.priority  # Higher priority first


@dataclass
class TargetInjectionEvent:
    """
    Specific parameters for target injection events.
    
    Attributes:
        target_type: Type of target to inject
        initial_position: Initial position [x, y, z] in meters
        initial_velocity: Initial velocity [vx, vy, vz] in m/s
        rcs: Radar cross section in m²
        threat_level: Threat assessment level (0-1)
        maneuver_profile: Optional maneuver profile name
        duration: How long target should remain active (seconds, None for indefinite)
        target_class: Classification class for the target
        additional_properties: Additional target properties
    """
    target_type: str
    initial_position: np.ndarray
    initial_velocity: np.ndarray
    rcs: float
    threat_level: float = 0.5
    maneuver_profile: Optional[str] = None
    duration: Optional[float] = None
    target_class: str = "unknown"
    additional_properties: Dict[str, Any] = field(default_factory=dict)


class EventProcessor(ABC):
    """Abstract base class for event processors."""
    
    @abstractmethod
    def process_event(self, event: ScenarioEvent, scenario_manager: 'DynamicScenarioManager') -> bool:
        """
        Process a scenario event.
        
        Args:
            event: Event to process
            scenario_manager: Reference to the scenario manager
            
        Returns:
            True if event was processed successfully
        """
        pass


class TargetInjectionProcessor(EventProcessor):
    """Processor for target injection events."""
    
    def process_event(self, event: ScenarioEvent, scenario_manager: 'DynamicScenarioManager') -> bool:
        """Process target injection event."""
        try:
            # Extract target injection parameters
            target_params = TargetInjectionEvent(**event.parameters)
            
            # Create new target
            target_id = f"dynamic_target_{uuid.uuid4().hex[:8]}"
            
            # Create Target object (simplified - would integrate with actual Target class)
            target = self._create_target(target_id, target_params)
            
            # Add to scenario manager
            scenario_manager.add_target(target_id, target, target_params)
            
            # Log injection
            logger.info(f"Injected target {target_id} at time {event.trigger_time:.2f}s")
            logger.info(f"  Position: {target_params.initial_position}")
            logger.info(f"  Velocity: {target_params.initial_velocity}")
            logger.info(f"  RCS: {target_params.rcs} m²")
            logger.info(f"  Threat Level: {target_params.threat_level}")
            
            # Schedule removal if duration is specified
            if target_params.duration is not None:
                removal_event = ScenarioEvent(
                    event_id=f"remove_{target_id}",
                    event_type=EventType.TARGET_REMOVAL,
                    trigger_time=event.trigger_time + target_params.duration,
                    parameters={"target_id": target_id},
                    priority=event.priority
                )
                scenario_manager.add_event(removal_event)
            
            # Trigger threat reassessment
            scenario_manager.trigger_threat_reassessment()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process target injection event: {e}")
            return False
    
    def _create_target(self, target_id: str, params: TargetInjectionEvent) -> Dict[str, Any]:
        """Create target object from parameters."""
        return {
            "id": target_id,
            "type": params.target_type,
            "position": params.initial_position.copy(),
            "velocity": params.initial_velocity.copy(),
            "rcs": params.rcs,
            "threat_level": params.threat_level,
            "target_class": params.target_class,
            "creation_time": time.time(),
            "maneuver_profile": params.maneuver_profile,
            "properties": params.additional_properties.copy()
        }


class TargetRemovalProcessor(EventProcessor):
    """Processor for target removal events."""
    
    def process_event(self, event: ScenarioEvent, scenario_manager: 'DynamicScenarioManager') -> bool:
        """Process target removal event."""
        try:
            target_id = event.parameters.get("target_id")
            if not target_id:
                logger.error("Target removal event missing target_id")
                return False
            
            if scenario_manager.remove_target(target_id):
                logger.info(f"Removed target {target_id} at time {event.trigger_time:.2f}s")
                scenario_manager.trigger_threat_reassessment()
                return True
            else:
                logger.warning(f"Failed to remove target {target_id} - not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to process target removal event: {e}")
            return False


class ParameterChangeProcessor(EventProcessor):
    """Processor for parameter change events."""
    
    def process_event(self, event: ScenarioEvent, scenario_manager: 'DynamicScenarioManager') -> bool:
        """Process parameter change event."""
        try:
            target_id = event.parameters.get("target_id")
            parameter_name = event.parameters.get("parameter")
            new_value = event.parameters.get("value")
            
            if not all([target_id, parameter_name, new_value is not None]):
                logger.error("Parameter change event missing required parameters")
                return False
            
            if scenario_manager.update_target_parameter(target_id, parameter_name, new_value):
                logger.info(f"Updated {parameter_name} for target {target_id} to {new_value}")
                return True
            else:
                logger.warning(f"Failed to update parameter for target {target_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to process parameter change event: {e}")
            return False


class DynamicScenarioManager:
    """
    Dynamic Scenario Manager for RadarSim.
    
    This class manages dynamic scenarios where targets and events can be
    injected at runtime based on time triggers. It provides comprehensive
    event scheduling, target management, and threat assessment capabilities.
    """
    
    def __init__(self, 
                 scenario_config: Optional[Dict[str, Any]] = None,
                 threat_assessor: Optional[ThreatAssessment] = None,
                 environment: Optional[Environment] = None):
        """
        Initialize dynamic scenario manager.
        
        Args:
            scenario_config: Optional scenario configuration dictionary
            threat_assessor: Optional threat assessment system
            environment: Optional environment configuration
        """
        # Core components
        self.scenario_config = scenario_config or {}
        self.threat_assessor = threat_assessor
        self.environment = environment
        
        # Event management
        self.event_queue: List[ScenarioEvent] = []
        self.processed_events: List[ScenarioEvent] = []
        self.event_processors: Dict[EventType, EventProcessor] = {
            EventType.TARGET_INJECTION: TargetInjectionProcessor(),
            EventType.TARGET_REMOVAL: TargetRemovalProcessor(),
            EventType.PARAMETER_CHANGE: ParameterChangeProcessor()
        }
        
        # Target management
        self.active_targets: Dict[str, Dict[str, Any]] = {}
        self.target_injection_params: Dict[str, TargetInjectionEvent] = {}
        self.target_creation_times: Dict[str, float] = {}
        
        # Timing and simulation state
        self.simulation_time: float = 0.0
        self.start_time: float = time.time()
        self.time_scale: float = 1.0  # Real-time multiplier
        self.is_running: bool = False
        
        # Threat assessment
        self.last_threat_assessment_time: float = 0.0
        self.threat_assessment_interval: float = 1.0  # seconds
        self.threat_levels: Dict[str, float] = {}
        
        # Callbacks and hooks
        self.target_injection_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        self.target_removal_callbacks: List[Callable[[str], None]] = []
        self.threat_update_callbacks: List[Callable[[Dict[str, float]], None]] = []
        self.event_callbacks: Dict[EventType, List[Callable[[ScenarioEvent], None]]] = {}
        
        # Statistics
        self.stats = {
            "targets_injected": 0,
            "targets_removed": 0,
            "events_processed": 0,
            "threat_assessments": 0
        }
        
        # Initialize from config if provided
        if scenario_config:
            self.load_scenario_config(scenario_config)
    
    def load_scenario_config(self, config: Dict[str, Any]) -> None:
        """
        Load scenario configuration.
        
        Args:
            config: Scenario configuration dictionary
        """
        self.scenario_config = config
        
        # Load events from config
        if "events" in config:
            for event_config in config["events"]:
                event = self._create_event_from_config(event_config)
                if event:
                    self.add_event(event)
        
        # Load scenario parameters
        if "parameters" in config:
            params = config["parameters"]
            self.time_scale = params.get("time_scale", 1.0)
            self.threat_assessment_interval = params.get("threat_assessment_interval", 1.0)
        
        logger.info(f"Loaded scenario config with {len(self.event_queue)} events")
    
    def load_scenario_from_yaml(self, yaml_file: str) -> None:
        """
        Load scenario from YAML file.
        
        Args:
            yaml_file: Path to YAML scenario file
        """
        try:
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
            
            self.load_scenario_config(config)
            logger.info(f"Loaded scenario from {yaml_file}")
            
        except Exception as e:
            logger.error(f"Failed to load scenario from {yaml_file}: {e}")
            raise
    
    def add_event(self, event: ScenarioEvent) -> None:
        """
        Add an event to the scenario.
        
        Args:
            event: Event to add
        """
        heapq.heappush(self.event_queue, event)
        logger.debug(f"Added event {event.event_id} scheduled for time {event.trigger_time:.2f}s")
    
    def remove_event(self, event_id: str) -> bool:
        """
        Remove an event from the scenario.
        
        Args:
            event_id: ID of event to remove
            
        Returns:
            True if event was found and removed
        """
        for i, event in enumerate(self.event_queue):
            if event.event_id == event_id:
                event.status = EventStatus.CANCELLED
                return True
        return False
    
    def update(self, current_time: float) -> List[ScenarioEvent]:
        """
        Update scenario state and process events.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of events that were processed
        """
        self.simulation_time = current_time
        processed_events = []
        
        # Process due events
        while self.event_queue and self.event_queue[0].trigger_time <= current_time:
            event = heapq.heappop(self.event_queue)
            
            if event.status == EventStatus.CANCELLED:
                continue
            
            # Check condition if specified
            if event.condition_func and not event.condition_func():
                # Re-schedule for later if condition not met
                event.trigger_time += 0.1  # Check again in 0.1 seconds
                heapq.heappush(self.event_queue, event)
                continue
            
            # Process event
            success = self._process_event(event)
            
            if success:
                event.status = EventStatus.COMPLETED
                processed_events.append(event)
                self.processed_events.append(event)
                self.stats["events_processed"] += 1
                
                # Handle repeating events
                if event.repeating and event.repeat_interval > 0:
                    new_event = ScenarioEvent(
                        event_id=f"{event.event_id}_repeat_{len(self.processed_events)}",
                        event_type=event.event_type,
                        trigger_time=current_time + event.repeat_interval,
                        parameters=event.parameters.copy(),
                        priority=event.priority,
                        repeating=event.repeating,
                        repeat_interval=event.repeat_interval,
                        condition_func=event.condition_func,
                        callback_func=event.callback_func,
                        metadata=event.metadata.copy()
                    )
                    self.add_event(new_event)
                
                # Execute callback if specified
                if event.callback_func:
                    try:
                        event.callback_func(event)
                    except Exception as e:
                        logger.error(f"Error in event callback: {e}")
                
                # Trigger event-type-specific callbacks
                if event.event_type in self.event_callbacks:
                    for callback in self.event_callbacks[event.event_type]:
                        try:
                            callback(event)
                        except Exception as e:
                            logger.error(f"Error in event type callback: {e}")
            else:
                event.status = EventStatus.CANCELLED
                logger.warning(f"Failed to process event {event.event_id}")
        
        # Periodic threat assessment
        if (current_time - self.last_threat_assessment_time >= self.threat_assessment_interval and
            self.active_targets):
            self._perform_threat_assessment()
            self.last_threat_assessment_time = current_time
        
        return processed_events
    
    def add_target(self, target_id: str, target: Dict[str, Any], 
                   injection_params: Optional[TargetInjectionEvent] = None) -> None:
        """
        Add a target to the scenario.
        
        Args:
            target_id: Unique target identifier
            target: Target object/dictionary
            injection_params: Original injection parameters
        """
        self.active_targets[target_id] = target
        self.target_creation_times[target_id] = self.simulation_time
        
        if injection_params:
            self.target_injection_params[target_id] = injection_params
        
        self.stats["targets_injected"] += 1
        
        # Notify callbacks
        for callback in self.target_injection_callbacks:
            try:
                callback(target_id, target)
            except Exception as e:
                logger.error(f"Error in target injection callback: {e}")
    
    def remove_target(self, target_id: str) -> bool:
        """
        Remove a target from the scenario.
        
        Args:
            target_id: ID of target to remove
            
        Returns:
            True if target was found and removed
        """
        if target_id in self.active_targets:
            target = self.active_targets.pop(target_id)
            self.target_creation_times.pop(target_id, None)
            self.target_injection_params.pop(target_id, None)
            self.threat_levels.pop(target_id, None)
            
            self.stats["targets_removed"] += 1
            
            # Notify callbacks
            for callback in self.target_removal_callbacks:
                try:
                    callback(target_id)
                except Exception as e:
                    logger.error(f"Error in target removal callback: {e}")
            
            return True
        
        return False
    
    def update_target_parameter(self, target_id: str, parameter: str, value: Any) -> bool:
        """
        Update a target parameter.
        
        Args:
            target_id: Target ID
            parameter: Parameter name
            value: New parameter value
            
        Returns:
            True if update was successful
        """
        if target_id not in self.active_targets:
            return False
        
        try:
            if parameter in self.active_targets[target_id]:
                old_value = self.active_targets[target_id][parameter]
                self.active_targets[target_id][parameter] = value
                logger.debug(f"Updated {parameter} for target {target_id}: {old_value} -> {value}")
                return True
            else:
                logger.warning(f"Parameter {parameter} not found for target {target_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to update parameter {parameter} for target {target_id}: {e}")
            return False
    
    def get_active_targets(self) -> Dict[str, Dict[str, Any]]:
        """Get dictionary of active targets."""
        return self.active_targets.copy()
    
    def get_target_measurements(self, current_time: float) -> List[Measurement]:
        """
        Generate measurements for active targets.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of measurements from active targets
        """
        measurements = []
        
        for target_id, target in self.active_targets.items():
            # Update target position based on velocity
            dt = current_time - self.target_creation_times.get(target_id, current_time)
            
            if dt > 0:
                # Simple kinematic update
                position = target["position"] + target["velocity"] * dt
                target["position"] = position
            
            # Create measurement with noise
            noise_std = 1.0  # meters
            measured_position = target["position"] + np.random.normal(0, noise_std, 3)
            
            measurement = Measurement(
                position=measured_position,
                timestamp=current_time,
                covariance=np.eye(3) * noise_std**2,
                metadata={
                    "source_target_id": target_id,
                    "true_position": target["position"].copy(),
                    "rcs": target["rcs"],
                    "target_type": target["type"]
                }
            )
            
            measurements.append(measurement)
        
        return measurements
    
    def trigger_threat_reassessment(self) -> None:
        """Trigger immediate threat reassessment."""
        if self.active_targets:
            self._perform_threat_assessment()
    
    def add_target_injection_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add callback for target injection events."""
        self.target_injection_callbacks.append(callback)
    
    def add_target_removal_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback for target removal events."""
        self.target_removal_callbacks.append(callback)
    
    def add_threat_update_callback(self, callback: Callable[[Dict[str, float]], None]) -> None:
        """Add callback for threat level updates."""
        self.threat_update_callbacks.append(callback)
    
    def add_event_callback(self, event_type: EventType, callback: Callable[[ScenarioEvent], None]) -> None:
        """Add callback for specific event types."""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    def get_scenario_statistics(self) -> Dict[str, Any]:
        """Get scenario statistics."""
        return {
            **self.stats,
            "active_targets": len(self.active_targets),
            "pending_events": len(self.event_queue),
            "processed_events": len(self.processed_events),
            "simulation_time": self.simulation_time,
            "threat_levels": self.threat_levels.copy()
        }
    
    def export_scenario_log(self) -> Dict[str, Any]:
        """Export complete scenario log for analysis."""
        return {
            "config": self.scenario_config,
            "processed_events": [
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "trigger_time": event.trigger_time,
                    "parameters": event.parameters,
                    "status": event.status.value
                }
                for event in self.processed_events
            ],
            "statistics": self.get_scenario_statistics(),
            "final_targets": self.active_targets.copy()
        }
    
    def _create_event_from_config(self, event_config: Dict[str, Any]) -> Optional[ScenarioEvent]:
        """Create event from configuration dictionary."""
        try:
            event_id = event_config.get("id", str(uuid.uuid4()))
            event_type_str = event_config.get("type")
            trigger_time = event_config.get("time", 0.0)
            
            if not event_type_str:
                logger.error("Event config missing type")
                return None
            
            try:
                event_type = EventType(event_type_str)
            except ValueError:
                logger.error(f"Unknown event type: {event_type_str}")
                return None
            
            event = ScenarioEvent(
                event_id=event_id,
                event_type=event_type,
                trigger_time=trigger_time,
                parameters=event_config.get("parameters", {}),
                priority=event_config.get("priority", 0),
                repeating=event_config.get("repeating", False),
                repeat_interval=event_config.get("repeat_interval", 0.0),
                metadata=event_config.get("metadata", {})
            )
            
            return event
            
        except Exception as e:
            logger.error(f"Failed to create event from config: {e}")
            return None
    
    def _process_event(self, event: ScenarioEvent) -> bool:
        """Process a single event."""
        if event.event_type in self.event_processors:
            return self.event_processors[event.event_type].process_event(event, self)
        else:
            logger.warning(f"No processor for event type {event.event_type}")
            return False
    
    def _perform_threat_assessment(self) -> None:
        """Perform threat assessment on active targets."""
        if not self.threat_assessor:
            # Simple threat assessment based on position and velocity
            for target_id, target in self.active_targets.items():
                # Simple threat calculation based on proximity and speed
                distance = np.linalg.norm(target["position"])
                speed = np.linalg.norm(target["velocity"])
                
                # Normalize and combine factors
                distance_threat = max(0, 1.0 - distance / 50000.0)  # 50km max range
                speed_threat = min(1.0, speed / 300.0)  # Normalize by 300 m/s
                rcs_threat = min(1.0, target["rcs"] / 100.0)  # Normalize by 100 m²
                
                # Weighted combination
                threat_level = (0.4 * distance_threat + 0.3 * speed_threat + 0.3 * rcs_threat)
                threat_level = max(0.0, min(1.0, threat_level))
                
                # Update with base threat level from injection
                base_threat = target.get("threat_level", 0.5)
                final_threat = 0.7 * threat_level + 0.3 * base_threat
                
                self.threat_levels[target_id] = final_threat
        else:
            # Use sophisticated threat assessor
            try:
                # Convert targets to format expected by threat assessor
                target_data = []
                for target_id, target in self.active_targets.items():
                    target_data.append({
                        "id": target_id,
                        "position": target["position"],
                        "velocity": target["velocity"],
                        "rcs": target["rcs"],
                        "target_class": target.get("target_class", "unknown")
                    })
                
                threat_results = self.threat_assessor.assess_threats(target_data)
                self.threat_levels.update(threat_results)
                
            except Exception as e:
                logger.error(f"Error in threat assessment: {e}")
        
        self.stats["threat_assessments"] += 1
        
        # Notify callbacks
        for callback in self.threat_update_callbacks:
            try:
                callback(self.threat_levels.copy())
            except Exception as e:
                logger.error(f"Error in threat update callback: {e}")


# Utility functions for creating common scenario events

def create_target_injection_event(event_id: str, 
                                trigger_time: float,
                                target_type: str,
                                position: np.ndarray,
                                velocity: np.ndarray,
                                rcs: float,
                                **kwargs) -> ScenarioEvent:
    """
    Create a target injection event.
    
    Args:
        event_id: Unique event identifier
        trigger_time: When to inject target (seconds)
        target_type: Type of target
        position: Initial position [x, y, z]
        velocity: Initial velocity [vx, vy, vz]
        rcs: Radar cross section
        **kwargs: Additional target parameters
        
    Returns:
        Target injection event
    """
    target_params = TargetInjectionEvent(
        target_type=target_type,
        initial_position=np.array(position),
        initial_velocity=np.array(velocity),
        rcs=rcs,
        **kwargs
    )
    
    return ScenarioEvent(
        event_id=event_id,
        event_type=EventType.TARGET_INJECTION,
        trigger_time=trigger_time,
        parameters=target_params.__dict__
    )


def create_target_removal_event(event_id: str,
                               trigger_time: float,
                               target_id: str) -> ScenarioEvent:
    """
    Create a target removal event.
    
    Args:
        event_id: Unique event identifier
        trigger_time: When to remove target (seconds)
        target_id: ID of target to remove
        
    Returns:
        Target removal event
    """
    return ScenarioEvent(
        event_id=event_id,
        event_type=EventType.TARGET_REMOVAL,
        trigger_time=trigger_time,
        parameters={"target_id": target_id}
    )


def load_scenario_from_yaml_file(yaml_path: str) -> DynamicScenarioManager:
    """
    Load a complete scenario from YAML file.
    
    Args:
        yaml_path: Path to YAML scenario file
        
    Returns:
        Configured dynamic scenario manager
    """
    manager = DynamicScenarioManager()
    manager.load_scenario_from_yaml(yaml_path)
    return manager


# Example scenario creation functions

def create_air_defense_scenario() -> DynamicScenarioManager:
    """
    Create a sample air defense scenario with multiple target injections.
    
    Returns:
        Configured scenario manager
    """
    manager = DynamicScenarioManager()
    
    # Fighter aircraft approach
    fighter_event = create_target_injection_event(
        event_id="fighter_1_injection",
        trigger_time=5.0,
        target_type="fighter",
        position=[35000, 0, 8000],
        velocity=[-250, 0, 0],
        rcs=5.0,
        threat_level=0.8,
        target_class="military_aircraft",
        duration=120.0  # Active for 2 minutes
    )
    manager.add_event(fighter_event)
    
    # Bomber with escort
    bomber_event = create_target_injection_event(
        event_id="bomber_injection",
        trigger_time=10.0,
        target_type="bomber",
        position=[45000, 5000, 10000],
        velocity=[-180, -20, 0],
        rcs=50.0,
        threat_level=0.9,
        target_class="military_aircraft",
        duration=180.0
    )
    manager.add_event(bomber_event)
    
    # Escort fighter
    escort_event = create_target_injection_event(
        event_id="escort_injection",
        trigger_time=12.0,
        target_type="fighter",
        position=[44000, 4000, 9500],
        velocity=[-200, -15, 0],
        rcs=3.0,
        threat_level=0.7,
        target_class="military_aircraft",
        duration=180.0
    )
    manager.add_event(escort_event)
    
    # Civilian aircraft (false alarm test)
    civilian_event = create_target_injection_event(
        event_id="civilian_injection",
        trigger_time=25.0,
        target_type="civilian",
        position=[30000, -10000, 11000],
        velocity=[-120, 50, 0],
        rcs=20.0,
        threat_level=0.1,
        target_class="civilian_aircraft",
        duration=300.0
    )
    manager.add_event(civilian_event)
    
    logger.info("Created air defense scenario with 4 target injection events")
    return manager


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Dynamic Scenario Manager...")
    
    # Create scenario manager
    scenario = DynamicScenarioManager()
    
    # Add some test events
    test_event = create_target_injection_event(
        event_id="test_target",
        trigger_time=1.0,
        target_type="test",
        position=[1000, 0, 500],
        velocity=[-50, 0, 0],
        rcs=1.0
    )
    scenario.add_event(test_event)
    
    # Simulate scenario execution
    print("Simulating scenario...")
    current_time = 0.0
    dt = 0.1
    
    for step in range(50):  # 5 seconds simulation
        current_time += dt
        
        # Update scenario
        processed_events = scenario.update(current_time)
        
        if processed_events:
            print(f"Time {current_time:.1f}s: Processed {len(processed_events)} events")
            for event in processed_events:
                print(f"  - {event.event_type.value}: {event.event_id}")
        
        # Get measurements
        measurements = scenario.get_target_measurements(current_time)
        if measurements:
            print(f"Time {current_time:.1f}s: {len(measurements)} measurements")
    
    # Print final statistics
    stats = scenario.get_scenario_statistics()
    print("\nFinal Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nDynamic Scenario Manager test completed successfully!")