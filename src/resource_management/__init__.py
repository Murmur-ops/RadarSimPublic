"""
Resource Management module for RadarSim.

This module provides comprehensive resource management capabilities including:
- Beam scheduling and allocation
- Priority calculation for targets
- Resource decision logging and analysis
- Performance monitoring and optimization
"""

try:
    from .resource_manager import (
        ResourceManager,
        BeamPosition,
        ResourceAllocation,
        TimeBudget
    )
    _resource_manager_available = True
except ImportError:
    _resource_manager_available = False

try:
    from .priority_calculator import (
        PriorityCalculator,
        TargetInfo,
        ThreatLevel
    )
    _priority_calculator_available = True
except ImportError:
    _priority_calculator_available = False

try:
    from .beam_scheduler import (
        BeamScheduler,
        BeamTask,
        BeamMode,
        BeamCapability
    )
    _beam_scheduler_available = True
except ImportError:
    _beam_scheduler_available = False

try:
    from .decision_logger import (
        DecisionLogger,
        DecisionRecord,
        PriorityChangeRecord,
        PerformanceSnapshot,
        DecisionType,
        LogLevel,
        create_decision_logger
    )
    _decision_logger_available = True
except ImportError:
    _decision_logger_available = False

# Build __all__ list dynamically based on available modules
__all__ = []

if _resource_manager_available:
    __all__.extend([
        'ResourceManager',
        'BeamPosition', 
        'ResourceAllocation',
        'TimeBudget'
    ])

if _priority_calculator_available:
    __all__.extend([
        'PriorityCalculator',
        'TargetInfo',
        'ThreatLevel'
    ])

if _beam_scheduler_available:
    __all__.extend([
        'BeamScheduler',
        'BeamTask',
        'BeamMode',
        'BeamCapability'
    ])

if _decision_logger_available:
    __all__.extend([
        'DecisionLogger',
        'DecisionRecord',
        'PriorityChangeRecord', 
        'PerformanceSnapshot',
        'DecisionType',
        'LogLevel',
        'create_decision_logger'
    ])

__version__ = "1.0.0"