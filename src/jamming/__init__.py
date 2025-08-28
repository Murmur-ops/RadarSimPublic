"""
Jamming and Electronic Warfare Module

This package provides comprehensive jamming and electronic countermeasures capabilities
including DRFM, false target generation, gate pull-off, and ECCM detection.
"""

# DRFM Jammer
from .drfm_jammer import (
    DRFMJammer,
    PulseDescriptorWord,
    FalseTarget,
    DRFMTechnique
)

# False Target Generator
from .false_target_generator import (
    FalseTargetGenerator,
    FalseTargetParams,
    SwarmParams,
    ScreenParams,
    DecoyChainParams,
    RCSProfile,
    PatternType,
    RCSModel
)

# Gate Pull-Off
from .gate_pull_off import (
    GatePullOff,
    RangeParams,
    VelocityParams,
    TransitionParams,
    PullOffProfile,
    PullOffDirection,
    GateFeedback,
    create_adaptive_range_params,
    create_adaptive_velocity_params
)

# ECCM Detection
from .eccm_detector import (
    ECCMDetector,
    ECCMDetectionResult,
    JammingType,
    PulseConsistencyMetrics,
    create_adaptive_eccm_suite
)

__all__ = [
    # DRFM
    'DRFMJammer',
    'PulseDescriptorWord',
    'FalseTarget',
    'DRFMTechnique',
    
    # False Targets
    'FalseTargetGenerator',
    'FalseTargetParams',
    'SwarmParams',
    'ScreenParams',
    'DecoyChainParams',
    'RCSProfile',
    'PatternType',
    'RCSModel',
    
    # Gate Pull-Off
    'GatePullOff',
    'RangeParams',
    'VelocityParams',
    'TransitionParams',
    'PullOffProfile',
    'PullOffDirection',
    'GateFeedback',
    'create_adaptive_range_params',
    'create_adaptive_velocity_params',
    
    # ECCM
    'ECCMDetector',
    'ECCMDetectionResult',
    'JammingType',
    'PulseConsistencyMetrics',
    'create_adaptive_eccm_suite'
]

# Version info
__version__ = '1.0.0'
__author__ = 'RadarSim Team'