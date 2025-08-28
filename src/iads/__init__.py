"""
Integrated Air Defense System (IADS) Package

Provides comprehensive air defense simulation including:
- IADS network C4I
- SAM systems (long, medium, short range)
- Layered defense architecture
- Sensor fusion and track management
"""

from .iads_network import (
    IADSNetwork,
    IADSTarget,
    ThreatLevel,
    EngagementStatus,
    EngagementZone,
    SensorType
)

from .sam_system import (
    SAMSite,
    SAMType,
    Missile,
    MissileGuidance,
    MissileKinematics
)

__all__ = [
    # IADS Network
    'IADSNetwork',
    'IADSTarget',
    'ThreatLevel',
    'EngagementStatus',
    'EngagementZone',
    'SensorType',
    
    # SAM Systems
    'SAMSite',
    'SAMType',
    'Missile',
    'MissileGuidance',
    'MissileKinematics'
]

__version__ = '1.0.0'