"""
Networked Radar System Package

Provides distributed and networked radar capabilities including:
- Bistatic and multistatic radar configurations
- Distributed tracking and data fusion
- Communication and synchronization
- Multiple fusion architectures
"""

from .networked_radar import (
    NetworkedRadar,
    NetworkNode,
    NetworkArchitecture,
    RadarNodeType,
    CommunicationLink,
    NetworkMessage
)

from .communication import (
    MessageRouter,
    TimeSync,
    TimeSyncMessage,
    DataCompression,
    NetworkProtocol
)

from .distributed_tracking import (
    NetworkTrack,
    TrackAssociator,
    DistributedTrackFusion,
    ConsensusTracker
)

from .fusion_center import (
    DataFusionCenter,
    FusionArchitecture,
    FusionNode
)

__all__ = [
    # Core networked radar
    'NetworkedRadar',
    'NetworkNode',
    'NetworkArchitecture',
    'RadarNodeType',
    'CommunicationLink',
    'NetworkMessage',
    
    # Communication
    'MessageRouter',
    'TimeSync',
    'TimeSyncMessage',
    'DataCompression',
    'NetworkProtocol',
    
    # Distributed tracking
    'NetworkTrack',
    'TrackAssociator',
    'DistributedTrackFusion',
    'ConsensusTracker',
    
    # Fusion center
    'DataFusionCenter',
    'FusionArchitecture',
    'FusionNode'
]

__version__ = '1.0.0'