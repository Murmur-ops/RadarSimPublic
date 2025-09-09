"""
RadarSim: A Python-based radar simulation library
"""

from .radar import Radar, RadarParameters
from .target import Target, TargetType, TargetMotion
from .environment import Environment, AtmosphericConditions

# Optional imports (may not exist yet)
try:
    from .signal import SignalProcessor
except ImportError:
    SignalProcessor = None

try:
    from .visualization import Visualizer
except ImportError:
    Visualizer = None

__version__ = "1.0.0"

__all__ = [
    "Radar",
    "RadarParameters", 
    "Target",
    "TargetType",
    "TargetMotion",
    "Environment",
    "AtmosphericConditions",
]

# Add optional modules if they exist
if SignalProcessor:
    __all__.append("SignalProcessor")
if Visualizer:
    __all__.append("Visualizer")