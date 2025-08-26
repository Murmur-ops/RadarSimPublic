"""
RadarSim: A Python-based radar simulation library
"""

from .radar import Radar
from .target import Target
from .signal import SignalProcessor
from .environment import Environment
from .visualization import Visualizer

__version__ = "0.1.0"
__all__ = ["Radar", "Target", "SignalProcessor", "Environment", "Visualizer"]