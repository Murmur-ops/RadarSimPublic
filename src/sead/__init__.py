"""
Suppression of Enemy Air Defenses (SEAD) Package

Provides SEAD/DEAD simulation capabilities including:
- Wild Weasel aircraft
- AGM-88 HARM anti-radiation missiles
- Radar warning receivers (RWR)
- Emitter location and targeting
"""

from .wild_weasel import (
    WildWeasel,
    HARM,
    EmitterDetection,
    SEADTactic,
    HARMMode
)

__all__ = [
    'WildWeasel',
    'HARM',
    'EmitterDetection',
    'SEADTactic',
    'HARMMode'
]

__version__ = '1.0.0'