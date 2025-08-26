"""
Physical and System Constants for Radar Simulation

This module contains all physical constants, system parameters, and 
standard values used throughout the radar simulation.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Tuple


# ============================================================================
# FUNDAMENTAL PHYSICAL CONSTANTS
# ============================================================================

# Speed of light in vacuum (m/s)
SPEED_OF_LIGHT = 299792458.0  # Exact value by definition

# Boltzmann constant (J/K)
BOLTZMANN_CONSTANT = 1.380649e-23  # 2019 SI definition

# Standard temperature (Kelvin)
STANDARD_TEMPERATURE = 290.0  # ~17°C, standard for noise calculations

# Earth radius (m)
EARTH_RADIUS = 6371000.0

# Standard atmospheric pressure at sea level (Pa)
STANDARD_PRESSURE = 101325.0

# Standard gravity (m/s²)
STANDARD_GRAVITY = 9.80665


# ============================================================================
# RADAR FREQUENCY BANDS
# ============================================================================

class RadarBand(Enum):
    """IEEE Standard Radar Frequency Bands"""
    HF = (3e6, 30e6, "High Frequency")
    VHF = (30e6, 300e6, "Very High Frequency")
    UHF = (300e6, 1e9, "Ultra High Frequency")
    L = (1e9, 2e9, "L-band")
    S = (2e9, 4e9, "S-band")
    C = (4e9, 8e9, "C-band")
    X = (8e9, 12e9, "X-band")
    Ku = (12e9, 18e9, "Ku-band")
    K = (18e9, 27e9, "K-band")
    Ka = (27e9, 40e9, "Ka-band")
    V = (40e9, 75e9, "V-band")
    W = (75e9, 110e9, "W-band")
    mm = (110e9, 300e9, "Millimeter wave")
    
    def __init__(self, min_freq: float, max_freq: float, description: str):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.description = description
    
    def contains(self, frequency: float) -> bool:
        """Check if frequency is within this band"""
        return self.min_freq <= frequency <= self.max_freq
    
    @classmethod
    def get_band(cls, frequency: float):
        """Get the band for a given frequency"""
        for band in cls:
            if band.contains(frequency):
                return band
        return None


# ============================================================================
# RADAR SYSTEM LIMITS
# ============================================================================

@dataclass
class RadarLimits:
    """Physical and practical limits for radar parameters"""
    
    # Frequency limits
    MIN_FREQUENCY = 1e6  # 1 MHz
    MAX_FREQUENCY = 100e9  # 100 GHz
    
    # Power limits
    MIN_POWER = 0.0  # Watts
    MAX_POWER = 10e6  # 10 MW (practical limit for most systems)
    
    # PRF limits
    MIN_PRF = 100.0  # 100 Hz
    MAX_PRF = 100e3  # 100 kHz
    
    # Antenna gain limits
    MIN_ANTENNA_GAIN = 0.0  # dB (isotropic)
    MAX_ANTENNA_GAIN = 60.0  # dB (very large phased array)
    
    # Pulse width limits
    MIN_PULSE_WIDTH = 10e-9  # 10 ns
    MAX_PULSE_WIDTH = 1e-3  # 1 ms
    
    # RCS limits
    MIN_RCS = 0.0001  # m² (insect)
    MAX_RCS = 100000.0  # m² (large ship)
    
    # Range limits
    MIN_RANGE = 1.0  # 1 meter
    MAX_RANGE = 1e6  # 1000 km (space surveillance)
    
    # Velocity limits
    MIN_VELOCITY = -10000.0  # m/s (hypersonic)
    MAX_VELOCITY = 10000.0  # m/s
    
    # Noise figure limits
    MIN_NOISE_FIGURE = 0.5  # dB (theoretical minimum)
    MAX_NOISE_FIGURE = 20.0  # dB (poor receiver)
    
    # Loss limits
    MIN_LOSSES = 0.0  # dB (lossless)
    MAX_LOSSES = 20.0  # dB (high losses)


# ============================================================================
# ATMOSPHERIC PARAMETERS
# ============================================================================

@dataclass
class AtmosphericConstants:
    """Standard atmospheric model parameters"""
    
    # ISA (International Standard Atmosphere) at sea level
    TEMPERATURE_SEA_LEVEL = 288.15  # K (15°C)
    PRESSURE_SEA_LEVEL = 101325.0  # Pa
    DENSITY_SEA_LEVEL = 1.225  # kg/m³
    
    # Temperature lapse rate
    TEMPERATURE_LAPSE_RATE = -0.0065  # K/m (troposphere)
    
    # Tropopause
    TROPOPAUSE_ALTITUDE = 11000.0  # m
    
    # Water vapor density at standard conditions
    WATER_VAPOR_DENSITY = 7.5  # g/m³ (typical)
    
    # Oxygen absorption peaks (GHz)
    OXYGEN_ABSORPTION_PEAKS = [60.0, 118.75]
    
    # Water vapor absorption peak (GHz)
    WATER_VAPOR_ABSORPTION_PEAK = 22.235


# ============================================================================
# SIGNAL PROCESSING CONSTANTS
# ============================================================================

@dataclass
class SignalConstants:
    """Signal processing related constants"""
    
    # CFAR parameters
    CFAR_GUARD_CELLS_DEFAULT = 4
    CFAR_TRAINING_CELLS_DEFAULT = 16
    CFAR_PFA_DEFAULT = 1e-6
    
    # FFT sizes (powers of 2)
    FFT_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]
    
    # Window functions loss factors (dB)
    WINDOW_LOSS = {
        'rectangular': 0.0,
        'hamming': 1.34,
        'hanning': 1.76,
        'blackman': 2.37,
        'kaiser': 1.5  # Depends on beta
    }
    
    # Matched filter loss (practical)
    MATCHED_FILTER_LOSS = 1.0  # dB
    
    # Integration improvement factors
    COHERENT_INTEGRATION_EFFICIENCY = 1.0
    NON_COHERENT_INTEGRATION_EFFICIENCY = 0.8


# ============================================================================
# TRACKING CONSTANTS
# ============================================================================

@dataclass
class TrackingConstants:
    """Tracking algorithm constants"""
    
    # Track confirmation/deletion thresholds
    TRACK_CONFIRMATION_DEFAULT = 3  # Detections to confirm
    TRACK_DELETION_DEFAULT = 5  # Misses to delete
    
    # Gate thresholds (chi-squared values)
    GATE_THRESHOLD_99 = 11.34  # 99% probability, 3D
    GATE_THRESHOLD_95 = 7.81  # 95% probability, 3D
    GATE_THRESHOLD_90 = 6.25  # 90% probability, 3D
    
    # IMM filter parameters
    IMM_MIN_MODEL_PROBABILITY = 0.01
    IMM_MAX_MODELS = 5
    
    # Association thresholds
    MIN_ASSOCIATION_PROBABILITY = 0.05
    
    # Maximum tracks
    MAX_TRACKS_DEFAULT = 100


# ============================================================================
# MACHINE LEARNING CONSTANTS
# ============================================================================

@dataclass
class MLConstants:
    """Machine learning related constants"""
    
    # Training parameters
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_BATCH_SIZE = 32
    MAX_GRADIENT_NORM = 5.0
    DROPOUT_RATE = 0.2
    
    # Model architecture
    TRANSFORMER_HEADS = 4
    TRANSFORMER_DIM = 64
    CNN_FILTERS = [32, 64, 128]
    
    # Feature dimensions
    PDW_BASE_FEATURES = 13
    ADVANCED_FEATURES = 19
    
    # Classification thresholds
    CLASSIFICATION_CONFIDENCE_MIN = 0.1
    
    # Inference constraints
    MAX_INFERENCE_TIME_MS = 10.0
    
    # Random seed for reproducibility
    DEFAULT_RANDOM_SEED = None  # None for random, int for reproducible


# ============================================================================
# ELECTRONIC WARFARE CONSTANTS
# ============================================================================

@dataclass
class EWConstants:
    """Electronic warfare parameters"""
    
    # Jamming types
    JAMMING_TYPES = ['noise', 'deception', 'drfm', 'false_target']
    
    # DRFM parameters
    DRFM_MIN_DELAY = 10e-9  # 10 ns minimum delay
    DRFM_MAX_DELAY = 1e-6  # 1 μs maximum delay
    
    # Jamming effectiveness
    MIN_JSR_EFFECTIVE = 0  # dB (minimum to have effect)
    BURNTHROUGH_JSR = -10  # dB (radar burns through jamming)
    
    # False target parameters
    MAX_FALSE_TARGETS = 10
    FALSE_TARGET_RCS_MULTIPLIER = (0.5, 2.0)  # Range of RCS variation


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_frequency(frequency: float) -> bool:
    """Validate if frequency is within acceptable range"""
    return RadarLimits.MIN_FREQUENCY <= frequency <= RadarLimits.MAX_FREQUENCY

def validate_power(power: float) -> bool:
    """Validate if power is within acceptable range"""
    return RadarLimits.MIN_POWER <= power <= RadarLimits.MAX_POWER

def validate_prf(prf: float) -> bool:
    """Validate if PRF is within acceptable range"""
    return RadarLimits.MIN_PRF <= prf <= RadarLimits.MAX_PRF

def get_wavelength(frequency: float) -> float:
    """Calculate wavelength from frequency"""
    return SPEED_OF_LIGHT / frequency

def get_max_unambiguous_range(prf: float) -> float:
    """Calculate maximum unambiguous range for given PRF"""
    return SPEED_OF_LIGHT / (2 * prf)

def get_max_unambiguous_velocity(frequency: float, prf: float) -> float:
    """Calculate maximum unambiguous velocity"""
    wavelength = get_wavelength(frequency)
    return wavelength * prf / 4

def db_to_linear(db_value: float) -> float:
    """Convert dB to linear scale"""
    return 10 ** (db_value / 10)

def linear_to_db(linear_value: float) -> float:
    """Convert linear to dB scale"""
    return 10 * np.log10(np.maximum(linear_value, 1e-10))

def dbm_to_watts(dbm: float) -> float:
    """Convert dBm to Watts"""
    return 10 ** ((dbm - 30) / 10)

def watts_to_dbm(watts: float) -> float:
    """Convert Watts to dBm"""
    return 10 * np.log10(watts) + 30


# ============================================================================
# DEPRECATED CONSTANTS (for backward compatibility)
# ============================================================================

# Old style constants - use new ones above
c = SPEED_OF_LIGHT  # Deprecated: use SPEED_OF_LIGHT
k_boltzmann = BOLTZMANN_CONSTANT  # Deprecated: use BOLTZMANN_CONSTANT