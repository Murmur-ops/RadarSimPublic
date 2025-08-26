"""
Input Validation Module for Radar Simulation

This module provides comprehensive validation for all radar parameters,
ensuring physical constraints and system limitations are respected.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings

from .constants import (
    RadarLimits, SPEED_OF_LIGHT, RadarBand,
    get_max_unambiguous_range, get_max_unambiguous_velocity
)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class ValidationError(Exception):
    """Base exception for validation errors"""
    pass

class ParameterOutOfRangeError(ValidationError):
    """Raised when a parameter is outside acceptable range"""
    def __init__(self, param_name: str, value: float, min_val: float, max_val: float):
        self.param_name = param_name
        self.value = value
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(
            f"{param_name} = {value} is outside valid range [{min_val}, {max_val}]"
        )

class PhysicalConstraintError(ValidationError):
    """Raised when physical constraints are violated"""
    pass

class AmbiguityError(ValidationError):
    """Raised when ambiguity constraints are violated"""
    pass


# ============================================================================
# VALIDATION RESULTS
# ============================================================================

@dataclass
class ValidationResult:
    """Container for validation results"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def add_error(self, message: str):
        """Add an error message"""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a warning message"""
        self.warnings.append(message)
    
    def raise_if_invalid(self):
        """Raise exception if validation failed"""
        if not self.is_valid:
            raise ValidationError("\n".join(self.errors))


# ============================================================================
# PARAMETER VALIDATORS
# ============================================================================

class RadarParameterValidator:
    """Validates radar system parameters"""
    
    @staticmethod
    def validate_frequency(frequency: float, strict: bool = True) -> ValidationResult:
        """
        Validate radar frequency
        
        Args:
            frequency: Frequency in Hz
            strict: If True, raise exception on failure
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if frequency <= 0:
            result.add_error(f"Frequency must be positive, got {frequency} Hz")
        elif frequency < RadarLimits.MIN_FREQUENCY:
            result.add_error(
                f"Frequency {frequency/1e6:.2f} MHz is below minimum {RadarLimits.MIN_FREQUENCY/1e6:.2f} MHz"
            )
        elif frequency > RadarLimits.MAX_FREQUENCY:
            result.add_error(
                f"Frequency {frequency/1e9:.2f} GHz exceeds maximum {RadarLimits.MAX_FREQUENCY/1e9:.2f} GHz"
            )
        
        # Check band and add info
        band = RadarBand.get_band(frequency)
        if band:
            result.warnings.append(f"Frequency is in {band.description}")
        
        if strict and not result.is_valid:
            result.raise_if_invalid()
        
        return result
    
    @staticmethod
    def validate_power(power: float, strict: bool = True) -> ValidationResult:
        """Validate transmit power"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if power < 0:
            result.add_error(f"Power cannot be negative, got {power} W")
        elif power > RadarLimits.MAX_POWER:
            result.add_error(
                f"Power {power/1e6:.2f} MW exceeds maximum {RadarLimits.MAX_POWER/1e6:.2f} MW"
            )
        elif power > 1e6:
            result.add_warning(f"Power {power/1e6:.2f} MW is very high for most systems")
        
        if strict and not result.is_valid:
            result.raise_if_invalid()
        
        return result
    
    @staticmethod
    def validate_prf(prf: float, strict: bool = True) -> ValidationResult:
        """Validate pulse repetition frequency"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if prf <= 0:
            result.add_error(f"PRF must be positive, got {prf} Hz")
        elif prf < RadarLimits.MIN_PRF:
            result.add_error(
                f"PRF {prf} Hz is below minimum {RadarLimits.MIN_PRF} Hz"
            )
        elif prf > RadarLimits.MAX_PRF:
            result.add_error(
                f"PRF {prf/1e3:.1f} kHz exceeds maximum {RadarLimits.MAX_PRF/1e3:.1f} kHz"
            )
        
        # Calculate unambiguous range
        max_range = get_max_unambiguous_range(prf)
        if max_range < 1000:
            result.add_warning(f"Maximum unambiguous range is only {max_range:.1f} m")
        
        if strict and not result.is_valid:
            result.raise_if_invalid()
        
        return result
    
    @staticmethod
    def validate_antenna_gain(gain: float, strict: bool = True) -> ValidationResult:
        """Validate antenna gain"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if gain < RadarLimits.MIN_ANTENNA_GAIN:
            result.add_error(f"Antenna gain {gain} dB cannot be negative")
        elif gain > RadarLimits.MAX_ANTENNA_GAIN:
            result.add_error(
                f"Antenna gain {gain} dB exceeds maximum {RadarLimits.MAX_ANTENNA_GAIN} dB"
            )
        elif gain > 50:
            result.add_warning(f"Antenna gain {gain} dB is very high (large array required)")
        
        if strict and not result.is_valid:
            result.raise_if_invalid()
        
        return result
    
    @staticmethod
    def validate_pulse_width(pulse_width: float, prf: float, strict: bool = True) -> ValidationResult:
        """Validate pulse width"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if pulse_width <= 0:
            result.add_error(f"Pulse width must be positive, got {pulse_width} s")
        elif pulse_width < RadarLimits.MIN_PULSE_WIDTH:
            result.add_error(
                f"Pulse width {pulse_width*1e9:.1f} ns is below minimum {RadarLimits.MIN_PULSE_WIDTH*1e9:.1f} ns"
            )
        elif pulse_width > RadarLimits.MAX_PULSE_WIDTH:
            result.add_error(
                f"Pulse width {pulse_width*1e3:.2f} ms exceeds maximum {RadarLimits.MAX_PULSE_WIDTH*1e3:.2f} ms"
            )
        
        # Check duty cycle
        if prf > 0:
            duty_cycle = pulse_width * prf
            if duty_cycle > 0.5:
                result.add_error(f"Duty cycle {duty_cycle:.2%} exceeds 50%")
            elif duty_cycle > 0.3:
                result.add_warning(f"Duty cycle {duty_cycle:.2%} is high")
        
        if strict and not result.is_valid:
            result.raise_if_invalid()
        
        return result
    
    @staticmethod
    def validate_complete_radar(params: Dict[str, Any]) -> ValidationResult:
        """Validate complete set of radar parameters"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Required parameters
        required = ['frequency', 'power', 'prf', 'antenna_gain', 'pulse_width']
        for param in required:
            if param not in params:
                result.add_error(f"Missing required parameter: {param}")
        
        if not result.is_valid:
            return result
        
        # Validate individual parameters
        freq_result = RadarParameterValidator.validate_frequency(params['frequency'], strict=False)
        power_result = RadarParameterValidator.validate_power(params['power'], strict=False)
        prf_result = RadarParameterValidator.validate_prf(params['prf'], strict=False)
        gain_result = RadarParameterValidator.validate_antenna_gain(params['antenna_gain'], strict=False)
        pw_result = RadarParameterValidator.validate_pulse_width(
            params['pulse_width'], params['prf'], strict=False
        )
        
        # Combine results
        for r in [freq_result, power_result, prf_result, gain_result, pw_result]:
            result.errors.extend(r.errors)
            result.warnings.extend(r.warnings)
            if not r.is_valid:
                result.is_valid = False
        
        # Cross-parameter checks
        if result.is_valid:
            # Check range-Doppler ambiguity
            max_range = get_max_unambiguous_range(params['prf'])
            max_velocity = get_max_unambiguous_velocity(params['frequency'], params['prf'])
            
            if 'max_range' in params and params['max_range'] > max_range:
                result.add_warning(
                    f"Requested max range {params['max_range']/1e3:.1f} km exceeds "
                    f"unambiguous range {max_range/1e3:.1f} km"
                )
            
            result.warnings.append(
                f"Max unambiguous velocity: ±{max_velocity:.1f} m/s"
            )
        
        return result


class TargetValidator:
    """Validates target parameters"""
    
    @staticmethod
    def validate_rcs(rcs: float, strict: bool = True) -> ValidationResult:
        """Validate radar cross section"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if rcs <= 0:
            result.add_error(f"RCS must be positive, got {rcs} m²")
        elif rcs < RadarLimits.MIN_RCS:
            result.add_warning(
                f"RCS {rcs} m² is extremely small (< {RadarLimits.MIN_RCS} m²)"
            )
        elif rcs > RadarLimits.MAX_RCS:
            result.add_warning(
                f"RCS {rcs} m² is extremely large (> {RadarLimits.MAX_RCS} m²)"
            )
        
        # Classify RCS
        if rcs < 0.01:
            result.warnings.append("RCS consistent with small drone/bird")
        elif rcs < 0.1:
            result.warnings.append("RCS consistent with missile")
        elif rcs < 1:
            result.warnings.append("RCS consistent with small aircraft")
        elif rcs < 10:
            result.warnings.append("RCS consistent with fighter aircraft")
        elif rcs < 100:
            result.warnings.append("RCS consistent with large aircraft")
        else:
            result.warnings.append("RCS consistent with ship/building")
        
        if strict and not result.is_valid:
            result.raise_if_invalid()
        
        return result
    
    @staticmethod
    def validate_velocity(velocity: float, frequency: float, prf: float, 
                         strict: bool = True) -> ValidationResult:
        """Validate target velocity against ambiguity limits"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if abs(velocity) > RadarLimits.MAX_VELOCITY:
            result.add_error(
                f"Velocity {velocity} m/s exceeds maximum {RadarLimits.MAX_VELOCITY} m/s"
            )
        
        # Check Doppler ambiguity
        max_unambiguous = get_max_unambiguous_velocity(frequency, prf)
        if abs(velocity) > max_unambiguous:
            result.add_warning(
                f"Velocity {velocity} m/s will cause Doppler ambiguity "
                f"(max unambiguous: ±{max_unambiguous:.1f} m/s)"
            )
        
        # Classify velocity
        speed = abs(velocity)
        if speed > 340:
            result.warnings.append(f"Target is supersonic (Mach {speed/340:.1f})")
        elif speed > 1020:
            result.warnings.append(f"Target is hypersonic (Mach {speed/340:.1f})")
        
        if strict and not result.is_valid:
            result.raise_if_invalid()
        
        return result
    
    @staticmethod
    def validate_range(range_m: float, prf: float, strict: bool = True) -> ValidationResult:
        """Validate target range"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if range_m <= 0:
            result.add_error(f"Range must be positive, got {range_m} m")
        elif range_m < RadarLimits.MIN_RANGE:
            result.add_error(f"Range {range_m} m is below minimum {RadarLimits.MIN_RANGE} m")
        elif range_m > RadarLimits.MAX_RANGE:
            result.add_warning(
                f"Range {range_m/1e3:.1f} km exceeds typical maximum {RadarLimits.MAX_RANGE/1e3:.1f} km"
            )
        
        # Check range ambiguity
        max_unambiguous = get_max_unambiguous_range(prf)
        if range_m > max_unambiguous:
            result.add_warning(
                f"Range {range_m/1e3:.1f} km will be ambiguous "
                f"(max unambiguous: {max_unambiguous/1e3:.1f} km)"
            )
        
        if strict and not result.is_valid:
            result.raise_if_invalid()
        
        return result


class ScenarioValidator:
    """Validates complete scenario configuration"""
    
    @staticmethod
    def validate_scenario(scenario_config: Dict[str, Any]) -> ValidationResult:
        """Validate complete scenario configuration"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Validate radar parameters
        if 'radar' not in scenario_config:
            result.add_error("Missing radar configuration")
            return result
        
        radar_result = RadarParameterValidator.validate_complete_radar(
            scenario_config['radar']
        )
        result.errors.extend(radar_result.errors)
        result.warnings.extend(radar_result.warnings)
        if not radar_result.is_valid:
            result.is_valid = False
        
        # Validate targets
        if 'targets' in scenario_config:
            for i, target in enumerate(scenario_config['targets']):
                # Validate RCS
                if 'rcs' in target:
                    rcs_result = TargetValidator.validate_rcs(target['rcs'], strict=False)
                    if not rcs_result.is_valid:
                        result.add_error(f"Target {i}: " + "; ".join(rcs_result.errors))
                
                # Validate velocity
                if 'velocity' in target and 'radar' in scenario_config:
                    vel_result = TargetValidator.validate_velocity(
                        target['velocity'],
                        scenario_config['radar'].get('frequency', 10e9),
                        scenario_config['radar'].get('prf', 1000),
                        strict=False
                    )
                    if not vel_result.is_valid:
                        result.add_error(f"Target {i}: " + "; ".join(vel_result.errors))
                    if vel_result.warnings:
                        result.add_warning(f"Target {i}: " + "; ".join(vel_result.warnings))
                
                # Validate range
                if 'range' in target and 'radar' in scenario_config:
                    range_result = TargetValidator.validate_range(
                        target['range'],
                        scenario_config['radar'].get('prf', 1000),
                        strict=False
                    )
                    if not range_result.is_valid:
                        result.add_error(f"Target {i}: " + "; ".join(range_result.errors))
                    if range_result.warnings:
                        result.add_warning(f"Target {i}: " + "; ".join(range_result.warnings))
        
        # Validate timing
        if 'duration' in scenario_config:
            if scenario_config['duration'] <= 0:
                result.add_error("Duration must be positive")
            elif scenario_config['duration'] > 3600:
                result.add_warning("Duration > 1 hour may require large memory")
        
        if 'time_step' in scenario_config:
            if scenario_config['time_step'] <= 0:
                result.add_error("Time step must be positive")
            elif scenario_config['time_step'] > 1.0:
                result.add_warning("Time step > 1s may cause tracking issues")
        
        return result


# ============================================================================
# UNIT CONVERSION UTILITIES
# ============================================================================

class UnitConverter:
    """Utilities for unit conversion"""
    
    @staticmethod
    def freq_to_band(frequency_hz: float) -> str:
        """Convert frequency to band name"""
        band = RadarBand.get_band(frequency_hz)
        return band.description if band else "Unknown"
    
    @staticmethod
    def mhz_to_hz(mhz: float) -> float:
        """Convert MHz to Hz"""
        return mhz * 1e6
    
    @staticmethod
    def ghz_to_hz(ghz: float) -> float:
        """Convert GHz to Hz"""
        return ghz * 1e9
    
    @staticmethod
    def km_to_m(km: float) -> float:
        """Convert km to meters"""
        return km * 1000
    
    @staticmethod
    def nmi_to_m(nmi: float) -> float:
        """Convert nautical miles to meters"""
        return nmi * 1852
    
    @staticmethod
    def mach_to_ms(mach: float, temperature_c: float = 15) -> float:
        """Convert Mach number to m/s"""
        # Speed of sound varies with temperature
        temp_k = temperature_c + 273.15
        sound_speed = 331.3 * np.sqrt(temp_k / 273.15)
        return mach * sound_speed
    
    @staticmethod
    def dbsm_to_m2(dbsm: float) -> float:
        """Convert dBsm to m²"""
        return 10 ** (dbsm / 10)
    
    @staticmethod
    def m2_to_dbsm(m2: float) -> float:
        """Convert m² to dBsm"""
        return 10 * np.log10(max(m2, 1e-10))


# ============================================================================
# VALIDATION MODE CONFIGURATION
# ============================================================================

class ValidationConfig:
    """Configuration for validation behavior"""
    
    def __init__(self, 
                 strict: bool = True,
                 fail_fast: bool = True,
                 warn_on_ambiguity: bool = True,
                 check_physics: bool = True):
        """
        Initialize validation configuration
        
        Args:
            strict: Raise exceptions on validation failure
            fail_fast: Stop on first error
            warn_on_ambiguity: Generate warnings for ambiguous conditions
            check_physics: Validate physical constraints
        """
        self.strict = strict
        self.fail_fast = fail_fast
        self.warn_on_ambiguity = warn_on_ambiguity
        self.check_physics = check_physics
    
    def validate_with_config(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate parameters according to configuration"""
        if self.check_physics:
            return ScenarioValidator.validate_scenario(params)
        else:
            # Minimal validation
            result = ValidationResult(is_valid=True, errors=[], warnings=[])
            if 'radar' in params:
                if params['radar'].get('power', 0) < 0:
                    result.add_error("Negative power is not allowed")
            return result


# Create default validator instance
default_validator = ValidationConfig(strict=False, warn_on_ambiguity=True)