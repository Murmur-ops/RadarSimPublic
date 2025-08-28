#!/usr/bin/env python3
"""
Observable-Based Target Classifier

This classifier maintains the non-cheating integrity of RadarSim by only
using information that can actually be determined from radar returns and
legitimate data sources (transponders, ADS-B, etc).

NO CHEATING: We cannot determine from radar returns alone:
- Whether a helicopter is medical vs military
- Passenger counts on aircraft
- Friend vs foe without IFF
- Intent or mission type

WHAT WE CAN DETERMINE:
- Physical characteristics (size, speed, altitude)
- Behavioral patterns (maneuvering, approach paths)
- Transponder/IFF codes (if transmitted)
- Micro-Doppler signatures (rotor/jet modulation)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class ObservableClass(Enum):
    """Classifications based purely on observable physics"""
    # Size-based (from RCS)
    VERY_SMALL = "very_small"      # RCS < 0.1 m² (bird, drone, small missile)
    SMALL = "small"                 # RCS 0.1-10 m² (missile, small aircraft)
    MEDIUM = "medium"               # RCS 10-50 m² (fighter, helicopter)
    LARGE = "large"                 # RCS 50-200 m² (transport, airliner)
    VERY_LARGE = "very_large"       # RCS > 200 m² (heavy aircraft, ship)
    
    # Speed-based (from Doppler)
    STATIONARY = "stationary"       # < 5 m/s
    SLOW = "slow"                   # 5-50 m/s
    MODERATE = "moderate"           # 50-150 m/s
    FAST = "fast"                   # 150-300 m/s
    VERY_FAST = "very_fast"         # > 300 m/s
    
    # Special signatures
    ROTORCRAFT = "rotorcraft"       # Has blade modulation
    BIRD_LIKE = "bird_like"         # Erratic, low RCS, biological pattern
    
    # Unknown
    UNKNOWN = "unknown"             # Cannot determine


@dataclass
class ObservableFeatures:
    """Features that can actually be measured by radar"""
    # Direct measurements
    rcs: float                      # Radar Cross Section (m²)
    range: float                    # Distance (m)
    azimuth: float                  # Horizontal angle (radians)
    elevation: float                # Vertical angle (radians)
    doppler_velocity: float         # Radial velocity (m/s)
    
    # Derived measurements
    altitude: Optional[float] = None           # From range and elevation
    ground_speed: Optional[float] = None       # From multiple observations
    heading: Optional[float] = None            # From track history
    
    # Signal characteristics
    snr: Optional[float] = None                # Signal-to-noise ratio (dB)
    doppler_spread: Optional[float] = None     # Spectral width (Hz)
    has_blade_flash: bool = False              # Rotor modulation detected
    has_jet_modulation: bool = False           # Jet engine modulation
    
    # Transponder data (if available)
    transponder_code: Optional[str] = None     # Squawk code
    mode_s_id: Optional[str] = None            # Aircraft ID
    adsb_data: Optional[Dict] = None           # ADS-B information
    iff_response: Optional[str] = None         # IFF mode/code
    
    # Track characteristics (from history)
    is_maneuvering: bool = False              # Changing course/speed
    is_accelerating: bool = False             # Changing velocity
    track_quality: float = 0.0                # Track confidence (0-1)
    time_since_first_detection: float = 0.0   # Track age (seconds)


@dataclass
class BehaviorPattern:
    """Observable behavior patterns"""
    pattern_type: str               # Type of behavior
    confidence: float              # Confidence in pattern (0-1)
    evidence: List[str]            # Supporting observations


class ObservableClassifier:
    """
    Classifier that only uses observable physics and legitimate data sources
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.5,
                 require_multiple_observations: bool = True):
        """
        Initialize the observable classifier
        
        Args:
            confidence_threshold: Minimum confidence to return classification
            require_multiple_observations: Need multiple looks for confidence
        """
        self.confidence_threshold = confidence_threshold
        self.require_multiple_observations = require_multiple_observations
        self.track_history: Dict[str, List[ObservableFeatures]] = {}
        
    def classify(self, 
                features: ObservableFeatures,
                track_id: Optional[str] = None) -> Tuple[Dict[str, Any], float]:
        """
        Classify target based solely on observable features
        
        Args:
            features: Observable radar measurements
            track_id: Track identifier for history
            
        Returns:
            (classification_dict, confidence) tuple
        """
        classification = {
            'size_class': self._classify_size(features.rcs),
            'speed_class': self._classify_speed(features.doppler_velocity),
            'altitude_band': self._classify_altitude(features.altitude),
            'special_signature': self._detect_special_signature(features),
            'transponder_status': self._check_transponder(features),
            'behavior': self._analyze_behavior(features, track_id)
        }
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(classification, features, track_id)
        
        # Add operational priority hint (based on behavior, not identity)
        classification['priority_hint'] = self._suggest_priority(classification, features)
        
        return classification, confidence
    
    def _classify_size(self, rcs: float) -> str:
        """Classify size based on RCS"""
        if rcs < 0.1:
            return "very_small"
        elif rcs < 10:
            return "small"
        elif rcs < 50:
            return "medium"
        elif rcs < 200:
            return "large"
        else:
            return "very_large"
    
    def _classify_speed(self, velocity: float) -> str:
        """Classify speed based on Doppler"""
        speed = abs(velocity)
        if speed < 5:
            return "stationary"
        elif speed < 50:
            return "slow"
        elif speed < 150:
            return "moderate"
        elif speed < 300:
            return "fast"
        else:
            return "very_fast"
    
    def _classify_altitude(self, altitude: Optional[float]) -> str:
        """Classify altitude band"""
        if altitude is None:
            return "unknown"
        elif altitude < 500:
            return "very_low"
        elif altitude < 3000:
            return "low"
        elif altitude < 10000:
            return "medium"
        elif altitude < 15000:
            return "high"
        else:
            return "very_high"
    
    def _detect_special_signature(self, features: ObservableFeatures) -> Optional[str]:
        """Detect special signatures from signal characteristics"""
        signatures = []
        
        if features.has_blade_flash:
            signatures.append("rotorcraft")
            
        if features.has_jet_modulation:
            signatures.append("jet_powered")
            
        if features.doppler_spread and features.doppler_spread > 50:
            if features.rcs < 1.0:
                signatures.append("possible_bird_flock")
                
        return signatures[0] if signatures else None
    
    def _check_transponder(self, features: ObservableFeatures) -> Dict[str, Any]:
        """Check transponder and IFF status"""
        status = {
            'has_transponder': False,
            'emergency': False,
            'identity': None
        }
        
        if features.transponder_code:
            status['has_transponder'] = True
            
            # Check for emergency codes
            if features.transponder_code in ['7700', '7600', '7500']:
                status['emergency'] = True
                status['emergency_type'] = {
                    '7700': 'general_emergency',
                    '7600': 'radio_failure',
                    '7500': 'hijack'
                }.get(features.transponder_code)
                
        if features.mode_s_id:
            status['identity'] = features.mode_s_id
            
        if features.iff_response:
            status['iff_mode'] = features.iff_response
            
        return status
    
    def _analyze_behavior(self, 
                         features: ObservableFeatures,
                         track_id: Optional[str]) -> Dict[str, Any]:
        """Analyze behavior patterns from track history"""
        behavior = {
            'maneuvering': features.is_maneuvering,
            'accelerating': features.is_accelerating,
            'pattern': None
        }
        
        if track_id and track_id in self.track_history:
            history = self.track_history[track_id]
            
            # Analyze trajectory pattern
            if len(history) >= 5:
                # Check for approach pattern
                if self._is_approaching_pattern(history):
                    behavior['pattern'] = 'approaching'
                    
                # Check for orbit/hold pattern
                elif self._is_orbiting_pattern(history):
                    behavior['pattern'] = 'orbiting'
                    
                # Check for search pattern
                elif self._is_search_pattern(history):
                    behavior['pattern'] = 'searching'
                    
        return behavior
    
    def _suggest_priority(self, 
                         classification: Dict,
                         features: ObservableFeatures) -> str:
        """
        Suggest operational priority based on observable behavior
        NOT based on assumed identity or mission
        """
        # Emergency always highest
        if classification['transponder_status'].get('emergency'):
            return "emergency"
            
        # Fast and approaching
        if (classification['speed_class'] in ['fast', 'very_fast'] and
            features.doppler_velocity < -100):  # Approaching fast
            return "high"
            
        # Large aircraft in approach pattern
        if (classification['size_class'] in ['large', 'very_large'] and
            classification['behavior'].get('pattern') == 'approaching'):
            return "high"
            
        # No transponder at low altitude (suspicious)
        if (not classification['transponder_status']['has_transponder'] and
            classification['altitude_band'] in ['very_low', 'low']):
            return "investigate"
            
        # Unknown or unusual
        if classification.get('special_signature') == 'possible_bird_flock':
            return "low"
            
        return "normal"
    
    def _calculate_confidence(self, 
                             classification: Dict,
                             features: ObservableFeatures,
                             track_id: Optional[str]) -> float:
        """Calculate classification confidence"""
        confidence = 0.5  # Base confidence
        
        # Higher SNR = higher confidence
        if features.snr:
            confidence += min(0.2, features.snr / 100)
            
        # Multiple observations increase confidence
        if track_id and track_id in self.track_history:
            num_observations = len(self.track_history[track_id])
            confidence += min(0.2, num_observations / 20)
            
        # Track quality affects confidence
        confidence *= (0.5 + 0.5 * features.track_quality)
        
        # Emergency codes are high confidence (explicit declaration)
        if classification['transponder_status'].get('emergency'):
            confidence = max(confidence, 0.9)
            
        return min(1.0, confidence)
    
    def _is_approaching_pattern(self, history: List[ObservableFeatures]) -> bool:
        """Check if track shows approach pattern"""
        if len(history) < 3:
            return False
            
        # Decreasing range and altitude
        ranges = [f.range for f in history]
        altitudes = [f.altitude for f in history if f.altitude]
        
        if ranges and altitudes:
            range_decreasing = ranges[-1] < ranges[0]
            altitude_decreasing = altitudes[-1] < altitudes[0] if altitudes else False
            return range_decreasing and altitude_decreasing
            
        return False
    
    def _is_orbiting_pattern(self, history: List[ObservableFeatures]) -> bool:
        """Check if track shows orbit/holding pattern"""
        if len(history) < 10:
            return False
            
        # Check for circular motion (azimuth changes while range stays similar)
        azimuths = [f.azimuth for f in history]
        ranges = [f.range for f in history]
        
        azimuth_change = max(azimuths) - min(azimuths)
        range_variation = np.std(ranges) / np.mean(ranges) if ranges else 1.0
        
        return azimuth_change > np.pi/2 and range_variation < 0.1
    
    def _is_search_pattern(self, history: List[ObservableFeatures]) -> bool:
        """Check if track shows search pattern"""
        if len(history) < 8:
            return False
            
        # Look for systematic back-and-forth motion
        headings = [f.heading for f in history if f.heading is not None]
        if len(headings) < 4:
            return False
            
        # Check for heading reversals
        heading_changes = np.diff(headings)
        reversals = sum(1 for i in range(len(heading_changes)-1) 
                       if heading_changes[i] * heading_changes[i+1] < 0)
        
        return reversals >= 2
    
    def update_history(self, track_id: str, features: ObservableFeatures):
        """Update track history for better classification over time"""
        if track_id not in self.track_history:
            self.track_history[track_id] = []
            
        self.track_history[track_id].append(features)
        
        # Keep only recent history (last 30 observations)
        if len(self.track_history[track_id]) > 30:
            self.track_history[track_id] = self.track_history[track_id][-30:]