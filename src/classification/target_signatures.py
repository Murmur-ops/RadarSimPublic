#!/usr/bin/env python3
"""
Target signature library with threat levels and characteristics
Defines signatures for various target types used in classification
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class ThreatLevel(Enum):
    """Threat level categories"""
    CRITICAL = 5  # Immediate lethal threat
    HIGH = 4      # Significant threat
    MEDIUM = 3    # Moderate threat
    LOW = 2       # Minor threat
    MINIMAL = 1   # Negligible threat
    UNKNOWN = 0   # Unclassified


class TargetClass(Enum):
    """Target classification categories"""
    # Military Aircraft
    FIGHTER = "fighter"
    BOMBER = "bomber"
    ATTACK_HELICOPTER = "attack_helicopter"
    TRANSPORT_HELICOPTER = "transport_helicopter"
    
    # Missiles
    ANTI_SHIP_MISSILE = "anti_ship_missile"
    ANTI_SHIP_BALLISTIC_MISSILE = "anti_ship_ballistic_missile"
    SUPERSONIC_ASM = "supersonic_anti_ship_missile"
    CRUISE_MISSILE = "cruise_missile"
    BALLISTIC_MISSILE = "ballistic_missile"
    AIR_TO_AIR_MISSILE = "air_to_air_missile"
    
    # UAVs
    ARMED_DRONE = "armed_drone"
    SURVEILLANCE_DRONE = "surveillance_drone"
    QUADCOPTER = "quadcopter"
    LOITERING_MUNITION = "loitering_munition"
    
    # Civilian
    COMMERCIAL_AIRCRAFT = "commercial_aircraft"
    GENERAL_AVIATION = "general_aviation"
    
    # Natural
    BIRD = "bird"
    BIRD_FLOCK = "bird_flock"
    WEATHER = "weather"
    
    # Ground/Sea
    GROUND_VEHICLE = "ground_vehicle"
    SURFACE_VESSEL = "surface_vessel"
    
    # Other
    DECOY = "decoy"
    CHAFF = "chaff"
    UNKNOWN = "unknown"


@dataclass
class TargetSignature:
    """Complete target signature definition"""
    # Classification
    target_class: TargetClass
    name: str
    threat_level: ThreatLevel
    
    # Kinematic bounds
    velocity_min: float  # m/s
    velocity_max: float  # m/s
    altitude_min: float  # meters
    altitude_max: float  # meters
    acceleration_max: float  # m/s²
    turn_rate_max: float  # degrees/s
    
    # RCS characteristics
    rcs_mean: float  # m²
    rcs_variance: float  # m²
    rcs_fluctuation_model: str  # Swerling model (0, 1, 2, 3, 4)
    
    # Micro-Doppler
    has_micro_doppler: bool
    micro_doppler_type: Optional[str] = None
    blade_count: Optional[int] = None
    rotation_rate: Optional[float] = None  # Hz
    
    # Behavioral characteristics
    typical_formation: bool = False
    evasive_capable: bool = False
    terrain_following: bool = False
    hover_capable: bool = False
    
    # Engagement parameters
    effective_range: float = 0  # meters (for weapons)
    reaction_time: float = 0  # seconds to engage
    
    def matches_observation(self, 
                           velocity: float,
                           altitude: float,
                           rcs: float,
                           tolerance: float = 0.3) -> float:
        """
        Calculate match score between observation and signature
        
        Args:
            velocity: Observed velocity (m/s)
            altitude: Observed altitude (m)
            rcs: Observed RCS (m²)
            tolerance: Matching tolerance factor
            
        Returns:
            Match score (0-1)
        """
        score = 1.0
        
        # Velocity match
        if self.velocity_min <= velocity <= self.velocity_max:
            vel_score = 1.0
        else:
            if velocity < self.velocity_min:
                vel_score = max(0, 1 - (self.velocity_min - velocity) / (self.velocity_min * tolerance))
            else:
                vel_score = max(0, 1 - (velocity - self.velocity_max) / (self.velocity_max * tolerance))
        score *= vel_score
        
        # Altitude match
        if self.altitude_min <= altitude <= self.altitude_max:
            alt_score = 1.0
        else:
            if altitude < self.altitude_min:
                alt_score = max(0, 1 - (self.altitude_min - altitude) / (self.altitude_min * tolerance))
            else:
                alt_score = max(0, 1 - (altitude - self.altitude_max) / (self.altitude_max * tolerance))
        score *= alt_score
        
        # RCS match (log scale)
        rcs_ratio = rcs / self.rcs_mean if self.rcs_mean > 0 else 0
        if 0.1 <= rcs_ratio <= 10:  # Within order of magnitude
            rcs_score = 1.0 - abs(np.log10(rcs_ratio)) / 2
        else:
            rcs_score = 0.1
        score *= rcs_score
        
        return score


class TargetSignatureLibrary:
    """Library of target signatures for classification"""
    
    def __init__(self):
        """Initialize signature library"""
        self.signatures = self._build_signature_library()
        self.threat_matrix = self._build_threat_matrix()
    
    def _build_signature_library(self) -> Dict[TargetClass, TargetSignature]:
        """Build comprehensive signature library"""
        
        signatures = {
            # Military Aircraft
            TargetClass.FIGHTER: TargetSignature(
                target_class=TargetClass.FIGHTER,
                name="Fighter Aircraft",
                threat_level=ThreatLevel.HIGH,
                velocity_min=100, velocity_max=600,
                altitude_min=100, altitude_max=15000,
                acceleration_max=100, turn_rate_max=20,
                rcs_mean=5, rcs_variance=2,
                rcs_fluctuation_model="Swerling-1",
                has_micro_doppler=True,
                micro_doppler_type="jet_engine",
                evasive_capable=True,
                effective_range=50000,
                reaction_time=2.0
            ),
            
            TargetClass.BOMBER: TargetSignature(
                target_class=TargetClass.BOMBER,
                name="Strategic Bomber",
                threat_level=ThreatLevel.HIGH,
                velocity_min=150, velocity_max=300,
                altitude_min=5000, altitude_max=15000,
                acceleration_max=20, turn_rate_max=5,
                rcs_mean=100, rcs_variance=20,
                rcs_fluctuation_model="Swerling-0",
                has_micro_doppler=True,
                micro_doppler_type="jet_engine",
                effective_range=100000,
                reaction_time=5.0
            ),
            
            TargetClass.ATTACK_HELICOPTER: TargetSignature(
                target_class=TargetClass.ATTACK_HELICOPTER,
                name="Attack Helicopter",
                threat_level=ThreatLevel.HIGH,
                velocity_min=0, velocity_max=100,
                altitude_min=10, altitude_max=3000,
                acceleration_max=20, turn_rate_max=30,
                rcs_mean=10, rcs_variance=5,
                rcs_fluctuation_model="Swerling-2",
                has_micro_doppler=True,
                micro_doppler_type="rotor_blade",
                blade_count=4, rotation_rate=4.5,
                hover_capable=True,
                terrain_following=True,
                effective_range=8000,
                reaction_time=3.0
            ),
            
            # Missiles
            TargetClass.ANTI_SHIP_MISSILE: TargetSignature(
                target_class=TargetClass.ANTI_SHIP_MISSILE,
                name="Anti-Ship Cruise Missile",
                threat_level=ThreatLevel.CRITICAL,
                velocity_min=200, velocity_max=350,
                altitude_min=5, altitude_max=100,  # Sea-skimming
                acceleration_max=50, turn_rate_max=10,
                rcs_mean=0.1, rcs_variance=0.05,
                rcs_fluctuation_model="Swerling-1",
                has_micro_doppler=False,
                terrain_following=True,
                evasive_capable=True,
                effective_range=0,  # Is a weapon itself
                reaction_time=1.0
            ),
            
            TargetClass.ANTI_SHIP_BALLISTIC_MISSILE: TargetSignature(
                target_class=TargetClass.ANTI_SHIP_BALLISTIC_MISSILE,
                name="Anti-Ship Ballistic Missile",
                threat_level=ThreatLevel.CRITICAL,
                velocity_min=1000, velocity_max=3000,  # Terminal velocity
                altitude_min=1000, altitude_max=100000,  # Ballistic trajectory
                acceleration_max=200, turn_rate_max=5,  # Limited terminal maneuver
                rcs_mean=0.5, rcs_variance=0.2,
                rcs_fluctuation_model="Swerling-0",
                has_micro_doppler=False,
                terrain_following=False,
                evasive_capable=True,  # Terminal maneuvering capability
                effective_range=0,
                reaction_time=0.5  # Very short reaction time due to speed
            ),
            
            TargetClass.SUPERSONIC_ASM: TargetSignature(
                target_class=TargetClass.SUPERSONIC_ASM,
                name="Supersonic Anti-Ship Missile",
                threat_level=ThreatLevel.CRITICAL,
                velocity_min=600, velocity_max=1000,  # Mach 2-3
                altitude_min=5, altitude_max=1000,  # High-low profile
                acceleration_max=100, turn_rate_max=15,
                rcs_mean=0.2, rcs_variance=0.1,
                rcs_fluctuation_model="Swerling-1",
                has_micro_doppler=True,
                micro_doppler_type="ramjet",  # Ramjet propulsion
                terrain_following=True,
                evasive_capable=True,
                effective_range=0,
                reaction_time=0.8  # Very short due to high speed
            ),
            
            TargetClass.CRUISE_MISSILE: TargetSignature(
                target_class=TargetClass.CRUISE_MISSILE,
                name="Land Attack Cruise Missile",
                threat_level=ThreatLevel.CRITICAL,
                velocity_min=200, velocity_max=300,
                altitude_min=30, altitude_max=500,
                acceleration_max=30, turn_rate_max=15,
                rcs_mean=0.5, rcs_variance=0.2,
                rcs_fluctuation_model="Swerling-1",
                has_micro_doppler=True,
                micro_doppler_type="jet_engine",
                terrain_following=True,
                reaction_time=1.0
            ),
            
            TargetClass.BALLISTIC_MISSILE: TargetSignature(
                target_class=TargetClass.BALLISTIC_MISSILE,
                name="Ballistic Missile",
                threat_level=ThreatLevel.CRITICAL,
                velocity_min=1000, velocity_max=7000,
                altitude_min=1000, altitude_max=100000,
                acceleration_max=100, turn_rate_max=2,
                rcs_mean=1.0, rcs_variance=0.5,
                rcs_fluctuation_model="Swerling-0",
                has_micro_doppler=False,
                reaction_time=0.5
            ),
            
            # UAVs
            TargetClass.ARMED_DRONE: TargetSignature(
                target_class=TargetClass.ARMED_DRONE,
                name="Armed UAV",
                threat_level=ThreatLevel.MEDIUM,
                velocity_min=20, velocity_max=80,
                altitude_min=100, altitude_max=8000,
                acceleration_max=10, turn_rate_max=15,
                rcs_mean=0.5, rcs_variance=0.2,
                rcs_fluctuation_model="Swerling-3",
                has_micro_doppler=True,
                micro_doppler_type="propeller",
                blade_count=2, rotation_rate=50,
                effective_range=10000,
                reaction_time=5.0
            ),
            
            TargetClass.QUADCOPTER: TargetSignature(
                target_class=TargetClass.QUADCOPTER,
                name="Quadcopter Drone",
                threat_level=ThreatLevel.LOW,
                velocity_min=0, velocity_max=30,
                altitude_min=0, altitude_max=500,
                acceleration_max=5, turn_rate_max=45,
                rcs_mean=0.01, rcs_variance=0.005,
                rcs_fluctuation_model="Swerling-4",
                has_micro_doppler=True,
                micro_doppler_type="drone_rotors",
                blade_count=4, rotation_rate=100,
                hover_capable=True,
                reaction_time=10.0
            ),
            
            TargetClass.LOITERING_MUNITION: TargetSignature(
                target_class=TargetClass.LOITERING_MUNITION,
                name="Loitering Munition",
                threat_level=ThreatLevel.HIGH,
                velocity_min=30, velocity_max=150,
                altitude_min=50, altitude_max=3000,
                acceleration_max=20, turn_rate_max=20,
                rcs_mean=0.05, rcs_variance=0.02,
                rcs_fluctuation_model="Swerling-3",
                has_micro_doppler=True,
                micro_doppler_type="propeller",
                evasive_capable=True,
                reaction_time=2.0
            ),
            
            # Civilian
            TargetClass.COMMERCIAL_AIRCRAFT: TargetSignature(
                target_class=TargetClass.COMMERCIAL_AIRCRAFT,
                name="Commercial Airliner",
                threat_level=ThreatLevel.MINIMAL,
                velocity_min=200, velocity_max=300,
                altitude_min=5000, altitude_max=12000,
                acceleration_max=10, turn_rate_max=3,
                rcs_mean=100, rcs_variance=10,
                rcs_fluctuation_model="Swerling-0",
                has_micro_doppler=True,
                micro_doppler_type="jet_engine",
                reaction_time=30.0
            ),
            
            # Natural
            TargetClass.BIRD: TargetSignature(
                target_class=TargetClass.BIRD,
                name="Bird",
                threat_level=ThreatLevel.MINIMAL,
                velocity_min=5, velocity_max=30,
                altitude_min=0, altitude_max=3000,
                acceleration_max=5, turn_rate_max=90,
                rcs_mean=0.001, rcs_variance=0.0005,
                rcs_fluctuation_model="Swerling-4",
                has_micro_doppler=True,
                micro_doppler_type="bird_wings",
                blade_count=2, rotation_rate=5,
                reaction_time=60.0
            ),
            
            # Decoys
            TargetClass.DECOY: TargetSignature(
                target_class=TargetClass.DECOY,
                name="Radar Decoy",
                threat_level=ThreatLevel.LOW,
                velocity_min=50, velocity_max=300,
                altitude_min=100, altitude_max=10000,
                acceleration_max=20, turn_rate_max=10,
                rcs_mean=10, rcs_variance=50,  # Highly variable
                rcs_fluctuation_model="Swerling-4",
                has_micro_doppler=False,
                reaction_time=10.0
            ),
        }
        
        return signatures
    
    def _build_threat_matrix(self) -> Dict[Tuple[str, TargetClass], ThreatLevel]:
        """
        Build context-dependent threat matrix
        Returns threat level based on (radar_type, target_class) tuple
        """
        matrix = {
            # Naval radar priorities
            ("naval", TargetClass.ANTI_SHIP_MISSILE): ThreatLevel.CRITICAL,
            ("naval", TargetClass.CRUISE_MISSILE): ThreatLevel.HIGH,
            ("naval", TargetClass.FIGHTER): ThreatLevel.HIGH,
            ("naval", TargetClass.ATTACK_HELICOPTER): ThreatLevel.HIGH,
            ("naval", TargetClass.SURFACE_VESSEL): ThreatLevel.MEDIUM,
            ("naval", TargetClass.BOMBER): ThreatLevel.MEDIUM,
            ("naval", TargetClass.QUADCOPTER): ThreatLevel.LOW,
            
            # Ground-based air defense priorities
            ("ground", TargetClass.BOMBER): ThreatLevel.CRITICAL,
            ("ground", TargetClass.CRUISE_MISSILE): ThreatLevel.CRITICAL,
            ("ground", TargetClass.BALLISTIC_MISSILE): ThreatLevel.CRITICAL,
            ("ground", TargetClass.FIGHTER): ThreatLevel.HIGH,
            ("ground", TargetClass.ATTACK_HELICOPTER): ThreatLevel.HIGH,
            ("ground", TargetClass.ARMED_DRONE): ThreatLevel.MEDIUM,
            
            # Airborne radar priorities
            ("airborne", TargetClass.FIGHTER): ThreatLevel.CRITICAL,
            ("airborne", TargetClass.AIR_TO_AIR_MISSILE): ThreatLevel.CRITICAL,
            ("airborne", TargetClass.BOMBER): ThreatLevel.HIGH,
            ("airborne", TargetClass.CRUISE_MISSILE): ThreatLevel.MEDIUM,
            
            # Counter-UAS priorities
            ("counter_uas", TargetClass.QUADCOPTER): ThreatLevel.HIGH,
            ("counter_uas", TargetClass.ARMED_DRONE): ThreatLevel.CRITICAL,
            ("counter_uas", TargetClass.LOITERING_MUNITION): ThreatLevel.CRITICAL,
            ("counter_uas", TargetClass.BIRD): ThreatLevel.MINIMAL,
        }
        
        return matrix
    
    def get_signature(self, target_class: TargetClass) -> Optional[TargetSignature]:
        """Get signature for a target class"""
        return self.signatures.get(target_class)
    
    def get_threat_level(self, radar_type: str, target_class: TargetClass) -> ThreatLevel:
        """
        Get context-dependent threat level
        
        Args:
            radar_type: Type of radar system
            target_class: Classification of target
            
        Returns:
            Threat level for this combination
        """
        # Check specific context
        specific = self.threat_matrix.get((radar_type, target_class))
        if specific:
            return specific
        
        # Fall back to signature default
        signature = self.signatures.get(target_class)
        if signature:
            return signature.threat_level
        
        return ThreatLevel.UNKNOWN
    
    def classify_by_features(self, 
                            velocity: float,
                            altitude: float,
                            rcs: float,
                            has_rotor: bool = False) -> List[Tuple[TargetClass, float]]:
        """
        Classify target based on observed features
        
        Args:
            velocity: Observed velocity (m/s)
            altitude: Observed altitude (m)
            rcs: Observed RCS (m²)
            has_rotor: Whether rotor modulation detected
            
        Returns:
            List of (target_class, confidence) tuples, sorted by confidence
        """
        candidates = []
        
        for target_class, signature in self.signatures.items():
            # Calculate match score
            score = signature.matches_observation(velocity, altitude, rcs)
            
            # Boost score if micro-Doppler matches
            if has_rotor and signature.micro_doppler_type in ["rotor_blade", "drone_rotors"]:
                score *= 1.5
            elif has_rotor and not signature.has_micro_doppler:
                score *= 0.5
            
            if score > 0.1:  # Minimum threshold
                candidates.append((target_class, min(score, 1.0)))
        
        # Sort by confidence
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates
    
    def get_engagement_priority(self, 
                               target_class: TargetClass,
                               range_km: float,
                               closing_velocity_ms: float,
                               radar_type: str = "ground") -> float:
        """
        Calculate engagement priority score
        
        Args:
            target_class: Target classification
            range_km: Range to target (km)
            closing_velocity_ms: Closing velocity (m/s, negative = approaching)
            radar_type: Type of radar system
            
        Returns:
            Priority score (0-100)
        """
        # Get base threat level
        threat_level = self.get_threat_level(radar_type, target_class)
        base_priority = threat_level.value * 20  # 0-100
        
        # Range factor (closer = higher priority)
        range_factor = max(0, 100 - range_km) / 100
        
        # Closing rate factor
        if closing_velocity_ms < 0:  # Approaching
            closing_factor = min(abs(closing_velocity_ms) / 100, 2.0)
        else:
            closing_factor = 0.5  # Receding
        
        # Get signature for reaction time
        signature = self.signatures.get(target_class)
        if signature:
            # Urgent if within reaction time
            time_to_impact = (range_km * 1000) / abs(closing_velocity_ms) if closing_velocity_ms != 0 else 999
            if time_to_impact < signature.reaction_time * 2:
                urgency_factor = 2.0
            else:
                urgency_factor = 1.0
        else:
            urgency_factor = 1.0
        
        # Calculate final priority
        priority = base_priority * range_factor * closing_factor * urgency_factor
        
        return min(priority, 100)  # Cap at 100