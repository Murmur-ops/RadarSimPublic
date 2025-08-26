"""
Classification module for radar target identification and threat assessment
"""

from .feature_extractor import FeatureExtractor, TargetFeatures
from .micro_doppler import MicroDopplerSimulator, MicroDopplerType, MicroDopplerParams
from .target_signatures import (
    TargetClass, 
    ThreatLevel, 
    TargetSignature, 
    TargetSignatureLibrary
)
from .threat_assessment import (
    ThreatAssessment,
    ThreatAssessmentContext,
    ThreatTrack,
    RadarMission
)

# Configuration compatibility mapping
CONFIG_TYPE_TO_TARGET_CLASS = {
    # Aircraft
    'fighter': TargetClass.FIGHTER,
    'bomber': TargetClass.BOMBER,
    'attack_helicopter': TargetClass.ATTACK_HELICOPTER,
    'helicopter': TargetClass.ATTACK_HELICOPTER,
    'transport': TargetClass.COMMERCIAL_AIRCRAFT,
    
    # Missiles
    'cruise_missile': TargetClass.CRUISE_MISSILE,
    'anti_ship_missile': TargetClass.ANTI_SHIP_MISSILE,
    'anti_ship_cruise_missile': TargetClass.ANTI_SHIP_MISSILE,
    'anti_ship_ballistic_missile': TargetClass.ANTI_SHIP_BALLISTIC_MISSILE,
    'supersonic_anti_ship_missile': TargetClass.SUPERSONIC_ASM,
    'ASM': TargetClass.ANTI_SHIP_MISSILE,
    'ASBM': TargetClass.ANTI_SHIP_BALLISTIC_MISSILE,
    'ballistic_missile': TargetClass.BALLISTIC_MISSILE,
    
    # Drones/UAVs
    'quadcopter': TargetClass.QUADCOPTER,
    'drone': TargetClass.QUADCOPTER,
    'fixed_wing_uav': TargetClass.ARMED_DRONE,
    'armed_drone': TargetClass.ARMED_DRONE,
    'surveillance_drone': TargetClass.SURVEILLANCE_DRONE,
    
    # Civilian
    'commercial_aircraft': TargetClass.COMMERCIAL_AIRCRAFT,
    'airliner': TargetClass.COMMERCIAL_AIRCRAFT,
    'general_aviation': TargetClass.GENERAL_AVIATION,
    
    # Natural/Other
    'bird': TargetClass.BIRD,
    'decoy': TargetClass.DECOY,
    'unknown': TargetClass.UNKNOWN,
    
    # Ground/Sea
    'vehicle': TargetClass.GROUND_VEHICLE,
    'ship': TargetClass.SURFACE_VESSEL,
}

# Radar type mapping for configuration files
RADAR_TYPE_TO_MISSION = {
    'ground_based': RadarMission.AIR_DEFENSE,
    'naval_3d': RadarMission.NAVAL_DEFENSE,
    'naval': RadarMission.NAVAL_DEFENSE,
    'airborne': RadarMission.AIR_DEFENSE,
    'counter_uas': RadarMission.COUNTER_UAS,
    'fire_control': RadarMission.FIRE_CONTROL,
}

def get_target_class_from_config(config_type: str) -> TargetClass:
    """
    Convert configuration file target type to TargetClass enum
    
    Args:
        config_type: Target type string from config file
        
    Returns:
        Corresponding TargetClass enum value
    """
    return CONFIG_TYPE_TO_TARGET_CLASS.get(
        config_type.lower(), 
        TargetClass.UNKNOWN
    )

def get_radar_mission_from_config(radar_type: str) -> RadarMission:
    """
    Convert configuration file radar type to RadarMission enum
    
    Args:
        radar_type: Radar type string from config file
        
    Returns:
        Corresponding RadarMission enum value
    """
    return RADAR_TYPE_TO_MISSION.get(
        radar_type.lower(),
        RadarMission.SURVEILLANCE
    )

__all__ = [
    'FeatureExtractor',
    'TargetFeatures',
    'MicroDopplerSimulator',
    'MicroDopplerType',
    'MicroDopplerParams',
    'TargetClass',
    'ThreatLevel',
    'TargetSignature',
    'TargetSignatureLibrary',
    'ThreatAssessment',
    'ThreatAssessmentContext', 
    'ThreatTrack',
    'RadarMission',
    'get_target_class_from_config',
    'get_radar_mission_from_config',
    'CONFIG_TYPE_TO_TARGET_CLASS',
    'RADAR_TYPE_TO_MISSION'
]