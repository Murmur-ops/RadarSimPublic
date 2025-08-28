#!/usr/bin/env python3
"""
Threat assessment and prioritization system
Context-aware threat scoring based on radar type and mission
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

from .target_signatures import TargetClass, ThreatLevel, TargetSignatureLibrary


class RadarMission(Enum):
    """Radar mission types"""
    AIR_DEFENSE = "air_defense"
    MISSILE_DEFENSE = "missile_defense"  
    NAVAL_DEFENSE = "naval_defense"
    COUNTER_UAS = "counter_uas"
    SURVEILLANCE = "surveillance"
    FIRE_CONTROL = "fire_control"


@dataclass
class ThreatAssessmentContext:
    """Context for threat assessment"""
    radar_type: str  # "naval", "ground", "airborne", etc.
    mission: RadarMission
    defended_asset_value: float = 1.0  # Relative value (0-1)
    rules_of_engagement: str = "defensive"  # "defensive", "offensive", "passive"
    threat_axis: Optional[float] = None  # Expected threat direction (degrees)
    
    # Environmental factors
    visibility: float = 1.0  # Weather impact (0-1)
    jamming_level: float = 0.0  # Jamming intensity (0-1)
    clutter_level: float = 0.0  # Clutter density (0-1)


@dataclass
class ThreatTrack:
    """Track with threat assessment"""
    track_id: int
    classification: TargetClass
    confidence: float
    
    # Kinematics
    range: float  # meters
    altitude: float  # meters
    velocity: float  # m/s
    heading: float  # degrees
    closing_velocity: float  # m/s (negative = approaching)
    
    # Threat metrics
    threat_level: ThreatLevel
    threat_score: float = 0.0
    time_to_intercept: float = float('inf')
    
    # Track quality
    track_quality: float = 1.0
    last_update_time: float = 0.0
    track_history_length: int = 0
    
    # Engagement status
    engaged: bool = False
    engagement_recommendation: Optional[str] = None
    
    def __lt__(self, other):
        """Compare tracks by threat score for sorting"""
        return self.threat_score > other.threat_score  # Higher score = higher priority


class ThreatAssessment:
    """Threat assessment and prioritization engine"""
    
    def __init__(self, context: ThreatAssessmentContext):
        """
        Initialize threat assessment system
        
        Args:
            context: Assessment context including radar type and mission
        """
        self.context = context
        self.signature_library = TargetSignatureLibrary()
        self.threat_history: Dict[int, List[float]] = {}  # Track ID -> threat scores over time
        
    def assess_threat(self, track: ThreatTrack) -> float:
        """
        Calculate comprehensive threat score for a track
        
        Args:
            track: Target track to assess
            
        Returns:
            Threat score (0-100)
        """
        # Get base threat level from library
        base_threat = self.signature_library.get_threat_level(
            self.context.radar_type, 
            track.classification
        )
        
        # Convert to numerical score
        threat_score = base_threat.value * 20  # 0-100
        
        # Apply context-specific modifiers
        threat_score = self._apply_kinematic_modifiers(threat_score, track)
        threat_score = self._apply_mission_modifiers(threat_score, track)
        threat_score = self._apply_environmental_modifiers(threat_score, track)
        threat_score = self._apply_behavioral_modifiers(threat_score, track)
        
        # Update track
        track.threat_level = base_threat
        track.threat_score = min(threat_score, 100)
        
        # Store in history
        if track.track_id not in self.threat_history:
            self.threat_history[track.track_id] = []
        self.threat_history[track.track_id].append(threat_score)
        
        return track.threat_score
    
    def _apply_kinematic_modifiers(self, score: float, track: ThreatTrack) -> float:
        """Apply kinematic-based threat modifiers"""
        
        # Range factor (closer = more threatening)
        range_km = track.range / 1000
        if range_km < 5:
            range_multiplier = 2.0
        elif range_km < 10:
            range_multiplier = 1.5
        elif range_km < 20:
            range_multiplier = 1.2
        elif range_km < 50:
            range_multiplier = 1.0
        else:
            range_multiplier = 0.8
        
        # Closing rate factor
        if track.closing_velocity < -200:  # Fast approach
            closing_multiplier = 2.0
        elif track.closing_velocity < -100:
            closing_multiplier = 1.5
        elif track.closing_velocity < 0:
            closing_multiplier = 1.2
        else:  # Receding
            closing_multiplier = 0.5
        
        # Altitude factor (context-dependent)
        if self.context.radar_type == "naval":
            # Sea-skimmers are high threat
            if track.altitude < 50:
                altitude_multiplier = 2.0
            elif track.altitude < 200:
                altitude_multiplier = 1.5
            else:
                altitude_multiplier = 1.0
        elif self.context.mission == RadarMission.AIR_DEFENSE:
            # Low altitude penetrators
            if track.altitude < 500:
                altitude_multiplier = 1.5
            else:
                altitude_multiplier = 1.0
        else:
            altitude_multiplier = 1.0
        
        # Time to intercept
        if track.closing_velocity < 0:
            track.time_to_intercept = track.range / abs(track.closing_velocity)
            if track.time_to_intercept < 30:  # Less than 30 seconds
                urgency_multiplier = 2.0
            elif track.time_to_intercept < 60:
                urgency_multiplier = 1.5
            else:
                urgency_multiplier = 1.0
        else:
            urgency_multiplier = 1.0
        
        # Apply all modifiers
        score *= range_multiplier * closing_multiplier * altitude_multiplier * urgency_multiplier
        
        return score
    
    def _apply_mission_modifiers(self, score: float, track: ThreatTrack) -> float:
        """Apply mission-specific threat modifiers"""
        
        if self.context.mission == RadarMission.MISSILE_DEFENSE:
            # Prioritize missiles
            if track.classification in [TargetClass.CRUISE_MISSILE, 
                                      TargetClass.BALLISTIC_MISSILE,
                                      TargetClass.ANTI_SHIP_MISSILE]:
                score *= 2.0
            elif track.classification == TargetClass.FIGHTER:
                score *= 1.2  # Could launch missiles
                
        elif self.context.mission == RadarMission.NAVAL_DEFENSE:
            # Prioritize anti-ship threats
            if track.classification == TargetClass.ANTI_SHIP_MISSILE:
                score *= 3.0
            elif track.classification == TargetClass.FIGHTER and track.altitude < 1000:
                score *= 1.5  # Low-flying fighter threat
            elif track.classification == TargetClass.SURFACE_VESSEL:
                score *= 1.2
                
        elif self.context.mission == RadarMission.COUNTER_UAS:
            # Prioritize drones
            if track.classification in [TargetClass.QUADCOPTER,
                                      TargetClass.ARMED_DRONE,
                                      TargetClass.LOITERING_MUNITION]:
                score *= 2.0
            elif track.classification == TargetClass.BIRD:
                score *= 0.1  # Depriotize birds
                
        elif self.context.mission == RadarMission.AIR_DEFENSE:
            # Prioritize aircraft and missiles
            if track.classification in [TargetClass.BOMBER, TargetClass.FIGHTER]:
                score *= 1.5
            elif track.classification in [TargetClass.CRUISE_MISSILE, TargetClass.BALLISTIC_MISSILE]:
                score *= 2.0
        
        # Asset value modifier
        score *= (0.5 + 0.5 * self.context.defended_asset_value)
        
        return score
    
    def _apply_environmental_modifiers(self, score: float, track: ThreatTrack) -> float:
        """Apply environmental condition modifiers"""
        
        # Poor visibility increases threat (harder to engage)
        if self.context.visibility < 0.5:
            score *= 1.2
        
        # Jamming presence increases threat
        if self.context.jamming_level > 0.5:
            score *= (1 + 0.3 * self.context.jamming_level)
        
        # High clutter reduces confidence but increases caution
        if self.context.clutter_level > 0.5:
            if track.track_quality < 0.7:
                score *= 0.8  # Low quality track in clutter
            else:
                score *= 1.1  # Confirmed track despite clutter
        
        return score
    
    def _apply_behavioral_modifiers(self, score: float, track: ThreatTrack) -> float:
        """Apply behavior-based threat modifiers"""
        
        # Check threat axis alignment
        if self.context.threat_axis is not None:
            angle_diff = abs(track.heading - self.context.threat_axis)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            if angle_diff < 30:  # On threat axis
                score *= 1.3
            elif angle_diff > 120:  # Away from threat axis
                score *= 0.7
        
        # Track quality affects confidence
        score *= (0.7 + 0.3 * track.track_quality)
        
        # Track age (stale tracks are less reliable)
        track_age = time.time() - track.last_update_time
        if track_age > 5:  # More than 5 seconds old
            score *= 0.8
        elif track_age > 10:
            score *= 0.5
        
        # Track history (consistent tracks are more threatening)
        if track.track_history_length > 10:
            score *= 1.1
        elif track.track_history_length < 3:
            score *= 0.9
        
        return score
    
    def prioritize_threats(self, tracks: List[ThreatTrack]) -> List[ThreatTrack]:
        """
        Prioritize list of tracks by threat level
        
        Args:
            tracks: List of tracks to prioritize
            
        Returns:
            Sorted list of tracks (highest threat first)
        """
        # Assess each track
        for track in tracks:
            self.assess_threat(track)
        
        # Sort by threat score
        prioritized = sorted(tracks, key=lambda t: t.threat_score, reverse=True)
        
        # Apply engagement recommendations
        self._recommend_engagements(prioritized)
        
        return prioritized
    
    def _recommend_engagements(self, tracks: List[ThreatTrack]):
        """Add engagement recommendations to tracks"""
        
        for i, track in enumerate(tracks):
            if track.threat_score > 80:
                track.engagement_recommendation = "ENGAGE_IMMEDIATE"
            elif track.threat_score > 60:
                track.engagement_recommendation = "ENGAGE_WHEN_ABLE"
            elif track.threat_score > 40:
                track.engagement_recommendation = "MONITOR_CLOSELY"
            elif track.threat_score > 20:
                track.engagement_recommendation = "TRACK_ONLY"
            else:
                track.engagement_recommendation = "IGNORE"
            
            # Special cases
            if track.classification == TargetClass.COMMERCIAL_AIRCRAFT:
                track.engagement_recommendation = "DO_NOT_ENGAGE"
            elif track.classification == TargetClass.UNKNOWN and track.threat_score > 50:
                track.engagement_recommendation = "INTERROGATE_IFF"
    
    def get_threat_picture(self, tracks: List[ThreatTrack]) -> Dict:
        """
        Generate comprehensive threat picture summary
        
        Args:
            tracks: Current tracks
            
        Returns:
            Dictionary with threat picture analysis
        """
        prioritized = self.prioritize_threats(tracks)
        
        # Count threats by level
        threat_counts = {
            ThreatLevel.CRITICAL: 0,
            ThreatLevel.HIGH: 0,
            ThreatLevel.MEDIUM: 0,
            ThreatLevel.LOW: 0,
            ThreatLevel.MINIMAL: 0
        }
        
        for track in prioritized:
            if track.threat_level in threat_counts:
                threat_counts[track.threat_level] += 1
        
        # Identify most dangerous threats
        critical_threats = [t for t in prioritized if t.threat_score > 80]
        imminent_threats = [t for t in prioritized if t.time_to_intercept < 60]
        
        # Calculate saturation level
        total_threats = len([t for t in prioritized if t.threat_score > 40])
        saturation_level = min(total_threats / 10, 1.0)  # Normalized to 0-1
        
        return {
            'total_tracks': len(tracks),
            'threat_counts': threat_counts,
            'critical_threats': critical_threats,
            'imminent_threats': imminent_threats,
            'highest_threat': prioritized[0] if prioritized else None,
            'saturation_level': saturation_level,
            'recommended_posture': self._recommend_posture(saturation_level, threat_counts)
        }
    
    def _recommend_posture(self, saturation: float, threat_counts: Dict) -> str:
        """Recommend defensive posture based on threat picture"""
        
        if threat_counts[ThreatLevel.CRITICAL] > 2 or saturation > 0.8:
            return "WEAPONS_FREE"  # Engage all threats
        elif threat_counts[ThreatLevel.CRITICAL] > 0:
            return "WEAPONS_TIGHT"  # Engage confirmed threats only
        elif threat_counts[ThreatLevel.HIGH] > 3:
            return "ELEVATED_ALERT"
        elif threat_counts[ThreatLevel.HIGH] > 0:
            return "ENHANCED_MONITORING"
        else:
            return "NORMAL_OPERATIONS"