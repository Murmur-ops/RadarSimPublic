"""
Integrated Air Defense System (IADS) Network

Implements a comprehensive IADS with C4I capabilities, sensor fusion,
and coordinated engagement management.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat classification levels"""
    CRITICAL = 5  # Ballistic missile, cruise missile
    HIGH = 4      # Fighter-bomber, attack aircraft
    MEDIUM = 3    # Fighter, armed UAV
    LOW = 2       # Reconnaissance, transport
    MINIMAL = 1   # Civilian, friendly


class EngagementStatus(Enum):
    """Engagement status for tracks"""
    MONITORING = "monitoring"
    TRACKING = "tracking"
    ENGAGING = "engaging"
    DESTROYED = "destroyed"
    LOST = "lost"


class SensorType(Enum):
    """Types of sensors in IADS"""
    EWR = "early_warning_radar"  # Long-range search
    TAR = "target_acquisition_radar"  # Medium-range acquisition
    FCR = "fire_control_radar"  # Engagement radar
    IRST = "infrared_search_track"  # Passive IR
    ACOUSTIC = "acoustic"  # Acoustic sensors


@dataclass
class IADSTarget:
    """Target track in IADS"""
    track_id: str
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    classification: str
    threat_level: ThreatLevel
    rcs: float  # Radar cross section
    
    # Tracking data
    first_detection_time: float
    last_update_time: float
    quality: float = 1.0  # Track quality 0-1
    
    # Engagement data
    engagement_status: EngagementStatus = EngagementStatus.MONITORING
    assigned_sam_sites: List[str] = field(default_factory=list)
    missiles_fired: int = 0
    time_to_intercept: Optional[float] = None
    
    # Sensor coverage
    tracking_sensors: Set[str] = field(default_factory=set)
    illuminating_radars: Set[str] = field(default_factory=set)


@dataclass
class EngagementZone:
    """Defines an engagement zone for coordination"""
    zone_id: str
    center: np.ndarray
    radius: float
    altitude_min: float
    altitude_max: float
    responsible_units: List[str]
    priority: int = 1


class IADSNetwork:
    """
    Integrated Air Defense System Network Controller
    
    Manages sensor fusion, threat assessment, and weapon assignment
    for a multi-layered air defense system.
    """
    
    def __init__(self, network_id: str = "IADS_MAIN"):
        """
        Initialize IADS network
        
        Args:
            network_id: Unique identifier for this IADS network
        """
        self.network_id = network_id
        
        # Component systems
        self.sensors: Dict[str, Any] = {}  # sensor_id -> sensor object
        self.sam_sites: Dict[str, Any] = {}  # site_id -> SAM site object
        self.command_posts: Dict[str, Any] = {}  # cp_id -> command post
        
        # Track management
        self.tracks: Dict[str, IADSTarget] = {}
        self.track_correlation: Dict[str, Set[str]] = defaultdict(set)  # Correlated tracks
        self.track_history: Dict[str, List[Dict]] = defaultdict(list)
        
        # Engagement management
        self.engagement_zones: List[EngagementZone] = []
        self.active_engagements: Dict[str, Dict] = {}
        self.engagement_doctrine: Dict = self._default_doctrine()
        
        # Network status
        self.network_health: Dict[str, float] = {}  # component_id -> health 0-1
        self.communication_links: Dict[Tuple[str, str], float] = {}  # (id1, id2) -> quality
        
        # Statistics
        self.statistics = {
            'tracks_detected': 0,
            'tracks_engaged': 0,
            'missiles_fired': 0,
            'intercepts_successful': 0,
            'sensor_handoffs': 0
        }
        
        logger.info(f"Initialized IADS network: {network_id}")
    
    def _default_doctrine(self) -> Dict:
        """Define default engagement doctrine"""
        return {
            'engagement_authority': 'distributed',  # 'centralized' or 'distributed'
            'fire_doctrine': 'shoot-look-shoot',  # or 'shoot-shoot-look'
            'missiles_per_target': {
                ThreatLevel.CRITICAL: 3,
                ThreatLevel.HIGH: 2,
                ThreatLevel.MEDIUM: 2,
                ThreatLevel.LOW: 1,
                ThreatLevel.MINIMAL: 0
            },
            'engagement_priorities': [
                ThreatLevel.CRITICAL,
                ThreatLevel.HIGH,
                ThreatLevel.MEDIUM,
                ThreatLevel.LOW
            ],
            'minimum_pk': 0.7,  # Minimum probability of kill for engagement
            'max_simultaneous_engagements': 10
        }
    
    def add_sensor(self, sensor_id: str, sensor_obj: Any, sensor_type: SensorType):
        """
        Add a sensor to the IADS network
        
        Args:
            sensor_id: Unique sensor identifier
            sensor_obj: Sensor object (radar, IRST, etc.)
            sensor_type: Type of sensor
        """
        self.sensors[sensor_id] = {
            'object': sensor_obj,
            'type': sensor_type,
            'status': 'active',
            'tracks': set()
        }
        self.network_health[sensor_id] = 1.0
        logger.debug(f"Added {sensor_type.value} sensor: {sensor_id}")
    
    def add_sam_site(self, site_id: str, sam_site: Any):
        """
        Add a SAM site to the network
        
        Args:
            site_id: Unique site identifier
            sam_site: SAM site object
        """
        self.sam_sites[site_id] = sam_site
        self.network_health[site_id] = 1.0
        logger.debug(f"Added SAM site: {site_id}")
    
    def process_sensor_data(self, sensor_id: str, detections: List[Dict]) -> List[str]:
        """
        Process raw sensor detections and update tracks
        
        Args:
            sensor_id: Source sensor ID
            detections: List of detection dictionaries
            
        Returns:
            List of updated track IDs
        """
        if sensor_id not in self.sensors:
            logger.warning(f"Unknown sensor: {sensor_id}")
            return []
        
        updated_tracks = []
        
        for detection in detections:
            # Try to correlate with existing tracks
            track_id = self._correlate_detection(detection, sensor_id)
            
            if track_id:
                # Update existing track
                self._update_track(track_id, detection, sensor_id)
            else:
                # Create new track
                track_id = self._create_track(detection, sensor_id)
                self.statistics['tracks_detected'] += 1
            
            updated_tracks.append(track_id)
            
            # Add sensor to track's sensor list
            if track_id in self.tracks:
                self.tracks[track_id].tracking_sensors.add(sensor_id)
        
        return updated_tracks
    
    def _correlate_detection(self, detection: Dict, sensor_id: str) -> Optional[str]:
        """
        Correlate detection with existing tracks
        
        Args:
            detection: Detection data
            sensor_id: Source sensor
            
        Returns:
            Track ID if correlated, None otherwise
        """
        position = np.array(detection['position'])
        velocity = np.array(detection.get('velocity', [0, 0, 0]))
        
        best_track = None
        min_distance = float('inf')
        
        # Simple nearest-neighbor correlation (would use more sophisticated method in practice)
        for track_id, track in self.tracks.items():
            # Predict track position
            dt = detection['timestamp'] - track.last_update_time
            predicted_pos = track.position + track.velocity * dt
            
            # Calculate distance
            distance = np.linalg.norm(predicted_pos - position)
            
            # Gate check (e.g., 3-sigma gate)
            gate_size = 500 + 100 * dt  # Simple adaptive gate
            
            if distance < gate_size and distance < min_distance:
                min_distance = distance
                best_track = track_id
        
        return best_track
    
    def _create_track(self, detection: Dict, sensor_id: str) -> str:
        """Create new track from detection"""
        track_id = f"TRK_{len(self.tracks):04d}"
        
        # Classify target (simplified)
        classification, threat_level = self._classify_target(detection)
        
        track = IADSTarget(
            track_id=track_id,
            position=np.array(detection['position']),
            velocity=np.array(detection.get('velocity', [0, 0, 0])),
            classification=classification,
            threat_level=threat_level,
            rcs=detection.get('rcs', 10.0),
            first_detection_time=detection['timestamp'],
            last_update_time=detection['timestamp']
        )
        
        track.tracking_sensors.add(sensor_id)
        self.tracks[track_id] = track
        
        logger.info(f"Created new track {track_id}: {classification} at {track.position/1000} km")
        
        return track_id
    
    def _update_track(self, track_id: str, detection: Dict, sensor_id: str):
        """Update existing track with new detection"""
        if track_id not in self.tracks:
            return
        
        track = self.tracks[track_id]
        
        # Simple alpha-beta filter update (would use Kalman filter in practice)
        alpha = 0.7  # Position smoothing
        beta = 0.4   # Velocity smoothing
        
        dt = detection['timestamp'] - track.last_update_time
        if dt > 0:
            # Predict
            predicted_pos = track.position + track.velocity * dt
            
            # Update
            position_error = np.array(detection['position']) - predicted_pos
            track.position = predicted_pos + alpha * position_error
            
            if 'velocity' in detection:
                track.velocity = (1 - beta) * track.velocity + beta * np.array(detection['velocity'])
            else:
                # Estimate velocity from position change
                track.velocity = position_error / dt
        
        track.last_update_time = detection['timestamp']
        track.quality = min(1.0, track.quality + 0.1)  # Improve quality with updates
    
    def _classify_target(self, detection: Dict) -> Tuple[str, ThreatLevel]:
        """
        Classify target and assess threat level
        
        Args:
            detection: Detection data
            
        Returns:
            Tuple of (classification, threat_level)
        """
        # Simplified classification based on kinematics and RCS
        velocity = np.linalg.norm(detection.get('velocity', [0, 0, 0]))
        altitude = detection['position'][2] if len(detection['position']) > 2 else 0
        rcs = detection.get('rcs', 10.0)
        
        # Speed-based classification (simplified)
        if velocity > 500:  # Very high speed
            if altitude > 15000:
                return "ballistic_missile", ThreatLevel.CRITICAL
            else:
                return "cruise_missile", ThreatLevel.CRITICAL
        elif velocity > 200:  # Fighter speed
            if rcs > 50:
                return "bomber", ThreatLevel.HIGH
            else:
                return "fighter", ThreatLevel.MEDIUM
        elif velocity > 50:  # Slow mover
            if altitude < 500:
                return "helicopter", ThreatLevel.LOW
            else:
                return "uav", ThreatLevel.MEDIUM
        else:
            return "unknown", ThreatLevel.LOW
    
    def perform_threat_assessment(self) -> List[str]:
        """
        Assess all tracks and prioritize threats
        
        Returns:
            List of track IDs sorted by threat priority
        """
        threat_scores = {}
        
        for track_id, track in self.tracks.items():
            if track.engagement_status == EngagementStatus.DESTROYED:
                continue
            
            # Calculate threat score
            score = self._calculate_threat_score(track)
            threat_scores[track_id] = score
        
        # Sort by threat score
        sorted_tracks = sorted(threat_scores.keys(), 
                              key=lambda x: threat_scores[x], 
                              reverse=True)
        
        return sorted_tracks
    
    def _calculate_threat_score(self, track: IADSTarget) -> float:
        """
        Calculate threat score for a track
        
        Args:
            track: Target track
            
        Returns:
            Threat score (0-100)
        """
        score = 0.0
        
        # Base threat level
        threat_multipliers = {
            ThreatLevel.CRITICAL: 5.0,
            ThreatLevel.HIGH: 3.0,
            ThreatLevel.MEDIUM: 2.0,
            ThreatLevel.LOW: 1.0,
            ThreatLevel.MINIMAL: 0.5
        }
        score += threat_multipliers[track.threat_level] * 20
        
        # Range factor (closer = higher threat)
        range_to_target = np.linalg.norm(track.position)
        if range_to_target < 10000:  # Within 10 km
            score += 30
        elif range_to_target < 50000:  # Within 50 km
            score += 20
        elif range_to_target < 100000:  # Within 100 km
            score += 10
        
        # Closing velocity (approaching = higher threat)
        radial_velocity = np.dot(track.velocity, -track.position / np.linalg.norm(track.position))
        if radial_velocity > 100:  # Fast approach
            score += 20
        elif radial_velocity > 0:  # Approaching
            score += 10
        
        # Altitude factor (low altitude = harder to engage)
        altitude = track.position[2] if len(track.position) > 2 else 0
        if altitude < 100:  # Very low
            score += 15
        elif altitude < 1000:  # Low
            score += 10
        
        # Track quality (better track = more confident threat)
        score *= track.quality
        
        return min(100.0, score)
    
    def assign_weapons(self, track_id: str) -> Optional[Dict]:
        """
        Assign SAM systems to engage a track
        
        Args:
            track_id: Track to engage
            
        Returns:
            Engagement assignment or None if cannot engage
        """
        if track_id not in self.tracks:
            return None
        
        track = self.tracks[track_id]
        
        # Check if already engaged
        if track.engagement_status == EngagementStatus.ENGAGING:
            return None
        
        # Find capable SAM sites
        capable_sites = self._find_capable_sam_sites(track)
        
        if not capable_sites:
            logger.warning(f"No SAM sites capable of engaging {track_id}")
            return None
        
        # Select best SAM site(s) based on doctrine
        num_missiles = self.engagement_doctrine['missiles_per_target'].get(
            track.threat_level, 1
        )
        
        selected_sites = self._select_sam_sites(capable_sites, track, num_missiles)
        
        if not selected_sites:
            return None
        
        # Create engagement assignment
        assignment = {
            'track_id': track_id,
            'sam_sites': selected_sites,
            'num_missiles': num_missiles,
            'engagement_time': None,  # Will be set by SAM controller
            'pk_estimated': self._estimate_pk(track, selected_sites[0])
        }
        
        # Update track status
        track.engagement_status = EngagementStatus.ENGAGING
        track.assigned_sam_sites = selected_sites
        self.active_engagements[track_id] = assignment
        self.statistics['tracks_engaged'] += 1
        
        logger.info(f"Assigned {selected_sites} to engage {track_id}")
        
        return assignment
    
    def _find_capable_sam_sites(self, track: IADSTarget) -> List[str]:
        """Find SAM sites capable of engaging target"""
        capable = []
        
        for site_id, sam_site in self.sam_sites.items():
            # Check if site is operational
            if self.network_health.get(site_id, 0) < 0.5:
                continue
            
            # Check if target is within engagement envelope
            if sam_site.can_engage(track.position, track.velocity):
                capable.append(site_id)
        
        return capable
    
    def _select_sam_sites(self, capable_sites: List[str], 
                         track: IADSTarget, 
                         num_missiles: int) -> List[str]:
        """Select best SAM sites for engagement"""
        if not capable_sites:
            return []
        
        # Score each site
        site_scores = {}
        for site_id in capable_sites:
            sam_site = self.sam_sites[site_id]
            
            # Range to target
            range_to_target = np.linalg.norm(
                track.position - sam_site.position
            )
            
            # Closer is better (normalized)
            range_score = 1.0 - min(1.0, range_to_target / sam_site.max_range)
            
            # Available missiles
            availability_score = min(1.0, sam_site.missiles_available / 10)
            
            # Site health
            health_score = self.network_health.get(site_id, 0.5)
            
            # Combined score
            site_scores[site_id] = (range_score + availability_score + health_score) / 3
        
        # Sort by score and select best
        sorted_sites = sorted(site_scores.keys(), 
                            key=lambda x: site_scores[x], 
                            reverse=True)
        
        # Select sites based on fire doctrine
        if self.engagement_doctrine['fire_doctrine'] == 'shoot-shoot-look':
            # Multiple sites fire simultaneously
            return sorted_sites[:min(2, len(sorted_sites))]
        else:
            # Single site fires first
            return sorted_sites[:1]
    
    def _estimate_pk(self, track: IADSTarget, sam_site_id: str) -> float:
        """
        Estimate probability of kill
        
        Args:
            track: Target track
            sam_site_id: SAM site ID
            
        Returns:
            Estimated Pk (0-1)
        """
        if sam_site_id not in self.sam_sites:
            return 0.0
        
        sam_site = self.sam_sites[sam_site_id]
        
        # Base Pk from SAM capability
        base_pk = 0.8  # Nominal for modern SAM
        
        # Range factor
        range_to_target = np.linalg.norm(track.position - sam_site.position)
        range_factor = 1.0 - (range_to_target / sam_site.max_range) ** 2
        
        # Altitude factor
        altitude = track.position[2] if len(track.position) > 2 else 0
        if altitude < 100:  # Very low altitude
            altitude_factor = 0.5
        elif altitude > 20000:  # Very high altitude
            altitude_factor = 0.7
        else:
            altitude_factor = 1.0
        
        # Speed factor
        speed = np.linalg.norm(track.velocity)
        if speed > 1000:  # Very fast
            speed_factor = 0.6
        elif speed < 50:  # Very slow/hovering
            speed_factor = 1.2
        else:
            speed_factor = 1.0
        
        # RCS factor
        if track.rcs < 0.1:  # Stealth
            rcs_factor = 0.3
        elif track.rcs > 10:  # Large RCS
            rcs_factor = 1.1
        else:
            rcs_factor = 0.8
        
        # Calculate final Pk
        pk = base_pk * range_factor * altitude_factor * speed_factor * rcs_factor
        
        return max(0.0, min(1.0, pk))
    
    def update_engagement_status(self, track_id: str, status: EngagementStatus, 
                                 result: Optional[str] = None):
        """
        Update engagement status for a track
        
        Args:
            track_id: Track ID
            status: New engagement status
            result: Engagement result (e.g., 'hit', 'miss')
        """
        if track_id not in self.tracks:
            return
        
        track = self.tracks[track_id]
        track.engagement_status = status
        
        if status == EngagementStatus.DESTROYED:
            self.statistics['intercepts_successful'] += 1
            logger.info(f"Target {track_id} destroyed")
        elif status == EngagementStatus.LOST:
            logger.warning(f"Lost track {track_id}")
        
        # Clean up active engagement
        if track_id in self.active_engagements:
            del self.active_engagements[track_id]
    
    def get_network_status(self) -> Dict:
        """
        Get comprehensive network status
        
        Returns:
            Dictionary containing network status information
        """
        operational_sensors = sum(1 for h in self.network_health.values() if h > 0.5)
        operational_sams = sum(1 for sid in self.sam_sites 
                              if self.network_health.get(sid, 0) > 0.5)
        
        return {
            'network_id': self.network_id,
            'operational_sensors': operational_sensors,
            'total_sensors': len(self.sensors),
            'operational_sam_sites': operational_sams,
            'total_sam_sites': len(self.sam_sites),
            'active_tracks': len(self.tracks),
            'active_engagements': len(self.active_engagements),
            'network_health': np.mean(list(self.network_health.values())),
            'statistics': self.statistics.copy()
        }
    
    def coordinate_handoff(self, track_id: str, from_sensor: str, to_sensor: str) -> bool:
        """
        Coordinate track handoff between sensors
        
        Args:
            track_id: Track to hand off
            from_sensor: Current tracking sensor
            to_sensor: New tracking sensor
            
        Returns:
            True if handoff successful
        """
        if track_id not in self.tracks:
            return False
        
        track = self.tracks[track_id]
        
        # Remove from old sensor
        if from_sensor in track.tracking_sensors:
            track.tracking_sensors.remove(from_sensor)
        
        # Add to new sensor
        track.tracking_sensors.add(to_sensor)
        
        self.statistics['sensor_handoffs'] += 1
        logger.debug(f"Handed off {track_id} from {from_sensor} to {to_sensor}")
        
        return True