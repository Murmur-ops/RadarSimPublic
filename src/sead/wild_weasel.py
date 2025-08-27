"""
Wild Weasel SEAD Aircraft and AGM-88 HARM Implementation

Models Suppression of Enemy Air Defenses (SEAD) aircraft with
radar warning receivers, emitter location, and anti-radiation missiles.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SEADTactic(Enum):
    """SEAD tactical approaches"""
    PREEMPTIVE = "preemptive"      # Fire before SAM launches
    REACTIVE = "reactive"          # Fire after radar detection
    STANDOFF = "standoff"          # Launch from max range
    PENETRATION = "penetration"    # Close-in attack


class HARMMode(Enum):
    """AGM-88 HARM operating modes"""
    POS = "pre_briefed"           # Pre-Briefed mode
    HAS = "harm_as_sensor"        # HARM as Sensor mode
    TOO = "target_of_opportunity" # Target of Opportunity mode
    EOM = "equations_of_motion"    # Equations of Motion mode (AGM-88E)


@dataclass
class EmitterDetection:
    """Detected radar emitter"""
    emitter_id: str
    emitter_type: str  # 'search', 'track', 'fcr'
    position_estimate: np.ndarray
    position_error: float  # Uncertainty in meters
    frequency: float  # Hz
    pri: float  # Pulse repetition interval
    signal_strength: float  # dBm
    bearing: float  # Degrees
    detected_time: float
    last_update_time: float
    classification_confidence: float


@dataclass
class HARM:
    """AGM-88 High-speed Anti-Radiation Missile"""
    missile_id: str
    position: np.ndarray
    velocity: np.ndarray
    target_emitter_id: str
    launch_time: float
    mode: HARMMode
    
    # Seeker parameters
    seeker_fov: float = 90.0  # degrees
    memory_mode: bool = False  # Can remember target location if radar shuts down
    target_last_position: Optional[np.ndarray] = None
    
    # Flight parameters
    max_range: float = 150000  # 150 km for AGM-88E
    max_speed: float = 680     # m/s (Mach 2+)
    fuel_remaining: float = 1.0
    time_to_impact: Optional[float] = None


class WildWeasel:
    """
    Wild Weasel SEAD Aircraft
    
    F-16CJ or EA-18G equivalent with RWR, HARM capability,
    and electronic warfare systems.
    """
    
    def __init__(self,
                 aircraft_id: str,
                 position: np.ndarray,
                 velocity: np.ndarray,
                 num_harms: int = 4):
        """
        Initialize Wild Weasel aircraft
        
        Args:
            aircraft_id: Unique aircraft identifier
            position: Initial position [x, y, z]
            velocity: Initial velocity [vx, vy, vz]
            num_harms: Number of HARMs carried
        """
        self.aircraft_id = aircraft_id
        self.position = np.asarray(position, dtype=np.float64)
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.num_harms = num_harms
        self.harms_remaining = num_harms
        
        # Radar Warning Receiver (RWR)
        self.rwr_detections: Dict[str, EmitterDetection] = {}
        self.rwr_range = 200000  # 200 km RWR range
        self.rwr_accuracy = 5.0  # degrees bearing accuracy
        
        # Emitter location system
        self.emitter_database: Dict[str, Dict] = {}
        self.triangulation_data: Dict[str, List] = {}
        
        # HARMs in flight
        self.harms_launched: Dict[str, HARM] = {}
        
        # Electronic warfare capabilities
        self.jamming_power = 100  # Watts
        self.jamming_active = False
        self.decoys_available = 4
        
        # Mission parameters
        self.tactic = SEADTactic.REACTIVE
        self.engagement_range = 80000  # 80 km typical HARM launch range
        self.minimum_range = 10000    # 10 km minimum
        
        # Status
        self.mission_time = 0.0
        self.threats_destroyed = 0
        self.status = "ingressing"
        
        logger.info(f"Wild Weasel {aircraft_id} initialized with {num_harms} HARMs")
    
    def update_position(self, dt: float):
        """Update aircraft position"""
        self.position += self.velocity * dt
        self.mission_time += dt
    
    def detect_emitters(self, emitters: Dict[str, Dict]) -> List[str]:
        """
        Detect radar emitters with RWR
        
        Args:
            emitters: Dictionary of active emitters
            
        Returns:
            List of detected emitter IDs
        """
        detected = []
        
        for emitter_id, emitter_data in emitters.items():
            if not emitter_data.get('emitting', False):
                continue
            
            emitter_pos = np.array(emitter_data['position'])
            range_to_emitter = np.linalg.norm(emitter_pos - self.position)
            
            # Check if in RWR range
            if range_to_emitter > self.rwr_range:
                continue
            
            # Calculate signal strength (simplified)
            emitter_power = emitter_data.get('power', 1000)
            signal_strength = emitter_power / (4 * np.pi * range_to_emitter ** 2)
            signal_strength_dbm = 10 * np.log10(signal_strength * 1000)
            
            # RWR detection threshold
            if signal_strength_dbm < -100:  # -100 dBm threshold
                continue
            
            # Calculate bearing
            relative_pos = emitter_pos - self.position
            bearing = np.degrees(np.arctan2(relative_pos[1], relative_pos[0]))
            
            # Add bearing error
            bearing += np.random.normal(0, self.rwr_accuracy)
            
            # Position estimate with error
            position_error = range_to_emitter * 0.1  # 10% range error initially
            position_estimate = self._estimate_emitter_position(
                bearing, range_to_emitter, position_error
            )
            
            # Create or update detection
            if emitter_id not in self.rwr_detections:
                detection = EmitterDetection(
                    emitter_id=emitter_id,
                    emitter_type=emitter_data.get('type', 'unknown'),
                    position_estimate=position_estimate,
                    position_error=position_error,
                    frequency=emitter_data.get('frequency', 10e9),
                    pri=1.0 / emitter_data.get('prf', 1000),
                    signal_strength=signal_strength_dbm,
                    bearing=bearing,
                    detected_time=self.mission_time,
                    last_update_time=self.mission_time,
                    classification_confidence=0.7
                )
                self.rwr_detections[emitter_id] = detection
                logger.info(f"{self.aircraft_id} detected emitter {emitter_id} at bearing {bearing:.1f}")
            else:
                # Update existing detection
                detection = self.rwr_detections[emitter_id]
                detection.bearing = bearing
                detection.signal_strength = signal_strength_dbm
                detection.last_update_time = self.mission_time
                
                # Improve position estimate through triangulation
                self._update_position_estimate(detection, bearing, range_to_emitter)
            
            detected.append(emitter_id)
        
        # Remove old detections
        stale_detections = []
        for emitter_id, detection in self.rwr_detections.items():
            if self.mission_time - detection.last_update_time > 5.0:  # 5 second timeout
                stale_detections.append(emitter_id)
        
        for emitter_id in stale_detections:
            del self.rwr_detections[emitter_id]
            logger.debug(f"{self.aircraft_id} lost track of emitter {emitter_id}")
        
        return detected
    
    def _estimate_emitter_position(self, bearing: float, 
                                  estimated_range: float,
                                  error: float) -> np.ndarray:
        """Estimate emitter position from bearing and range"""
        bearing_rad = np.radians(bearing)
        
        # Add error to range estimate
        range_with_error = estimated_range + np.random.normal(0, error)
        
        # Calculate position
        x = self.position[0] + range_with_error * np.cos(bearing_rad)
        y = self.position[1] + range_with_error * np.sin(bearing_rad)
        z = 0  # Assume ground-based
        
        return np.array([x, y, z])
    
    def _update_position_estimate(self, detection: EmitterDetection,
                                 new_bearing: float,
                                 estimated_range: float):
        """Improve position estimate through triangulation"""
        # Store bearing measurements for triangulation
        if detection.emitter_id not in self.triangulation_data:
            self.triangulation_data[detection.emitter_id] = []
        
        self.triangulation_data[detection.emitter_id].append({
            'aircraft_position': self.position.copy(),
            'bearing': new_bearing,
            'time': self.mission_time
        })
        
        # Need at least 2 measurements from different positions for triangulation
        if len(self.triangulation_data[detection.emitter_id]) >= 2:
            # Simple triangulation (would use least squares in practice)
            measurements = self.triangulation_data[detection.emitter_id][-2:]
            
            pos1 = measurements[0]['aircraft_position']
            bearing1 = np.radians(measurements[0]['bearing'])
            
            pos2 = measurements[1]['aircraft_position']
            bearing2 = np.radians(measurements[1]['bearing'])
            
            # Calculate intersection point (simplified 2D)
            # Line 1: pos1 + t1 * [cos(bearing1), sin(bearing1)]
            # Line 2: pos2 + t2 * [cos(bearing2), sin(bearing2)]
            
            A = np.array([[np.cos(bearing1), -np.cos(bearing2)],
                         [np.sin(bearing1), -np.sin(bearing2)]])
            b = pos2[:2] - pos1[:2]
            
            try:
                t = np.linalg.solve(A, b)
                if t[0] > 0:  # Forward direction
                    new_estimate = pos1[:2] + t[0] * np.array([np.cos(bearing1), 
                                                               np.sin(bearing1)])
                    detection.position_estimate[:2] = new_estimate
                    detection.position_error *= 0.9  # Improve error estimate
            except np.linalg.LinAlgError:
                pass  # Lines parallel, keep previous estimate
    
    def select_target(self) -> Optional[str]:
        """
        Select highest priority target for engagement
        
        Returns:
            Emitter ID to engage or None
        """
        if not self.rwr_detections or self.harms_remaining == 0:
            return None
        
        # Score each detected emitter
        target_scores = {}
        
        for emitter_id, detection in self.rwr_detections.items():
            score = 0.0
            
            # Threat type priority
            if 'fcr' in detection.emitter_type.lower():  # Fire control radar
                score += 100
            elif 'track' in detection.emitter_type.lower():  # Tracking radar
                score += 70
            elif 'search' in detection.emitter_type.lower():  # Search radar
                score += 40
            
            # Signal strength (stronger = closer or more powerful)
            score += max(0, detection.signal_strength + 100) / 2
            
            # Classification confidence
            score *= detection.classification_confidence
            
            # Range factor (prefer targets in optimal engagement envelope)
            estimated_range = np.linalg.norm(detection.position_estimate - self.position)
            if self.minimum_range < estimated_range < self.engagement_range:
                score *= 1.2
            elif estimated_range > self.engagement_range:
                score *= 0.7
            
            # Time tracked (prefer well-established tracks)
            track_time = self.mission_time - detection.detected_time
            if track_time > 10:
                score *= 1.1
            
            target_scores[emitter_id] = score
        
        # Select highest scoring target
        if target_scores:
            best_target = max(target_scores.keys(), key=lambda x: target_scores[x])
            return best_target
        
        return None
    
    def launch_harm(self, target_emitter_id: str, 
                   mode: HARMMode = HARMMode.TOO) -> Optional[str]:
        """
        Launch HARM at detected emitter
        
        Args:
            target_emitter_id: Target emitter ID
            mode: HARM operating mode
            
        Returns:
            HARM ID if launched, None otherwise
        """
        if self.harms_remaining == 0:
            logger.warning(f"{self.aircraft_id} out of HARMs")
            return None
        
        if target_emitter_id not in self.rwr_detections:
            logger.warning(f"No detection for target {target_emitter_id}")
            return None
        
        detection = self.rwr_detections[target_emitter_id]
        
        # Check range
        range_to_target = np.linalg.norm(detection.position_estimate - self.position)
        if range_to_target > 150000:  # Max HARM range
            logger.warning(f"Target {target_emitter_id} out of range ({range_to_target/1000:.1f} km)")
            return None
        
        # Create HARM
        harm_id = f"{self.aircraft_id}_HARM_{self.num_harms - self.harms_remaining + 1}"
        
        # Calculate launch vector
        target_vector = detection.position_estimate - self.position
        target_vector = target_vector / np.linalg.norm(target_vector)
        
        # Initial velocity (aircraft velocity + launch velocity)
        launch_velocity = self.velocity + target_vector * 200  # 200 m/s launch speed
        
        harm = HARM(
            missile_id=harm_id,
            position=self.position.copy(),
            velocity=launch_velocity,
            target_emitter_id=target_emitter_id,
            launch_time=self.mission_time,
            mode=mode,
            target_last_position=detection.position_estimate.copy()
        )
        
        # AGM-88E has memory mode
        if mode == HARMMode.EOM:
            harm.memory_mode = True
        
        self.harms_launched[harm_id] = harm
        self.harms_remaining -= 1
        
        logger.info(f"{self.aircraft_id} launched {harm_id} at {target_emitter_id} "
                   f"from {range_to_target/1000:.1f} km")
        
        return harm_id
    
    def update_harms(self, dt: float, 
                    emitters: Dict[str, Dict]) -> List[Dict]:
        """
        Update HARM positions and guidance
        
        Args:
            dt: Time step
            emitters: Current emitter states
            
        Returns:
            List of impact results
        """
        impacts = []
        harms_to_remove = []
        
        for harm_id, harm in self.harms_launched.items():
            # Check if target still emitting
            target_emitting = False
            target_position = harm.target_last_position
            
            if harm.target_emitter_id in emitters:
                emitter = emitters[harm.target_emitter_id]
                target_emitting = emitter.get('emitting', False)
                if target_emitting:
                    target_position = np.array(emitter['position'])
                    harm.target_last_position = target_position.copy()
            
            # Update HARM physics
            flight_time = self.mission_time - harm.launch_time
            
            # Simple speed model
            if flight_time < 3:  # Boost phase
                speed = min(harm.max_speed, 340 + flight_time * 100)
            else:
                speed = harm.max_speed
            
            # Update guidance
            if target_emitting or harm.memory_mode:
                # Home on target (or last known position)
                relative_pos = target_position - harm.position
                range_to_target = np.linalg.norm(relative_pos)
                
                if range_to_target > 0:
                    # Proportional navigation
                    desired_velocity = relative_pos / range_to_target * speed
                    
                    # Smooth turn (limited by G-force)
                    max_turn_rate = 30 * 9.81 * dt  # 30G max
                    velocity_change = desired_velocity - harm.velocity
                    
                    if np.linalg.norm(velocity_change) > max_turn_rate:
                        velocity_change = velocity_change / np.linalg.norm(velocity_change) * max_turn_rate
                    
                    harm.velocity += velocity_change
                    harm.velocity = harm.velocity / np.linalg.norm(harm.velocity) * speed
                
                # Check for impact
                if range_to_target < 10:  # Within 10m
                    impacts.append({
                        'harm_id': harm_id,
                        'target_id': harm.target_emitter_id,
                        'impact_position': harm.position.copy(),
                        'result': 'hit'
                    })
                    harms_to_remove.append(harm_id)
                    self.threats_destroyed += 1
                    logger.info(f"HARM {harm_id} destroyed {harm.target_emitter_id}")
                    continue
            else:
                # Lost target, continue on last vector
                pass
            
            # Update position
            harm.position += harm.velocity * dt
            
            # Check max range
            distance_traveled = np.linalg.norm(harm.position - self.position)
            if distance_traveled > harm.max_range:
                impacts.append({
                    'harm_id': harm_id,
                    'target_id': harm.target_emitter_id,
                    'result': 'miss_range'
                })
                harms_to_remove.append(harm_id)
                logger.debug(f"HARM {harm_id} exceeded max range")
        
        # Remove impacted HARMs
        for harm_id in harms_to_remove:
            del self.harms_launched[harm_id]
        
        return impacts
    
    def execute_sead_tactic(self, emitters: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Execute SEAD tactics based on current situation
        
        Args:
            emitters: Current emitter states
            
        Returns:
            Action taken
        """
        action = {'type': 'none', 'details': {}}
        
        # Detect emitters
        detected = self.detect_emitters(emitters)
        
        if not detected:
            return action
        
        # Select tactic based on situation
        if self.tactic == SEADTactic.PREEMPTIVE:
            # Launch at all detected FCRs immediately
            for emitter_id in detected:
                detection = self.rwr_detections[emitter_id]
                if 'fcr' in detection.emitter_type.lower() and self.harms_remaining > 0:
                    harm_id = self.launch_harm(emitter_id, HARMMode.HAS)
                    if harm_id:
                        action = {'type': 'harm_launch', 
                                'details': {'harm_id': harm_id, 'target': emitter_id}}
                        break
        
        elif self.tactic == SEADTactic.REACTIVE:
            # Launch at highest threat
            target = self.select_target()
            if target and self.harms_remaining > 0:
                harm_id = self.launch_harm(target, HARMMode.TOO)
                if harm_id:
                    action = {'type': 'harm_launch',
                            'details': {'harm_id': harm_id, 'target': target}}
        
        elif self.tactic == SEADTactic.STANDOFF:
            # Launch from max range
            for emitter_id in detected:
                detection = self.rwr_detections[emitter_id]
                range_to_emitter = np.linalg.norm(detection.position_estimate - self.position)
                if 70000 < range_to_emitter < 90000 and self.harms_remaining > 0:
                    harm_id = self.launch_harm(emitter_id, HARMMode.POS)
                    if harm_id:
                        action = {'type': 'harm_launch',
                                'details': {'harm_id': harm_id, 'target': emitter_id}}
                        break
        
        return action
    
    def get_status(self) -> Dict:
        """Get Wild Weasel status"""
        return {
            'aircraft_id': self.aircraft_id,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'harms_remaining': self.harms_remaining,
            'harms_in_flight': len(self.harms_launched),
            'emitters_detected': len(self.rwr_detections),
            'threats_destroyed': self.threats_destroyed,
            'status': self.status,
            'mission_time': self.mission_time
        }