"""
Surface-to-Air Missile (SAM) System Implementation

Models various SAM systems with realistic engagement envelopes,
missile kinematics, and guidance systems.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SAMType(Enum):
    """Types of SAM systems"""
    LONG_RANGE = "long_range"      # S-400/Patriot class (400km)
    MEDIUM_RANGE = "medium_range"  # Buk/NASAMS class (50km)
    SHORT_RANGE = "short_range"    # Pantsir/C-RAM class (20km)
    MANPADS = "manpads"            # Stinger/Igla class (5km)


class MissileGuidance(Enum):
    """Missile guidance types"""
    COMMAND = "command"                    # Command guidance
    SEMI_ACTIVE = "semi_active_radar"     # SARH
    ACTIVE = "active_radar"                # Active radar homing
    INFRARED = "infrared"                  # IR homing
    BEAM_RIDING = "beam_riding"           # Beam rider


@dataclass
class MissileKinematics:
    """Missile performance parameters"""
    max_speed: float           # m/s
    max_range: float          # meters
    min_range: float          # meters
    max_altitude: float       # meters
    min_altitude: float       # meters
    max_g: float             # Maximum G-force for maneuvering
    burn_time: float         # Motor burn time in seconds
    cruise_speed: float      # Speed after burn


@dataclass
class Missile:
    """Individual missile in flight"""
    missile_id: str
    position: np.ndarray
    velocity: np.ndarray
    target_id: str
    launch_time: float
    guidance_type: MissileGuidance
    kinematics: MissileKinematics
    
    # Flight parameters
    fuel_remaining: float = 1.0
    seeker_active: bool = False
    terminal_phase: bool = False
    
    # Guidance data
    last_command_time: float = 0.0
    predicted_intercept_point: Optional[np.ndarray] = None
    time_to_intercept: Optional[float] = None


class SAMSite:
    """
    Surface-to-Air Missile Site
    
    Models a complete SAM battery with radar, launchers, and missiles.
    """
    
    def __init__(self,
                 site_id: str,
                 position: np.ndarray,
                 sam_type: SAMType,
                 num_launchers: int = 4,
                 missiles_per_launcher: int = 4):
        """
        Initialize SAM site
        
        Args:
            site_id: Unique site identifier
            position: Site position [x, y, z] in meters
            sam_type: Type of SAM system
            num_launchers: Number of launch rails/tubes
            missiles_per_launcher: Missiles per launcher
        """
        self.site_id = site_id
        self.position = np.asarray(position, dtype=np.float64)
        self.sam_type = sam_type
        self.num_launchers = num_launchers
        self.missiles_per_launcher = missiles_per_launcher
        
        # Initialize missile inventory
        self.missiles_available = num_launchers * missiles_per_launcher
        self.missiles_in_flight: Dict[str, Missile] = {}
        
        # Get system capabilities
        self.capabilities = self._get_sam_capabilities(sam_type)
        self.kinematics = self.capabilities['kinematics']
        self.max_range = self.kinematics.max_range
        self.min_range = self.kinematics.min_range
        self.max_altitude = self.kinematics.max_altitude
        self.min_altitude = self.kinematics.min_altitude
        
        # Radar parameters
        self.radar_range = self.capabilities['radar_range']
        self.radar_active = True
        self.emitting = False  # For EMCON management
        
        # Engagement parameters
        self.max_simultaneous_engagements = self.capabilities['max_engagements']
        self.current_engagements: Dict[str, List[str]] = {}  # target_id -> missile_ids
        self.engagement_history: List[Dict] = []
        
        # Status
        self.operational = True
        self.reload_time = self.capabilities['reload_time']
        self.last_launch_time = 0.0
        
        logger.info(f"Initialized {sam_type.value} SAM site {site_id} at {position/1000} km")
    
    def _get_sam_capabilities(self, sam_type: SAMType) -> Dict:
        """Get capabilities based on SAM type"""
        
        if sam_type == SAMType.LONG_RANGE:
            # S-400/Patriot class
            return {
                'kinematics': MissileKinematics(
                    max_speed=2000,      # Mach 6
                    max_range=400000,    # 400 km
                    min_range=5000,      # 5 km
                    max_altitude=30000,  # 30 km
                    min_altitude=10,     # 10 m
                    max_g=30,
                    burn_time=10,
                    cruise_speed=1500
                ),
                'radar_range': 600000,  # 600 km search range
                'max_engagements': 12,
                'reload_time': 300,     # 5 minutes
                'guidance': MissileGuidance.ACTIVE
            }
        
        elif sam_type == SAMType.MEDIUM_RANGE:
            # Buk/NASAMS class
            return {
                'kinematics': MissileKinematics(
                    max_speed=1200,      # Mach 3.5
                    max_range=50000,     # 50 km
                    min_range=3000,      # 3 km
                    max_altitude=25000,  # 25 km
                    min_altitude=15,     # 15 m
                    max_g=25,
                    burn_time=6,
                    cruise_speed=900
                ),
                'radar_range': 150000,  # 150 km search range
                'max_engagements': 6,
                'reload_time': 180,     # 3 minutes
                'guidance': MissileGuidance.SEMI_ACTIVE
            }
        
        elif sam_type == SAMType.SHORT_RANGE:
            # Pantsir/C-RAM class
            return {
                'kinematics': MissileKinematics(
                    max_speed=900,       # Mach 2.6
                    max_range=20000,     # 20 km
                    min_range=200,       # 200 m
                    max_altitude=15000,  # 15 km
                    min_altitude=0,      # 0 m (can engage ground targets)
                    max_g=20,
                    burn_time=4,
                    cruise_speed=700
                ),
                'radar_range': 40000,   # 40 km search range
                'max_engagements': 4,
                'reload_time': 60,      # 1 minute
                'guidance': MissileGuidance.COMMAND
            }
        
        else:  # MANPADS
            return {
                'kinematics': MissileKinematics(
                    max_speed=650,       # Mach 2
                    max_range=5000,      # 5 km
                    min_range=500,       # 500 m
                    max_altitude=3500,   # 3.5 km
                    min_altitude=10,     # 10 m
                    max_g=15,
                    burn_time=2,
                    cruise_speed=500
                ),
                'radar_range': 10000,   # 10 km (optical/IR)
                'max_engagements': 1,
                'reload_time': 30,      # 30 seconds
                'guidance': MissileGuidance.INFRARED
            }
    
    def can_engage(self, target_position: np.ndarray, 
                  target_velocity: np.ndarray) -> bool:
        """
        Check if target is within engagement envelope
        
        Args:
            target_position: Target position [x, y, z]
            target_velocity: Target velocity [vx, vy, vz]
            
        Returns:
            True if target can be engaged
        """
        if not self.operational or self.missiles_available == 0:
            return False
        
        # Calculate range and altitude
        relative_pos = target_position - self.position
        range_to_target = np.linalg.norm(relative_pos[:2])  # Horizontal range
        altitude = target_position[2] if len(target_position) > 2 else 0
        
        # Check envelope
        if range_to_target < self.min_range or range_to_target > self.max_range:
            return False
        
        if altitude < self.min_altitude or altitude > self.max_altitude:
            return False
        
        # Check if we can guide more missiles
        if len(self.current_engagements) >= self.max_simultaneous_engagements:
            return False
        
        # Check predicted intercept point
        intercept_point = self._predict_intercept(target_position, target_velocity)
        if intercept_point is not None:
            intercept_range = np.linalg.norm(intercept_point - self.position)
            if intercept_range > self.max_range * 0.8:  # 80% max range for margin
                return False
        
        return True
    
    def _predict_intercept(self, target_pos: np.ndarray, 
                          target_vel: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict intercept point for target
        
        Args:
            target_pos: Target position
            target_vel: Target velocity
            
        Returns:
            Predicted intercept point or None
        """
        # Simplified constant velocity prediction
        avg_missile_speed = (self.kinematics.max_speed + self.kinematics.cruise_speed) / 2
        
        # Iterative prediction (2-3 iterations usually sufficient)
        intercept_point = target_pos.copy()
        for _ in range(3):
            range_to_intercept = np.linalg.norm(intercept_point - self.position)
            time_to_intercept = range_to_intercept / avg_missile_speed
            intercept_point = target_pos + target_vel * time_to_intercept
        
        return intercept_point
    
    def engage_target(self, target_id: str, 
                     target_position: np.ndarray,
                     target_velocity: np.ndarray,
                     num_missiles: int = 2,
                     current_time: float = 0.0) -> List[str]:
        """
        Launch missiles at target
        
        Args:
            target_id: Target identifier
            target_position: Target position
            target_velocity: Target velocity
            num_missiles: Number of missiles to launch
            current_time: Current simulation time
            
        Returns:
            List of launched missile IDs
        """
        if not self.can_engage(target_position, target_velocity):
            logger.warning(f"{self.site_id} cannot engage {target_id}")
            return []
        
        # Limit missiles to available
        num_missiles = min(num_missiles, self.missiles_available)
        if num_missiles == 0:
            return []
        
        launched_missiles = []
        
        for i in range(num_missiles):
            # Create missile
            missile_id = f"{self.site_id}_M{len(self.missiles_in_flight):03d}"
            
            # Calculate launch direction (simplified)
            intercept_point = self._predict_intercept(target_position, target_velocity)
            if intercept_point is None:
                intercept_point = target_position
            
            launch_direction = intercept_point - self.position
            launch_direction = launch_direction / np.linalg.norm(launch_direction)
            
            # Initial missile velocity (vertical launch then turn)
            initial_velocity = launch_direction * 50  # 50 m/s initial
            initial_velocity[2] = 100  # Strong vertical component
            
            missile = Missile(
                missile_id=missile_id,
                position=self.position.copy(),
                velocity=initial_velocity,
                target_id=target_id,
                launch_time=current_time,
                guidance_type=self.capabilities['guidance'],
                kinematics=self.kinematics,
                predicted_intercept_point=intercept_point
            )
            
            self.missiles_in_flight[missile_id] = missile
            launched_missiles.append(missile_id)
            self.missiles_available -= 1
            
            # Add launch delay between missiles
            current_time += 0.5
        
        # Track engagement
        if target_id not in self.current_engagements:
            self.current_engagements[target_id] = []
        self.current_engagements[target_id].extend(launched_missiles)
        
        self.last_launch_time = current_time
        
        logger.info(f"{self.site_id} launched {num_missiles} missiles at {target_id}")
        
        return launched_missiles
    
    def update_missiles(self, dt: float, 
                       target_updates: Dict[str, Dict],
                       current_time: float) -> List[Dict]:
        """
        Update missile positions and guidance
        
        Args:
            dt: Time step
            target_updates: Dictionary of target positions/velocities
            current_time: Current simulation time
            
        Returns:
            List of intercept results
        """
        intercepts = []
        missiles_to_remove = []
        
        for missile_id, missile in self.missiles_in_flight.items():
            # Get target data
            if missile.target_id not in target_updates:
                # Lost target
                missiles_to_remove.append(missile_id)
                continue
            
            target_data = target_updates[missile.target_id]
            target_pos = np.array(target_data['position'])
            target_vel = np.array(target_data.get('velocity', [0, 0, 0]))
            
            # Update missile physics
            self._update_missile_physics(missile, dt, current_time)
            
            # Update guidance
            self._update_missile_guidance(missile, target_pos, target_vel, dt)
            
            # Check for intercept
            distance_to_target = np.linalg.norm(missile.position - target_pos)
            
            if distance_to_target < 10:  # Within 10m = hit
                intercepts.append({
                    'missile_id': missile_id,
                    'target_id': missile.target_id,
                    'impact_point': missile.position.copy(),
                    'time': current_time,
                    'result': 'hit'
                })
                missiles_to_remove.append(missile_id)
                logger.info(f"Missile {missile_id} intercepted {missile.target_id}")
            
            # Check if missile is out of fuel or range
            elif missile.fuel_remaining <= 0:
                intercepts.append({
                    'missile_id': missile_id,
                    'target_id': missile.target_id,
                    'result': 'miss_fuel'
                })
                missiles_to_remove.append(missile_id)
            
            elif np.linalg.norm(missile.position - self.position) > self.max_range * 1.2:
                intercepts.append({
                    'missile_id': missile_id,
                    'target_id': missile.target_id,
                    'result': 'miss_range'
                })
                missiles_to_remove.append(missile_id)
        
        # Remove completed missiles
        for missile_id in missiles_to_remove:
            del self.missiles_in_flight[missile_id]
            # Update engagement tracking
            for target_id, missile_list in self.current_engagements.items():
                if missile_id in missile_list:
                    missile_list.remove(missile_id)
        
        # Clean up completed engagements
        completed_engagements = [tid for tid, mlist in self.current_engagements.items() 
                                if len(mlist) == 0]
        for target_id in completed_engagements:
            del self.current_engagements[target_id]
        
        return intercepts
    
    def _update_missile_physics(self, missile: Missile, dt: float, current_time: float):
        """Update missile position and velocity"""
        flight_time = current_time - missile.launch_time
        
        # Boost phase
        if flight_time < missile.kinematics.burn_time:
            # Accelerating
            acceleration = (missile.kinematics.max_speed - 50) / missile.kinematics.burn_time
            speed = min(missile.kinematics.max_speed, 50 + acceleration * flight_time)
            missile.fuel_remaining = 1.0 - (flight_time / missile.kinematics.burn_time) * 0.7
        else:
            # Cruise/coast phase
            speed = missile.kinematics.cruise_speed
            missile.fuel_remaining = max(0, 0.3 - (flight_time - missile.kinematics.burn_time) * 0.01)
        
        # Update velocity magnitude
        if np.linalg.norm(missile.velocity) > 0:
            missile.velocity = missile.velocity / np.linalg.norm(missile.velocity) * speed
        
        # Update position
        missile.position += missile.velocity * dt
        
        # Simple gravity effect (reduced for missiles)
        missile.velocity[2] -= 2.0 * dt  # Reduced gravity effect
    
    def _update_missile_guidance(self, missile: Missile, 
                                target_pos: np.ndarray,
                                target_vel: np.ndarray,
                                dt: float):
        """Update missile guidance commands"""
        
        # Calculate lead point (proportional navigation)
        relative_pos = target_pos - missile.position
        range_to_target = np.linalg.norm(relative_pos)
        
        # Proportional navigation constant
        N = 3.0
        
        # Line of sight rate
        los_direction = relative_pos / range_to_target
        relative_vel = target_vel - missile.velocity
        closing_velocity = -np.dot(relative_vel, los_direction)
        
        if closing_velocity > 0:  # Approaching target
            # Calculate required acceleration
            los_rate = np.cross(relative_pos, relative_vel) / (range_to_target ** 2)
            commanded_accel = N * closing_velocity * np.cross(los_direction, los_rate)
            
            # Limit to max G
            max_accel = missile.kinematics.max_g * 9.81  # Convert G to m/s²
            accel_magnitude = np.linalg.norm(commanded_accel)
            if accel_magnitude > max_accel:
                commanded_accel = commanded_accel / accel_magnitude * max_accel
            
            # Apply acceleration
            missile.velocity += commanded_accel * dt
        
        # Terminal guidance (last 2 seconds)
        time_to_impact = range_to_target / max(closing_velocity, 1.0)
        if time_to_impact < 2.0:
            missile.terminal_phase = True
            # More aggressive guidance in terminal phase
            direct_vector = relative_pos / range_to_target
            missile.velocity = missile.velocity * 0.9 + direct_vector * np.linalg.norm(missile.velocity) * 0.1
    
    def set_emcon(self, emitting: bool):
        """
        Set emission control (EMCON) status
        
        Args:
            emitting: True to emit, False for silent
        """
        self.emitting = emitting
        self.radar_active = emitting
        
        if not emitting:
            logger.info(f"{self.site_id} entering EMCON (radar off)")
        else:
            logger.info(f"{self.site_id} radar active")
    
    def get_status(self) -> Dict:
        """Get SAM site status"""
        return {
            'site_id': self.site_id,
            'type': self.sam_type.value,
            'operational': self.operational,
            'missiles_available': self.missiles_available,
            'missiles_in_flight': len(self.missiles_in_flight),
            'active_engagements': len(self.current_engagements),
            'radar_active': self.radar_active,
            'position': self.position.tolist()
        }