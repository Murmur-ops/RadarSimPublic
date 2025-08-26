"""
Target modeling for radar simulation
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class TargetType(Enum):
    """Enumeration of target types"""
    AIRCRAFT = "aircraft"
    MISSILE = "missile"
    SHIP = "ship"
    VEHICLE = "vehicle"
    DRONE = "drone"
    BIRD = "bird"
    CUSTOM = "custom"


@dataclass
class TargetMotion:
    """Target motion parameters"""
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    acceleration: np.ndarray = None  # [ax, ay, az] in m/s^2
    
    def __post_init__(self):
        # Ensure arrays are float type
        self.position = self.position.astype(np.float64)
        self.velocity = self.velocity.astype(np.float64)
        if self.acceleration is None:
            self.acceleration = np.zeros(3, dtype=np.float64)
        else:
            self.acceleration = self.acceleration.astype(np.float64)
    
    def update(self, dt: float):
        """Update position and velocity based on time step"""
        self.position += self.velocity * dt + 0.5 * self.acceleration * dt**2
        self.velocity += self.acceleration * dt
    
    @property
    def speed(self) -> float:
        """Calculate speed magnitude"""
        return np.linalg.norm(self.velocity)
    
    @property
    def range(self) -> float:
        """Calculate range from origin"""
        return np.linalg.norm(self.position)


class Target:
    """Radar target model"""
    
    # Typical RCS values (m^2)
    RCS_VALUES = {
        TargetType.AIRCRAFT: 10.0,      # Fighter aircraft
        TargetType.MISSILE: 0.1,         # Small missile
        TargetType.SHIP: 10000.0,        # Large ship
        TargetType.VEHICLE: 5.0,         # Ground vehicle
        TargetType.DRONE: 0.01,          # Small drone
        TargetType.BIRD: 0.001,          # Bird
        TargetType.CUSTOM: 1.0           # Default custom
    }
    
    def __init__(self, 
                 target_type: TargetType = TargetType.AIRCRAFT,
                 rcs: Optional[float] = None,
                 motion: Optional[TargetMotion] = None,
                 fluctuation_model: str = "swerling1"):
        """
        Initialize target
        
        Args:
            target_type: Type of target
            rcs: Radar Cross Section in m^2 (if None, uses typical value)
            motion: Target motion parameters
            fluctuation_model: RCS fluctuation model
        """
        self.target_type = target_type
        self.rcs_mean = rcs if rcs is not None else self.RCS_VALUES[target_type]
        self.motion = motion if motion is not None else TargetMotion(
            position=np.array([1000.0, 0.0, 1000.0]),
            velocity=np.zeros(3)
        )
        self.fluctuation_model = fluctuation_model
        self.track_history = []
        
    def get_rcs(self, aspect_angle: Optional[float] = None) -> float:
        """
        Get RCS value with fluctuation
        
        Args:
            aspect_angle: Aspect angle in radians (for aspect-dependent RCS)
            
        Returns:
            RCS value in m^2
        """
        # Base RCS
        rcs = self.rcs_mean
        
        # Apply aspect dependency if angle provided
        if aspect_angle is not None:
            # Simple aspect model: maximum at broadside, minimum at nose/tail
            aspect_factor = 0.3 + 0.7 * abs(np.sin(aspect_angle))
            rcs *= aspect_factor
        
        # Apply fluctuation model
        if self.fluctuation_model == "swerling1":
            # Swerling 1: Slow fluctuation, exponential distribution
            rcs *= np.random.exponential(1.0)
        elif self.fluctuation_model == "swerling2":
            # Swerling 2: Fast fluctuation, exponential distribution
            rcs *= np.random.exponential(1.0)
        elif self.fluctuation_model == "swerling3":
            # Swerling 3: Slow fluctuation, chi-squared with 4 DOF
            rcs *= np.random.gamma(2, 0.5)
        elif self.fluctuation_model == "swerling4":
            # Swerling 4: Fast fluctuation, chi-squared with 4 DOF
            rcs *= np.random.gamma(2, 0.5)
        elif self.fluctuation_model == "constant":
            # No fluctuation (Swerling 0)
            pass
            
        return rcs
    
    def get_position_spherical(self) -> Tuple[float, float, float]:
        """
        Get target position in spherical coordinates
        
        Returns:
            (range, azimuth, elevation) in (meters, radians, radians)
        """
        x, y, z = self.motion.position
        
        range_val = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.arctan2(y, x)
        elevation = np.arcsin(z / range_val) if range_val > 0 else 0
        
        return range_val, azimuth, elevation
    
    def get_radial_velocity(self, radar_position: np.ndarray = None) -> float:
        """
        Calculate radial velocity relative to radar
        
        Args:
            radar_position: Radar position (default is origin)
            
        Returns:
            Radial velocity in m/s (positive = approaching)
        """
        if radar_position is None:
            radar_position = np.zeros(3)
        
        # Vector from radar to target
        los_vector = self.motion.position - radar_position
        los_distance = np.linalg.norm(los_vector)
        
        if los_distance == 0:
            return 0
        
        # Unit vector along line of sight
        los_unit = los_vector / los_distance
        
        # Radial velocity is projection of velocity onto LOS
        radial_vel = -np.dot(self.motion.velocity, los_unit)
        
        return radial_vel
    
    def simulate_trajectory(self, duration: float, dt: float = 0.1) -> List[np.ndarray]:
        """
        Simulate target trajectory
        
        Args:
            duration: Simulation duration in seconds
            dt: Time step in seconds
            
        Returns:
            List of positions over time
        """
        trajectory = []
        t = 0
        
        # Store initial state
        initial_position = self.motion.position.copy()
        initial_velocity = self.motion.velocity.copy()
        
        while t < duration:
            trajectory.append(self.motion.position.copy())
            self.motion.update(dt)
            t += dt
        
        # Store trajectory history
        self.track_history = trajectory
        
        # Restore initial state
        self.motion.position = initial_position
        self.motion.velocity = initial_velocity
        
        return trajectory
    
    def apply_maneuver(self, maneuver_type: str, g_force: float = 1.0):
        """
        Apply a maneuver to the target
        
        Args:
            maneuver_type: Type of maneuver ('turn', 'climb', 'dive', 'accelerate')
            g_force: G-force magnitude
        """
        g = 9.81  # m/s^2
        acceleration_magnitude = g_force * g
        
        if maneuver_type == "turn":
            # Horizontal turn
            heading = np.arctan2(self.motion.velocity[1], self.motion.velocity[0])
            self.motion.acceleration[0] = -acceleration_magnitude * np.sin(heading)
            self.motion.acceleration[1] = acceleration_magnitude * np.cos(heading)
            
        elif maneuver_type == "climb":
            # Vertical climb
            self.motion.acceleration[2] = acceleration_magnitude
            
        elif maneuver_type == "dive":
            # Vertical dive
            self.motion.acceleration[2] = -acceleration_magnitude
            
        elif maneuver_type == "accelerate":
            # Accelerate along current velocity vector
            if self.motion.speed > 0:
                vel_unit = self.motion.velocity / self.motion.speed
                self.motion.acceleration = acceleration_magnitude * vel_unit
            
    def is_in_radar_coverage(self, max_range: float, 
                           min_elevation: float = -np.pi/6,
                           max_elevation: float = np.pi/3,
                           azimuth_limits: Optional[Tuple[float, float]] = None) -> bool:
        """
        Check if target is within radar coverage
        
        Args:
            max_range: Maximum radar range in meters
            min_elevation: Minimum elevation angle in radians
            max_elevation: Maximum elevation angle in radians
            azimuth_limits: Optional (min, max) azimuth limits in radians
            
        Returns:
            True if target is in coverage
        """
        range_val, azimuth, elevation = self.get_position_spherical()
        
        # Check range
        if range_val > max_range:
            return False
        
        # Check elevation
        if elevation < min_elevation or elevation > max_elevation:
            return False
        
        # Check azimuth if limits provided
        if azimuth_limits is not None:
            min_az, max_az = azimuth_limits
            # Handle angle wrap-around
            if min_az <= max_az:
                if azimuth < min_az or azimuth > max_az:
                    return False
            else:  # Wrapped around
                if azimuth < min_az and azimuth > max_az:
                    return False
        
        return True


class TargetGenerator:
    """Generate multiple targets for simulation scenarios"""
    
    @staticmethod
    def create_formation(num_targets: int, 
                        formation_type: str = "line",
                        spacing: float = 100.0,
                        base_position: np.ndarray = None,
                        base_velocity: np.ndarray = None) -> List[Target]:
        """
        Create a formation of targets
        
        Args:
            num_targets: Number of targets
            formation_type: Type of formation ('line', 'wedge', 'diamond')
            spacing: Spacing between targets in meters
            base_position: Base position for formation
            base_velocity: Base velocity for all targets
            
        Returns:
            List of targets in formation
        """
        if base_position is None:
            base_position = np.array([5000.0, 0.0, 3000.0])
        if base_velocity is None:
            base_velocity = np.array([200.0, 0.0, 0.0])
        
        targets = []
        
        if formation_type == "line":
            # Line abreast formation
            for i in range(num_targets):
                offset = (i - num_targets // 2) * spacing
                position = base_position + np.array([0, offset, 0])
                motion = TargetMotion(position, base_velocity.copy())
                targets.append(Target(TargetType.AIRCRAFT, motion=motion))
                
        elif formation_type == "wedge":
            # V formation
            for i in range(num_targets):
                if i == 0:
                    position = base_position
                else:
                    side = 1 if i % 2 == 1 else -1
                    row = (i + 1) // 2
                    position = base_position + np.array([-row * spacing, side * row * spacing, 0])
                motion = TargetMotion(position, base_velocity.copy())
                targets.append(Target(TargetType.AIRCRAFT, motion=motion))
                
        elif formation_type == "diamond":
            # Diamond formation
            positions = [
                np.array([0, 0, 0]),
                np.array([-spacing, spacing, 0]),
                np.array([-spacing, -spacing, 0]),
                np.array([-2*spacing, 0, 0])
            ]
            for i in range(min(num_targets, 4)):
                position = base_position + positions[i]
                motion = TargetMotion(position, base_velocity.copy())
                targets.append(Target(TargetType.AIRCRAFT, motion=motion))
        
        return targets
    
    @staticmethod
    def create_random_scenario(num_targets: int,
                              range_limits: Tuple[float, float] = (1000, 10000),
                              altitude_limits: Tuple[float, float] = (100, 10000),
                              speed_limits: Tuple[float, float] = (50, 300)) -> List[Target]:
        """
        Create random targets within specified limits
        
        Args:
            num_targets: Number of targets to create
            range_limits: Min and max range in meters
            altitude_limits: Min and max altitude in meters
            speed_limits: Min and max speed in m/s
            
        Returns:
            List of random targets
        """
        targets = []
        target_types = list(TargetType)
        
        for _ in range(num_targets):
            # Random position
            range_val = np.random.uniform(*range_limits)
            azimuth = np.random.uniform(-np.pi, np.pi)
            altitude = np.random.uniform(*altitude_limits)
            
            x = range_val * np.cos(azimuth)
            y = range_val * np.sin(azimuth)
            z = altitude
            
            # Random velocity
            speed = np.random.uniform(*speed_limits)
            heading = np.random.uniform(-np.pi, np.pi)
            climb_angle = np.random.uniform(-np.pi/6, np.pi/6)
            
            vx = speed * np.cos(heading) * np.cos(climb_angle)
            vy = speed * np.sin(heading) * np.cos(climb_angle)
            vz = speed * np.sin(climb_angle)
            
            # Random target type
            target_type = np.random.choice(target_types)
            
            motion = TargetMotion(
                position=np.array([x, y, z]),
                velocity=np.array([vx, vy, vz])
            )
            
            targets.append(Target(target_type, motion=motion))
        
        return targets