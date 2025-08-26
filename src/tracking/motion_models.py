"""
Motion models for radar target tracking

This module provides various motion models for target tracking in radar systems,
including constant velocity, constant acceleration, coordinated turn, and Singer
acceleration models. All models support 2D and 3D coordinate systems and include
proper Jacobians for Extended Kalman Filter (EKF) implementations.

Author: RadarSim Project
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum
import warnings


class CoordinateSystem(Enum):
    """Enumeration of supported coordinate systems"""
    CARTESIAN_2D = "cartesian_2d"  # [x, y, vx, vy] or [x, y, vx, vy, ax, ay]
    CARTESIAN_3D = "cartesian_3d"  # [x, y, z, vx, vy, vz] or [x, y, z, vx, vy, vz, ax, ay, az]
    POLAR = "polar"                # [range, azimuth, range_rate, azimuth_rate]
    SPHERICAL = "spherical"        # [range, azimuth, elevation, range_rate, azimuth_rate, elevation_rate]


@dataclass
class ModelParameters:
    """Parameters for motion models"""
    dt: float                      # Time step
    process_noise_std: float = 1.0 # Standard deviation of process noise
    coordinate_system: CoordinateSystem = CoordinateSystem.CARTESIAN_2D
    additional_params: Dict[str, Any] = None  # Model-specific parameters
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


class MotionModel(ABC):
    """
    Abstract base class for motion models
    
    This class defines the interface for all motion models used in target tracking.
    Each model must implement methods for state transition, process noise covariance,
    and Jacobian computation for EKF applications.
    """
    
    def __init__(self, params: ModelParameters):
        """
        Initialize motion model
        
        Args:
            params: Model parameters including time step and coordinate system
        """
        self.params = params
        self.dt = params.dt
        self.coordinate_system = params.coordinate_system
        self.state_dim = self._get_state_dimension()
        
    @abstractmethod
    def _get_state_dimension(self) -> int:
        """Get the dimension of the state vector"""
        pass
    
    @abstractmethod
    def get_transition_matrix(self, state: np.ndarray) -> np.ndarray:
        """
        Get state transition matrix F
        
        Args:
            state: Current state vector
            
        Returns:
            State transition matrix F
        """
        pass
    
    @abstractmethod
    def get_process_noise_covariance(self, state: np.ndarray) -> np.ndarray:
        """
        Get process noise covariance matrix Q
        
        Args:
            state: Current state vector
            
        Returns:
            Process noise covariance matrix Q
        """
        pass
    
    @abstractmethod
    def predict_state(self, state: np.ndarray) -> np.ndarray:
        """
        Predict next state using motion model
        
        Args:
            state: Current state vector
            
        Returns:
            Predicted state vector
        """
        pass
    
    def get_jacobian(self, state: np.ndarray) -> np.ndarray:
        """
        Get Jacobian matrix for nonlinear models (default: return transition matrix)
        
        Args:
            state: Current state vector
            
        Returns:
            Jacobian matrix
        """
        return self.get_transition_matrix(state)
    
    def validate_state(self, state: np.ndarray) -> bool:
        """
        Validate state vector dimensions
        
        Args:
            state: State vector to validate
            
        Returns:
            True if state is valid
        """
        return len(state) == self.state_dim


class ConstantVelocityModel(MotionModel):
    """
    Constant Velocity (CV) motion model
    
    Assumes target moves with constant velocity. Works in 2D and 3D Cartesian coordinates.
    
    State vectors:
    - 2D: [x, y, vx, vy]
    - 3D: [x, y, z, vx, vy, vz]
    """
    
    def _get_state_dimension(self) -> int:
        """Get state dimension based on coordinate system"""
        if self.coordinate_system == CoordinateSystem.CARTESIAN_2D:
            return 4
        elif self.coordinate_system == CoordinateSystem.CARTESIAN_3D:
            return 6
        else:
            raise ValueError(f"Unsupported coordinate system: {self.coordinate_system}")
    
    def get_transition_matrix(self, state: np.ndarray) -> np.ndarray:
        """
        Get state transition matrix for constant velocity model
        
        Args:
            state: Current state vector (not used in linear model)
            
        Returns:
            State transition matrix F
        """
        if self.coordinate_system == CoordinateSystem.CARTESIAN_2D:
            F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        elif self.coordinate_system == CoordinateSystem.CARTESIAN_3D:
            F = np.array([
                [1, 0, 0, self.dt, 0, 0],
                [0, 1, 0, 0, self.dt, 0],
                [0, 0, 1, 0, 0, self.dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])
        else:
            raise ValueError(f"Unsupported coordinate system: {self.coordinate_system}")
        
        return F
    
    def get_process_noise_covariance(self, state: np.ndarray) -> np.ndarray:
        """
        Get process noise covariance matrix for constant velocity model
        
        Args:
            state: Current state vector (not used)
            
        Returns:
            Process noise covariance matrix Q
        """
        sigma = self.params.process_noise_std
        dt = self.dt
        
        if self.coordinate_system == CoordinateSystem.CARTESIAN_2D:
            # Discrete white noise acceleration model
            Q = np.array([
                [dt**4/4, 0, dt**3/2, 0],
                [0, dt**4/4, 0, dt**3/2],
                [dt**3/2, 0, dt**2, 0],
                [0, dt**3/2, 0, dt**2]
            ]) * sigma**2
        elif self.coordinate_system == CoordinateSystem.CARTESIAN_3D:
            Q = np.array([
                [dt**4/4, 0, 0, dt**3/2, 0, 0],
                [0, dt**4/4, 0, 0, dt**3/2, 0],
                [0, 0, dt**4/4, 0, 0, dt**3/2],
                [dt**3/2, 0, 0, dt**2, 0, 0],
                [0, dt**3/2, 0, 0, dt**2, 0],
                [0, 0, dt**3/2, 0, 0, dt**2]
            ]) * sigma**2
        else:
            raise ValueError(f"Unsupported coordinate system: {self.coordinate_system}")
        
        return Q
    
    def predict_state(self, state: np.ndarray) -> np.ndarray:
        """
        Predict next state using constant velocity model
        
        Args:
            state: Current state vector
            
        Returns:
            Predicted state vector
        """
        F = self.get_transition_matrix(state)
        return F @ state


class ConstantAccelerationModel(MotionModel):
    """
    Constant Acceleration (CA) motion model
    
    Assumes target moves with constant acceleration. Works in 2D and 3D Cartesian coordinates.
    
    State vectors:
    - 2D: [x, y, vx, vy, ax, ay]
    - 3D: [x, y, z, vx, vy, vz, ax, ay, az]
    """
    
    def _get_state_dimension(self) -> int:
        """Get state dimension based on coordinate system"""
        if self.coordinate_system == CoordinateSystem.CARTESIAN_2D:
            return 6
        elif self.coordinate_system == CoordinateSystem.CARTESIAN_3D:
            return 9
        else:
            raise ValueError(f"Unsupported coordinate system: {self.coordinate_system}")
    
    def get_transition_matrix(self, state: np.ndarray) -> np.ndarray:
        """
        Get state transition matrix for constant acceleration model
        
        Args:
            state: Current state vector (not used in linear model)
            
        Returns:
            State transition matrix F
        """
        dt = self.dt
        
        if self.coordinate_system == CoordinateSystem.CARTESIAN_2D:
            F = np.array([
                [1, 0, dt, 0, dt**2/2, 0],
                [0, 1, 0, dt, 0, dt**2/2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])
        elif self.coordinate_system == CoordinateSystem.CARTESIAN_3D:
            F = np.array([
                [1, 0, 0, dt, 0, 0, dt**2/2, 0, 0],
                [0, 1, 0, 0, dt, 0, 0, dt**2/2, 0],
                [0, 0, 1, 0, 0, dt, 0, 0, dt**2/2],
                [0, 0, 0, 1, 0, 0, dt, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, dt, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, dt],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1]
            ])
        else:
            raise ValueError(f"Unsupported coordinate system: {self.coordinate_system}")
        
        return F
    
    def get_process_noise_covariance(self, state: np.ndarray) -> np.ndarray:
        """
        Get process noise covariance matrix for constant acceleration model
        
        Args:
            state: Current state vector (not used)
            
        Returns:
            Process noise covariance matrix Q
        """
        sigma = self.params.process_noise_std
        dt = self.dt
        
        if self.coordinate_system == CoordinateSystem.CARTESIAN_2D:
            # Discrete white noise jerk model
            Q = np.array([
                [dt**6/36, 0, dt**5/12, 0, dt**4/6, 0],
                [0, dt**6/36, 0, dt**5/12, 0, dt**4/6],
                [dt**5/12, 0, dt**4/4, 0, dt**3/2, 0],
                [0, dt**5/12, 0, dt**4/4, 0, dt**3/2],
                [dt**4/6, 0, dt**3/2, 0, dt**2, 0],
                [0, dt**4/6, 0, dt**3/2, 0, dt**2]
            ]) * sigma**2
        elif self.coordinate_system == CoordinateSystem.CARTESIAN_3D:
            Q = np.zeros((9, 9))
            # Fill diagonal blocks for each axis
            for i in range(3):
                base_idx = i * 3
                Q[base_idx:base_idx+3, base_idx:base_idx+3] = np.array([
                    [dt**6/36, dt**5/12, dt**4/6],
                    [dt**5/12, dt**4/4, dt**3/2],
                    [dt**4/6, dt**3/2, dt**2]
                ]) * sigma**2
        else:
            raise ValueError(f"Unsupported coordinate system: {self.coordinate_system}")
        
        return Q
    
    def predict_state(self, state: np.ndarray) -> np.ndarray:
        """
        Predict next state using constant acceleration model
        
        Args:
            state: Current state vector
            
        Returns:
            Predicted state vector
        """
        F = self.get_transition_matrix(state)
        return F @ state


class CoordinatedTurnModel(MotionModel):
    """
    Coordinated Turn (CT) motion model
    
    Models target motion with coordinated turns at constant speed and turn rate.
    Supports both known and unknown turn rate scenarios.
    
    State vectors:
    - 2D with known turn rate: [x, y, vx, vy]
    - 2D with unknown turn rate: [x, y, vx, vy, omega]
    - 3D: [x, y, z, vx, vy, vz, omega] (turn in horizontal plane)
    """
    
    def __init__(self, params: ModelParameters):
        """
        Initialize coordinated turn model
        
        Args:
            params: Model parameters. additional_params should contain:
                   - 'turn_rate': Known turn rate in rad/s (if None, turn rate is estimated)
                   - 'turn_rate_noise': Standard deviation of turn rate noise
        """
        self.known_turn_rate = params.additional_params.get('turn_rate', None)
        self.turn_rate_noise = params.additional_params.get('turn_rate_noise', 0.1)
        super().__init__(params)
    
    def _get_state_dimension(self) -> int:
        """Get state dimension based on coordinate system and turn rate knowledge"""
        if self.coordinate_system == CoordinateSystem.CARTESIAN_2D:
            return 4 if self.known_turn_rate is not None else 5
        elif self.coordinate_system == CoordinateSystem.CARTESIAN_3D:
            return 7  # Always estimate turn rate in 3D
        else:
            raise ValueError(f"Unsupported coordinate system: {self.coordinate_system}")
    
    def get_transition_matrix(self, state: np.ndarray) -> np.ndarray:
        """
        Get state transition matrix for coordinated turn model
        
        Args:
            state: Current state vector
            
        Returns:
            State transition matrix F (linearized around current state)
        """
        dt = self.dt
        
        if self.coordinate_system == CoordinateSystem.CARTESIAN_2D:
            if self.known_turn_rate is not None:
                omega = self.known_turn_rate
            else:
                omega = state[4]  # Turn rate from state
            
            if abs(omega) < 1e-6:  # Straight line motion
                F = np.array([
                    [1, 0, dt, 0],
                    [0, 1, 0, dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                if self.known_turn_rate is None:
                    # Add turn rate dimension
                    F_aug = np.eye(5)
                    F_aug[:4, :4] = F
                    F = F_aug
            else:
                sin_omega_dt = np.sin(omega * dt)
                cos_omega_dt = np.cos(omega * dt)
                
                F = np.array([
                    [1, 0, sin_omega_dt/omega, -(1-cos_omega_dt)/omega],
                    [0, 1, (1-cos_omega_dt)/omega, sin_omega_dt/omega],
                    [0, 0, cos_omega_dt, -sin_omega_dt],
                    [0, 0, sin_omega_dt, cos_omega_dt]
                ])
                
                if self.known_turn_rate is None:
                    # Add partial derivatives w.r.t. turn rate
                    vx, vy = state[2], state[3]
                    F_aug = np.zeros((5, 5))
                    F_aug[:4, :4] = F
                    F_aug[4, 4] = 1  # Turn rate evolution
                    
                    # Partial derivatives w.r.t. omega
                    omega_sq = omega**2
                    F_aug[0, 4] = vx * (dt * cos_omega_dt - sin_omega_dt/omega) / omega + \
                                  vy * (dt * sin_omega_dt - (1-cos_omega_dt)/omega) / omega
                    F_aug[1, 4] = vx * (dt * sin_omega_dt - (1-cos_omega_dt)/omega) / omega + \
                                  vy * (dt * cos_omega_dt - sin_omega_dt/omega) / omega
                    F_aug[2, 4] = -vx * dt * sin_omega_dt - vy * dt * cos_omega_dt
                    F_aug[3, 4] = vx * dt * cos_omega_dt - vy * dt * sin_omega_dt
                    F = F_aug
        
        elif self.coordinate_system == CoordinateSystem.CARTESIAN_3D:
            omega = state[6]  # Turn rate from state
            
            if abs(omega) < 1e-6:  # Straight line motion
                F = np.eye(7)
                F[0, 3] = dt
                F[1, 4] = dt
                F[2, 5] = dt
            else:
                sin_omega_dt = np.sin(omega * dt)
                cos_omega_dt = np.cos(omega * dt)
                
                F = np.eye(7)
                # Position updates
                F[0, 3] = sin_omega_dt/omega
                F[0, 4] = -(1-cos_omega_dt)/omega
                F[1, 3] = (1-cos_omega_dt)/omega
                F[1, 4] = sin_omega_dt/omega
                F[2, 5] = dt  # z-axis is unaffected by horizontal turn
                
                # Velocity updates
                F[3, 3] = cos_omega_dt
                F[3, 4] = -sin_omega_dt
                F[4, 3] = sin_omega_dt
                F[4, 4] = cos_omega_dt
                
                # Partial derivatives w.r.t. turn rate
                vx, vy = state[3], state[4]
                omega_sq = omega**2
                F[0, 6] = vx * (dt * cos_omega_dt - sin_omega_dt/omega) / omega + \
                          vy * (dt * sin_omega_dt - (1-cos_omega_dt)/omega) / omega
                F[1, 6] = vx * (dt * sin_omega_dt - (1-cos_omega_dt)/omega) / omega + \
                          vy * (dt * cos_omega_dt - sin_omega_dt/omega) / omega
                F[3, 6] = -vx * dt * sin_omega_dt - vy * dt * cos_omega_dt
                F[4, 6] = vx * dt * cos_omega_dt - vy * dt * sin_omega_dt
        
        return F
    
    def get_process_noise_covariance(self, state: np.ndarray) -> np.ndarray:
        """
        Get process noise covariance matrix for coordinated turn model
        
        Args:
            state: Current state vector
            
        Returns:
            Process noise covariance matrix Q
        """
        sigma = self.params.process_noise_std
        dt = self.dt
        
        if self.coordinate_system == CoordinateSystem.CARTESIAN_2D:
            if self.known_turn_rate is not None:
                # Process noise for position and velocity
                Q = np.array([
                    [dt**4/4, 0, dt**3/2, 0],
                    [0, dt**4/4, 0, dt**3/2],
                    [dt**3/2, 0, dt**2, 0],
                    [0, dt**3/2, 0, dt**2]
                ]) * sigma**2
            else:
                # Include turn rate noise
                Q = np.diag([
                    (dt**4/4) * sigma**2,  # x
                    (dt**4/4) * sigma**2,  # y
                    dt**2 * sigma**2,      # vx
                    dt**2 * sigma**2,      # vy
                    dt * self.turn_rate_noise**2  # omega
                ])
        
        elif self.coordinate_system == CoordinateSystem.CARTESIAN_3D:
            Q = np.diag([
                (dt**4/4) * sigma**2,  # x
                (dt**4/4) * sigma**2,  # y
                (dt**4/4) * sigma**2,  # z
                dt**2 * sigma**2,      # vx
                dt**2 * sigma**2,      # vy
                dt**2 * sigma**2,      # vz
                dt * self.turn_rate_noise**2  # omega
            ])
        
        return Q
    
    def predict_state(self, state: np.ndarray) -> np.ndarray:
        """
        Predict next state using coordinated turn model
        
        Args:
            state: Current state vector
            
        Returns:
            Predicted state vector
        """
        dt = self.dt
        predicted_state = state.copy()
        
        if self.coordinate_system == CoordinateSystem.CARTESIAN_2D:
            x, y, vx, vy = state[:4]
            
            if self.known_turn_rate is not None:
                omega = self.known_turn_rate
            else:
                omega = state[4]
            
            if abs(omega) < 1e-6:  # Straight line motion
                predicted_state[0] = x + vx * dt
                predicted_state[1] = y + vy * dt
            else:
                sin_omega_dt = np.sin(omega * dt)
                cos_omega_dt = np.cos(omega * dt)
                
                predicted_state[0] = x + (vx * sin_omega_dt - vy * (1 - cos_omega_dt)) / omega
                predicted_state[1] = y + (vx * (1 - cos_omega_dt) + vy * sin_omega_dt) / omega
                predicted_state[2] = vx * cos_omega_dt - vy * sin_omega_dt
                predicted_state[3] = vx * sin_omega_dt + vy * cos_omega_dt
        
        elif self.coordinate_system == CoordinateSystem.CARTESIAN_3D:
            x, y, z, vx, vy, vz, omega = state
            
            if abs(omega) < 1e-6:  # Straight line motion
                predicted_state[0] = x + vx * dt
                predicted_state[1] = y + vy * dt
                predicted_state[2] = z + vz * dt
            else:
                sin_omega_dt = np.sin(omega * dt)
                cos_omega_dt = np.cos(omega * dt)
                
                predicted_state[0] = x + (vx * sin_omega_dt - vy * (1 - cos_omega_dt)) / omega
                predicted_state[1] = y + (vx * (1 - cos_omega_dt) + vy * sin_omega_dt) / omega
                predicted_state[2] = z + vz * dt  # z-motion unaffected
                predicted_state[3] = vx * cos_omega_dt - vy * sin_omega_dt
                predicted_state[4] = vx * sin_omega_dt + vy * cos_omega_dt
                # vz and omega remain unchanged
        
        return predicted_state


class SingerAccelerationModel(MotionModel):
    """
    Singer acceleration model with adaptive maneuver detection
    
    Models target acceleration as a time-correlated random process with
    exponential autocorrelation. Includes adaptive maneuver detection
    to adjust process noise based on detected maneuvers.
    
    State vectors:
    - 2D: [x, y, vx, vy, ax, ay]
    - 3D: [x, y, z, vx, vy, vz, ax, ay, az]
    """
    
    def __init__(self, params: ModelParameters):
        """
        Initialize Singer acceleration model
        
        Args:
            params: Model parameters. additional_params should contain:
                   - 'alpha': Reciprocal of maneuver time constant (1/tau)
                   - 'max_acceleration': Maximum expected acceleration magnitude
                   - 'maneuver_threshold': Threshold for maneuver detection
        """
        self.alpha = params.additional_params.get('alpha', 0.1)  # 1/tau
        self.max_acceleration = params.additional_params.get('max_acceleration', 5.0)  # m/s^2
        self.maneuver_threshold = params.additional_params.get('maneuver_threshold', 2.0)
        self.maneuver_detected = False
        super().__init__(params)
        
    def _get_state_dimension(self) -> int:
        """Get state dimension based on coordinate system"""
        if self.coordinate_system == CoordinateSystem.CARTESIAN_2D:
            return 6
        elif self.coordinate_system == CoordinateSystem.CARTESIAN_3D:
            return 9
        else:
            raise ValueError(f"Unsupported coordinate system: {self.coordinate_system}")
    
    def get_transition_matrix(self, state: np.ndarray) -> np.ndarray:
        """
        Get state transition matrix for Singer model
        
        Args:
            state: Current state vector
            
        Returns:
            State transition matrix F
        """
        dt = self.dt
        alpha = self.alpha
        
        # Singer model components
        exp_alpha_dt = np.exp(-alpha * dt)
        c1 = (1 - exp_alpha_dt) / alpha
        c2 = (alpha * dt - 1 + exp_alpha_dt) / (alpha**2)
        
        if self.coordinate_system == CoordinateSystem.CARTESIAN_2D:
            F = np.array([
                [1, 0, dt, 0, c2, 0],
                [0, 1, 0, dt, 0, c2],
                [0, 0, 1, 0, c1, 0],
                [0, 0, 0, 1, 0, c1],
                [0, 0, 0, 0, exp_alpha_dt, 0],
                [0, 0, 0, 0, 0, exp_alpha_dt]
            ])
        elif self.coordinate_system == CoordinateSystem.CARTESIAN_3D:
            F = np.array([
                [1, 0, 0, dt, 0, 0, c2, 0, 0],
                [0, 1, 0, 0, dt, 0, 0, c2, 0],
                [0, 0, 1, 0, 0, dt, 0, 0, c2],
                [0, 0, 0, 1, 0, 0, c1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, c1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, c1],
                [0, 0, 0, 0, 0, 0, exp_alpha_dt, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, exp_alpha_dt, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, exp_alpha_dt]
            ])
        
        return F
    
    def get_process_noise_covariance(self, state: np.ndarray) -> np.ndarray:
        """
        Get process noise covariance matrix for Singer model
        
        Args:
            state: Current state vector
            
        Returns:
            Process noise covariance matrix Q
        """
        dt = self.dt
        alpha = self.alpha
        sigma_a = self.max_acceleration
        
        # Adjust noise based on maneuver detection
        if self.maneuver_detected:
            sigma_a *= 2.0  # Increase noise during maneuvers
        
        # Singer model noise components
        exp_alpha_dt = np.exp(-alpha * dt)
        exp_2alpha_dt = np.exp(-2 * alpha * dt)
        
        q11 = (dt**5/20) - (dt**4/(6*alpha)) + (dt**3/(2*alpha**2)) - \
              (dt**2/(2*alpha**3)) + (1/(2*alpha**4)) - (exp_alpha_dt/(2*alpha**4))
        q12 = (dt**4/8) - (dt**3/(3*alpha)) + (dt**2/(2*alpha**2)) - \
              (1/(2*alpha**3)) + (exp_alpha_dt/(2*alpha**3))
        q13 = (dt**3/6) - (dt**2/(2*alpha)) + (1/(2*alpha**2)) - (exp_alpha_dt/(2*alpha**2))
        q22 = (dt**3/3) - (dt**2/alpha) + (1/alpha**2) - (exp_2alpha_dt/(2*alpha**2))
        q23 = (dt**2/2) - (1/alpha) + (exp_alpha_dt/alpha)
        q33 = 1 - exp_2alpha_dt
        
        # Scale by acceleration variance
        variance = (2 * alpha * sigma_a**2)
        
        if self.coordinate_system == CoordinateSystem.CARTESIAN_2D:
            Q = np.zeros((6, 6))
            # Fill blocks for x and y axes
            for i in range(2):
                base = i * 3
                Q[base:base+3, base:base+3] = np.array([
                    [q11, q12, q13],
                    [q12, q22, q23],
                    [q13, q23, q33]
                ]) * variance
        
        elif self.coordinate_system == CoordinateSystem.CARTESIAN_3D:
            Q = np.zeros((9, 9))
            # Fill blocks for x, y, and z axes
            for i in range(3):
                base = i * 3
                Q[base:base+3, base:base+3] = np.array([
                    [q11, q12, q13],
                    [q12, q22, q23],
                    [q13, q23, q33]
                ]) * variance
        
        return Q
    
    def predict_state(self, state: np.ndarray) -> np.ndarray:
        """
        Predict next state using Singer model
        
        Args:
            state: Current state vector
            
        Returns:
            Predicted state vector
        """
        F = self.get_transition_matrix(state)
        return F @ state
    
    def detect_maneuver(self, innovation: np.ndarray, innovation_covariance: np.ndarray) -> bool:
        """
        Detect maneuvers using innovation sequence
        
        Args:
            innovation: Innovation vector from Kalman filter
            innovation_covariance: Innovation covariance matrix
            
        Returns:
            True if maneuver is detected
        """
        try:
            # Normalized innovation squared (NIS) test
            inv_cov = np.linalg.inv(innovation_covariance)
            nis = innovation.T @ inv_cov @ innovation
            
            # Chi-squared test with degrees of freedom equal to measurement dimension
            threshold = self.maneuver_threshold * len(innovation)
            self.maneuver_detected = nis > threshold
            
        except np.linalg.LinAlgError:
            # Fallback if covariance is singular
            self.maneuver_detected = np.linalg.norm(innovation) > self.maneuver_threshold
        
        return self.maneuver_detected


# Utility functions for coordinate transformations and motion analysis

def cartesian_to_polar(x: float, y: float, vx: float = 0, vy: float = 0) -> Tuple[float, float, float, float]:
    """
    Convert 2D Cartesian coordinates to polar coordinates
    
    Args:
        x, y: Cartesian position
        vx, vy: Cartesian velocity (optional)
        
    Returns:
        (range, azimuth, range_rate, azimuth_rate) in (m, rad, m/s, rad/s)
    """
    range_val = np.sqrt(x**2 + y**2)
    azimuth = np.arctan2(y, x)
    
    if range_val > 0:
        range_rate = (x * vx + y * vy) / range_val
        azimuth_rate = (x * vy - y * vx) / (range_val**2)
    else:
        range_rate = 0
        azimuth_rate = 0
    
    return range_val, azimuth, range_rate, azimuth_rate


def polar_to_cartesian(range_val: float, azimuth: float, 
                      range_rate: float = 0, azimuth_rate: float = 0) -> Tuple[float, float, float, float]:
    """
    Convert polar coordinates to 2D Cartesian coordinates
    
    Args:
        range_val: Range in meters
        azimuth: Azimuth in radians
        range_rate: Range rate in m/s (optional)
        azimuth_rate: Azimuth rate in rad/s (optional)
        
    Returns:
        (x, y, vx, vy) in (m, m, m/s, m/s)
    """
    x = range_val * np.cos(azimuth)
    y = range_val * np.sin(azimuth)
    
    vx = range_rate * np.cos(azimuth) - range_val * azimuth_rate * np.sin(azimuth)
    vy = range_rate * np.sin(azimuth) + range_val * azimuth_rate * np.cos(azimuth)
    
    return x, y, vx, vy


def cartesian_to_spherical(x: float, y: float, z: float, 
                          vx: float = 0, vy: float = 0, vz: float = 0) -> Tuple[float, float, float, float, float, float]:
    """
    Convert 3D Cartesian coordinates to spherical coordinates
    
    Args:
        x, y, z: Cartesian position
        vx, vy, vz: Cartesian velocity (optional)
        
    Returns:
        (range, azimuth, elevation, range_rate, azimuth_rate, elevation_rate)
        in (m, rad, rad, m/s, rad/s, rad/s)
    """
    range_val = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(y, x)
    
    xy_range = np.sqrt(x**2 + y**2)
    elevation = np.arctan2(z, xy_range) if xy_range > 0 else (np.pi/2 if z > 0 else -np.pi/2)
    
    if range_val > 0:
        range_rate = (x * vx + y * vy + z * vz) / range_val
        
        if xy_range > 0:
            azimuth_rate = (x * vy - y * vx) / (xy_range**2)
            elevation_rate = (xy_range * vz - z * (x * vx + y * vy) / xy_range) / (range_val**2)
        else:
            azimuth_rate = 0
            elevation_rate = 0
    else:
        range_rate = 0
        azimuth_rate = 0
        elevation_rate = 0
    
    return range_val, azimuth, elevation, range_rate, azimuth_rate, elevation_rate


def spherical_to_cartesian(range_val: float, azimuth: float, elevation: float,
                          range_rate: float = 0, azimuth_rate: float = 0, 
                          elevation_rate: float = 0) -> Tuple[float, float, float, float, float, float]:
    """
    Convert spherical coordinates to 3D Cartesian coordinates
    
    Args:
        range_val: Range in meters
        azimuth: Azimuth in radians
        elevation: Elevation in radians
        range_rate: Range rate in m/s (optional)
        azimuth_rate: Azimuth rate in rad/s (optional)
        elevation_rate: Elevation rate in rad/s (optional)
        
    Returns:
        (x, y, z, vx, vy, vz) in (m, m, m, m/s, m/s, m/s)
    """
    cos_el = np.cos(elevation)
    sin_el = np.sin(elevation)
    cos_az = np.cos(azimuth)
    sin_az = np.sin(azimuth)
    
    x = range_val * cos_el * cos_az
    y = range_val * cos_el * sin_az
    z = range_val * sin_el
    
    vx = (range_rate * cos_el * cos_az - 
          range_val * elevation_rate * sin_el * cos_az -
          range_val * azimuth_rate * cos_el * sin_az)
    
    vy = (range_rate * cos_el * sin_az - 
          range_val * elevation_rate * sin_el * sin_az +
          range_val * azimuth_rate * cos_el * cos_az)
    
    vz = (range_rate * sin_el + 
          range_val * elevation_rate * cos_el)
    
    return x, y, z, vx, vy, vz


def compute_turn_rate(state: np.ndarray, coordinate_system: CoordinateSystem) -> float:
    """
    Compute turn rate from state vector
    
    Args:
        state: State vector
        coordinate_system: Coordinate system of the state
        
    Returns:
        Turn rate in rad/s
    """
    if coordinate_system == CoordinateSystem.CARTESIAN_2D:
        if len(state) >= 4:
            vx, vy = state[2], state[3]
            speed = np.sqrt(vx**2 + vy**2)
            if speed > 1e-6:
                # Assuming constant speed coordinated turn
                if len(state) >= 6:  # Has acceleration
                    ax, ay = state[4], state[5]
                    # Centripetal acceleration gives turn rate
                    a_centripetal = abs(vx * ay - vy * ax) / speed
                    return a_centripetal / speed
            return 0.0
    elif coordinate_system == CoordinateSystem.CARTESIAN_3D:
        if len(state) >= 6:
            vx, vy = state[3], state[4]
            speed_horizontal = np.sqrt(vx**2 + vy**2)
            if speed_horizontal > 1e-6:
                if len(state) >= 9:  # Has acceleration
                    ax, ay = state[6], state[7]
                    a_centripetal = abs(vx * ay - vy * ax) / speed_horizontal
                    return a_centripetal / speed_horizontal
            return 0.0
    
    return 0.0


def estimate_adaptive_process_noise(innovation_history: list, 
                                  window_size: int = 10, 
                                  base_noise: float = 1.0) -> float:
    """
    Estimate adaptive process noise based on innovation history
    
    Args:
        innovation_history: List of recent innovation vectors
        window_size: Size of the moving window for estimation
        base_noise: Base process noise level
        
    Returns:
        Adapted process noise standard deviation
    """
    if len(innovation_history) < 2:
        return base_noise
    
    # Use recent innovations
    recent_innovations = innovation_history[-window_size:]
    
    # Compute average innovation magnitude
    avg_innovation = np.mean([np.linalg.norm(innov) for innov in recent_innovations])
    
    # Adaptive scaling based on innovation magnitude
    # Higher innovations suggest model mismatch, increase process noise
    scaling_factor = 1.0 + np.tanh(avg_innovation - 1.0)  # Smooth scaling
    
    return base_noise * scaling_factor


def get_measurement_jacobian(state: np.ndarray, 
                           measurement_type: str = "cartesian") -> np.ndarray:
    """
    Get measurement Jacobian matrix for different measurement types
    
    Args:
        state: Current state vector [x, y, z?, vx, vy, vz?, ax?, ay?, az?]
        measurement_type: Type of measurement ("cartesian", "polar", "spherical")
        
    Returns:
        Measurement Jacobian matrix H
    """
    if measurement_type == "cartesian":
        # Direct position measurements
        if len(state) == 4:  # 2D CV
            H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        elif len(state) == 6 and state.shape[0] == 6:  # 2D CA or 3D CV
            if np.allclose(state[2:], 0):  # Likely 3D CV
                H = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0]])
            else:  # 2D CA
                H = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0]])
        elif len(state) == 9:  # 3D CA
            H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0]])
        else:
            raise ValueError(f"Unsupported state dimension for cartesian measurements: {len(state)}")
    
    elif measurement_type == "polar":
        # Polar measurements: [range, azimuth]
        x, y = state[0], state[1]
        range_val = np.sqrt(x**2 + y**2)
        
        if range_val < 1e-6:
            # Avoid division by zero
            H = np.zeros((2, len(state)))
        else:
            H = np.zeros((2, len(state)))
            # Range derivatives
            H[0, 0] = x / range_val
            H[0, 1] = y / range_val
            # Azimuth derivatives
            H[1, 0] = -y / (range_val**2)
            H[1, 1] = x / (range_val**2)
    
    elif measurement_type == "spherical":
        # Spherical measurements: [range, azimuth, elevation]
        x, y, z = state[0], state[1], state[2]
        range_val = np.sqrt(x**2 + y**2 + z**2)
        xy_range = np.sqrt(x**2 + y**2)
        
        if range_val < 1e-6:
            H = np.zeros((3, len(state)))
        else:
            H = np.zeros((3, len(state)))
            # Range derivatives
            H[0, 0] = x / range_val
            H[0, 1] = y / range_val
            H[0, 2] = z / range_val
            
            if xy_range > 1e-6:
                # Azimuth derivatives
                H[1, 0] = -y / (xy_range**2)
                H[1, 1] = x / (xy_range**2)
                # Elevation derivatives
                H[2, 0] = -x * z / (range_val**2 * xy_range)
                H[2, 1] = -y * z / (range_val**2 * xy_range)
                H[2, 2] = xy_range / (range_val**2)
    
    else:
        raise ValueError(f"Unsupported measurement type: {measurement_type}")
    
    return H


# Model factory function
def create_motion_model(model_type: str, params: ModelParameters) -> MotionModel:
    """
    Factory function to create motion models
    
    Args:
        model_type: Type of motion model ("cv", "ca", "ct", "singer")
        params: Model parameters
        
    Returns:
        Motion model instance
    """
    model_map = {
        "cv": ConstantVelocityModel,
        "ca": ConstantAccelerationModel,
        "ct": CoordinatedTurnModel,
        "singer": SingerAccelerationModel
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_map.keys())}")
    
    return model_map[model_type](params)