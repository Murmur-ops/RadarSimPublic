"""
Comprehensive Kalman Filter implementations for radar tracking.

This module provides implementations of various Kalman filter variants:
- Standard Kalman Filter (KF) for linear systems
- Extended Kalman Filter (EKF) for nonlinear systems
- Unscented Kalman Filter (UKF) for highly nonlinear systems

Author: RadarSim Project
"""

import numpy as np
from typing import Callable, Tuple, Optional, Union
from abc import ABC, abstractmethod
import warnings


class BaseKalmanFilter(ABC):
    """Abstract base class for all Kalman filter implementations."""
    
    def __init__(self, dim_x: int, dim_z: int):
        """
        Initialize base Kalman filter.
        
        Args:
            dim_x: Dimension of state vector
            dim_z: Dimension of measurement vector
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        # State vector and covariance
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        
        # Process noise covariance
        self.Q = np.eye(dim_x)
        
        # Measurement noise covariance
        self.R = np.eye(dim_z)
        
        # Innovation and likelihood
        self.y = np.zeros(dim_z)  # Innovation (residual)
        self.S = np.eye(dim_z)    # Innovation covariance
        self.log_likelihood = 0.0
        
        # Mahalanobis distance for gating
        self.mahalanobis_distance = 0.0
        
    @abstractmethod
    def predict(self, dt: float, u: Optional[np.ndarray] = None) -> None:
        """Predict the next state."""
        pass
    
    @abstractmethod
    def update(self, z: np.ndarray) -> None:
        """Update state with measurement."""
        pass
    
    def compute_innovation(self, z: np.ndarray, Hx: np.ndarray) -> None:
        """Compute innovation and its covariance."""
        self.y = z - Hx
        
    def compute_log_likelihood(self) -> float:
        """Compute log-likelihood of the current measurement."""
        try:
            # Compute log-likelihood using multivariate normal distribution
            det_S = np.linalg.det(self.S)
            if det_S <= 0:
                return -np.inf
            
            inv_S = np.linalg.inv(self.S)
            self.log_likelihood = -0.5 * (
                len(self.y) * np.log(2 * np.pi) +
                np.log(det_S) +
                self.y.T @ inv_S @ self.y
            )
            return self.log_likelihood
        except np.linalg.LinAlgError:
            return -np.inf
    
    def mahalanobis(self, z: np.ndarray, Hx: np.ndarray) -> float:
        """Compute Mahalanobis distance for gating."""
        y = z - Hx
        try:
            inv_S = np.linalg.inv(self.S)
            self.mahalanobis_distance = np.sqrt(y.T @ inv_S @ y)
            return self.mahalanobis_distance
        except np.linalg.LinAlgError:
            return np.inf
    
    def is_measurement_valid(self, z: np.ndarray, Hx: np.ndarray, 
                           gate_threshold: float = 9.21) -> bool:
        """
        Validate measurement using chi-squared gating.
        
        Args:
            z: Measurement vector
            Hx: Predicted measurement
            gate_threshold: Chi-squared threshold (default for 95% confidence, 2 DOF)
            
        Returns:
            True if measurement passes gating test
        """
        distance = self.mahalanobis(z, Hx)
        return distance <= gate_threshold


class KalmanFilter(BaseKalmanFilter):
    """
    Standard Kalman Filter for linear systems.
    
    Assumes linear dynamics: x(k+1) = F @ x(k) + B @ u(k) + w(k)
    And linear measurements: z(k) = H @ x(k) + v(k)
    """
    
    def __init__(self, dim_x: int, dim_z: int, dim_u: int = 0):
        """
        Initialize Kalman Filter.
        
        Args:
            dim_x: Dimension of state vector
            dim_z: Dimension of measurement vector
            dim_u: Dimension of control input (default: 0)
        """
        super().__init__(dim_x, dim_z)
        self.dim_u = dim_u
        
        # System matrices
        self.F = np.eye(dim_x)  # State transition matrix
        self.H = np.eye(dim_z, dim_x)  # Measurement matrix
        self.B = None  # Control matrix
        
        if dim_u > 0:
            self.B = np.zeros((dim_x, dim_u))
        
        # Kalman gain
        self.K = np.zeros((dim_x, dim_z))
        
        # Prior state and covariance (for storage)
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        
    def predict(self, dt: float, u: Optional[np.ndarray] = None) -> None:
        """
        Predict the next state using linear dynamics.
        
        Args:
            dt: Time step (used for time-varying F matrix)
            u: Control input vector
        """
        # Store prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        
        # State prediction
        self.x = self.F @ self.x
        if self.B is not None and u is not None:
            self.x += self.B @ u
            
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, z: np.ndarray) -> None:
        """
        Update state estimate with measurement.
        
        Args:
            z: Measurement vector
        """
        # Predicted measurement
        Hx = self.H @ self.x
        
        # Innovation
        self.compute_innovation(z, Hx)
        
        # Innovation covariance
        self.S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        try:
            self.K = self.P @ self.H.T @ np.linalg.inv(self.S)
        except np.linalg.LinAlgError:
            warnings.warn("Singular innovation covariance matrix")
            return
        
        # State update
        self.x = self.x + self.K @ self.y
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.dim_x) - self.K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + self.K @ self.R @ self.K.T
        
        # Compute likelihood
        self.compute_log_likelihood()


class ExtendedKalmanFilter(BaseKalmanFilter):
    """
    Extended Kalman Filter for nonlinear systems.
    
    Handles nonlinear dynamics: x(k+1) = f(x(k), u(k)) + w(k)
    And nonlinear measurements: z(k) = h(x(k)) + v(k)
    """
    
    def __init__(self, dim_x: int, dim_z: int, dt: float = 1.0):
        """
        Initialize Extended Kalman Filter.
        
        Args:
            dim_x: Dimension of state vector
            dim_z: Dimension of measurement vector
            dt: Time step
        """
        super().__init__(dim_x, dim_z)
        self.dt = dt
        
        # Nonlinear functions (to be set by user)
        self.f_func: Optional[Callable] = None  # State transition function
        self.h_func: Optional[Callable] = None  # Measurement function
        self.F_jac: Optional[Callable] = None   # Jacobian of f
        self.H_jac: Optional[Callable] = None   # Jacobian of h
        
        # Linearized matrices
        self.F = np.eye(dim_x)
        self.H = np.eye(dim_z, dim_x)
        
        # Kalman gain
        self.K = np.zeros((dim_x, dim_z))
        
        # Prior state and covariance
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
    
    def set_state_transition(self, f_func: Callable, F_jac: Callable) -> None:
        """
        Set nonlinear state transition function and its Jacobian.
        
        Args:
            f_func: State transition function f(x, u, dt)
            F_jac: Jacobian of f with respect to x
        """
        self.f_func = f_func
        self.F_jac = F_jac
    
    def set_measurement_function(self, h_func: Callable, H_jac: Callable) -> None:
        """
        Set nonlinear measurement function and its Jacobian.
        
        Args:
            h_func: Measurement function h(x)
            H_jac: Jacobian of h with respect to x
        """
        self.h_func = h_func
        self.H_jac = H_jac
    
    def predict(self, dt: float, u: Optional[np.ndarray] = None) -> None:
        """
        Predict using nonlinear state transition.
        
        Args:
            dt: Time step
            u: Control input
        """
        if self.f_func is None or self.F_jac is None:
            raise ValueError("State transition function and Jacobian must be set")
        
        # Store prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        
        # Nonlinear state prediction
        self.x = self.f_func(self.x, u, dt)
        
        # Linearize at current state
        self.F = self.F_jac(self.x_prior, u, dt)
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z: np.ndarray) -> None:
        """
        Update state using nonlinear measurement model.
        
        Args:
            z: Measurement vector
        """
        if self.h_func is None or self.H_jac is None:
            raise ValueError("Measurement function and Jacobian must be set")
        
        # Nonlinear measurement prediction
        Hx = self.h_func(self.x)
        
        # Linearize measurement model
        self.H = self.H_jac(self.x)
        
        # Innovation
        self.compute_innovation(z, Hx)
        
        # Innovation covariance
        self.S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        try:
            self.K = self.P @ self.H.T @ np.linalg.inv(self.S)
        except np.linalg.LinAlgError:
            warnings.warn("Singular innovation covariance matrix")
            return
        
        # State update
        self.x = self.x + self.K @ self.y
        
        # Covariance update
        I_KH = np.eye(self.dim_x) - self.K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + self.K @ self.R @ self.K.T
        
        # Compute likelihood
        self.compute_log_likelihood()


class UnscentedKalmanFilter(BaseKalmanFilter):
    """
    Unscented Kalman Filter for highly nonlinear systems.
    
    Uses the unscented transform to propagate sigma points through
    nonlinear functions, providing better approximations than linearization.
    """
    
    def __init__(self, dim_x: int, dim_z: int, dt: float = 1.0, 
                 hx: Optional[Callable] = None, fx: Optional[Callable] = None,
                 kappa: float = 0.0, alpha: float = 0.001, beta: float = 2.0,
                 augment_state: bool = True):
        """
        Initialize Unscented Kalman Filter.
        
        Args:
            dim_x: Dimension of state vector
            dim_z: Dimension of measurement vector
            dt: Time step
            hx: Measurement function h(x)
            fx: State transition function f(x, dt)
            kappa: Secondary scaling parameter (usually 0 or 3-n)
            alpha: Spread of sigma points (usually small, 1e-3)
            beta: Prior knowledge parameter (2 is optimal for Gaussian)
            augment_state: Whether to use augmented sigma points
        """
        super().__init__(dim_x, dim_z)
        self.dt = dt
        self.hx = hx
        self.fx = fx
        
        # UKF parameters
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta
        self.augment_state = augment_state
        
        # Augmented dimensions
        if augment_state:
            self.dim_aug = dim_x + dim_x + dim_z  # state + process noise + measurement noise
        else:
            self.dim_aug = dim_x
        
        # Number of sigma points
        self.n_sigma = 2 * self.dim_aug + 1
        
        # Compute weights
        self._compute_weights()
        
        # Sigma points
        self.sigma_points_x = np.zeros((self.n_sigma, dim_x))
        self.sigma_points_z = np.zeros((self.n_sigma, dim_z))
        
        # Innovation
        self.K = np.zeros((dim_x, dim_z))
        
        # Cross correlation matrix
        self.Pxz = np.zeros((dim_x, dim_z))
        
        # Prior state and covariance
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
    
    def _compute_weights(self) -> None:
        """Compute sigma point weights."""
        lambda_val = self.alpha**2 * (self.dim_aug + self.kappa) - self.dim_aug
        
        # Mean weights
        self.Wm = np.full(self.n_sigma, 1.0 / (2 * (self.dim_aug + lambda_val)))
        self.Wm[0] = lambda_val / (self.dim_aug + lambda_val)
        
        # Covariance weights
        self.Wc = self.Wm.copy()
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
    
    def _generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Generate sigma points using augmented or standard approach.
        
        Args:
            x: State vector
            P: Covariance matrix
            
        Returns:
            Sigma points matrix
        """
        if self.augment_state:
            # Augmented sigma points
            x_aug = np.zeros(self.dim_aug)
            x_aug[:self.dim_x] = x
            
            P_aug = np.zeros((self.dim_aug, self.dim_aug))
            P_aug[:self.dim_x, :self.dim_x] = P
            P_aug[self.dim_x:self.dim_x+self.dim_x, 
                  self.dim_x:self.dim_x+self.dim_x] = self.Q
            P_aug[self.dim_x+self.dim_x:, self.dim_x+self.dim_x:] = self.R
            
            lambda_val = self.alpha**2 * (self.dim_aug + self.kappa) - self.dim_aug
            
        else:
            # Standard sigma points
            x_aug = x
            P_aug = P
            lambda_val = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x
        
        # Cholesky decomposition
        try:
            sqrt = np.linalg.cholesky((len(x_aug) + lambda_val) * P_aug)
        except np.linalg.LinAlgError:
            # Fall back to eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(P_aug)
            eigenvals = np.maximum(eigenvals, 1e-12)  # Ensure positive
            sqrt = eigenvecs @ np.diag(np.sqrt(eigenvals * (len(x_aug) + lambda_val)))
        
        # Generate sigma points
        sigma_points = np.zeros((self.n_sigma, len(x_aug)))
        sigma_points[0] = x_aug
        
        for i in range(len(x_aug)):
            sigma_points[i + 1] = x_aug + sqrt[i]
            sigma_points[i + 1 + len(x_aug)] = x_aug - sqrt[i]
        
        return sigma_points
    
    def _unscented_transform(self, sigma_points: np.ndarray, 
                           noise_cov: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply unscented transform to sigma points.
        
        Args:
            sigma_points: Transformed sigma points
            noise_cov: Additional noise covariance to add
            
        Returns:
            Mean and covariance
        """
        # Compute mean
        mean = np.average(sigma_points, axis=0, weights=self.Wm)
        
        # Compute covariance
        cov = np.zeros((sigma_points.shape[1], sigma_points.shape[1]))
        for i, point in enumerate(sigma_points):
            diff = point - mean
            cov += self.Wc[i] * np.outer(diff, diff)
        
        if noise_cov is not None:
            cov += noise_cov
        
        return mean, cov
    
    def predict(self, dt: float, u: Optional[np.ndarray] = None) -> None:
        """
        Predict using unscented transform.
        
        Args:
            dt: Time step
            u: Control input
        """
        if self.fx is None:
            raise ValueError("State transition function fx must be set")
        
        # Store prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        
        # Generate sigma points
        sigma_points_aug = self._generate_sigma_points(self.x, self.P)
        
        # Propagate through state transition
        for i, point in enumerate(sigma_points_aug):
            if self.augment_state:
                # Extract state and process noise
                x_point = point[:self.dim_x]
                w_point = point[self.dim_x:self.dim_x+self.dim_x]
                self.sigma_points_x[i] = self.fx(x_point, dt) + w_point
            else:
                self.sigma_points_x[i] = self.fx(point, dt)
        
        # Apply unscented transform
        if self.augment_state:
            self.x, self.P = self._unscented_transform(self.sigma_points_x)
        else:
            self.x, self.P = self._unscented_transform(self.sigma_points_x, self.Q)
    
    def update(self, z: np.ndarray) -> None:
        """
        Update using unscented transform.
        
        Args:
            z: Measurement vector
        """
        if self.hx is None:
            raise ValueError("Measurement function hx must be set")
        
        # Generate sigma points for current state
        sigma_points_aug = self._generate_sigma_points(self.x, self.P)
        
        # Propagate through measurement function
        for i, point in enumerate(sigma_points_aug):
            if self.augment_state:
                # Extract state and measurement noise
                x_point = point[:self.dim_x]
                v_point = point[self.dim_x+self.dim_x:]
                self.sigma_points_z[i] = self.hx(x_point) + v_point
            else:
                self.sigma_points_z[i] = self.hx(point)
        
        # Predicted measurement and covariance
        if self.augment_state:
            z_pred, self.S = self._unscented_transform(self.sigma_points_z)
        else:
            z_pred, self.S = self._unscented_transform(self.sigma_points_z, self.R)
        
        # Innovation
        self.compute_innovation(z, z_pred)
        
        # Cross correlation
        self.Pxz = np.zeros((self.dim_x, self.dim_z))
        for i in range(self.n_sigma):
            dx = self.sigma_points_x[i] - self.x
            dz = self.sigma_points_z[i] - z_pred
            self.Pxz += self.Wc[i] * np.outer(dx, dz)
        
        # Kalman gain
        try:
            self.K = self.Pxz @ np.linalg.inv(self.S)
        except np.linalg.LinAlgError:
            warnings.warn("Singular innovation covariance matrix")
            return
        
        # State update
        self.x = self.x + self.K @ self.y
        
        # Covariance update
        self.P = self.P - self.K @ self.S @ self.K.T
        
        # Compute likelihood
        self.compute_log_likelihood()


# Utility functions for filter initialization and common operations

def initialize_constant_velocity_filter(dim: int = 2, dt: float = 1.0, 
                                      process_noise_std: float = 1.0,
                                      measurement_noise_std: float = 1.0) -> KalmanFilter:
    """
    Initialize a Kalman filter for constant velocity motion model.
    
    Args:
        dim: Spatial dimensions (2 for 2D, 3 for 3D)
        dt: Time step
        process_noise_std: Standard deviation of process noise
        measurement_noise_std: Standard deviation of measurement noise
        
    Returns:
        Configured Kalman filter
    """
    dim_x = 2 * dim  # position and velocity
    dim_z = dim      # position measurements only
    
    kf = KalmanFilter(dim_x, dim_z)
    
    # State transition matrix (constant velocity)
    kf.F = np.eye(dim_x)
    for i in range(dim):
        kf.F[i, i + dim] = dt
    
    # Measurement matrix (observe positions only)
    kf.H = np.zeros((dim_z, dim_x))
    for i in range(dim):
        kf.H[i, i] = 1.0
    
    # Process noise (discrete white noise model)
    q = process_noise_std**2
    dt2 = dt**2
    dt3 = dt**3
    dt4 = dt**4
    
    Q_block = np.array([[dt4/4, dt3/2],
                        [dt3/2, dt2]]) * q
    
    kf.Q = np.zeros((dim_x, dim_x))
    for i in range(dim):
        kf.Q[i*2:(i+1)*2, i*2:(i+1)*2] = Q_block
    
    # Measurement noise
    kf.R = np.eye(dim_z) * measurement_noise_std**2
    
    return kf


def initialize_constant_acceleration_filter(dim: int = 2, dt: float = 1.0,
                                          process_noise_std: float = 1.0,
                                          measurement_noise_std: float = 1.0) -> KalmanFilter:
    """
    Initialize a Kalman filter for constant acceleration motion model.
    
    Args:
        dim: Spatial dimensions (2 for 2D, 3 for 3D)
        dt: Time step
        process_noise_std: Standard deviation of process noise
        measurement_noise_std: Standard deviation of measurement noise
        
    Returns:
        Configured Kalman filter
    """
    dim_x = 3 * dim  # position, velocity, and acceleration
    dim_z = dim      # position measurements only
    
    kf = KalmanFilter(dim_x, dim_z)
    
    # State transition matrix (constant acceleration)
    kf.F = np.eye(dim_x)
    for i in range(dim):
        kf.F[i, i + dim] = dt              # position += velocity * dt
        kf.F[i, i + 2*dim] = 0.5 * dt**2   # position += 0.5 * acceleration * dt^2
        kf.F[i + dim, i + 2*dim] = dt      # velocity += acceleration * dt
    
    # Measurement matrix (observe positions only)
    kf.H = np.zeros((dim_z, dim_x))
    for i in range(dim):
        kf.H[i, i] = 1.0
    
    # Process noise
    q = process_noise_std**2
    dt2 = dt**2
    dt3 = dt**3
    dt4 = dt**4
    dt5 = dt**5
    dt6 = dt**6
    
    Q_block = np.array([[dt6/36, dt5/12, dt4/6],
                        [dt5/12, dt4/4,  dt3/2],
                        [dt4/6,  dt3/2,  dt2]]) * q
    
    kf.Q = np.zeros((dim_x, dim_x))
    for i in range(dim):
        kf.Q[i*3:(i+1)*3, i*3:(i+1)*3] = Q_block
    
    # Measurement noise
    kf.R = np.eye(dim_z) * measurement_noise_std**2
    
    return kf


def adaptive_noise_estimation(filter_obj: BaseKalmanFilter, 
                            innovation_history: list,
                            window_size: int = 10) -> None:
    """
    Adaptively estimate measurement noise covariance based on innovation sequence.
    
    Args:
        filter_obj: Kalman filter object
        innovation_history: List of recent innovation vectors
        window_size: Number of recent innovations to use
    """
    if len(innovation_history) < window_size:
        return
    
    # Use recent innovations
    recent_innovations = innovation_history[-window_size:]
    innovations_array = np.array(recent_innovations)
    
    # Estimate innovation covariance
    innovation_cov = np.cov(innovations_array.T)
    
    # Theoretical innovation covariance is H*P*H' + R
    # So R ≈ innovation_cov - H*P*H'
    if hasattr(filter_obj, 'H'):
        H = filter_obj.H
        theoretical_cov = H @ filter_obj.P @ H.T
        estimated_R = innovation_cov - theoretical_cov
        
        # Ensure positive definiteness
        eigenvals, eigenvecs = np.linalg.eigh(estimated_R)
        eigenvals = np.maximum(eigenvals, 1e-6)
        filter_obj.R = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T


def predict_multiple_steps(filter_obj: BaseKalmanFilter, 
                          n_steps: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict multiple steps ahead without measurements.
    
    Args:
        filter_obj: Kalman filter object
        n_steps: Number of prediction steps
        dt: Time step
        
    Returns:
        Predicted states and covariances
    """
    states = []
    covariances = []
    
    # Save current state
    x_saved = filter_obj.x.copy()
    P_saved = filter_obj.P.copy()
    
    for _ in range(n_steps):
        filter_obj.predict(dt)
        states.append(filter_obj.x.copy())
        covariances.append(filter_obj.P.copy())
    
    # Restore original state
    filter_obj.x = x_saved
    filter_obj.P = P_saved
    
    return np.array(states), np.array(covariances)


def compute_nees(true_state: np.ndarray, estimated_state: np.ndarray, 
                covariance: np.ndarray) -> float:
    """
    Compute Normalized Estimation Error Squared (NEES) for filter evaluation.
    
    Args:
        true_state: True state vector
        estimated_state: Estimated state vector
        covariance: State covariance matrix
        
    Returns:
        NEES value
    """
    error = true_state - estimated_state
    try:
        inv_cov = np.linalg.inv(covariance)
        nees = error.T @ inv_cov @ error
        return float(nees)
    except np.linalg.LinAlgError:
        return np.inf


def compute_nis(innovation: np.ndarray, innovation_cov: np.ndarray) -> float:
    """
    Compute Normalized Innovation Squared (NIS) for filter evaluation.
    
    Args:
        innovation: Innovation vector
        innovation_cov: Innovation covariance matrix
        
    Returns:
        NIS value
    """
    try:
        inv_cov = np.linalg.inv(innovation_cov)
        nis = innovation.T @ inv_cov @ innovation
        return float(nis)
    except np.linalg.LinAlgError:
        return np.inf