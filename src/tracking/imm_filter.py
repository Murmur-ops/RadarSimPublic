"""
Interacting Multiple Model (IMM) Filter Implementation for Radar Tracking

This module provides a comprehensive implementation of the Interacting Multiple Model (IMM) 
filter for tracking maneuvering targets in radar systems. The IMM filter combines multiple 
motion models with adaptive model probabilities to handle various target behaviors including 
constant velocity, constant acceleration, and coordinated turn maneuvers.

Key Features:
- Support for multiple motion models (CV, CA, CT, Singer)
- Adaptive model probability computation
- Parallel Kalman filtering for each model
- Model mixing and state combination
- Performance metrics and likelihood computation
- Configurable transition probability matrix

Author: RadarSim Project
"""

import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from enum import Enum

# Import existing tracking components
from .kalman_filters import BaseKalmanFilter, KalmanFilter, ExtendedKalmanFilter
from .motion_models import (
    MotionModel, CoordinateSystem, ModelParameters,
    ConstantVelocityModel, ConstantAccelerationModel, CoordinatedTurnModel,
    create_motion_model
)


@dataclass
class IMMParameters:
    """Parameters for IMM filter configuration."""
    dt: float                                          # Time step
    model_types: List[str]                            # List of model types ['cv', 'ca', 'ct']
    transition_probabilities: np.ndarray              # Model transition probability matrix
    initial_model_probabilities: Optional[np.ndarray] = None  # Initial model probabilities
    coordinate_system: CoordinateSystem = CoordinateSystem.CARTESIAN_2D
    process_noise_stds: Optional[List[float]] = None  # Process noise for each model
    measurement_noise_std: float = 1.0                # Measurement noise standard deviation
    gate_threshold: float = 9.21                      # Chi-squared gating threshold
    min_probability: float = 1e-6                     # Minimum model probability
    normalize_probabilities: bool = True              # Normalize probabilities
    model_specific_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class ModelState:
    """Container for individual model filter state."""
    
    def __init__(self, model_id: str, filter_obj: BaseKalmanFilter, 
                 motion_model: MotionModel, probability: float = 0.0):
        """
        Initialize model state.
        
        Args:
            model_id: Unique identifier for the model
            filter_obj: Kalman filter instance
            motion_model: Motion model instance
            probability: Current model probability
        """
        self.model_id = model_id
        self.filter = filter_obj
        self.motion_model = motion_model
        self.probability = probability
        self.likelihood = 0.0
        self.mixed_state = None
        self.mixed_covariance = None
        
        # History tracking
        self.state_history = []
        self.probability_history = []
        self.likelihood_history = []


class IMMFilter:
    """
    Interacting Multiple Model (IMM) Filter for maneuvering target tracking.
    
    The IMM filter maintains multiple motion models simultaneously and computes
    adaptive model probabilities based on how well each model fits the incoming
    measurements. This allows robust tracking of targets with varying motion
    characteristics.
    """
    
    def __init__(self, params: IMMParameters):
        """
        Initialize IMM filter.
        
        Args:
            params: IMM filter parameters
        """
        self.params = params
        self.dt = params.dt
        self.num_models = len(params.model_types)
        
        # Validate parameters
        self._validate_parameters()
        
        # Determine measurement dimension based on coordinate system
        if params.coordinate_system == CoordinateSystem.CARTESIAN_2D:
            self.measurement_dim = 2  # x, y position measurements
        elif params.coordinate_system == CoordinateSystem.CARTESIAN_3D:
            self.measurement_dim = 3  # x, y, z position measurements
        else:
            self.measurement_dim = 2  # Default to 2D
        
        # Initialize models first to determine state dimensions
        self.models = self._initialize_models()
        
        # Set state dimensions based on the first model (they should be compatible)
        self.state_dim = max(model.filter.dim_x for model in self.models)
        self.combined_state = np.zeros(self.state_dim)
        self.combined_covariance = np.eye(self.state_dim)
        
        # Model interaction matrices and probabilities
        self.transition_probs = params.transition_probabilities.copy()
        self.model_probs = self._initialize_model_probabilities()
        self.mixed_probs = np.zeros((self.num_models, self.num_models))
        self.normalization_factors = np.zeros(self.num_models)
        
        # Performance tracking
        self.innovation_history = []
        self.model_likelihood_history = []
        self.combined_likelihood = 0.0
        self.step_count = 0
        
        # Gating
        self.gate_threshold = params.gate_threshold
        self.last_valid_measurement = None
        
    def _validate_parameters(self) -> None:
        """Validate IMM parameters for consistency."""
        if self.num_models < 2:
            raise ValueError("IMM filter requires at least 2 models")
        
        # Check transition probability matrix
        if self.params.transition_probabilities.shape != (self.num_models, self.num_models):
            raise ValueError(f"Transition matrix must be {self.num_models}x{self.num_models}")
        
        # Check if transition matrix rows sum to 1
        row_sums = np.sum(self.params.transition_probabilities, axis=1)
        if not np.allclose(row_sums, 1.0):
            warnings.warn("Transition probability matrix rows should sum to 1")
        
        # Validate model types
        valid_models = ['cv', 'ca', 'ct', 'singer']
        for model_type in self.params.model_types:
            if model_type not in valid_models:
                raise ValueError(f"Unknown model type: {model_type}")
    
    def _initialize_models(self) -> List[ModelState]:
        """Initialize motion models and associated Kalman filters."""
        models = []
        
        for i, model_type in enumerate(self.params.model_types):
            # Create motion model parameters
            if self.params.process_noise_stds:
                process_noise = self.params.process_noise_stds[i]
            else:
                process_noise = 1.0
            
            model_params = ModelParameters(
                dt=self.dt,
                process_noise_std=process_noise,
                coordinate_system=self.params.coordinate_system,
                additional_params=self.params.model_specific_params.get(model_type, {})
            )
            
            # Create motion model
            motion_model = create_motion_model(model_type, model_params)
            
            # Create appropriate Kalman filter
            if model_type in ['cv', 'ca']:
                # Linear models use standard Kalman filter
                kf = KalmanFilter(motion_model.state_dim, self.measurement_dim)
            else:
                # Nonlinear models use Extended Kalman filter
                kf = ExtendedKalmanFilter(motion_model.state_dim, self.measurement_dim, self.dt)
                # Set up nonlinear functions
                kf.set_state_transition(
                    lambda x, u, dt: motion_model.predict_state(x),
                    lambda x, u, dt: motion_model.get_transition_matrix(x)
                )
                # Set up measurement functions (position measurements)
                def measurement_function(x):
                    return x[:self.measurement_dim]  # Extract position from state
                
                def measurement_jacobian(x):
                    H = np.zeros((self.measurement_dim, len(x)))
                    for i in range(self.measurement_dim):
                        H[i, i] = 1.0
                    return H
                
                kf.set_measurement_function(measurement_function, measurement_jacobian)
            
            # Set initial matrices
            kf.F = motion_model.get_transition_matrix(np.zeros(motion_model.state_dim))
            kf.Q = motion_model.get_process_noise_covariance(np.zeros(motion_model.state_dim))
            kf.R = np.eye(self.measurement_dim) * self.params.measurement_noise_std**2
            
            # Set measurement matrix (assuming position measurements)
            kf.H = self._create_measurement_matrix(motion_model.state_dim, self.measurement_dim)
            
            # Create model state
            model_state = ModelState(f"model_{i}_{model_type}", kf, motion_model)
            models.append(model_state)
        
        return models
    
    def _create_measurement_matrix(self, state_dim: int, measurement_dim: int) -> np.ndarray:
        """Create measurement matrix assuming position measurements."""
        H = np.zeros((measurement_dim, state_dim))
        for i in range(measurement_dim):
            H[i, i] = 1.0
        return H
    
    def _initialize_model_probabilities(self) -> np.ndarray:
        """Initialize model probabilities."""
        if self.params.initial_model_probabilities is not None:
            probs = self.params.initial_model_probabilities.copy()
        else:
            # Equal initial probabilities
            probs = np.ones(self.num_models) / self.num_models
        
        # Set initial probabilities in model states
        for i, model in enumerate(self.models):
            model.probability = probs[i]
        
        return probs
    
    def initialize_state(self, initial_state: np.ndarray, 
                        initial_covariance: Optional[np.ndarray] = None) -> None:
        """
        Initialize state for all models.
        
        Args:
            initial_state: Initial state vector
            initial_covariance: Initial covariance matrix (optional)
        """
        if initial_covariance is None:
            initial_covariance = np.eye(len(initial_state)) * 100.0
        
        # Initialize all models with the same state
        for model in self.models:
            # Pad or truncate state if needed
            if len(initial_state) != model.filter.dim_x:
                if len(initial_state) < model.filter.dim_x:
                    # Pad with zeros
                    padded_state = np.zeros(model.filter.dim_x)
                    padded_state[:len(initial_state)] = initial_state
                    model.filter.x = padded_state
                    
                    # Pad covariance
                    padded_cov = np.eye(model.filter.dim_x) * 100.0
                    min_dim = min(initial_covariance.shape[0], model.filter.dim_x)
                    padded_cov[:min_dim, :min_dim] = initial_covariance[:min_dim, :min_dim]
                    model.filter.P = padded_cov
                else:
                    # Truncate
                    model.filter.x = initial_state[:model.filter.dim_x]
                    model.filter.P = initial_covariance[:model.filter.dim_x, :model.filter.dim_x]
            else:
                model.filter.x = initial_state.copy()
                model.filter.P = initial_covariance.copy()
        
        # Update combined state
        self._update_combined_estimate()
    
    def predict(self) -> None:
        """
        Perform IMM prediction step.
        
        This involves:
        1. Computing mixing probabilities
        2. Mixing states and covariances
        3. Performing prediction for each model
        """
        # Step 1: Compute mixing probabilities
        self._compute_mixing_probabilities()
        
        # Step 2: Mix states and covariances
        self._mix_states()
        
        # Step 3: Predict each model
        for model in self.models:
            # Update filter matrices if using linear models
            if isinstance(model.filter, KalmanFilter):
                model.filter.F = model.motion_model.get_transition_matrix(model.filter.x)
                model.filter.Q = model.motion_model.get_process_noise_covariance(model.filter.x)
            
            # Perform prediction
            model.filter.predict(self.dt)
    
    def update(self, measurement: np.ndarray) -> bool:
        """
        Perform IMM update step with measurement.
        
        Args:
            measurement: Measurement vector
            
        Returns:
            True if update was successful
        """
        if measurement is None:
            return False
        
        # Update each model with the measurement
        likelihoods = np.zeros(self.num_models)
        valid_updates = 0
        
        for i, model in enumerate(self.models):
            # Check measurement validity using gating
            if isinstance(model.filter, KalmanFilter):
                predicted_measurement = model.filter.H @ model.filter.x
            else:
                # For EKF, use nonlinear measurement function if available
                if hasattr(model.filter, 'hx') and model.filter.hx is not None:
                    predicted_measurement = model.filter.hx(model.filter.x)
                else:
                    predicted_measurement = model.filter.H @ model.filter.x
            
            # Gating test
            if self._is_measurement_valid(measurement, predicted_measurement, model.filter):
                model.filter.update(measurement)
                likelihoods[i] = np.exp(model.filter.log_likelihood)
                model.likelihood = likelihoods[i]
                valid_updates += 1
            else:
                # Set very low likelihood for gated measurements
                likelihoods[i] = self.params.min_probability
                model.likelihood = likelihoods[i]
        
        if valid_updates == 0:
            warnings.warn("No models passed gating test")
            return False
        
        # Step 4: Update model probabilities
        self._update_model_probabilities(likelihoods)
        
        # Step 5: Compute combined estimate
        self._update_combined_estimate()
        
        # Update history
        self.innovation_history.append(self.models[0].filter.y.copy())
        self.model_likelihood_history.append(likelihoods.copy())
        self.step_count += 1
        self.last_valid_measurement = measurement.copy()
        
        return True
    
    def _compute_mixing_probabilities(self) -> None:
        """Compute mixing probabilities for model interaction."""
        # Compute normalization factors: c_j = sum_i(π_ij * μ_i)
        for j in range(self.num_models):
            self.normalization_factors[j] = np.sum(
                self.transition_probs[:, j] * self.model_probs
            )
            # Avoid division by zero
            if self.normalization_factors[j] < self.params.min_probability:
                self.normalization_factors[j] = self.params.min_probability
        
        # Compute mixing probabilities: ω_ij = (π_ij * μ_i) / c_j
        for i in range(self.num_models):
            for j in range(self.num_models):
                self.mixed_probs[i, j] = (
                    self.transition_probs[i, j] * self.model_probs[i] / 
                    self.normalization_factors[j]
                )
    
    def _mix_states(self) -> None:
        """Mix states and covariances for each model."""
        for j in range(self.num_models):
            # Mixed state: x̂_j^0 = sum_i(ω_ij * x̂_i)
            mixed_state = np.zeros(self.models[j].filter.dim_x)
            for i in range(self.num_models):
                # Handle different state dimensions
                state_i = self._adapt_state_dimension(
                    self.models[i].filter.x, self.models[j].filter.dim_x
                )
                mixed_state += self.mixed_probs[i, j] * state_i
            
            # Mixed covariance: P_j^0 = sum_i(ω_ij * [P_i + (x̂_i - x̂_j^0)(x̂_i - x̂_j^0)'])
            mixed_cov = np.zeros((self.models[j].filter.dim_x, self.models[j].filter.dim_x))
            for i in range(self.num_models):
                # Adapt covariance dimension
                cov_i = self._adapt_covariance_dimension(
                    self.models[i].filter.P, self.models[j].filter.dim_x
                )
                state_i = self._adapt_state_dimension(
                    self.models[i].filter.x, self.models[j].filter.dim_x
                )
                
                state_diff = state_i - mixed_state
                mixed_cov += self.mixed_probs[i, j] * (
                    cov_i + np.outer(state_diff, state_diff)
                )
            
            # Store mixed initial conditions
            self.models[j].mixed_state = mixed_state.copy()
            self.models[j].mixed_covariance = mixed_cov.copy()
            
            # Set as initial conditions for prediction
            self.models[j].filter.x = mixed_state
            self.models[j].filter.P = mixed_cov
    
    def _adapt_state_dimension(self, state: np.ndarray, target_dim: int) -> np.ndarray:
        """Adapt state vector to target dimension."""
        if len(state) == target_dim:
            return state.copy()
        elif len(state) < target_dim:
            # Pad with zeros
            adapted = np.zeros(target_dim)
            adapted[:len(state)] = state
            return adapted
        else:
            # Truncate
            return state[:target_dim].copy()
    
    def _adapt_covariance_dimension(self, cov: np.ndarray, target_dim: int) -> np.ndarray:
        """Adapt covariance matrix to target dimension."""
        if cov.shape[0] == target_dim:
            return cov.copy()
        elif cov.shape[0] < target_dim:
            # Pad with identity
            adapted = np.eye(target_dim) * 100.0  # Large uncertainty for new dimensions
            adapted[:cov.shape[0], :cov.shape[1]] = cov
            return adapted
        else:
            # Truncate
            return cov[:target_dim, :target_dim].copy()
    
    def _update_model_probabilities(self, likelihoods: np.ndarray) -> None:
        """Update model probabilities based on likelihoods."""
        # Compute new model probabilities: μ_j = (λ_j * c_j) / c
        new_probs = likelihoods * self.normalization_factors
        
        # Normalize
        total_prob = np.sum(new_probs)
        if total_prob > self.params.min_probability:
            new_probs /= total_prob
        else:
            # Fallback to equal probabilities
            new_probs = np.ones(self.num_models) / self.num_models
        
        # Apply minimum probability constraint
        new_probs = np.maximum(new_probs, self.params.min_probability)
        
        # Renormalize if needed
        if self.params.normalize_probabilities:
            new_probs /= np.sum(new_probs)
        
        # Update model probabilities
        self.model_probs = new_probs
        for i, model in enumerate(self.models):
            model.probability = new_probs[i]
            model.probability_history.append(new_probs[i])
            model.likelihood_history.append(likelihoods[i])
    
    def _update_combined_estimate(self) -> None:
        """Compute combined state estimate and covariance."""
        # Combined state: x̂ = sum_j(μ_j * x̂_j)
        self.combined_state = np.zeros(self.state_dim)
        for i, model in enumerate(self.models):
            state_adapted = self._adapt_state_dimension(model.filter.x, self.state_dim)
            self.combined_state += model.probability * state_adapted
        
        # Combined covariance: P = sum_j(μ_j * [P_j + (x̂_j - x̂)(x̂_j - x̂)'])
        self.combined_covariance = np.zeros((self.state_dim, self.state_dim))
        for i, model in enumerate(self.models):
            cov_adapted = self._adapt_covariance_dimension(model.filter.P, self.state_dim)
            state_adapted = self._adapt_state_dimension(model.filter.x, self.state_dim)
            
            state_diff = state_adapted - self.combined_state
            self.combined_covariance += model.probability * (
                cov_adapted + np.outer(state_diff, state_diff)
            )
        
        # Compute combined likelihood
        self.combined_likelihood = np.sum([
            model.probability * model.likelihood for model in self.models
        ])
    
    def _is_measurement_valid(self, measurement: np.ndarray, 
                            predicted_measurement: np.ndarray,
                            filter_obj: BaseKalmanFilter) -> bool:
        """Check if measurement passes gating test."""
        try:
            return filter_obj.is_measurement_valid(
                measurement, predicted_measurement, self.gate_threshold
            )
        except:
            # Fallback to simple distance check
            distance = np.linalg.norm(measurement - predicted_measurement)
            return distance < np.sqrt(self.gate_threshold)
    
    def get_state_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get combined state estimate and covariance.
        
        Returns:
            (state, covariance) tuple
        """
        return self.combined_state.copy(), self.combined_covariance.copy()
    
    def get_model_probabilities(self) -> np.ndarray:
        """Get current model probabilities."""
        return self.model_probs.copy()
    
    def get_most_likely_model(self) -> Tuple[int, str, float]:
        """
        Get the most likely model.
        
        Returns:
            (index, model_id, probability) tuple
        """
        idx = np.argmax(self.model_probs)
        return idx, self.models[idx].model_id, self.model_probs[idx]
    
    def get_model_states(self) -> List[Tuple[str, np.ndarray, np.ndarray, float]]:
        """
        Get states for all models.
        
        Returns:
            List of (model_id, state, covariance, probability) tuples
        """
        return [
            (model.model_id, model.filter.x.copy(), 
             model.filter.P.copy(), model.probability)
            for model in self.models
        ]
    
    def predict_trajectory(self, n_steps: int) -> Dict[str, np.ndarray]:
        """
        Predict future trajectory for each model.
        
        Args:
            n_steps: Number of prediction steps
            
        Returns:
            Dictionary with predicted trajectories for each model
        """
        trajectories = {}
        
        for model in self.models:
            # Save current state
            x_saved = model.filter.x.copy()
            P_saved = model.filter.P.copy()
            
            # Predict multiple steps
            predicted_states = []
            for _ in range(n_steps):
                model.filter.predict(self.dt)
                predicted_states.append(model.filter.x.copy())
            
            # Restore state
            model.filter.x = x_saved
            model.filter.P = P_saved
            
            trajectories[model.model_id] = np.array(predicted_states)
        
        return trajectories
    
    def get_innovation_statistics(self) -> Dict[str, float]:
        """Get innovation statistics for filter evaluation."""
        if not self.innovation_history:
            return {}
        
        innovations = np.array(self.innovation_history)
        
        return {
            'mean_innovation_norm': np.mean([np.linalg.norm(inn) for inn in innovations]),
            'std_innovation_norm': np.std([np.linalg.norm(inn) for inn in innovations]),
            'max_innovation_norm': np.max([np.linalg.norm(inn) for inn in innovations]),
            'num_steps': len(self.innovation_history)
        }
    
    def get_model_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for each model."""
        metrics = {}
        
        for model in self.models:
            if model.likelihood_history:
                metrics[model.model_id] = {
                    'mean_probability': np.mean(model.probability_history),
                    'std_probability': np.std(model.probability_history),
                    'mean_likelihood': np.mean(model.likelihood_history),
                    'max_probability': np.max(model.probability_history),
                    'active_percentage': np.mean(np.array(model.probability_history) > 0.1)
                }
        
        return metrics
    
    def reset_history(self) -> None:
        """Reset tracking history."""
        self.innovation_history.clear()
        self.model_likelihood_history.clear()
        self.step_count = 0
        
        for model in self.models:
            model.probability_history.clear()
            model.likelihood_history.clear()
            model.state_history.clear()


# Utility functions for IMM filter setup and analysis

def create_imm_filter(model_types: List[str], dt: float = 1.0,
                     coordinate_system: CoordinateSystem = CoordinateSystem.CARTESIAN_2D,
                     process_noise_stds: Optional[List[float]] = None,
                     measurement_noise_std: float = 1.0,
                     transition_matrix: Optional[np.ndarray] = None) -> IMMFilter:
    """
    Create a pre-configured IMM filter.
    
    Args:
        model_types: List of model types ['cv', 'ca', 'ct']
        dt: Time step
        coordinate_system: Coordinate system
        process_noise_stds: Process noise for each model
        measurement_noise_std: Measurement noise standard deviation
        transition_matrix: Model transition probability matrix
        
    Returns:
        Configured IMM filter
    """
    num_models = len(model_types)
    
    # Default transition matrix (slight preference for staying in same model)
    if transition_matrix is None:
        # Higher probability of staying in same model
        stay_prob = 0.9
        switch_prob = (1.0 - stay_prob) / (num_models - 1)
        
        transition_matrix = np.full((num_models, num_models), switch_prob)
        np.fill_diagonal(transition_matrix, stay_prob)
    
    # Default process noise
    if process_noise_stds is None:
        process_noise_stds = [1.0] * num_models
    
    # Create parameters
    params = IMMParameters(
        dt=dt,
        model_types=model_types,
        transition_probabilities=transition_matrix,
        coordinate_system=coordinate_system,
        process_noise_stds=process_noise_stds,
        measurement_noise_std=measurement_noise_std
    )
    
    return IMMFilter(params)


def create_standard_imm_cv_ca_ct(dt: float = 1.0, 
                                process_noise_cv: float = 1.0,
                                process_noise_ca: float = 2.0,
                                process_noise_ct: float = 1.5,
                                measurement_noise_std: float = 1.0,
                                turn_rate_std: float = 0.1) -> IMMFilter:
    """
    Create a standard IMM filter with CV, CA, and CT models.
    
    Args:
        dt: Time step
        process_noise_cv: Process noise for CV model
        process_noise_ca: Process noise for CA model  
        process_noise_ct: Process noise for CT model
        measurement_noise_std: Measurement noise standard deviation
        turn_rate_std: Turn rate noise for CT model
        
    Returns:
        Configured IMM filter
    """
    # Transition matrix favoring CV model with occasional maneuvers
    transition_matrix = np.array([
        [0.85, 0.10, 0.05],  # CV -> CV, CA, CT
        [0.15, 0.80, 0.05],  # CA -> CV, CA, CT
        [0.15, 0.05, 0.80]   # CT -> CV, CA, CT
    ])
    
    # Model-specific parameters
    model_specific_params = {
        'ct': {
            'turn_rate': None,  # Estimate turn rate
            'turn_rate_noise': turn_rate_std
        }
    }
    
    params = IMMParameters(
        dt=dt,
        model_types=['cv', 'ca', 'ct'],
        transition_probabilities=transition_matrix,
        process_noise_stds=[process_noise_cv, process_noise_ca, process_noise_ct],
        measurement_noise_std=measurement_noise_std,
        model_specific_params=model_specific_params
    )
    
    return IMMFilter(params)


def analyze_model_switching(imm_filter: IMMFilter, 
                          window_size: int = 20) -> Dict[str, Any]:
    """
    Analyze model switching behavior in IMM filter.
    
    Args:
        imm_filter: IMM filter instance
        window_size: Size of sliding window for analysis
        
    Returns:
        Analysis results dictionary
    """
    if imm_filter.step_count < window_size:
        return {"error": "Insufficient history for analysis"}
    
    # Get probability histories
    prob_histories = []
    for model in imm_filter.models:
        if len(model.probability_history) >= window_size:
            prob_histories.append(np.array(model.probability_history[-window_size:]))
        else:
            prob_histories.append(np.array(model.probability_history))
    
    if not prob_histories:
        return {"error": "No probability history available"}
    
    prob_array = np.array(prob_histories).T  # Shape: (time, models)
    
    # Detect model switches (when most likely model changes)
    most_likely_models = np.argmax(prob_array, axis=1)
    switches = np.sum(np.diff(most_likely_models) != 0)
    
    # Compute model dominance periods
    dominance_periods = []
    current_model = most_likely_models[0]
    current_period = 1
    
    for i in range(1, len(most_likely_models)):
        if most_likely_models[i] == current_model:
            current_period += 1
        else:
            dominance_periods.append(current_period)
            current_model = most_likely_models[i]
            current_period = 1
    dominance_periods.append(current_period)
    
    # Compute model utilization
    model_utilization = np.mean(prob_array, axis=0)
    
    return {
        "num_switches": switches,
        "switch_rate": switches / window_size,
        "mean_dominance_period": np.mean(dominance_periods),
        "model_utilization": model_utilization,
        "dominant_model_sequence": most_likely_models,
        "model_names": [model.model_id for model in imm_filter.models]
    }


def compute_model_likelihood_ratios(imm_filter: IMMFilter) -> np.ndarray:
    """
    Compute likelihood ratios between models.
    
    Args:
        imm_filter: IMM filter instance
        
    Returns:
        Matrix of likelihood ratios
    """
    num_models = len(imm_filter.models)
    ratios = np.zeros((num_models, num_models))
    
    likelihoods = [model.likelihood for model in imm_filter.models]
    
    for i in range(num_models):
        for j in range(num_models):
            if likelihoods[j] > 1e-12:
                ratios[i, j] = likelihoods[i] / likelihoods[j]
            else:
                ratios[i, j] = np.inf if likelihoods[i] > 1e-12 else 1.0
    
    return ratios


# Example usage and demonstration functions

def demonstrate_imm_maneuvering_target():
    """
    Demonstrate IMM filter tracking a maneuvering target.
    
    This function creates a synthetic trajectory with different motion phases
    and shows how the IMM filter adapts its model probabilities.
    """
    # Create IMM filter
    imm = create_standard_imm_cv_ca_ct(dt=1.0)
    
    # Initialize with a simple state [x, y, vx, vy]
    initial_state = np.array([0.0, 0.0, 10.0, 0.0])
    imm.initialize_state(initial_state)
    
    # Simulate trajectory with different phases
    true_states = []
    measurements = []
    
    # Phase 1: Constant velocity (20 steps)
    state = initial_state.copy()
    for t in range(20):
        state[0] += state[2]  # x += vx
        state[1] += state[3]  # y += vy
        
        # Add noise to measurement
        measurement = state[:2] + np.random.normal(0, 1.0, 2)
        
        true_states.append(state.copy())
        measurements.append(measurement)
    
    # Phase 2: Constant acceleration turn (30 steps)
    for t in range(30):
        # Add centripetal acceleration for turn
        speed = np.sqrt(state[2]**2 + state[3]**2)
        if speed > 0:
            turn_rate = 0.1  # rad/s
            ax = -turn_rate * state[3]  # centripetal acceleration
            ay = turn_rate * state[2]
            
            state[2] += ax  # vx += ax
            state[3] += ay  # vy += ay
        
        state[0] += state[2]
        state[1] += state[3]
        
        measurement = state[:2] + np.random.normal(0, 1.0, 2)
        
        true_states.append(state.copy())
        measurements.append(measurement)
    
    # Phase 3: Return to constant velocity (20 steps)
    for t in range(20):
        state[0] += state[2]
        state[1] += state[3]
        
        measurement = state[:2] + np.random.normal(0, 1.0, 2)
        
        true_states.append(state.copy())
        measurements.append(measurement)
    
    # Track with IMM filter
    estimated_states = []
    model_probabilities = []
    
    for measurement in measurements:
        imm.predict()
        imm.update(measurement)
        
        state_est, _ = imm.get_state_estimate()
        estimated_states.append(state_est.copy())
        model_probabilities.append(imm.get_model_probabilities().copy())
    
    return {
        'true_states': np.array(true_states),
        'measurements': np.array(measurements),
        'estimated_states': np.array(estimated_states),
        'model_probabilities': np.array(model_probabilities),
        'model_names': [model.model_id for model in imm.models],
        'performance_metrics': imm.get_model_performance_metrics(),
        'innovation_stats': imm.get_innovation_statistics()
    }


if __name__ == "__main__":
    # Example usage
    print("Testing IMM Filter Implementation...")
    
    # Basic test
    imm = create_standard_imm_cv_ca_ct()
    print(f"Created IMM filter with {len(imm.models)} models")
    print(f"Model types: {[model.model_id for model in imm.models]}")
    
    # Initialize and test basic functionality
    initial_state = np.array([0.0, 0.0, 5.0, 2.0])
    imm.initialize_state(initial_state)
    
    # Simulate a few updates
    for i in range(5):
        imm.predict()
        measurement = np.array([i * 5.0, i * 2.0]) + np.random.normal(0, 0.5, 2)
        imm.update(measurement)
        
        state, cov = imm.get_state_estimate()
        probs = imm.get_model_probabilities()
        
        print(f"Step {i+1}: State = {state[:4]}")
        print(f"  Model probs = {probs}")
        print(f"  Most likely model: {imm.get_most_likely_model()[1]}")
    
    print("\nIMM Filter test completed successfully!")