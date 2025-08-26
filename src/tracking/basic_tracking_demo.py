#!/usr/bin/env python3
"""
Basic Single-Target Tracking Demonstration

This script demonstrates fundamental single-target tracking using Kalman filters
with different motion models (CV and CA). It shows the effects of measurement
noise and process noise on tracking performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tracking.kalman_filters import KalmanFilter, initialize_constant_velocity_filter, initialize_constant_acceleration_filter
from src.tracking.motion_models import ConstantVelocityModel, ConstantAccelerationModel


def generate_cv_trajectory(duration: float, dt: float, initial_state: np.ndarray, 
                          process_noise_std: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a constant velocity trajectory with process noise."""
    n_steps = int(duration / dt)
    dim = len(initial_state) // 2  # Position dimensions
    
    # True states
    states = np.zeros((n_steps, len(initial_state)))
    states[0] = initial_state
    
    # State transition for CV model
    F = np.eye(2 * dim)
    for i in range(dim):
        F[i, dim + i] = dt
    
    # Process noise
    Q = np.zeros((2 * dim, 2 * dim))
    for i in range(dim):
        Q[i, i] = (dt**3 / 3) * process_noise_std**2
        Q[i, dim + i] = (dt**2 / 2) * process_noise_std**2
        Q[dim + i, i] = (dt**2 / 2) * process_noise_std**2
        Q[dim + i, dim + i] = dt * process_noise_std**2
    
    # Generate trajectory
    for k in range(1, n_steps):
        process_noise = np.random.multivariate_normal(np.zeros(2 * dim), Q)
        states[k] = F @ states[k-1] + process_noise
    
    # Extract positions for plotting
    positions = states[:, :dim]
    
    return states, positions


def generate_ca_trajectory(duration: float, dt: float, initial_state: np.ndarray,
                          acceleration: np.ndarray, process_noise_std: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a constant acceleration trajectory."""
    n_steps = int(duration / dt)
    dim = len(initial_state) // 2
    
    # Extend state to include acceleration
    extended_state = np.zeros(3 * dim)
    extended_state[:2*dim] = initial_state
    extended_state[2*dim:] = acceleration
    
    states = np.zeros((n_steps, 3 * dim))
    states[0] = extended_state
    
    # State transition for CA model
    F = np.eye(3 * dim)
    for i in range(dim):
        F[i, dim + i] = dt
        F[i, 2*dim + i] = dt**2 / 2
        F[dim + i, 2*dim + i] = dt
    
    # Process noise (small acceleration changes)
    Q = np.zeros((3 * dim, 3 * dim))
    for i in range(dim):
        Q[2*dim + i, 2*dim + i] = process_noise_std**2
    
    # Generate trajectory
    for k in range(1, n_steps):
        process_noise = np.random.multivariate_normal(np.zeros(3 * dim), Q)
        states[k] = F @ states[k-1] + process_noise
    
    # Extract positions
    positions = states[:, :dim]
    
    return states, positions


def generate_measurements(positions: np.ndarray, measurement_noise_std: float = 1.0,
                         detection_probability: float = 0.95) -> List[np.ndarray]:
    """Generate noisy measurements from true positions."""
    measurements = []
    dim = positions.shape[1]
    
    for pos in positions:
        if np.random.rand() < detection_probability:
            noise = np.random.randn(dim) * measurement_noise_std
            measurements.append(pos + noise)
        else:
            measurements.append(None)  # Missed detection
    
    return measurements


def plot_tracking_results(true_positions: np.ndarray, measurements: List[np.ndarray],
                         cv_estimates: np.ndarray, ca_estimates: np.ndarray,
                         cv_errors: np.ndarray, ca_errors: np.ndarray,
                         cv_covariances: List[np.ndarray], ca_covariances: List[np.ndarray]):
    """Create comprehensive visualization of tracking results."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # 2D Trajectory Plot
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(true_positions[:, 0], true_positions[:, 1], 'g-', linewidth=2, label='True Trajectory')
    
    # Plot measurements
    meas_x = [m[0] for m in measurements if m is not None]
    meas_y = [m[1] for m in measurements if m is not None]
    ax1.scatter(meas_x, meas_y, c='gray', s=20, alpha=0.5, label='Measurements')
    
    # Plot estimates
    ax1.plot(cv_estimates[:, 0], cv_estimates[:, 1], 'b--', linewidth=1.5, label='CV Estimate')
    ax1.plot(ca_estimates[:, 0], ca_estimates[:, 1], 'r--', linewidth=1.5, label='CA Estimate')
    
    # Plot uncertainty ellipses for last position
    for estimates, covariances, color in [(cv_estimates, cv_covariances, 'blue'), 
                                          (ca_estimates, ca_covariances, 'red')]:
        if len(covariances) > 0 and covariances[-1] is not None:
            cov_2d = covariances[-1][:2, :2]
            eigenvalues, eigenvectors = np.linalg.eig(cov_2d)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * np.sqrt(eigenvalues) * 2  # 2-sigma ellipse
            ellipse = Ellipse(estimates[-1, :2], width, height, angle=angle,
                             facecolor='none', edgecolor=color, alpha=0.5, linewidth=2)
            ax1.add_patch(ellipse)
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('2D Trajectory Tracking')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # X Position over Time
    ax2 = plt.subplot(2, 3, 2)
    time = np.arange(len(true_positions)) * 0.1
    ax2.plot(time, true_positions[:, 0], 'g-', label='True')
    ax2.plot(time, cv_estimates[:, 0], 'b--', label='CV')
    ax2.plot(time, ca_estimates[:, 0], 'r--', label='CA')
    if len(meas_x) > 0:
        meas_time = [time[i] for i, m in enumerate(measurements) if m is not None]
        ax2.scatter(meas_time, meas_x, c='gray', s=10, alpha=0.3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('X Position (m)')
    ax2.set_title('X Position vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Y Position over Time
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(time, true_positions[:, 1], 'g-', label='True')
    ax3.plot(time, cv_estimates[:, 1], 'b--', label='CV')
    ax3.plot(time, ca_estimates[:, 1], 'r--', label='CA')
    if len(meas_y) > 0:
        meas_time = [time[i] for i, m in enumerate(measurements) if m is not None]
        ax3.scatter(meas_time, meas_y, c='gray', s=10, alpha=0.3)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Y Position (m)')
    ax3.set_title('Y Position vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Position Error Over Time
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(time, cv_errors, 'b-', label='CV Error')
    ax4.plot(time, ca_errors, 'r-', label='CA Error')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position Error (m)')
    ax4.set_title('Position Estimation Error')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Error Histogram
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(cv_errors[cv_errors > 0], bins=30, alpha=0.5, color='blue', label=f'CV (mean={np.mean(cv_errors):.2f}m)')
    ax5.hist(ca_errors[ca_errors > 0], bins=30, alpha=0.5, color='red', label=f'CA (mean={np.mean(ca_errors):.2f}m)')
    ax5.set_xlabel('Position Error (m)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Error Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Performance Metrics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate metrics
    cv_rmse = np.sqrt(np.mean(cv_errors**2))
    ca_rmse = np.sqrt(np.mean(ca_errors**2))
    cv_max_error = np.max(cv_errors)
    ca_max_error = np.max(ca_errors)
    
    metrics_text = f"""Performance Metrics:

Constant Velocity (CV) Model:
  • RMSE: {cv_rmse:.3f} m
  • Mean Error: {np.mean(cv_errors):.3f} m
  • Max Error: {cv_max_error:.3f} m
  • Std Dev: {np.std(cv_errors):.3f} m

Constant Acceleration (CA) Model:
  • RMSE: {ca_rmse:.3f} m
  • Mean Error: {np.mean(ca_errors):.3f} m
  • Max Error: {ca_max_error:.3f} m
  • Std Dev: {np.std(ca_errors):.3f} m

Simulation Parameters:
  • Duration: {len(true_positions)*0.1:.1f} s
  • Measurement Noise: 1.0 m
  • Process Noise: 0.1 m/s²
  • Detection Rate: 95%"""
    
    ax6.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.suptitle('Basic Single-Target Tracking Demonstration', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('basic_tracking_demo.png', dpi=100, bbox_inches='tight')
    print("Plot saved as 'basic_tracking_demo.png'")
    plt.close()


def run_basic_tracking_demo():
    """Run the basic tracking demonstration."""
    
    print("=" * 60)
    print("Basic Single-Target Tracking Demonstration")
    print("=" * 60)
    
    # Simulation parameters
    duration = 20.0  # seconds
    dt = 0.1  # time step
    measurement_noise_std = 1.0  # meters
    process_noise_std = 0.1  # m/s^2
    
    # Initial state [x, y, vx, vy]
    initial_state = np.array([0.0, 0.0, 10.0, 5.0])
    
    print(f"\nSimulation Parameters:")
    print(f"  • Duration: {duration} seconds")
    print(f"  • Time Step: {dt} seconds")
    print(f"  • Initial Position: ({initial_state[0]:.1f}, {initial_state[1]:.1f}) m")
    print(f"  • Initial Velocity: ({initial_state[2]:.1f}, {initial_state[3]:.1f}) m/s")
    print(f"  • Measurement Noise: {measurement_noise_std} m")
    print(f"  • Process Noise: {process_noise_std} m/s²")
    
    # Generate true trajectory (with slight acceleration for realism)
    acceleration = np.array([0.5, -0.2])  # Small acceleration
    true_states, true_positions = generate_ca_trajectory(
        duration, dt, initial_state, acceleration, process_noise_std
    )
    
    # Generate measurements
    measurements = generate_measurements(true_positions, measurement_noise_std)
    
    print(f"\nGenerated {len(measurements)} measurements")
    print(f"  • Detected: {sum(1 for m in measurements if m is not None)}")
    print(f"  • Missed: {sum(1 for m in measurements if m is None)}")
    
    # Initialize Kalman filters
    print("\nInitializing Kalman Filters...")
    
    # CV Filter
    cv_filter = initialize_constant_velocity_filter(
        dim=2,
        dt=dt,
        process_noise_std=process_noise_std * 10,  # Higher process noise for CV
        measurement_noise_std=measurement_noise_std
    )
    cv_filter.x = initial_state.copy()
    
    # CA Filter
    ca_filter = initialize_constant_acceleration_filter(
        dim=2,
        dt=dt,
        process_noise_std=process_noise_std,
        measurement_noise_std=measurement_noise_std
    )
    # Initialize CA filter state
    ca_state = np.zeros(6)
    ca_state[:4] = initial_state
    ca_filter.x = ca_state
    
    # Run tracking
    print("\nRunning tracking algorithms...")
    
    cv_estimates = []
    ca_estimates = []
    cv_covariances = []
    ca_covariances = []
    
    for k, measurement in enumerate(measurements):
        # Predict
        cv_filter.predict(dt)
        ca_filter.predict(dt)
        
        # Update if measurement available
        if measurement is not None:
            cv_filter.update(measurement)
            ca_filter.update(measurement)
        
        # Store estimates
        cv_estimates.append(cv_filter.x[:2].copy())
        ca_estimates.append(ca_filter.x[:2].copy())
        cv_covariances.append(cv_filter.P[:2, :2].copy())
        ca_covariances.append(ca_filter.P[:2, :2].copy())
    
    cv_estimates = np.array(cv_estimates)
    ca_estimates = np.array(ca_estimates)
    
    # Calculate errors
    cv_errors = np.linalg.norm(cv_estimates - true_positions, axis=1)
    ca_errors = np.linalg.norm(ca_estimates - true_positions, axis=1)
    
    # Print summary statistics
    print("\nTracking Performance Summary:")
    print("-" * 40)
    print("Constant Velocity (CV) Model:")
    print(f"  • RMSE: {np.sqrt(np.mean(cv_errors**2)):.3f} m")
    print(f"  • Mean Error: {np.mean(cv_errors):.3f} m")
    print(f"  • Max Error: {np.max(cv_errors):.3f} m")
    
    print("\nConstant Acceleration (CA) Model:")
    print(f"  • RMSE: {np.sqrt(np.mean(ca_errors**2)):.3f} m")
    print(f"  • Mean Error: {np.mean(ca_errors):.3f} m")
    print(f"  • Max Error: {np.max(ca_errors):.3f} m")
    
    print("\nComparison:")
    cv_rmse = np.sqrt(np.mean(cv_errors**2))
    ca_rmse = np.sqrt(np.mean(ca_errors**2))
    improvement = (cv_rmse - ca_rmse) / cv_rmse * 100
    print(f"  • CA model shows {improvement:.1f}% improvement over CV model")
    
    # Visualize results
    print("\nGenerating visualization...")
    plot_tracking_results(
        true_positions, measurements,
        cv_estimates, ca_estimates,
        cv_errors, ca_errors,
        cv_covariances, ca_covariances
    )
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the demonstration
    run_basic_tracking_demo()