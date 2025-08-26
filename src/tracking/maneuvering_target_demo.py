#!/usr/bin/env python3
"""
Maneuvering Target Tracking Demonstration

This script demonstrates tracking of maneuvering targets using the Interacting
Multiple Model (IMM) filter. It shows how the IMM adapts to different motion
patterns and compares its performance against single-model filters.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch
from typing import List, Tuple, Dict
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tracking.imm_filter import IMMFilter, IMMParameters
from src.tracking.kalman_filters import KalmanFilter, initialize_constant_velocity_filter
from src.tracking.motion_models import ConstantVelocityModel, ConstantAccelerationModel, CoordinatedTurnModel


def generate_maneuvering_trajectory(duration: float, dt: float) -> Tuple[np.ndarray, List[str]]:
    """
    Generate a complex maneuvering trajectory with multiple motion phases.
    
    Returns:
        states: Array of true states [x, y, vx, vy]
        phases: List of phase descriptions
    """
    n_steps = int(duration / dt)
    states = np.zeros((n_steps, 4))
    phases = []
    
    # Initial state
    states[0] = [0, 0, 10, 0]  # Start at origin, moving right
    
    # Define motion phases
    phase_definitions = [
        (0.0, 5.0, "Constant Velocity", "cv"),
        (5.0, 10.0, "Right Turn", "ct_right"),
        (10.0, 15.0, "Acceleration", "ca"),
        (15.0, 20.0, "Left Turn", "ct_left"),
        (20.0, 25.0, "Deceleration", "ca_neg"),
        (25.0, 30.0, "Straight Line", "cv"),
        (30.0, 35.0, "Sharp Right Turn", "ct_sharp"),
        (35.0, 40.0, "Constant Velocity", "cv")
    ]
    
    current_idx = 0
    
    for start_time, end_time, description, motion_type in phase_definitions:
        start_idx = int(start_time / dt)
        end_idx = min(int(end_time / dt), n_steps)
        phases.append(f"t={start_time:.0f}-{end_time:.0f}s: {description}")
        
        for k in range(max(1, start_idx), end_idx):
            if k >= n_steps:
                break
                
            # Get current state
            x, y, vx, vy = states[k-1]
            
            if motion_type == "cv":
                # Constant velocity
                states[k, 0] = x + vx * dt
                states[k, 1] = y + vy * dt
                states[k, 2] = vx
                states[k, 3] = vy
                
            elif motion_type == "ct_right":
                # Coordinated turn (right)
                omega = -0.1  # rad/s (negative for right turn)
                cos_wt = np.cos(omega * dt)
                sin_wt = np.sin(omega * dt)
                states[k, 0] = x + (sin_wt * vx - (1 - cos_wt) * vy) / omega
                states[k, 1] = y + ((1 - cos_wt) * vx + sin_wt * vy) / omega
                states[k, 2] = cos_wt * vx - sin_wt * vy
                states[k, 3] = sin_wt * vx + cos_wt * vy
                
            elif motion_type == "ct_left":
                # Coordinated turn (left)
                omega = 0.1  # rad/s (positive for left turn)
                cos_wt = np.cos(omega * dt)
                sin_wt = np.sin(omega * dt)
                states[k, 0] = x + (sin_wt * vx - (1 - cos_wt) * vy) / omega
                states[k, 1] = y + ((1 - cos_wt) * vx + sin_wt * vy) / omega
                states[k, 2] = cos_wt * vx - sin_wt * vy
                states[k, 3] = sin_wt * vx + cos_wt * vy
                
            elif motion_type == "ct_sharp":
                # Sharp coordinated turn
                omega = -0.2  # rad/s (sharper turn)
                cos_wt = np.cos(omega * dt)
                sin_wt = np.sin(omega * dt)
                states[k, 0] = x + (sin_wt * vx - (1 - cos_wt) * vy) / omega
                states[k, 1] = y + ((1 - cos_wt) * vx + sin_wt * vy) / omega
                states[k, 2] = cos_wt * vx - sin_wt * vy
                states[k, 3] = sin_wt * vx + cos_wt * vy
                
            elif motion_type == "ca":
                # Constant acceleration
                ax, ay = 2.0, 1.0  # m/s^2
                states[k, 0] = x + vx * dt + 0.5 * ax * dt**2
                states[k, 1] = y + vy * dt + 0.5 * ay * dt**2
                states[k, 2] = vx + ax * dt
                states[k, 3] = vy + ay * dt
                
            elif motion_type == "ca_neg":
                # Deceleration
                ax, ay = -2.0, -0.5  # m/s^2
                states[k, 2] = max(2.0, vx + ax * dt)  # Don't go below minimum speed
                states[k, 3] = vy + ay * dt
                states[k, 0] = x + vx * dt + 0.5 * ax * dt**2
                states[k, 1] = y + vy * dt + 0.5 * ay * dt**2
            
            # Add small process noise
            states[k] += np.random.randn(4) * np.array([0.05, 0.05, 0.01, 0.01])
    
    return states, phases


def generate_noisy_measurements(states: np.ndarray, measurement_noise_std: float = 1.0,
                               detection_prob: float = 0.95) -> List[np.ndarray]:
    """Generate noisy position measurements from true states."""
    measurements = []
    
    for state in states:
        if np.random.rand() < detection_prob:
            noise = np.random.randn(2) * measurement_noise_std
            measurements.append(state[:2] + noise)
        else:
            measurements.append(None)
    
    return measurements


def run_single_model_filter(measurements: List[np.ndarray], model_type: str, 
                          dt: float, initial_state: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Run a single-model Kalman filter."""
    
    # Initialize filter based on model type
    if model_type == 'cv':
        filter = initialize_constant_velocity_filter(
            dim=2, dt=dt, process_noise_std=1.0, measurement_noise_std=1.0
        )
    else:  # ca
        from src.tracking.kalman_filters import initialize_constant_acceleration_filter
        filter = initialize_constant_acceleration_filter(
            dim=2, dt=dt, process_noise_std=0.5, measurement_noise_std=1.0
        )
        # Extend initial state for CA model
        extended_state = np.zeros(6)
        extended_state[:4] = initial_state
        initial_state = extended_state
    
    filter.x = initial_state.copy()
    
    # Run filter
    estimates = []
    covariances = []
    
    for measurement in measurements:
        filter.predict(dt)
        if measurement is not None:
            filter.update(measurement)
        
        estimates.append(filter.x[:2].copy())
        covariances.append(filter.P[:2, :2].copy())
    
    return np.array(estimates), covariances


def plot_maneuvering_results(true_states: np.ndarray, measurements: List[np.ndarray],
                            imm_estimates: np.ndarray, cv_estimates: np.ndarray,
                            ca_estimates: np.ndarray, model_probs: np.ndarray,
                            phases: List[str], dt: float):
    """Create comprehensive visualization of maneuvering target tracking."""
    
    fig = plt.figure(figsize=(18, 12))
    
    # Trajectory comparison
    ax1 = plt.subplot(3, 3, (1, 4))
    
    # True trajectory with phase colors
    phase_colors = ['green', 'orange', 'red', 'orange', 'blue', 'green', 'red', 'green']
    phase_duration = 5.0  # seconds
    samples_per_phase = int(phase_duration / dt)
    
    for i, phase in enumerate(phases):
        start_idx = i * samples_per_phase
        end_idx = min((i + 1) * samples_per_phase, len(true_states))
        if start_idx < len(true_states):
            ax1.plot(true_states[start_idx:end_idx, 0], true_states[start_idx:end_idx, 1],
                    '-', color=phase_colors[i], linewidth=3, alpha=0.7)
    
    # Plot measurements
    meas_x = [m[0] for m in measurements if m is not None]
    meas_y = [m[1] for m in measurements if m is not None]
    ax1.scatter(meas_x, meas_y, c='gray', s=10, alpha=0.3, label='Measurements')
    
    # Plot estimates
    ax1.plot(imm_estimates[:, 0], imm_estimates[:, 1], 'b-', linewidth=2, label='IMM', alpha=0.8)
    ax1.plot(cv_estimates[:, 0], cv_estimates[:, 1], 'g--', linewidth=1, label='CV Only', alpha=0.6)
    ax1.plot(ca_estimates[:, 0], ca_estimates[:, 1], 'r--', linewidth=1, label='CA Only', alpha=0.6)
    
    # Mark start and end
    ax1.plot(true_states[0, 0], true_states[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(true_states[-1, 0], true_states[-1, 1], 'rs', markersize=10, label='End')
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Maneuvering Target Trajectory')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Model probabilities over time
    ax2 = plt.subplot(3, 3, (2, 3))
    time = np.arange(len(model_probs)) * dt
    
    # Add phase boundaries
    for i in range(1, len(phases)):
        ax2.axvline(x=i * phase_duration, color='gray', linestyle='--', alpha=0.5)
    
    ax2.fill_between(time, 0, model_probs[:, 0], alpha=0.5, color='blue', label='CV Model')
    ax2.fill_between(time, model_probs[:, 0], model_probs[:, 0] + model_probs[:, 1], 
                     alpha=0.5, color='red', label='CA Model')
    ax2.fill_between(time, model_probs[:, 0] + model_probs[:, 1], 1, 
                     alpha=0.5, color='orange', label='CT Model')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Model Probability')
    ax2.set_title('IMM Model Probabilities Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Position errors
    ax3 = plt.subplot(3, 3, 5)
    imm_errors = np.linalg.norm(imm_estimates - true_states[:, :2], axis=1)
    cv_errors = np.linalg.norm(cv_estimates - true_states[:, :2], axis=1)
    ca_errors = np.linalg.norm(ca_estimates - true_states[:, :2], axis=1)
    
    ax3.plot(time, imm_errors, 'b-', label=f'IMM (avg={np.mean(imm_errors):.2f}m)')
    ax3.plot(time, cv_errors, 'g--', label=f'CV (avg={np.mean(cv_errors):.2f}m)', alpha=0.6)
    ax3.plot(time, ca_errors, 'r--', label=f'CA (avg={np.mean(ca_errors):.2f}m)', alpha=0.6)
    
    # Add phase boundaries
    for i in range(1, len(phases)):
        ax3.axvline(x=i * phase_duration, color='gray', linestyle='--', alpha=0.5)
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position Error (m)')
    ax3.set_title('Tracking Errors Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Velocity comparison
    ax4 = plt.subplot(3, 3, 6)
    true_speed = np.linalg.norm(true_states[:, 2:4], axis=1)
    ax4.plot(time, true_speed, 'g-', linewidth=2, label='True Speed')
    
    # Add phase boundaries
    for i in range(1, len(phases)):
        ax4.axvline(x=i * phase_duration, color='gray', linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Speed (m/s)')
    ax4.set_title('Target Speed Profile')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Turn rate analysis
    ax5 = plt.subplot(3, 3, 7)
    
    # Calculate turn rates
    turn_rates = []
    for i in range(1, len(true_states)):
        v_prev = true_states[i-1, 2:4]
        v_curr = true_states[i, 2:4]
        
        # Calculate angle change
        angle_prev = np.arctan2(v_prev[1], v_prev[0])
        angle_curr = np.arctan2(v_curr[1], v_curr[0])
        angle_diff = angle_curr - angle_prev
        
        # Normalize to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        turn_rate = angle_diff / dt
        turn_rates.append(turn_rate)
    
    ax5.plot(time[1:], np.array(turn_rates), 'g-', label='Turn Rate')
    ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add phase boundaries
    for i in range(1, len(phases)):
        ax5.axvline(x=i * phase_duration, color='gray', linestyle='--', alpha=0.5)
    
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Turn Rate (rad/s)')
    ax5.set_title('Target Turn Rate')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Error statistics by phase
    ax6 = plt.subplot(3, 3, 8)
    
    phase_errors = {'IMM': [], 'CV': [], 'CA': []}
    phase_labels = []
    
    for i in range(len(phases)):
        start_idx = i * samples_per_phase
        end_idx = min((i + 1) * samples_per_phase, len(imm_errors))
        
        if start_idx < len(imm_errors):
            phase_errors['IMM'].append(np.mean(imm_errors[start_idx:end_idx]))
            phase_errors['CV'].append(np.mean(cv_errors[start_idx:end_idx]))
            phase_errors['CA'].append(np.mean(ca_errors[start_idx:end_idx]))
            phase_labels.append(f"Phase {i+1}")
    
    x = np.arange(len(phase_labels))
    width = 0.25
    
    ax6.bar(x - width, phase_errors['IMM'], width, label='IMM', color='blue')
    ax6.bar(x, phase_errors['CV'], width, label='CV', color='green')
    ax6.bar(x + width, phase_errors['CA'], width, label='CA', color='red')
    
    ax6.set_xlabel('Motion Phase')
    ax6.set_ylabel('Mean Error (m)')
    ax6.set_title('Error by Motion Phase')
    ax6.set_xticks(x)
    ax6.set_xticklabels(phase_labels, rotation=45)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Phase descriptions
    ax7 = plt.subplot(3, 3, 9)
    ax7.axis('off')
    
    phase_text = "Motion Phases:\n\n"
    for i, phase in enumerate(phases):
        phase_text += f"{i+1}. {phase}\n"
    
    phase_text += f"\n\nPerformance Summary:\n"
    phase_text += f"IMM RMSE: {np.sqrt(np.mean(imm_errors**2)):.3f} m\n"
    phase_text += f"CV RMSE:  {np.sqrt(np.mean(cv_errors**2)):.3f} m\n"
    phase_text += f"CA RMSE:  {np.sqrt(np.mean(ca_errors**2)):.3f} m\n\n"
    
    imm_improvement_cv = (np.mean(cv_errors) - np.mean(imm_errors)) / np.mean(cv_errors) * 100
    imm_improvement_ca = (np.mean(ca_errors) - np.mean(imm_errors)) / np.mean(ca_errors) * 100
    
    phase_text += f"IMM Improvement:\n"
    phase_text += f"  vs CV: {imm_improvement_cv:.1f}%\n"
    phase_text += f"  vs CA: {imm_improvement_ca:.1f}%"
    
    ax7.text(0.1, 0.5, phase_text, fontsize=9, family='monospace',
            verticalalignment='center')
    
    plt.suptitle('Maneuvering Target Tracking with IMM Filter', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('maneuvering_target_demo.png', dpi=100, bbox_inches='tight')
    print("Plot saved as 'maneuvering_target_demo.png'")
    plt.close()


def run_maneuvering_target_demo():
    """Run the maneuvering target tracking demonstration."""
    
    print("=" * 60)
    print("Maneuvering Target Tracking Demonstration")
    print("=" * 60)
    
    # Simulation parameters
    duration = 40.0  # seconds
    dt = 0.1  # time step
    measurement_noise_std = 1.0
    
    print(f"\nSimulation Parameters:")
    print(f"  • Duration: {duration} seconds")
    print(f"  • Time Step: {dt} seconds")
    print(f"  • Measurement Noise: {measurement_noise_std} m")
    
    # Generate maneuvering trajectory
    print("\nGenerating maneuvering trajectory...")
    true_states, phases = generate_maneuvering_trajectory(duration, dt)
    
    print("\nMotion phases:")
    for i, phase in enumerate(phases):
        print(f"  {i+1}. {phase}")
    
    # Generate measurements
    print("\nGenerating noisy measurements...")
    measurements = generate_noisy_measurements(true_states, measurement_noise_std)
    
    detected = sum(1 for m in measurements if m is not None)
    print(f"  • Generated {len(measurements)} time steps")
    print(f"  • Detected: {detected} ({detected/len(measurements)*100:.1f}%)")
    
    # Initialize IMM filter
    print("\nInitializing IMM filter with three models (CV, CA, CT)...")
    
    # Create IMM parameters
    params = IMMParameters(
        dt=dt,
        model_types=['cv', 'ca', 'ct'],
        transition_probabilities=np.array([
            [0.80, 0.10, 0.10],  # From CV
            [0.15, 0.70, 0.15],  # From CA
            [0.15, 0.15, 0.70]   # From CT
        ]),
        initial_model_probabilities=np.array([0.70, 0.20, 0.10]),
        process_noise_stds=[0.5, 1.0, 0.3],
        measurement_noise_std=measurement_noise_std
    )
    
    imm_filter = IMMFilter(params)
    
    # Initialize state
    initial_state = true_states[0].copy()
    initial_cov = np.diag([1.0, 1.0, 0.1, 0.1])
    imm_filter.initialize_state(initial_state, initial_cov)
    
    # Run IMM filter
    print("\nRunning IMM filter...")
    imm_estimates = []
    model_probabilities = []
    
    for measurement in measurements:
        imm_filter.predict()
        if measurement is not None:
            imm_filter.update(measurement)
        
        state, _ = imm_filter.get_state_estimate()
        imm_estimates.append(state[:2].copy())
        model_probabilities.append(imm_filter.get_model_probabilities().copy())
    
    imm_estimates = np.array(imm_estimates)
    model_probabilities = np.array(model_probabilities)
    
    # Run single-model filters for comparison
    print("\nRunning single-model filters for comparison...")
    
    # CV filter
    cv_estimates, _ = run_single_model_filter(measurements, 'cv', dt, initial_state)
    
    # CA filter
    ca_estimates, _ = run_single_model_filter(measurements, 'ca', dt, initial_state)
    
    # Calculate performance metrics
    print("\nPerformance Analysis:")
    print("-" * 40)
    
    imm_errors = np.linalg.norm(imm_estimates - true_states[:, :2], axis=1)
    cv_errors = np.linalg.norm(cv_estimates - true_states[:, :2], axis=1)
    ca_errors = np.linalg.norm(ca_estimates - true_states[:, :2], axis=1)
    
    print(f"Root Mean Square Error (RMSE):")
    print(f"  • IMM Filter: {np.sqrt(np.mean(imm_errors**2)):.3f} m")
    print(f"  • CV Only:    {np.sqrt(np.mean(cv_errors**2)):.3f} m")
    print(f"  • CA Only:    {np.sqrt(np.mean(ca_errors**2)):.3f} m")
    
    print(f"\nMean Absolute Error:")
    print(f"  • IMM Filter: {np.mean(imm_errors):.3f} m")
    print(f"  • CV Only:    {np.mean(cv_errors):.3f} m")
    print(f"  • CA Only:    {np.mean(ca_errors):.3f} m")
    
    print(f"\nMaximum Error:")
    print(f"  • IMM Filter: {np.max(imm_errors):.3f} m")
    print(f"  • CV Only:    {np.max(cv_errors):.3f} m")
    print(f"  • CA Only:    {np.max(ca_errors):.3f} m")
    
    # Model probability analysis
    print(f"\nModel Probability Statistics:")
    print(f"  • CV Model: Mean={np.mean(model_probabilities[:, 0]):.3f}, "
          f"Max={np.max(model_probabilities[:, 0]):.3f}")
    print(f"  • CA Model: Mean={np.mean(model_probabilities[:, 1]):.3f}, "
          f"Max={np.max(model_probabilities[:, 1]):.3f}")
    print(f"  • CT Model: Mean={np.mean(model_probabilities[:, 2]):.3f}, "
          f"Max={np.max(model_probabilities[:, 2]):.3f}")
    
    # Visualize results
    print("\nGenerating visualization...")
    plot_maneuvering_results(
        true_states, measurements,
        imm_estimates, cv_estimates, ca_estimates,
        model_probabilities, phases, dt
    )
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the demonstration
    run_maneuvering_target_demo()