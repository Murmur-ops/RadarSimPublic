#!/usr/bin/env python3
"""
Simple demonstration of RadarSim capabilities
This script works with the existing RadarSim architecture
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.radar import Radar, RadarParameters
from src.target import Target, TargetType, TargetMotion

def radar_equation(power, gain, wavelength, sigma, range_m, losses=1.0):
    """
    Calculate received power using radar equation
    
    P_r = (P_t * G^2 * λ^2 * σ) / ((4π)^3 * R^4 * L)
    """
    numerator = power * (10**(gain/10))**2 * wavelength**2 * sigma
    denominator = (4 * np.pi)**3 * range_m**4 * (10**(losses/10))
    return numerator / denominator

def calculate_snr(radar_params, target, noise_temp=290):
    """Calculate SNR for a target"""
    # Get range to target
    range_m = np.linalg.norm(target.motion.position)
    
    # Received power
    pr = radar_equation(
        radar_params.power,
        radar_params.antenna_gain,
        radar_params.wavelength,
        target.rcs_mean,
        range_m,
        radar_params.losses
    )
    
    # Noise power
    k_b = 1.38e-23  # Boltzmann constant
    noise_power = k_b * noise_temp * radar_params.bandwidth * (10**(radar_params.noise_figure/10))
    
    # SNR
    snr_linear = pr / noise_power
    snr_db = 10 * np.log10(snr_linear)
    
    return snr_db

def main():
    print("=" * 60)
    print("RadarSim - Simple Demonstration")
    print("=" * 60)
    
    # Create radar parameters
    print("\n1. Creating radar system...")
    params = RadarParameters(
        frequency=10e9,      # 10 GHz
        power=1000,          # 1 kW
        antenna_gain=35,     # 35 dB
        pulse_width=1e-6,    # 1 μs
        prf=1000,            # 1 kHz
        bandwidth=1e6,       # 1 MHz
        noise_figure=3,      # 3 dB
        losses=3             # 3 dB
    )
    
    radar = Radar(params)
    
    print(f"   Frequency: {params.frequency/1e9:.1f} GHz")
    print(f"   Wavelength: {params.wavelength:.3f} m")
    print(f"   Power: {params.power} W")
    print(f"   Max unambiguous range: {params.max_unambiguous_range/1000:.1f} km")
    print(f"   Range resolution: {params.range_resolution:.1f} m")
    
    # Create targets
    print("\n2. Creating targets...")
    targets = [
        Target(
            target_type=TargetType.AIRCRAFT,
            rcs=10.0,
            motion=TargetMotion(
                position=np.array([10000.0, 5000.0, 3000.0]),
                velocity=np.array([-200.0, 0.0, 0.0])
            )
        ),
        Target(
            target_type=TargetType.AIRCRAFT,
            rcs=50.0,
            motion=TargetMotion(
                position=np.array([25000.0, 10000.0, 8000.0]),
                velocity=np.array([-150.0, -50.0, 0.0])
            )
        ),
        Target(
            target_type=TargetType.MISSILE,
            rcs=0.1,
            motion=TargetMotion(
                position=np.array([5000.0, 2000.0, 1000.0]),
                velocity=np.array([-300.0, 0.0, 0.0])
            )
        )
    ]
    
    target_names = ["Aircraft-1", "Aircraft-2", "Missile"]
    for target, name in zip(targets, target_names):
        range_km = np.linalg.norm(target.motion.position) / 1000
        print(f"   {name}: Range={range_km:.1f} km, RCS={target.rcs_mean} m²")
    
    # Simulate detection over time
    print("\n3. Running detection simulation...")
    simulation_time = 10.0
    dt = 0.1
    num_steps = int(simulation_time / dt)
    
    detection_threshold = 13.0  # 13 dB SNR
    
    # Storage
    detection_history = {name: [] for name in target_names}
    range_history = {name: [] for name in target_names}
    snr_history = {name: [] for name in target_names}
    
    for step in range(num_steps):
        time = step * dt
        
        for target, name in zip(targets, target_names):
            # Update position
            target.motion.position = target.motion.position + target.motion.velocity * dt
            
            # Calculate SNR
            snr = calculate_snr(params, target)
            
            # Store data
            range_km = np.linalg.norm(target.motion.position) / 1000
            range_history[name].append(range_km)
            snr_history[name].append(snr)
            detection_history[name].append(snr > detection_threshold)
    
    # Print detection statistics
    print("\n4. Detection Statistics:")
    for target, name in zip(targets, target_names):
        detections = sum(detection_history[name])
        rate = 100 * detections / num_steps
        avg_snr = np.mean(snr_history[name])
        print(f"   {name}:")
        print(f"     Detection rate: {rate:.1f}%")
        print(f"     Average SNR: {avg_snr:.1f} dB")
    
    # Create visualization
    print("\n5. Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    time_vec = np.arange(num_steps) * dt
    
    # Plot 1: SNR vs Time
    ax = axes[0, 0]
    for name in target_names:
        ax.plot(time_vec, snr_history[name], 
               label=name, linewidth=2)
    ax.axhline(y=detection_threshold, color='red', linestyle='--', 
              label='Detection Threshold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('Signal-to-Noise Ratio')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Range vs Time
    ax = axes[0, 1]
    for name in target_names:
        ax.plot(time_vec, range_history[name], 
               label=name, linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Range (km)')
    ax.set_title('Target Range')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Detection Status
    ax = axes[1, 0]
    y_positions = list(range(len(targets)))
    for i, name in enumerate(target_names):
        detections = detection_history[name]
        detect_times = [t for t, d in zip(time_vec, detections) if d]
        ax.scatter(detect_times, [i] * len(detect_times), 
                  label=name, s=20, alpha=0.6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Target')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(target_names)
    ax.set_title('Detection Events')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, simulation_time)
    
    # Plot 4: Detection Probability vs Range
    ax = axes[1, 1]
    ax.set_title('Detection Performance')
    ax.axis('off')
    
    info_text = "Radar Parameters:\n"
    info_text += f"Frequency: {params.frequency/1e9:.1f} GHz\n"
    info_text += f"Power: {params.power} W\n"
    info_text += f"Antenna Gain: {params.antenna_gain} dB\n"
    info_text += f"PRF: {params.prf} Hz\n"
    info_text += f"Detection Threshold: {detection_threshold} dB\n\n"
    info_text += "Detection Results:\n"
    
    for name in target_names:
        rate = 100 * sum(detection_history[name]) / num_steps
        info_text += f"{name}: {rate:.0f}%\n"
    
    ax.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
           verticalalignment='center')
    
    plt.suptitle('RadarSim - Simple Detection Demo', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_dir = Path('../results/getting_started')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'simple_demo.png', dpi=150)
    print(f"   Saved to: {output_dir / 'simple_demo.png'}")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("Next: Try running 'python ../run_scenario.py first_scenario.yaml'")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())