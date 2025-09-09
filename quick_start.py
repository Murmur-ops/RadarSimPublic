#!/usr/bin/env python3
"""
RadarSimPublic Quick Start Example
Demonstrates basic radar simulation with target detection
"""

import numpy as np
import matplotlib.pyplot as plt
from src.radar import Radar, RadarParameters
from src.target import Target, TargetType, TargetMotion
from src.environment import Environment, AtmosphericConditions
from src.signal import SignalProcessor


def main():
    """Run a simple radar simulation"""
    
    print("=" * 60)
    print("RadarSimPublic - Quick Start Demonstration")
    print("=" * 60)
    
    # 1. Configure Radar System
    print("\n1. Setting up X-band radar...")
    radar_params = RadarParameters(
        frequency=10e9,      # 10 GHz (X-band)
        power=100000,        # 100 kW peak power (more realistic for air surveillance)
        antenna_gain=35,     # 35 dB antenna gain
        pulse_width=2e-6,    # 2 microsecond pulse
        prf=2000,            # 2 kHz PRF
        bandwidth=50e6,      # 50 MHz bandwidth
        noise_figure=3.0,    # 3 dB noise figure
        losses=3.0           # 3 dB system losses
    )
    radar = Radar(radar_params)
    print(f"   - Frequency: {radar_params.frequency/1e9:.1f} GHz")
    print(f"   - Max range: {radar_params.max_unambiguous_range/1000:.1f} km")
    print(f"   - Range resolution: {radar_params.range_resolution:.1f} m")
    
    # 2. Create Targets
    print("\n2. Creating targets...")
    targets = []
    
    # Fighter aircraft
    fighter = Target(
        target_type=TargetType.AIRCRAFT,
        rcs=2.0,  # 2 m² RCS
        motion=TargetMotion(
            position=np.array([15000, 8000, 5000]),  # 15km range
            velocity=np.array([-250, -50, 0])        # Approaching
        )
    )
    targets.append(fighter)
    print(f"   - Fighter: Range={fighter.motion.range/1000:.1f} km, RCS={fighter.rcs_mean:.1f} m²")
    
    # Commercial aircraft
    airliner = Target(
        target_type=TargetType.AIRCRAFT,
        rcs=100.0,  # 100 m² RCS
        motion=TargetMotion(
            position=np.array([30000, 15000, 10000]),  # 30km range
            velocity=np.array([-200, 0, 0])            # Straight approach
        )
    )
    targets.append(airliner)
    print(f"   - Airliner: Range={airliner.motion.range/1000:.1f} km, RCS={airliner.rcs_mean:.1f} m²")
    
    # Small drone
    drone = Target(
        target_type=TargetType.DRONE,
        rcs=0.01,  # 0.01 m² RCS
        motion=TargetMotion(
            position=np.array([3000, 2000, 500]),  # 3km range
            velocity=np.array([20, 15, 0])         # Slow moving
        )
    )
    targets.append(drone)
    print(f"   - Drone: Range={drone.motion.range/1000:.1f} km, RCS={drone.rcs_mean:.1f} m²")
    
    # 3. Set Environment
    print("\n3. Setting environment conditions...")
    environment = Environment()
    print(f"   - Temperature: {environment.conditions.temperature}°C")
    print(f"   - Humidity: {environment.conditions.humidity}%")
    
    # 4. Calculate Detection Performance
    print("\n4. Calculating detection performance...")
    print("\n   Target Detection Analysis:")
    print("   " + "-" * 50)
    print(f"   {'Target':<12} {'Range(km)':<10} {'SNR(dB)':<10} {'Detectable':<12}")
    print("   " + "-" * 50)
    
    detection_results = []
    for i, target in enumerate(targets):
        # Calculate SNR
        range_m = target.motion.range
        snr_db = radar.snr(range_m, target.rcs_mean)
        
        # Apply atmospheric losses
        atten_db = environment.atmospheric_attenuation(
            radar_params.frequency, range_m
        )
        snr_db -= atten_db
        
        # Detection decision (threshold = 13 dB)
        detectable = snr_db >= 13.0
        
        target_names = ['Fighter', 'Airliner', 'Drone']
        print(f"   {target_names[i]:<12} {range_m/1000:<10.1f} {snr_db:<10.1f} "
              f"{'✅ Yes' if detectable else '❌ No':<12}")
        
        detection_results.append({
            'name': target_names[i],
            'range': range_m,
            'snr': snr_db,
            'detectable': detectable
        })
    
    # 5. Visualize Results
    print("\n5. Generating visualization...")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: SNR vs Range
    ax1.set_title('SNR vs Range for Different Targets')
    ranges = np.linspace(1000, 50000, 100)
    
    # Plot SNR curves for different RCS values
    for rcs, label in [(0.01, 'Small Drone'), (2.0, 'Fighter'), (100.0, 'Airliner')]:
        snr_curve = []
        for r in ranges:
            snr = radar.snr(r, rcs)
            atten = environment.atmospheric_attenuation(radar_params.frequency, r)
            snr_curve.append(snr - atten)
        ax1.plot(ranges/1000, snr_curve, label=f'{label} (RCS={rcs} m²)', linewidth=2)
    
    # Add detection threshold
    ax1.axhline(y=13.0, color='r', linestyle='--', label='Detection Threshold')
    ax1.set_xlabel('Range (km)')
    ax1.set_ylabel('SNR (dB)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim([0, 50])
    ax1.set_ylim([-10, 50])
    
    # Right plot: PPI Display
    ax2 = plt.subplot(122, projection='polar')
    ax2.set_title('Plan Position Indicator (PPI)')
    
    # Plot range rings
    for r in [10, 20, 30, 40]:
        circle = plt.Circle((0, 0), r, fill=False, edgecolor='green', 
                           alpha=0.3, transform=ax2.transData._b)
        ax2.add_patch(circle)
    
    # Plot targets
    for i, target in enumerate(targets):
        r = target.motion.range / 1000  # Convert to km
        theta = np.arctan2(target.motion.position[1], target.motion.position[0])
        
        # Color based on detectability
        color = 'green' if detection_results[i]['detectable'] else 'red'
        marker = ['v', 'o', 's'][i]  # Different markers for each target
        
        ax2.plot(theta, r, marker=marker, color=color, markersize=10, 
                label=detection_results[i]['name'])
    
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_rlim(0, 40)
    ax2.set_rlabel_position(45)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('RadarSimPublic - Target Detection Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 6. Summary
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"✅ Detected: {sum(1 for r in detection_results if r['detectable'])} targets")
    print(f"❌ Missed: {sum(1 for r in detection_results if not r['detectable'])} targets")
    print("\nKey Insights:")
    print("- Large RCS targets (airliners) detectable at long range")
    print("- Small drones challenging to detect beyond a few km")
    print("- Fighter aircraft detectable at medium ranges")
    
    plt.show()
    
    return detection_results


if __name__ == "__main__":
    results = main()
    print("\n👉 Try modifying target positions, RCS values, or radar parameters!")
    print("👉 Explore more examples in the 'examples/' directory")