#!/usr/bin/env python3
"""
Getting Started - Your First Radar Simulation

This script demonstrates the basics of RadarSim by:
1. Creating a simple radar system
2. Generating a target
3. Running detection and tracking
4. Visualizing the results

Run this script from the GETTING_STARTED directory:
    python run_first_simulation.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import RadarSim modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.radar import Radar, RadarParameters
from src.target import Target
from src.environment import Environment
from src.tracking.kalman_filters import KalmanFilter, initialize_constant_velocity_filter

def create_simple_radar():
    """Create a basic pulse-Doppler radar system"""
    # Create radar parameters
    params = RadarParameters(
        frequency=10e9,         # X-band (10 GHz)
        bandwidth=1e6,          # 1 MHz bandwidth
        power=1000,             # 1 kW transmit power
        antenna_gain=35,        # 35 dB antenna gain
        losses=3,               # 3 dB system losses
        noise_figure=3,         # 3 dB noise figure
        prf=1000,               # 1 kHz PRF
        pulse_width=1e-6        # 1 microsecond pulse
    )
    
    radar = Radar(params)
    # Set additional attributes for compatibility
    radar.position = np.array([0.0, 0.0, 10.0])
    radar.detection_threshold = 13  # 13 dB SNR threshold
    return radar

def create_target_scenario():
    """Create a simple target scenario with an aircraft"""
    from src.target import Target, TargetType, TargetMotion
    
    # Commercial aircraft flying at 10km altitude
    motion = TargetMotion(
        position=np.array([30000.0, 15000.0, 10000.0]),  # 30km range
        velocity=np.array([-200.0, -50.0, 0.0]),         # 206 m/s ground speed
    )
    
    target = Target(
        target_type=TargetType.AIRCRAFT,
        rcs=100.0,              # 100 m² RCS (large aircraft)
        motion=motion
    )
    
    # Add attributes needed by the simulation
    target.target_id = "Flight-001"
    target.position = motion.position
    target.velocity = motion.velocity
    target.rcs = target.rcs_mean  # Use the mean RCS value
    
    return target

def run_basic_simulation():
    """Run a basic radar simulation"""
    print("=" * 60)
    print("RadarSim - Getting Started Tutorial")
    print("=" * 60)
    
    # Step 1: Create radar and target
    print("\n1. Setting up radar system...")
    radar = create_simple_radar()
    print(f"   Radar at position: {radar.position}")
    print(f"   Operating frequency: {radar.frequency/1e9:.1f} GHz")
    print(f"   Transmit power: {radar.power:.0f} W")
    
    print("\n2. Creating target...")
    target = create_target_scenario()
    print(f"   Target ID: {target.target_id}")
    print(f"   Initial position: {target.position/1000} km")
    print(f"   Velocity: {np.linalg.norm(target.velocity):.1f} m/s")
    print(f"   RCS: {target.rcs} m²")
    
    # Step 2: Create environment
    print("\n3. Setting up environment...")
    from src.environment import Environment, AtmosphericConditions
    conditions = AtmosphericConditions(
        temperature=15,
        pressure=1013.25,
        humidity=60
    )
    environment = Environment(conditions=conditions)
    
    # Step 3: Initialize tracking
    print("\n4. Initializing Kalman filter for tracking...")
    kf = initialize_constant_velocity_filter(
        dim=2,  # 2D tracking (x, y)
        dt=0.1,
        process_noise_std=1.0,
        measurement_noise_std=10.0
    )
    
    # Set initial state
    kf.x[0] = target.position[0]  # x position
    kf.x[1] = target.velocity[0]  # x velocity
    kf.x[2] = target.position[1]  # y position
    kf.x[3] = target.velocity[1]  # y velocity
    
    # Step 4: Run simulation
    print("\n5. Running simulation...")
    simulation_time = 30.0  # 30 seconds
    dt = 0.1  # 100ms update rate
    num_steps = int(simulation_time / dt)
    
    # Storage for results
    true_positions = []
    detections = []
    tracks = []
    snr_history = []
    
    print(f"   Duration: {simulation_time}s")
    print(f"   Time step: {dt}s")
    print(f"   Total steps: {num_steps}")
    
    print("\n6. Starting simulation loop...")
    for step in range(num_steps):
        current_time = step * dt
        
        # Update target position
        target.position = target.position + target.velocity * dt
        true_positions.append(target.position.copy())
        
        # Calculate SNR
        snr = radar.calculate_snr(target, environment)
        snr_history.append(snr)
        
        # Detection logic
        if snr > radar.detection_threshold:
            # Add measurement noise
            range_to_target = np.linalg.norm(target.position - radar.position)
            azimuth = np.arctan2(target.position[1], target.position[0])
            
            # Add realistic measurement errors
            range_error = np.random.normal(0, 30)  # 30m range error
            azimuth_error = np.random.normal(0, np.radians(0.1))  # 0.1° azimuth error
            
            measured_range = range_to_target + range_error
            measured_azimuth = azimuth + azimuth_error
            
            # Convert to Cartesian for tracking
            measured_x = measured_range * np.cos(measured_azimuth)
            measured_y = measured_range * np.sin(measured_azimuth)
            
            detection = {
                'time': current_time,
                'range': measured_range,
                'azimuth': np.degrees(measured_azimuth),
                'position': np.array([measured_x, measured_y, target.position[2]]),
                'snr': snr
            }
            detections.append(detection)
            
            # Update Kalman filter
            kf.predict(dt)
            measurement = np.array([measured_x, measured_y])
            kf.update(measurement)
            
            # Store track
            track_state = kf.state.copy()
            track = {
                'time': current_time,
                'position': np.array([track_state[0], track_state[2], target.position[2]]),
                'velocity': np.array([track_state[1], track_state[3], 0])
            }
            tracks.append(track)
        
        # Progress indicator
        if step % 50 == 0:
            print(f"   Step {step}/{num_steps}: "
                  f"Range={np.linalg.norm(target.position - radar.position)/1000:.1f}km, "
                  f"SNR={snr:.1f}dB, "
                  f"Detected={'Yes' if snr > radar.detection_threshold else 'No'}")
    
    print(f"\n7. Simulation complete!")
    print(f"   Total detections: {len(detections)}")
    print(f"   Detection rate: {100*len(detections)/num_steps:.1f}%")
    print(f"   Average SNR: {np.mean(snr_history):.1f} dB")
    
    return true_positions, detections, tracks, snr_history

def visualize_results(true_positions, detections, tracks, snr_history):
    """Create visualization of simulation results"""
    print("\n8. Creating visualizations...")
    
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: PPI Display (Plan Position Indicator)
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_title('PPI Display - Detections', fontsize=12, fontweight='bold')
    ax1.set_xlabel('East (km)')
    ax1.set_ylabel('North (km)')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot range rings
    for r in [10, 20, 30, 40]:
        circle = plt.Circle((0, 0), r, fill=False, color='gray', alpha=0.3, linestyle='--')
        ax1.add_patch(circle)
    
    # Plot radar position
    ax1.plot(0, 0, 'k^', markersize=10, label='Radar')
    
    # Plot true trajectory
    true_positions = np.array(true_positions)
    ax1.plot(true_positions[:, 0]/1000, true_positions[:, 1]/1000, 
             'b-', alpha=0.5, linewidth=2, label='True Path')
    
    # Plot detections
    if detections:
        det_positions = np.array([d['position'] for d in detections])
        ax1.scatter(det_positions[:, 0]/1000, det_positions[:, 1]/1000, 
                   c='red', s=10, alpha=0.6, label='Detections')
    
    ax1.legend(loc='upper right')
    ax1.set_xlim(-5, 35)
    ax1.set_ylim(-5, 20)
    
    # Plot 2: Tracking Performance
    ax2 = plt.subplot(2, 3, 2)
    ax2.set_title('Tracking Performance', fontsize=12, fontweight='bold')
    ax2.set_xlabel('East (km)')
    ax2.set_ylabel('North (km)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Plot true path
    ax2.plot(true_positions[:, 0]/1000, true_positions[:, 1]/1000, 
             'b-', alpha=0.5, linewidth=2, label='True Path')
    
    # Plot tracked path
    if tracks:
        track_positions = np.array([t['position'] for t in tracks])
        ax2.plot(track_positions[:, 0]/1000, track_positions[:, 1]/1000, 
                'g--', linewidth=2, label='Tracked Path')
    
    ax2.legend(loc='upper right')
    ax2.set_xlim(-5, 35)
    ax2.set_ylim(-5, 20)
    
    # Plot 3: SNR History
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_title('SNR History', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('SNR (dB)')
    ax3.grid(True, alpha=0.3)
    
    time_vector = np.arange(len(snr_history)) * 0.1
    ax3.plot(time_vector, snr_history, 'b-', linewidth=2)
    ax3.axhline(y=13, color='r', linestyle='--', label='Detection Threshold')
    ax3.legend(loc='upper right')
    
    # Plot 4: Range vs Time
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_title('Range vs Time', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Range (km)')
    ax4.grid(True, alpha=0.3)
    
    ranges = [np.linalg.norm(pos) / 1000 for pos in true_positions]
    ax4.plot(time_vector, ranges, 'b-', linewidth=2, label='True Range')
    
    if detections:
        det_times = [d['time'] for d in detections]
        det_ranges = [d['range']/1000 for d in detections]
        ax4.scatter(det_times, det_ranges, c='red', s=10, alpha=0.6, label='Measured')
    
    ax4.legend(loc='upper right')
    
    # Plot 5: Tracking Error
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_title('Tracking Error', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Position Error (m)')
    ax5.grid(True, alpha=0.3)
    
    if tracks:
        errors = []
        error_times = []
        for i, track in enumerate(tracks):
            # Find corresponding true position
            track_time = track['time']
            true_index = int(track_time / 0.1)
            if true_index < len(true_positions):
                true_pos = true_positions[true_index]
                track_pos = track['position']
                error = np.linalg.norm(true_pos[:2] - track_pos[:2])
                errors.append(error)
                error_times.append(track_time)
        
        if errors:
            ax5.plot(error_times, errors, 'g-', linewidth=2)
            avg_error = np.mean(errors)
            ax5.axhline(y=avg_error, color='r', linestyle='--', 
                       label=f'Avg: {avg_error:.1f}m')
            ax5.legend(loc='upper right')
    
    # Plot 6: Detection Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.set_title('Detection Statistics', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # Calculate statistics
    total_steps = len(snr_history)
    num_detections = len(detections)
    detection_rate = 100 * num_detections / total_steps
    avg_snr = np.mean(snr_history)
    max_snr = np.max(snr_history)
    min_snr = np.min(snr_history)
    
    stats_text = f"""
    Simulation Statistics:
    ━━━━━━━━━━━━━━━━━━━━━
    Total Time Steps: {total_steps}
    Detections: {num_detections}
    Detection Rate: {detection_rate:.1f}%
    
    SNR Statistics:
    Average: {avg_snr:.1f} dB
    Maximum: {max_snr:.1f} dB
    Minimum: {min_snr:.1f} dB
    
    Tracking Performance:
    Track Updates: {len(tracks)}
    """
    
    if tracks and len(errors) > 0:
        stats_text += f"Avg Position Error: {np.mean(errors):.1f} m\n"
        stats_text += f"Max Position Error: {np.max(errors):.1f} m\n"
    
    ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.suptitle('RadarSim - Getting Started Tutorial Results', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('results/getting_started')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'first_simulation.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Saved visualization to: {output_file}")
    
    plt.show()
    
    return fig

def main():
    """Main function to run the getting started tutorial"""
    try:
        # Run simulation
        true_positions, detections, tracks, snr_history = run_basic_simulation()
        
        # Visualize results
        visualize_results(true_positions, detections, tracks, snr_history)
        
        print("\n" + "=" * 60)
        print("Tutorial Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Try modifying radar parameters in create_simple_radar()")
        print("2. Change target trajectory in create_target_scenario()")
        print("3. Run a YAML scenario: python ../run_scenario.py first_scenario.yaml")
        print("4. Explore advanced features in the demos/ directory")
        print("\nHappy simulating!")
        
    except Exception as e:
        print(f"\nError during simulation: {e}")
        print("Please check your installation and try again.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())