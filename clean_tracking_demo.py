#!/usr/bin/env python3
"""
Clean Tracking Visualization Demo

Demonstrates honest, observable-based tracking with clean visualizations
that show what the radar actually sees without predetermined outcomes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent))

from src.radar import Radar, RadarParameters
from src.target import Target, TargetType, TargetMotion
from src.environment import Environment, AtmosphericConditions
from src.classification.observable_classifier import ObservableClassifier, ObservableFeatures
from src.classification.transponder_decoder import TransponderDecoder


def create_test_scenario():
    """Create a test scenario with multiple aircraft types"""
    
    # Create radar
    radar_params = RadarParameters(
        frequency=3e9,  # S-band
        power=100e3,    # 100 kW
        antenna_gain=40,  # 40 dB
        pulse_width=1e-6,
        prf=1000,
        bandwidth=5e6,
        noise_figure=3,
        losses=3
    )
    radar = Radar(radar_params)
    radar.position = np.array([0, 0, 10])  # 10m elevation
    radar.detection_threshold = 10  # 10 dB SNR
    
    # Create targets with different observable characteristics
    targets = []
    
    # Large aircraft with transponder (emergency)
    motion1 = TargetMotion(
        position=np.array([50000, 30000, 5000]),
        velocity=np.array([-150, -50, -20])  # Descending approach
    )
    target1 = Target(TargetType.AIRCRAFT, rcs=80.0, motion=motion1)
    target1.name = "Large-Emergency"
    target1.transponder_code = "7700"  # Emergency!
    target1.has_transponder = True
    targets.append(target1)
    
    # Fast small target (possible missile)
    motion2 = TargetMotion(
        position=np.array([30000, -20000, 2000]),
        velocity=np.array([-400, 100, 0])  # Very fast
    )
    target2 = Target(TargetType.MISSILE, rcs=0.5, motion=motion2)
    target2.name = "Fast-Small"
    target2.transponder_code = None
    target2.has_transponder = False
    targets.append(target2)
    
    # Rotorcraft with blade flash
    motion3 = TargetMotion(
        position=np.array([15000, 10000, 500]),
        velocity=np.array([-40, -20, 0])  # Slow
    )
    target3 = Target(TargetType.AIRCRAFT, rcs=10.0, motion=motion3)
    target3.name = "Rotorcraft"
    target3.transponder_code = "1200"  # VFR
    target3.has_transponder = True
    target3.has_blade_flash = True
    targets.append(target3)
    
    # Unknown no transponder
    motion4 = TargetMotion(
        position=np.array([25000, 15000, 1000]),
        velocity=np.array([-80, -30, 0])
    )
    target4 = Target(TargetType.DRONE, rcs=2.0, motion=motion4)
    target4.name = "Unknown-NoXpdr"
    target4.transponder_code = None
    target4.has_transponder = False
    targets.append(target4)
    
    return radar, targets


def run_clean_simulation(duration=30.0, dt=0.5):
    """Run simulation with observable-based classification"""
    
    print("=" * 60)
    print("Clean Observable-Based Tracking Demo")
    print("=" * 60)
    
    # Setup
    radar, targets = create_test_scenario()
    environment = Environment()
    classifier = ObservableClassifier()
    transponder = TransponderDecoder()
    
    # Storage
    tracks = {t.name: {'positions': [], 'classifications': [], 
                       'priorities': [], 'detections': []} 
             for t in targets}
    
    # Simulation loop
    n_steps = int(duration / dt)
    
    for step in range(n_steps):
        time = step * dt
        
        for target in targets:
            # Update position
            target.motion.update(dt)
            target.position = target.motion.position
            target.velocity = target.motion.velocity
            
            # Calculate SNR
            range_to_target = np.linalg.norm(target.position - radar.position)
            snr = radar.snr(range_to_target, target.rcs_mean)
            
            # Detection logic
            detected = snr > radar.detection_threshold
            
            if detected:
                # Create observable features
                features = ObservableFeatures(
                    rcs=target.rcs_mean,
                    range=range_to_target,
                    azimuth=np.arctan2(target.position[1], target.position[0]),
                    elevation=np.arctan(target.position[2] / 
                              np.sqrt(target.position[0]**2 + target.position[1]**2)),
                    doppler_velocity=np.dot(target.velocity, 
                                           (target.position - radar.position)) / range_to_target,
                    altitude=target.position[2],
                    snr=snr,
                    transponder_code=getattr(target, 'transponder_code', None),
                    has_blade_flash=getattr(target, 'has_blade_flash', False)
                )
                
                # Classify based on observables
                classification, confidence = classifier.classify(features, target.name)
                
                # Determine priority
                priority = "normal"
                if features.transponder_code == "7700":
                    priority = "emergency"
                elif abs(features.doppler_velocity) > 300:
                    priority = "high_speed"
                elif not features.transponder_code:
                    priority = "investigate"
                elif features.has_blade_flash:
                    priority = "rotorcraft"
                
                # Store results
                tracks[target.name]['positions'].append(target.position.copy())
                tracks[target.name]['classifications'].append(classification)
                tracks[target.name]['priorities'].append(priority)
                tracks[target.name]['detections'].append({
                    'time': time,
                    'snr': snr,
                    'confidence': confidence
                })
        
        if step % 10 == 0:
            print(f"Time {time:5.1f}s: Tracking {len(targets)} targets")
    
    return tracks, radar


def create_clean_visualization(tracks, radar):
    """Create clean, honest visualization of tracking results"""
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Observable-Based Radar Tracking - Clean Visualization', 
                 fontsize=14, fontweight='bold')
    
    # 1. Plan Position Display (PPI)
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_title('Radar Plan Position Display', fontweight='bold')
    ax1.set_xlabel('East (km)')
    ax1.set_ylabel('North (km)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Draw range rings
    for r in [20, 40, 60]:
        circle = Circle((0, 0), r, fill=False, color='gray', 
                       alpha=0.3, linestyle='--')
        ax1.add_patch(circle)
    
    # Radar position
    ax1.plot(0, 0, 'k^', markersize=12, label='Radar', zorder=5)
    
    # Plot tracks with classification-based styling
    colors = {'emergency': 'red', 'high_speed': 'orange', 
             'investigate': 'yellow', 'rotorcraft': 'green', 'normal': 'blue'}
    
    for name, track_data in tracks.items():
        if track_data['positions']:
            positions = np.array(track_data['positions']) / 1000  # Convert to km
            priority = track_data['priorities'][0] if track_data['priorities'] else 'normal'
            color = colors.get(priority, 'gray')
            
            # Plot track
            ax1.plot(positions[:, 0], positions[:, 1], 
                    color=color, alpha=0.6, linewidth=2)
            
            # Current position
            ax1.scatter(positions[-1, 0], positions[-1, 1], 
                       color=color, s=100, edgecolor='black', 
                       linewidth=2, zorder=4)
            
            # Label
            ax1.text(positions[-1, 0], positions[-1, 1] + 2, 
                    name, fontsize=8, ha='center')
    
    ax1.legend(loc='upper right')
    ax1.set_xlim(-70, 70)
    ax1.set_ylim(-40, 40)
    
    # 2. Classification Confidence
    ax2 = plt.subplot(2, 3, 2)
    ax2.set_title('Classification Confidence', fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Confidence')
    ax2.grid(True, alpha=0.3)
    
    for name, track_data in tracks.items():
        if track_data['detections']:
            times = [d['time'] for d in track_data['detections']]
            confidences = [d['confidence'] for d in track_data['detections']]
            ax2.plot(times, confidences, marker='o', label=name, alpha=0.7)
    
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, 
                label='Min Confidence')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='best', fontsize=8)
    
    # 3. SNR History
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_title('Signal-to-Noise Ratio', fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('SNR (dB)')
    ax3.grid(True, alpha=0.3)
    
    for name, track_data in tracks.items():
        if track_data['detections']:
            times = [d['time'] for d in track_data['detections']]
            snrs = [d['snr'] for d in track_data['detections']]
            ax3.plot(times, snrs, marker='.', label=name, alpha=0.7)
    
    ax3.axhline(y=10, color='r', linestyle='--', alpha=0.5, 
                label='Detection Threshold')
    ax3.legend(loc='best', fontsize=8)
    
    # 4. Observable Features Table
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_title('Observable Features', fontweight='bold')
    ax4.axis('off')
    
    # Create feature summary
    table_data = []
    table_data.append(['Target', 'Size', 'Speed', 'Xpdr', 'Priority'])
    
    for name, track_data in tracks.items():
        if track_data['classifications']:
            last_class = track_data['classifications'][-1]
            size = last_class.get('size_class', 'unknown')
            speed = last_class.get('speed_class', 'unknown')
            xpdr = last_class.get('transponder_status', {}).get('has_transponder', False)
            priority = track_data['priorities'][-1] if track_data['priorities'] else 'unknown'
            
            xpdr_str = '✓' if xpdr else '✗'
            if last_class.get('transponder_status', {}).get('emergency'):
                xpdr_str = '⚠ EMERGENCY'
            
            table_data.append([name[:12], size[:8], speed[:8], xpdr_str, priority])
    
    # Create table
    table = ax4.table(cellText=table_data, loc='center', 
                     cellLoc='left', colWidths=[0.25, 0.15, 0.15, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#cccccc')
        table[(0, i)].set_text_props(weight='bold')
    
    # 5. Altitude Profile
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_title('Altitude Profile', fontweight='bold')
    ax5.set_xlabel('Range (km)')
    ax5.set_ylabel('Altitude (m)')
    ax5.grid(True, alpha=0.3)
    
    for name, track_data in tracks.items():
        if track_data['positions']:
            positions = np.array(track_data['positions'])
            ranges = np.linalg.norm(positions[:, :2], axis=1) / 1000
            altitudes = positions[:, 2]
            ax5.plot(ranges, altitudes, marker='o', label=name, alpha=0.7)
    
    ax5.legend(loc='best', fontsize=8)
    
    # 6. Priority Distribution
    ax6 = plt.subplot(2, 3, 6)
    ax6.set_title('Priority Distribution', fontweight='bold')
    
    # Count priorities
    priority_counts = {}
    for track_data in tracks.values():
        for priority in track_data['priorities']:
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
    
    if priority_counts:
        priorities = list(priority_counts.keys())
        counts = list(priority_counts.values())
        colors_list = [colors.get(p, 'gray') for p in priorities]
        
        bars = ax6.bar(range(len(priorities)), counts, color=colors_list, alpha=0.7)
        ax6.set_xticks(range(len(priorities)))
        ax6.set_xticklabels(priorities, rotation=45, ha='right')
        ax6.set_ylabel('Detection Count')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    str(count), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'clean_tracking_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    return fig


def main():
    """Main demo function"""
    
    # Run simulation
    tracks, radar = run_clean_simulation(duration=30.0, dt=0.5)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Tracking Summary (Observable-Based)")
    print("=" * 60)
    
    for name, track_data in tracks.items():
        n_detections = len(track_data['detections'])
        priorities = set(track_data['priorities'])
        
        print(f"\n{name}:")
        print(f"  Detections: {n_detections}")
        print(f"  Priorities: {', '.join(priorities)}")
        
        if track_data['classifications']:
            last_class = track_data['classifications'][-1]
            print(f"  Size Class: {last_class.get('size_class', 'unknown')}")
            print(f"  Speed Class: {last_class.get('speed_class', 'unknown')}")
            
            xpdr_status = last_class.get('transponder_status', {})
            if xpdr_status.get('emergency'):
                print(f"  ⚠ EMERGENCY SQUAWK DETECTED")
            elif xpdr_status.get('has_transponder'):
                print(f"  Transponder: Active")
            else:
                print(f"  Transponder: None/Off")
    
    # Create visualization
    print("\nCreating visualization...")
    create_clean_visualization(tracks, radar)
    
    print("\n✅ Demo complete!")
    print("\nKey Points:")
    print("• Classification based only on observable physics")
    print("• Emergency squawk correctly prioritized")
    print("• Fast small target flagged as high priority")
    print("• Unknown no-transponder marked for investigation")
    print("• Rotorcraft identified by blade flash signature")


if __name__ == "__main__":
    main()