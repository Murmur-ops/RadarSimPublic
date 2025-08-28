#!/usr/bin/env python3
"""
Visualize Multi-Target Radar Scenario

This script creates an informative visualization of the multi-target tracking scenario,
showing target trajectories, radar location, detection zones, and clutter distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from typing import List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def generate_crossing_trajectories(n_targets: int = 5, duration: float = 30.0, dt: float = 0.1) -> List[dict]:
    """Generate multiple targets with crossing trajectories."""
    n_steps = int(duration / dt)
    targets = []
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i in range(n_targets):
        # Random birth and death times
        birth_time = np.random.randint(0, n_steps // 4)
        death_time = np.random.randint(3 * n_steps // 4, n_steps)
        
        # Generate trajectory
        trajectory = np.zeros((n_steps, 4))  # [x, y, vx, vy]
        
        # Initial state - targets start from different angles around the surveillance area
        angle = 2 * np.pi * i / n_targets + np.random.randn() * 0.2
        radius = 50 + np.random.randn() * 10
        speed = 5 + np.random.randn() * 2
        
        trajectory[birth_time, 0] = radius * np.cos(angle)  # x
        trajectory[birth_time, 1] = radius * np.sin(angle)  # y
        trajectory[birth_time, 2] = -speed * np.cos(angle)  # vx (toward center)
        trajectory[birth_time, 3] = -speed * np.sin(angle)  # vy
        
        # Add some curvature
        turn_rate = (np.random.rand() - 0.5) * 0.02
        
        # Propagate trajectory
        for k in range(birth_time + 1, death_time):
            # Update velocity with turn
            vx = trajectory[k-1, 2]
            vy = trajectory[k-1, 3]
            trajectory[k, 2] = vx * np.cos(turn_rate) - vy * np.sin(turn_rate)
            trajectory[k, 3] = vx * np.sin(turn_rate) + vy * np.cos(turn_rate)
            
            # Update position
            trajectory[k, 0] = trajectory[k-1, 0] + trajectory[k, 2] * dt
            trajectory[k, 1] = trajectory[k-1, 1] + trajectory[k, 3] * dt
            
            # Add small process noise
            trajectory[k, :2] += np.random.randn(2) * 0.1
        
        targets.append({
            'id': i,
            'trajectory': trajectory,
            'birth_time': birth_time,
            'death_time': death_time,
            'color': colors[i % len(colors)],
            'birth_time_sec': birth_time * dt,
            'death_time_sec': death_time * dt
        })
    
    return targets


def create_radar_scenario_visualization():
    """Create comprehensive visualization of multi-target radar scenario."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate scenario
    duration = 30.0  # seconds
    dt = 0.1
    n_targets = 5
    targets = generate_crossing_trajectories(n_targets, duration, dt)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Main radar view (large subplot on left)
    ax1 = plt.subplot(2, 3, (1, 4))
    
    # Radar at origin
    radar_pos = [0, 0]
    
    # Draw radar coverage area
    max_range = 100
    coverage_circle = Circle(radar_pos, max_range, fill=False, edgecolor='gray', 
                            linestyle='--', linewidth=1, alpha=0.5)
    ax1.add_patch(coverage_circle)
    
    # Draw range rings
    for r in [25, 50, 75, 100]:
        ring = Circle(radar_pos, r, fill=False, edgecolor='gray', 
                     linestyle=':', linewidth=0.5, alpha=0.3)
        ax1.add_patch(ring)
        ax1.text(r-5, 0, f'{r}m', fontsize=8, alpha=0.5)
    
    # Draw radar beam sectors (azimuth coverage)
    beam_width = 30  # degrees
    for angle in range(0, 360, beam_width):
        wedge = Wedge(radar_pos, max_range, angle, angle + beam_width,
                     fill=True, facecolor='cyan', alpha=0.05, edgecolor='cyan',
                     linewidth=0.5, linestyle='-')
        ax1.add_patch(wedge)
    
    # Draw radar icon
    radar_icon = Circle(radar_pos, 3, fill=True, facecolor='black', edgecolor='gold', linewidth=2)
    ax1.add_patch(radar_icon)
    ax1.plot([0], [0], 'y*', markersize=15, label='Radar', zorder=10)
    
    # Plot all target trajectories
    for target in targets:
        traj = target['trajectory']
        birth = target['birth_time']
        death = target['death_time']
        
        # Full trajectory (faded)
        ax1.plot(traj[birth:death, 0], traj[birth:death, 1], 
                color=target['color'], alpha=0.3, linewidth=1, linestyle='-')
        
        # Current position (at middle of scenario)
        mid_time = (birth + death) // 2
        if birth <= mid_time < death:
            ax1.plot(traj[mid_time, 0], traj[mid_time, 1], 'o',
                    color=target['color'], markersize=8, 
                    label=f"Target {target['id']+1}")
            
            # Velocity vector
            vel_scale = 3
            ax1.arrow(traj[mid_time, 0], traj[mid_time, 1],
                     traj[mid_time, 2] * vel_scale, traj[mid_time, 3] * vel_scale,
                     head_width=2, head_length=1, fc=target['color'], 
                     ec=target['color'], alpha=0.7)
        
        # Mark birth and death positions
        ax1.plot(traj[birth, 0], traj[birth, 1], '^', 
                color=target['color'], markersize=10, alpha=0.7)
        ax1.plot(traj[death-1, 0], traj[death-1, 1], 'v', 
                color=target['color'], markersize=10, alpha=0.7)
    
    # Add clutter region
    clutter_region = Rectangle((-100, -100), 200, 200, fill=True,
                              facecolor='red', alpha=0.05, 
                              edgecolor='red', linestyle='--', linewidth=1)
    ax1.add_patch(clutter_region)
    
    # Labels and formatting
    ax1.set_xlabel('East-West Position (m)', fontsize=12)
    ax1.set_ylabel('North-South Position (m)', fontsize=12)
    ax1.set_title('Multi-Target Radar Scenario - Spatial View', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_xlim([-110, 110])
    ax1.set_ylim([-110, 110])
    ax1.set_aspect('equal')
    
    # Add compass
    ax1.annotate('N', xy=(0, 105), ha='center', fontsize=12, fontweight='bold')
    ax1.annotate('E', xy=(105, 0), ha='center', fontsize=12, fontweight='bold')
    ax1.annotate('S', xy=(0, -105), ha='center', fontsize=12, fontweight='bold')
    ax1.annotate('W', xy=(-105, 0), ha='center', fontsize=12, fontweight='bold')
    
    # Range-Time plot (top right)
    ax2 = plt.subplot(2, 3, 2)
    time_axis = np.arange(0, duration, dt)
    
    for target in targets:
        traj = target['trajectory']
        birth = target['birth_time']
        death = target['death_time']
        
        # Calculate range from radar
        ranges = np.sqrt(traj[birth:death, 0]**2 + traj[birth:death, 1]**2)
        times = time_axis[birth:death]
        
        ax2.plot(times, ranges, color=target['color'], linewidth=2,
                label=f"Target {target['id']+1}", alpha=0.8)
    
    ax2.axhline(y=max_range, color='red', linestyle='--', alpha=0.5, label='Max Range')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Range from Radar (m)', fontsize=11)
    ax2.set_title('Target Range vs Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    ax2.set_xlim([0, duration])
    
    # Doppler (Radial Velocity) plot (middle right)
    ax3 = plt.subplot(2, 3, 3)
    
    for target in targets:
        traj = target['trajectory']
        birth = target['birth_time']
        death = target['death_time']
        
        # Calculate radial velocity (velocity component toward radar)
        radial_velocities = []
        for k in range(birth, death):
            pos = traj[k, :2]
            vel = traj[k, 2:4]
            range_vec = -pos / np.linalg.norm(pos)  # Unit vector from target to radar
            radial_vel = np.dot(vel, range_vec)  # Positive = approaching
            radial_velocities.append(radial_vel)
        
        times = time_axis[birth:death]
        ax3.plot(times, radial_velocities, color=target['color'], 
                linewidth=2, alpha=0.8)
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.fill_between([0, duration], [0, 0], [10, 10], alpha=0.1, color='red', label='Approaching')
    ax3.fill_between([0, duration], [0, 0], [-10, -10], alpha=0.1, color='blue', label='Receding')
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Radial Velocity (m/s)', fontsize=11)
    ax3.set_title('Doppler (Radial Velocity) vs Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=9)
    ax3.set_xlim([0, duration])
    
    # Target timeline (bottom right)
    ax4 = plt.subplot(2, 3, 5)
    
    for i, target in enumerate(targets):
        y_pos = i
        birth_sec = target['birth_time_sec']
        death_sec = target['death_time_sec']
        
        # Draw timeline bar
        ax4.barh(y_pos, death_sec - birth_sec, left=birth_sec, height=0.6,
                color=target['color'], alpha=0.7, 
                label=f"Target {target['id']+1}")
        
        # Mark key events
        ax4.plot(birth_sec, y_pos, '^', color=target['color'], markersize=10)
        ax4.plot(death_sec, y_pos, 'v', color=target['color'], markersize=10)
        
        # Add text
        ax4.text(birth_sec - 1, y_pos, f'T{target["id"]+1}', 
                fontsize=10, ha='right', va='center')
    
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('Target ID', fontsize=11)
    ax4.set_title('Target Lifetimes', fontsize=12, fontweight='bold')
    ax4.set_xlim([-2, duration + 2])
    ax4.set_ylim([-0.5, n_targets - 0.5])
    ax4.grid(True, axis='x', alpha=0.3)
    ax4.set_yticks(range(n_targets))
    ax4.set_yticklabels([f'T{i+1}' for i in range(n_targets)])
    
    # Scenario statistics (bottom right corner)
    ax5 = plt.subplot(2, 3, 6)
    ax5.axis('off')
    
    # Calculate statistics
    total_tracks = len(targets)
    avg_lifetime = np.mean([t['death_time_sec'] - t['birth_time_sec'] for t in targets])
    max_concurrent = 0
    for t in range(int(duration/dt)):
        concurrent = sum(1 for target in targets 
                        if target['birth_time'] <= t < target['death_time'])
        max_concurrent = max(max_concurrent, concurrent)
    
    # Calculate crossing points (simplified - when targets are within 20m of each other)
    crossing_events = 0
    min_separation = float('inf')
    for t in range(int(duration/dt)):
        active_targets = [target for target in targets 
                         if target['birth_time'] <= t < target['death_time']]
        for i in range(len(active_targets)):
            for j in range(i+1, len(active_targets)):
                dist = np.linalg.norm(
                    active_targets[i]['trajectory'][t, :2] - 
                    active_targets[j]['trajectory'][t, :2]
                )
                min_separation = min(min_separation, dist)
                if dist < 20:
                    crossing_events += 1
    
    stats_text = f"""Scenario Statistics:
    
Temporal:
  • Duration: {duration:.1f} seconds
  • Time Step: {dt:.3f} seconds
  • Total Targets: {total_tracks}
  • Avg Lifetime: {avg_lifetime:.1f} seconds
  • Max Concurrent: {max_concurrent} targets

Spatial:
  • Surveillance Area: 200m × 200m
  • Max Range: {max_range}m
  • Min Separation: {min_separation:.1f}m
  • Close Encounters: {crossing_events} events
  
Radar Parameters:
  • Position: (0, 0) - Center
  • Coverage: 360° azimuth
  • Range Resolution: ~1m
  • Update Rate: {1/dt:.1f} Hz
  
Clutter Model:
  • Type: Uniform distribution
  • Density: ~5 false alarms/scan
  • Region: Full surveillance area"""
    
    ax5.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    # Main title
    plt.suptitle('Multi-Target Radar Tracking Scenario Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('multi_target_scenario_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'multi_target_scenario_visualization.png'")
    plt.close()
    
    # Create a second figure showing measurement density and association challenges
    fig2, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Measurement density heatmap
    ax = axes[0, 0]
    
    # Generate synthetic measurements with clutter
    all_measurements = []
    for t in range(int(duration/dt)):
        # Target measurements
        for target in targets:
            if target['birth_time'] <= t < target['death_time']:
                if np.random.rand() < 0.9:  # 90% detection probability
                    pos = target['trajectory'][t, :2] + np.random.randn(2) * 1.0
                    all_measurements.append(pos)
        
        # Clutter
        n_clutter = np.random.poisson(5)
        for _ in range(n_clutter):
            clutter_pos = np.random.uniform(-100, 100, 2)
            all_measurements.append(clutter_pos)
    
    all_measurements = np.array(all_measurements)
    
    # Create 2D histogram
    h = ax.hist2d(all_measurements[:, 0], all_measurements[:, 1], 
                  bins=30, cmap='YlOrRd', cmin=1)
    plt.colorbar(h[3], ax=ax, label='Measurement Count')
    
    # Overlay true tracks
    for target in targets:
        traj = target['trajectory']
        birth = target['birth_time']
        death = target['death_time']
        ax.plot(traj[birth:death, 0], traj[birth:death, 1], 
               color='blue', alpha=0.5, linewidth=2, linestyle='--')
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Measurement Density (Targets + Clutter)')
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    ax.grid(True, alpha=0.3)
    
    # Association complexity over time
    ax = axes[0, 1]
    
    complexity_scores = []
    time_points = []
    
    for t in range(0, int(duration/dt), 5):  # Sample every 0.5 seconds
        active_targets = sum(1 for target in targets 
                           if target['birth_time'] <= t < target['death_time'])
        
        # Calculate minimum pairwise distances
        if active_targets > 1:
            active_trajs = [target['trajectory'][t, :2] for target in targets 
                          if target['birth_time'] <= t < target['death_time']]
            
            min_dist = float('inf')
            for i in range(len(active_trajs)):
                for j in range(i+1, len(active_trajs)):
                    dist = np.linalg.norm(active_trajs[i] - active_trajs[j])
                    min_dist = min(min_dist, dist)
            
            # Complexity score based on number of targets and proximity
            complexity = active_targets * (1 + 20/max(min_dist, 1))
        else:
            complexity = active_targets
        
        complexity_scores.append(complexity)
        time_points.append(t * dt)
    
    ax.plot(time_points, complexity_scores, 'b-', linewidth=2)
    ax.fill_between(time_points, 0, complexity_scores, alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Association Complexity Score')
    ax.set_title('Data Association Complexity Over Time')
    ax.grid(True, alpha=0.3)
    
    # SNR vs Range
    ax = axes[1, 0]
    
    # Radar equation: SNR ∝ 1/R^4
    ranges = np.linspace(10, 100, 100)
    snr_db = 40 - 40 * np.log10(ranges/10)  # Reference: 40 dB at 10m
    
    ax.plot(ranges, snr_db, 'g-', linewidth=2, label='Theoretical')
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Detection Threshold')
    ax.fill_between(ranges, -10, 10, alpha=0.2, color='red', label='Low SNR Region')
    
    # Plot actual target SNRs at mid-time
    for target in targets:
        mid_time = (target['birth_time'] + target['death_time']) // 2
        if target['birth_time'] <= mid_time < target['death_time']:
            target_range = np.linalg.norm(target['trajectory'][mid_time, :2])
            target_snr = 40 - 40 * np.log10(target_range/10)
            ax.plot(target_range, target_snr, 'o', color=target['color'], 
                   markersize=10, label=f"Target {target['id']+1}")
    
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('Signal-to-Noise Ratio vs Range')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim([0, 110])
    
    # Tracking challenges summary
    ax = axes[1, 1]
    ax.axis('off')
    
    challenges_text = """Tracking Challenges in This Scenario:

1. Crossing Trajectories
   • Multiple targets converge toward center
   • Risk of track swapping at intersections
   • Association ambiguity when targets are close

2. Clutter Environment
   • ~5 false alarms per scan
   • Total ~1500 false alarms over scenario
   • Must distinguish targets from clutter

3. Variable Target Density
   • 0-5 concurrent targets
   • Targets appear/disappear at different times
   • Track initiation and termination logic critical

4. Measurement Uncertainty
   • Position noise: 1m standard deviation
   • Missed detections: ~10% probability
   • Lower SNR at longer ranges

5. Association Complexity
   • GNN: O(n³) with Hungarian algorithm
   • JPDA: O(2^n) hypothesis enumeration
   • Real-time processing requirements

Solution Approaches:
   ✓ GNN: Fast, works well in sparse scenarios
   ✓ JPDA: Robust in ambiguous situations
   ✓ MHT: Maintains multiple hypotheses
   ✓ IMM: Adapts to changing dynamics"""
    
    ax.text(0.05, 0.5, challenges_text, fontsize=10, family='monospace',
           verticalalignment='center')
    
    plt.suptitle('Multi-Target Tracking Challenges Analysis', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('multi_target_challenges.png', dpi=150, bbox_inches='tight')
    print("Challenges analysis saved as 'multi_target_challenges.png'")
    plt.close()


if __name__ == "__main__":
    create_radar_scenario_visualization()
    print("\nVisualization complete! Generated two comprehensive plots:")
    print("1. 'multi_target_scenario_visualization.png' - Spatial and temporal analysis")
    print("2. 'multi_target_challenges.png' - Tracking challenges and complexity")