#!/usr/bin/env python3
"""
Networked Radar System Demonstration

Demonstrates a 3-node radar network tracking multiple targets with:
- Distributed tracking and data fusion
- Covariance Intersection for consistent fusion
- Comparison of centralized vs decentralized architectures
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
from matplotlib.animation import FuncAnimation
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.networked_radar import (
    NetworkedRadar, NetworkNode, RadarNodeType,
    CommunicationLink, NetworkArchitecture
)
from src.networked_radar import (
    DataFusionCenter, FusionArchitecture,
    NetworkTrack, DistributedTrackFusion
)
from src.radar import RadarParameters


def create_radar_network():
    """Create a 3-node radar network configuration."""
    
    # Radar parameters (same for all nodes for simplicity)
    radar_params = RadarParameters(
        frequency=10e9,  # X-band
        power=1000,  # 1 kW
        antenna_gain=30,  # dB
        pulse_width=1e-6,
        prf=1000,
        bandwidth=10e6,
        noise_figure=3,
        losses=2
    )
    
    # Create network nodes
    nodes = [
        NetworkNode(
            node_id="radar_1",
            position=np.array([-20000, 0, 0]),  # 20 km west
            node_type=RadarNodeType.MONOSTATIC,
            radar_params=radar_params
        ),
        NetworkNode(
            node_id="radar_2",
            position=np.array([20000, 0, 0]),  # 20 km east
            node_type=RadarNodeType.MONOSTATIC,
            radar_params=radar_params
        ),
        NetworkNode(
            node_id="radar_3",
            position=np.array([0, 20000, 0]),  # 20 km north
            node_type=RadarNodeType.MONOSTATIC,
            radar_params=radar_params
        )
    ]
    
    # Create networked radars
    radars = []
    for node in nodes:
        radar = NetworkedRadar(
            params=radar_params,
            node=node,
            network_architecture=NetworkArchitecture.CENTRALIZED
        )
        radars.append(radar)
    
    # Create communication links (full connectivity)
    for i, radar1 in enumerate(radars):
        for j, radar2 in enumerate(radars):
            if i != j:
                link = CommunicationLink(
                    source_id=radar1.node.node_id,
                    target_id=radar2.node.node_id,
                    bandwidth=1e6,  # 1 Mbps
                    latency=0.01,  # 10 ms
                    packet_loss_rate=0.01  # 1% loss
                )
                radar1.add_connection(radar2.node, link)
    
    return radars, nodes


def simulate_targets(t):
    """Generate target positions and velocities."""
    targets = []
    
    # Target 1: Straight line trajectory
    targets.append({
        'id': 'target_1',
        'position': np.array([-10000 + 200*t, 15000, 5000]),
        'velocity': np.array([200, 0, 0]),
        'rcs': 10.0
    })
    
    # Target 2: Curved trajectory
    omega = 0.05
    targets.append({
        'id': 'target_2',
        'position': np.array([
            15000 * np.cos(omega * t),
            15000 * np.sin(omega * t),
            3000
        ]),
        'velocity': np.array([
            -15000 * omega * np.sin(omega * t),
            15000 * omega * np.cos(omega * t),
            0
        ]),
        'rcs': 5.0
    })
    
    # Target 3: Crossing trajectory
    targets.append({
        'id': 'target_3',
        'position': np.array([10000, -10000 + 150*t, 4000]),
        'velocity': np.array([0, 150, 0]),
        'rcs': 8.0
    })
    
    return targets


def generate_measurements(radars, targets, noise_std=50.0):
    """Generate noisy measurements from each radar."""
    measurements = {}
    
    for radar in radars:
        radar_measurements = []
        
        for target in targets:
            # Calculate range from radar to target
            range_vec = target['position'] - radar.node.position
            range_m = np.linalg.norm(range_vec)
            
            # Check if in detection range (100 km)
            if range_m > 100000:
                continue
            
            # Calculate SNR
            snr = radar.snr(range_m, target['rcs'])
            
            # Simple detection model
            if snr > 10:  # 10 dB threshold
                # Add measurement noise
                measured_pos = target['position'] + np.random.randn(3) * noise_std
                
                # Create measurement
                measurement = {
                    'position': measured_pos,
                    'timestamp': 0,  # Will be set by caller
                    'snr': snr,
                    'target_id_truth': target['id']  # For evaluation only
                }
                radar_measurements.append(measurement)
        
        measurements[radar.node.node_id] = radar_measurements
    
    return measurements


def create_network_tracks(measurements, timestamp):
    """Convert measurements to network tracks."""
    tracks = {}
    
    for node_id, node_measurements in measurements.items():
        node_tracks = []
        
        for i, meas in enumerate(node_measurements):
            # Simple track initialization (would use Kalman filter in practice)
            track = NetworkTrack(
                track_id=f"{node_id}_track_{i}",
                local_id=f"local_{i}",
                source_node=node_id,
                state=np.concatenate([meas['position'], np.zeros(3)]),  # [x,y,z,vx,vy,vz]
                covariance=np.diag([100, 100, 100, 10, 10, 10]),  # Position and velocity uncertainty
                timestamp=timestamp,
                quality=min(1.0, meas['snr'] / 30.0)  # Quality based on SNR
            )
            track.associated_nodes.add(node_id)
            node_tracks.append(track)
        
        tracks[node_id] = node_tracks
    
    return tracks


def compare_fusion_architectures(radars, simulation_time=60.0, dt=1.0):
    """Compare different fusion architectures."""
    
    # Create fusion centers for each architecture
    centralized_center = DataFusionCenter(
        center_id="central_fusion",
        architecture=FusionArchitecture.CENTRALIZED
    )
    
    decentralized_center = DataFusionCenter(
        center_id="decentral_fusion",
        architecture=FusionArchitecture.DECENTRALIZED
    )
    
    # Results storage
    results = {
        'centralized': {'tracks': [], 'rmse': [], 'consistency': []},
        'decentralized': {'tracks': [], 'rmse': [], 'consistency': []},
        'truth': {'positions': []}
    }
    
    # Simulation loop
    time_steps = np.arange(0, simulation_time, dt)
    
    for t in time_steps:
        # Generate true target positions
        true_targets = simulate_targets(t)
        results['truth']['positions'].append(true_targets)
        
        # Generate measurements from each radar
        measurements = generate_measurements(radars, true_targets)
        
        # Convert to network tracks
        network_tracks = create_network_tracks(measurements, t)
        
        # Centralized fusion
        for node_id, tracks in network_tracks.items():
            centralized_center.receive_local_tracks(node_id, tracks)
        
        centralized_fused = centralized_center.perform_fusion()
        results['centralized']['tracks'].append(centralized_fused)
        
        # Decentralized fusion
        for node_id, tracks in network_tracks.items():
            decentralized_center.receive_local_tracks(node_id, tracks)
        
        decentralized_fused = decentralized_center.perform_fusion()
        results['decentralized']['tracks'].append(decentralized_fused)
        
        # Calculate RMSE for evaluation
        if centralized_fused:
            rmse = calculate_rmse(centralized_fused, true_targets)
            results['centralized']['rmse'].append(rmse)
        
        if decentralized_fused:
            rmse = calculate_rmse(decentralized_fused, true_targets)
            results['decentralized']['rmse'].append(rmse)
    
    return results, time_steps


def calculate_rmse(fused_tracks, true_targets):
    """Calculate root mean square error."""
    errors = []
    
    for track in fused_tracks.values():
        # Find closest true target
        track_pos = track.state[:3]
        min_error = float('inf')
        
        for target in true_targets:
            error = np.linalg.norm(track_pos - target['position'])
            min_error = min(min_error, error)
        
        errors.append(min_error)
    
    return np.sqrt(np.mean(np.square(errors))) if errors else 0.0


def visualize_network(radars, results, time_steps):
    """Visualize the radar network and tracking results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Network topology and coverage
    ax1 = axes[0, 0]
    ax1.set_title('Radar Network Topology and Coverage', fontsize=12, fontweight='bold')
    ax1.set_xlabel('East-West (km)')
    ax1.set_ylabel('North-South (km)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot radar nodes and coverage
    colors = ['red', 'blue', 'green']
    for i, radar in enumerate(radars):
        pos = radar.node.position / 1000  # Convert to km
        ax1.scatter(pos[0], pos[1], s=200, c=colors[i], marker='^', 
                   label=radar.node.node_id, zorder=5)
        
        # Coverage circle (100 km range)
        coverage = Circle((pos[0], pos[1]), 100, fill=False, 
                         edgecolor=colors[i], linestyle='--', alpha=0.3)
        ax1.add_patch(coverage)
    
    # Plot communication links
    for i, radar1 in enumerate(radars):
        for j, radar2 in enumerate(radars):
            if i < j:
                pos1 = radar1.node.position / 1000
                pos2 = radar2.node.position / 1000
                ax1.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                        'k--', alpha=0.2, linewidth=1)
    
    ax1.legend(loc='upper right')
    ax1.set_xlim([-150, 150])
    ax1.set_ylim([-150, 150])
    
    # 2. Target trajectories and fused tracks
    ax2 = axes[0, 1]
    ax2.set_title('Target Trajectories and Fused Tracks', fontsize=12, fontweight='bold')
    ax2.set_xlabel('East-West (km)')
    ax2.set_ylabel('North-South (km)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Plot true trajectories
    for target_idx in range(3):
        trajectory = []
        for targets in results['truth']['positions']:
            if target_idx < len(targets):
                trajectory.append(targets[target_idx]['position'] / 1000)
        
        if trajectory:
            trajectory = np.array(trajectory)
            ax2.plot(trajectory[:, 0], trajectory[:, 1], 'k-', 
                    alpha=0.5, linewidth=2, label=f'True Target {target_idx+1}')
    
    # Plot centralized fusion results (sample every 5th point)
    for i in range(0, len(results['centralized']['tracks']), 5):
        tracks = results['centralized']['tracks'][i]
        for track in tracks.values():
            pos = track.state[:2] / 1000
            ax2.scatter(pos[0], pos[1], c='blue', s=10, alpha=0.3)
    
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_xlim([-50, 50])
    ax2.set_ylim([-50, 50])
    
    # 3. RMSE comparison
    ax3 = axes[1, 0]
    ax3.set_title('Tracking Error Comparison', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('RMSE (m)')
    ax3.grid(True, alpha=0.3)
    
    if results['centralized']['rmse']:
        ax3.plot(time_steps[:len(results['centralized']['rmse'])], 
                results['centralized']['rmse'], 
                'b-', label='Centralized', linewidth=2)
    
    if results['decentralized']['rmse']:
        ax3.plot(time_steps[:len(results['decentralized']['rmse'])], 
                results['decentralized']['rmse'], 
                'r--', label='Decentralized', linewidth=2)
    
    ax3.legend(loc='upper right')
    ax3.set_ylim([0, 500])
    
    # 4. Fusion statistics
    ax4 = axes[1, 1]
    ax4.set_title('Fusion Architecture Comparison', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # Calculate statistics
    cent_rmse = np.mean(results['centralized']['rmse']) if results['centralized']['rmse'] else 0
    decent_rmse = np.mean(results['decentralized']['rmse']) if results['decentralized']['rmse'] else 0
    
    stats_text = f"""
    Centralized Architecture:
    • Average RMSE: {cent_rmse:.1f} m
    • Communication: All-to-center
    • Processing: Single point
    • Latency: Higher
    • Robustness: Single point of failure
    
    Decentralized Architecture:
    • Average RMSE: {decent_rmse:.1f} m
    • Communication: Neighbor-to-neighbor
    • Processing: Distributed
    • Latency: Lower
    • Robustness: Graceful degradation
    
    Network Configuration:
    • 3 monostatic radars
    • 100 km detection range
    • 1 Mbps data links
    • 10 ms latency
    • Covariance Intersection fusion
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def main():
    """Run the networked radar demonstration."""
    print("=" * 60)
    print("NETWORKED RADAR SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Create radar network
    print("\nCreating 3-node radar network...")
    radars, nodes = create_radar_network()
    
    # Run simulation
    print("Running simulation (60 seconds)...")
    results, time_steps = compare_fusion_architectures(radars, simulation_time=60.0)
    
    # Visualize results
    print("Generating visualization...")
    fig = visualize_network(radars, results, time_steps)
    
    # Save figure
    output_dir = "results/networked_radar"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "networked_radar_demo.png")
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    
    cent_rmse = np.mean(results['centralized']['rmse']) if results['centralized']['rmse'] else 0
    decent_rmse = np.mean(results['decentralized']['rmse']) if results['decentralized']['rmse'] else 0
    
    print(f"\nCentralized Fusion:")
    print(f"  Average RMSE: {cent_rmse:.1f} m")
    print(f"  Final tracks: {len(results['centralized']['tracks'][-1]) if results['centralized']['tracks'] else 0}")
    
    print(f"\nDecentralized Fusion:")
    print(f"  Average RMSE: {decent_rmse:.1f} m")
    print(f"  Final tracks: {len(results['decentralized']['tracks'][-1]) if results['decentralized']['tracks'] else 0}")
    
    print(f"\nPerformance Comparison:")
    if cent_rmse > 0 and decent_rmse > 0:
        improvement = (decent_rmse - cent_rmse) / cent_rmse * 100
        if improvement > 0:
            print(f"  Centralized is {abs(improvement):.1f}% better")
        else:
            print(f"  Decentralized is {abs(improvement):.1f}% better")
    
    plt.show()


if __name__ == "__main__":
    main()