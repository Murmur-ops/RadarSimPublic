#!/usr/bin/env python3
"""
Simple Networked Radar Demonstration

A lightweight example showing networked radar tracking with data fusion.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.networked_radar import (
    NetworkedRadar, NetworkNode, RadarNodeType,
    CommunicationLink, NetworkArchitecture,
    DataFusionCenter, FusionArchitecture,
    NetworkTrack, DistributedTrackFusion
)
from src.radar import RadarParameters


def main():
    """Run simple networked radar demonstration."""
    print("\n" + "="*60)
    print("SIMPLE NETWORKED RADAR DEMONSTRATION")
    print("="*60)
    
    # 1. Setup: Create 3-node radar network
    print("\n1. NETWORK SETUP")
    print("-" * 40)
    
    radar_params = RadarParameters(
        frequency=10e9,  # X-band
        power=1000,
        antenna_gain=30,
        pulse_width=1e-6,
        prf=1000,
        bandwidth=10e6,
        noise_figure=3,
        losses=2
    )
    
    # Create three radar nodes in a triangle
    nodes = [
        NetworkNode("Radar_West", np.array([-20000, 0, 0]), node_type=RadarNodeType.MONOSTATIC),
        NetworkNode("Radar_East", np.array([20000, 0, 0]), node_type=RadarNodeType.MONOSTATIC),
        NetworkNode("Radar_North", np.array([0, 30000, 0]), node_type=RadarNodeType.MONOSTATIC)
    ]
    
    radars = [NetworkedRadar(radar_params, node) for node in nodes]
    
    for node in nodes:
        print(f"  • {node.node_id}: Position {node.position/1000} km")
    
    # 2. Target Scenario
    print("\n2. TARGET SCENARIO")
    print("-" * 40)
    
    # Single target moving across the surveillance area
    target_pos = np.array([5000, 10000, 5000])  # Initial position
    target_vel = np.array([100, -50, 0])  # Velocity m/s
    target_rcs = 10.0  # m^2
    
    print(f"  • Target position: {target_pos/1000} km")
    print(f"  • Target velocity: {target_vel} m/s")
    print(f"  • Target RCS: {target_rcs} m²")
    
    # 3. Generate Measurements
    print("\n3. RADAR MEASUREMENTS")
    print("-" * 40)
    
    measurements = {}
    for radar in radars:
        # Calculate range and SNR
        range_vec = target_pos - radar.node.position
        range_m = np.linalg.norm(range_vec)
        snr = radar.snr(range_m, target_rcs)
        
        # Add measurement noise
        noise_std = 50.0  # meters
        measured_pos = target_pos + np.random.randn(3) * noise_std
        
        measurements[radar.node.node_id] = {
            'position': measured_pos,
            'range': range_m,
            'snr': snr
        }
        
        print(f"  • {radar.node.node_id}:")
        print(f"    - Range: {range_m/1000:.1f} km")
        print(f"    - SNR: {snr:.1f} dB")
        print(f"    - Measurement error: {np.linalg.norm(measured_pos - target_pos):.1f} m")
    
    # 4. Create Network Tracks
    print("\n4. TRACK CREATION")
    print("-" * 40)
    
    tracks = []
    for node_id, meas in measurements.items():
        track = NetworkTrack(
            track_id=f"{node_id}_track",
            local_id="T1",
            source_node=node_id,
            state=np.concatenate([meas['position'], np.zeros(3)]),
            covariance=np.diag([100, 100, 100, 10, 10, 10]),
            timestamp=0.0,
            quality=min(1.0, meas['snr'] / 30.0)
        )
        tracks.append(track)
        print(f"  • {node_id}: Track quality {track.quality:.2f}")
    
    # 5. Bistatic Calculations
    print("\n5. BISTATIC GEOMETRY")
    print("-" * 40)
    
    # Calculate bistatic parameters between first two radars
    bistatic_range, tx_range, rx_range = radars[1].calculate_bistatic_range(
        target_pos, radars[0].node.position
    )
    
    bistatic_doppler = radars[1].calculate_bistatic_doppler(
        target_pos, target_vel, radars[0].node.position
    )
    
    print(f"  • TX (Radar_West) to Target: {tx_range/1000:.1f} km")
    print(f"  • Target to RX (Radar_East): {rx_range/1000:.1f} km")
    print(f"  • Total bistatic range: {bistatic_range/1000:.1f} km")
    print(f"  • Bistatic Doppler: {bistatic_doppler:.1f} Hz")
    
    # 6. Data Fusion
    print("\n6. DATA FUSION")
    print("-" * 40)
    
    # Perform Covariance Intersection fusion
    fusion = DistributedTrackFusion()
    
    print("\n  Covariance Intersection Fusion:")
    fused_track = fusion.covariance_intersection(tracks, optimize_omega=True)
    fusion_error = np.linalg.norm(fused_track.state[:3] - target_pos)
    print(f"    • Fused position: {fused_track.state[:3]/1000} km")
    print(f"    • Fusion error: {fusion_error:.1f} m")
    print(f"    • Covariance trace: {np.trace(fused_track.covariance):.1f}")
    
    # Compare with Information Matrix Fusion
    print("\n  Information Matrix Fusion:")
    info_fused = fusion.information_matrix_fusion(tracks)
    info_error = np.linalg.norm(info_fused.state[:3] - target_pos)
    print(f"    • Fused position: {info_fused.state[:3]/1000} km")
    print(f"    • Fusion error: {info_error:.1f} m")
    print(f"    • Covariance trace: {np.trace(info_fused.covariance):.1f}")
    
    # 7. Fusion Center Comparison
    print("\n7. ARCHITECTURE COMPARISON")
    print("-" * 40)
    
    # Centralized fusion
    central_fusion = DataFusionCenter(
        "central", 
        architecture=FusionArchitecture.CENTRALIZED
    )
    
    for i, track in enumerate(tracks):
        central_fusion.receive_local_tracks(nodes[i].node_id, [track])
    
    central_result = central_fusion.perform_fusion()
    
    # Decentralized fusion  
    decentral_fusion = DataFusionCenter(
        "decentral",
        architecture=FusionArchitecture.DECENTRALIZED
    )
    
    for i, track in enumerate(tracks):
        decentral_fusion.receive_local_tracks(nodes[i].node_id, [track])
    
    decentral_result = decentral_fusion.perform_fusion()
    
    print(f"  • Centralized: {len(central_result)} fused tracks")
    print(f"  • Decentralized: {len(decentral_result)} fused tracks")
    
    central_metrics = central_fusion.get_fusion_metrics()
    print(f"  • Fusion cycles: {central_metrics['fusion_cycles']}")
    print(f"  • Tracks fused: {central_metrics['tracks_fused']}")
    
    # 8. Visualization
    print("\n8. GENERATING VISUALIZATION")
    print("-" * 40)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Network topology
    ax1 = axes[0]
    ax1.set_title('Radar Network Topology', fontweight='bold')
    ax1.set_xlabel('East-West (km)')
    ax1.set_ylabel('North-South (km)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot radar positions
    for i, node in enumerate(nodes):
        pos = node.position / 1000
        ax1.scatter(pos[0], pos[1], s=200, c=f'C{i}', marker='^', 
                   label=node.node_id, zorder=5)
        
        # Add coverage circles
        circle = plt.Circle((pos[0], pos[1]), 100, fill=False, 
                           edgecolor=f'C{i}', linestyle='--', alpha=0.3)
        ax1.add_patch(circle)
    
    # Plot target
    ax1.scatter(target_pos[0]/1000, target_pos[1]/1000, 
               s=100, c='red', marker='o', label='Target')
    
    # Plot connections
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            pos1 = nodes[i].position / 1000
            pos2 = nodes[j].position / 1000
            ax1.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                    'k--', alpha=0.2, linewidth=1)
    
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlim([-50, 50])
    ax1.set_ylim([-20, 50])
    
    # Plot 2: Measurement comparison
    ax2 = axes[1]
    ax2.set_title('Measurement vs Fusion', fontweight='bold')
    ax2.set_xlabel('East-West (km)')
    ax2.set_ylabel('North-South (km)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Plot true position
    ax2.scatter(target_pos[0]/1000, target_pos[1]/1000, 
               s=150, c='red', marker='*', label='True Position', zorder=5)
    
    # Plot individual measurements
    for i, (node_id, meas) in enumerate(measurements.items()):
        pos = meas['position'] / 1000
        ax2.scatter(pos[0], pos[1], s=80, c=f'C{i}', marker='o',
                   label=f'{node_id} measurement', alpha=0.7)
    
    # Plot fused result
    ax2.scatter(fused_track.state[0]/1000, fused_track.state[1]/1000,
               s=120, c='green', marker='D', label='CI Fused', zorder=4)
    
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlim([0, 10])
    ax2.set_ylim([5, 15])
    
    # Plot 3: Error comparison
    ax3 = axes[2]
    ax3.set_title('Tracking Errors', fontweight='bold')
    ax3.set_ylabel('Position Error (m)')
    
    # Calculate errors
    errors = []
    labels = []
    
    for node_id, meas in measurements.items():
        error = np.linalg.norm(meas['position'] - target_pos)
        errors.append(error)
        labels.append(node_id.replace('Radar_', ''))
    
    errors.append(fusion_error)
    labels.append('CI Fusion')
    
    errors.append(info_error)
    labels.append('Info Fusion')
    
    # Create bar chart
    colors = ['C0', 'C1', 'C2', 'green', 'orange']
    bars = ax3.bar(range(len(errors)), errors, color=colors[:len(errors)])
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.0f}m', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = "results/networked_radar"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "networked_radar_simple.png")
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  • Saved to: {output_file}")
    
    # 9. Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nFusion Performance:")
    print(f"  • Individual radar errors: {np.mean([np.linalg.norm(m['position'] - target_pos) for m in measurements.values()]):.1f} m (average)")
    print(f"  • CI fusion error: {fusion_error:.1f} m")
    print(f"  • Information fusion error: {info_error:.1f} m")
    
    improvement = (np.mean([np.linalg.norm(m['position'] - target_pos) for m in measurements.values()]) - fusion_error) / np.mean([np.linalg.norm(m['position'] - target_pos) for m in measurements.values()]) * 100
    print(f"  • Improvement: {improvement:.1f}%")
    
    print("\nKey Benefits Demonstrated:")
    print("  ✓ Multiple viewpoints reduce measurement uncertainty")
    print("  ✓ Covariance Intersection handles correlation")
    print("  ✓ Fusion improves position accuracy")
    print("  ✓ Bistatic geometry provides additional information")
    
    plt.show()
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Demonstration completed successfully!")
    else:
        print("\n❌ Demonstration failed!")
        sys.exit(1)