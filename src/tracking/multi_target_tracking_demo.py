#!/usr/bin/env python3
"""
Multi-Target Tracking Demonstration

This script demonstrates multi-target tracking with data association algorithms
including GNN, JPDA, and MHT. It shows track initiation, maintenance, and
termination in cluttered environments with crossing trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Dict, Optional
import sys
import os
from dataclasses import dataclass
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tracking.kalman_filters import KalmanFilter, initialize_constant_velocity_filter
from src.tracking.association import (
    GlobalNearestNeighbor, JointProbabilisticDataAssociation,
    Detection, Track as AssocTrack, AssociationResult
)
from src.tracking.tracker_base import TrackState, Track, TrackingMetrics


@dataclass
class Target:
    """Represents a true target."""
    id: int
    trajectory: np.ndarray
    birth_time: int
    death_time: int
    color: str


def generate_crossing_trajectories(n_targets: int, duration: float, dt: float) -> List[Target]:
    """Generate multiple targets with crossing trajectories."""
    n_steps = int(duration / dt)
    targets = []
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i in range(n_targets):
        # Random birth and death times
        birth_time = np.random.randint(0, n_steps // 4)
        death_time = np.random.randint(3 * n_steps // 4, n_steps)
        
        # Generate trajectory
        trajectory = np.zeros((n_steps, 4))  # [x, y, vx, vy]
        
        # Initial state
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
            
            # Add process noise
            trajectory[k, :2] += np.random.randn(2) * 0.1
        
        targets.append(Target(
            id=i,
            trajectory=trajectory,
            birth_time=birth_time,
            death_time=death_time,
            color=colors[i % len(colors)]
        ))
    
    return targets


def generate_measurements_with_clutter(targets: List[Target], time_step: int,
                                      measurement_noise_std: float = 1.0,
                                      detection_prob: float = 0.9,
                                      clutter_rate: float = 5.0,
                                      surveillance_region: Tuple[float, float, float, float] = (-100, 100, -100, 100)) -> List[Detection]:
    """Generate measurements from targets plus clutter."""
    measurements = []
    
    # Target measurements
    for target in targets:
        if target.birth_time <= time_step < target.death_time:
            if np.random.rand() < detection_prob:
                true_pos = target.trajectory[time_step, :2]
                noise = np.random.randn(2) * measurement_noise_std
                meas_pos = true_pos + noise
                
                measurements.append(Detection(
                    position=meas_pos,
                    timestamp=time_step,
                    measurement_noise=np.eye(2) * measurement_noise_std**2
                ))
    
    # Clutter measurements
    n_clutter = np.random.poisson(clutter_rate)
    x_min, x_max, y_min, y_max = surveillance_region
    
    for _ in range(n_clutter):
        clutter_pos = np.array([
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max)
        ])
        measurements.append(Detection(
            position=clutter_pos,
            timestamp=time_step,
            measurement_noise=np.eye(2) * measurement_noise_std**2
        ))
    
    return measurements


class MultiTargetTracker:
    """Multi-target tracker with configurable association algorithm."""
    
    def __init__(self, association_method: str = 'gnn', dt: float = 0.1):
        self.association_method = association_method
        self.dt = dt
        self.tracks: Dict[int, Dict] = {}
        self.next_track_id = 0
        self.metrics = TrackingMetrics()
        
        # Initialize association algorithm
        if association_method == 'gnn':
            self.associator = GlobalNearestNeighbor(gate_threshold=9.21)  # 99% gate
        elif association_method == 'jpda':
            self.associator = JointProbabilisticDataAssociation(
                gate_threshold=9.21,
                prob_detection=0.9,
                clutter_density=1e-4
            )
        else:
            raise ValueError(f"Unknown association method: {association_method}")
        
        # Track management parameters
        self.confirmation_threshold = 3  # Detections needed to confirm
        self.deletion_threshold = 5  # Missed detections before deletion
        
    def process_measurements(self, measurements: List[Detection]) -> Dict[int, np.ndarray]:
        """Process measurements and return current track estimates."""
        
        # Predict existing tracks
        for track_id, track_data in self.tracks.items():
            track_data['filter'].predict(self.dt)
            track_data['predicted_state'] = track_data['filter'].x.copy()
            track_data['predicted_cov'] = track_data['filter'].P.copy()
        
        if len(self.tracks) > 0 and len(measurements) > 0:
            # Prepare tracks for association
            assoc_tracks = []
            track_id_map = {}
            
            for idx, (track_id, track_data) in enumerate(self.tracks.items()):
                if track_data['state'] == TrackState.CONFIRMED:
                    assoc_track = AssocTrack(
                        id=idx,
                        state=track_data['predicted_state'][:2],  # Position only
                        covariance=track_data['predicted_cov'][:2, :2]
                    )
                    assoc_tracks.append(assoc_track)
                    track_id_map[idx] = track_id
            
            # Perform association
            if len(assoc_tracks) > 0:
                if self.association_method == 'gnn':
                    result = self.associator.associate(assoc_tracks, measurements)
                    associations = result.associations
                elif self.association_method == 'jpda':
                    result = self.associator.associate(assoc_tracks, measurements)
                    associations = result.associations
                else:
                    associations = {}
                
                # Update associated tracks
                associated_measurements = set()
                for assoc_idx, meas_idx in associations.items():
                    if meas_idx is not None:
                        track_id = track_id_map[assoc_idx]
                        track_data = self.tracks[track_id]
                        track_data['filter'].update(measurements[meas_idx].position)
                        track_data['hits'] += 1
                        track_data['misses'] = 0
                        associated_measurements.add(meas_idx)
                        
                        # Confirm tentative tracks
                        if track_data['state'] == TrackState.TENTATIVE and track_data['hits'] >= self.confirmation_threshold:
                            track_data['state'] = TrackState.CONFIRMED
                            self.metrics.total_tracks_confirmed += 1
                
                # Handle missed detections
                for track_id, track_data in self.tracks.items():
                    if track_id not in [track_id_map[a] for a in associations if associations[a] is not None]:
                        track_data['misses'] += 1
                        track_data['filter'].update(None)  # Prediction only
                        
                # Initiate new tracks from unassociated measurements
                for meas_idx, meas in enumerate(measurements):
                    if meas_idx not in associated_measurements:
                        self.initiate_track(meas)
            else:
                # No confirmed tracks, initiate from all measurements
                for meas in measurements:
                    self.initiate_track(meas)
        elif len(measurements) > 0:
            # No existing tracks, initiate from measurements
            for meas in measurements:
                self.initiate_track(meas)
        else:
            # No measurements, increment miss counts
            for track_data in self.tracks.values():
                track_data['misses'] += 1
        
        # Delete tracks with too many misses
        tracks_to_delete = []
        for track_id, track_data in self.tracks.items():
            if track_data['misses'] >= self.deletion_threshold:
                tracks_to_delete.append(track_id)
                self.metrics.total_tracks_deleted += 1
        
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
        
        # Return current estimates
        estimates = {}
        for track_id, track_data in self.tracks.items():
            if track_data['state'] == TrackState.CONFIRMED:
                estimates[track_id] = track_data['filter'].x[:2].copy()
        
        return estimates
    
    def initiate_track(self, measurement: Detection):
        """Initiate a new track from a measurement."""
        # Create new Kalman filter
        kf = initialize_constant_velocity_filter(
            dim=2,
            dt=self.dt,
            process_noise_std=0.5,
            measurement_noise_std=1.0
        )
        
        # Initialize state with measurement
        initial_state = np.zeros(4)
        initial_state[:2] = measurement.position
        kf.x = initial_state
        
        # Create track data
        self.tracks[self.next_track_id] = {
            'filter': kf,
            'state': TrackState.TENTATIVE,
            'hits': 1,
            'misses': 0,
            'birth_time': measurement.timestamp,
            'predicted_state': None,
            'predicted_cov': None
        }
        
        self.metrics.total_tracks_initiated += 1
        self.next_track_id += 1


def plot_multi_target_results(targets: List[Target], all_measurements: List[List[Detection]],
                             gnn_tracks: List[Dict], jpda_tracks: List[Dict],
                             metrics_gnn: Dict, metrics_jpda: Dict):
    """Create comprehensive visualization of multi-target tracking results."""
    
    fig = plt.figure(figsize=(18, 10))
    
    # GNN Tracking Results
    ax1 = plt.subplot(2, 3, 1)
    
    # Plot true trajectories
    for target in targets:
        traj = target.trajectory[target.birth_time:target.death_time]
        ax1.plot(traj[:, 0], traj[:, 1], '-', color=target.color, alpha=0.3, linewidth=2)
        ax1.plot(traj[0, 0], traj[0, 1], 'o', color=target.color, markersize=8)
        ax1.plot(traj[-1, 0], traj[-1, 1], 's', color=target.color, markersize=8)
    
    # Plot GNN tracks
    track_colors = {}
    for track_id in set().union(*[set(t.keys()) for t in gnn_tracks]):
        track_positions = []
        for frame_tracks in gnn_tracks:
            if track_id in frame_tracks:
                track_positions.append(frame_tracks[track_id])
        
        if len(track_positions) > 1:
            track_positions = np.array(track_positions)
            if track_id not in track_colors:
                track_colors[track_id] = np.random.rand(3,)
            ax1.plot(track_positions[:, 0], track_positions[:, 1], '--',
                    color=track_colors[track_id], linewidth=1.5, alpha=0.8)
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('GNN Tracking Results')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-100, 100)
    ax1.set_ylim(-100, 100)
    
    # JPDA Tracking Results
    ax2 = plt.subplot(2, 3, 2)
    
    # Plot true trajectories
    for target in targets:
        traj = target.trajectory[target.birth_time:target.death_time]
        ax2.plot(traj[:, 0], traj[:, 1], '-', color=target.color, alpha=0.3, linewidth=2)
        ax2.plot(traj[0, 0], traj[0, 1], 'o', color=target.color, markersize=8)
        ax2.plot(traj[-1, 0], traj[-1, 1], 's', color=target.color, markersize=8)
    
    # Plot JPDA tracks
    track_colors = {}
    for track_id in set().union(*[set(t.keys()) for t in jpda_tracks]):
        track_positions = []
        for frame_tracks in jpda_tracks:
            if track_id in frame_tracks:
                track_positions.append(frame_tracks[track_id])
        
        if len(track_positions) > 1:
            track_positions = np.array(track_positions)
            if track_id not in track_colors:
                track_colors[track_id] = np.random.rand(3,)
            ax2.plot(track_positions[:, 0], track_positions[:, 1], '--',
                    color=track_colors[track_id], linewidth=1.5, alpha=0.8)
    
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('JPDA Tracking Results')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-100, 100)
    ax2.set_ylim(-100, 100)
    
    # Measurement Density Plot
    ax3 = plt.subplot(2, 3, 3)
    all_meas_positions = []
    for measurements in all_measurements:
        for meas in measurements:
            all_meas_positions.append(meas.position)
    
    if all_meas_positions:
        all_meas_positions = np.array(all_meas_positions)
        h = ax3.hist2d(all_meas_positions[:, 0], all_meas_positions[:, 1],
                      bins=30, cmap='YlOrRd', cmin=1)
        plt.colorbar(h[3], ax=ax3, label='Count')
    
    ax3.set_xlabel('X Position (m)')
    ax3.set_ylabel('Y Position (m)')
    ax3.set_title('Measurement Density (Targets + Clutter)')
    ax3.set_xlim(-100, 100)
    ax3.set_ylim(-100, 100)
    
    # Track Count Over Time
    ax4 = plt.subplot(2, 3, 4)
    time_steps = np.arange(len(gnn_tracks))
    gnn_track_counts = [len(tracks) for tracks in gnn_tracks]
    jpda_track_counts = [len(tracks) for tracks in jpda_tracks]
    
    # Count true targets at each time
    true_counts = []
    for t in range(len(gnn_tracks)):
        count = sum(1 for target in targets if target.birth_time <= t < target.death_time)
        true_counts.append(count)
    
    ax4.plot(time_steps, true_counts, 'g-', linewidth=2, label='True Targets')
    ax4.plot(time_steps, gnn_track_counts, 'b--', label='GNN Tracks')
    ax4.plot(time_steps, jpda_track_counts, 'r--', label='JPDA Tracks')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Number of Tracks')
    ax4.set_title('Track Count Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Clutter vs Target Measurements
    ax5 = plt.subplot(2, 3, 5)
    target_meas_count = []
    clutter_meas_count = []
    
    for t, measurements in enumerate(all_measurements):
        # Approximate classification based on proximity to true targets
        target_count = 0
        for meas in measurements:
            is_target = False
            for target in targets:
                if target.birth_time <= t < target.death_time:
                    true_pos = target.trajectory[t, :2]
                    if np.linalg.norm(meas.position - true_pos) < 3.0:  # Within 3 sigma
                        is_target = True
                        break
            if is_target:
                target_count += 1
        
        target_meas_count.append(target_count)
        clutter_meas_count.append(len(measurements) - target_count)
    
    ax5.plot(time_steps, target_meas_count, 'g-', label='Target Measurements')
    ax5.plot(time_steps, clutter_meas_count, 'r-', label='Clutter')
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Number of Measurements')
    ax5.set_title('Target vs Clutter Measurements')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Performance Metrics Comparison
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    metrics_text = f"""Performance Metrics Comparison:

Global Nearest Neighbor (GNN):
  • Track Initiations: {metrics_gnn['initiations']}
  • Track Confirmations: {metrics_gnn['confirmations']}
  • Track Deletions: {metrics_gnn['deletions']}
  • Avg Tracks/Frame: {np.mean(gnn_track_counts):.1f}
  • Max Tracks: {max(gnn_track_counts)}

Joint Probabilistic DA (JPDA):
  • Track Initiations: {metrics_jpda['initiations']}
  • Track Confirmations: {metrics_jpda['confirmations']}
  • Track Deletions: {metrics_jpda['deletions']}
  • Avg Tracks/Frame: {np.mean(jpda_track_counts):.1f}
  • Max Tracks: {max(jpda_track_counts)}

Simulation Parameters:
  • True Targets: {len(targets)}
  • Detection Probability: 90%
  • Clutter Rate: 5 false alarms/frame
  • Measurement Noise: 1.0 m"""
    
    ax6.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.suptitle('Multi-Target Tracking with Data Association', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('multi_target_tracking_demo.png', dpi=100, bbox_inches='tight')
    print("Plot saved as 'multi_target_tracking_demo.png'")
    plt.close()


def run_multi_target_tracking_demo():
    """Run the multi-target tracking demonstration."""
    
    print("=" * 60)
    print("Multi-Target Tracking Demonstration")
    print("=" * 60)
    
    # Simulation parameters
    n_targets = 5
    duration = 30.0  # seconds
    dt = 0.1
    measurement_noise_std = 1.0
    detection_prob = 0.9
    clutter_rate = 5.0
    
    print(f"\nSimulation Parameters:")
    print(f"  • Number of Targets: {n_targets}")
    print(f"  • Duration: {duration} seconds")
    print(f"  • Time Step: {dt} seconds")
    print(f"  • Detection Probability: {detection_prob*100:.0f}%")
    print(f"  • Clutter Rate: {clutter_rate} false alarms/frame")
    print(f"  • Measurement Noise: {measurement_noise_std} m")
    
    # Generate target trajectories
    print("\nGenerating target trajectories...")
    targets = generate_crossing_trajectories(n_targets, duration, dt)
    
    for i, target in enumerate(targets):
        print(f"  • Target {i}: Birth at t={target.birth_time*dt:.1f}s, Death at t={target.death_time*dt:.1f}s")
    
    # Generate measurements
    print("\nGenerating measurements with clutter...")
    all_measurements = []
    n_steps = int(duration / dt)
    
    for t in range(n_steps):
        measurements = generate_measurements_with_clutter(
            targets, t, measurement_noise_std, detection_prob, clutter_rate
        )
        all_measurements.append(measurements)
    
    total_measurements = sum(len(m) for m in all_measurements)
    print(f"  • Total measurements: {total_measurements}")
    print(f"  • Average per frame: {total_measurements/n_steps:.1f}")
    
    # Run GNN tracking
    print("\nRunning Global Nearest Neighbor (GNN) tracking...")
    gnn_tracker = MultiTargetTracker(association_method='gnn', dt=dt)
    gnn_tracks = []
    
    for measurements in all_measurements:
        estimates = gnn_tracker.process_measurements(measurements)
        gnn_tracks.append(estimates.copy())
    
    gnn_metrics = {
        'initiations': gnn_tracker.metrics.total_tracks_initiated,
        'confirmations': gnn_tracker.metrics.total_tracks_confirmed,
        'deletions': gnn_tracker.metrics.total_tracks_deleted
    }
    
    print(f"  • Tracks initiated: {gnn_metrics['initiations']}")
    print(f"  • Tracks confirmed: {gnn_metrics['confirmations']}")
    print(f"  • Tracks deleted: {gnn_metrics['deletions']}")
    
    # Run JPDA tracking
    print("\nRunning Joint Probabilistic Data Association (JPDA) tracking...")
    jpda_tracker = MultiTargetTracker(association_method='jpda', dt=dt)
    jpda_tracks = []
    
    for measurements in all_measurements:
        estimates = jpda_tracker.process_measurements(measurements)
        jpda_tracks.append(estimates.copy())
    
    jpda_metrics = {
        'initiations': jpda_tracker.metrics.total_tracks_initiated,
        'confirmations': jpda_tracker.metrics.total_tracks_confirmed,
        'deletions': jpda_tracker.metrics.total_tracks_deleted
    }
    
    print(f"  • Tracks initiated: {jpda_metrics['initiations']}")
    print(f"  • Tracks confirmed: {jpda_metrics['confirmations']}")
    print(f"  • Tracks deleted: {jpda_metrics['deletions']}")
    
    # Compare performance
    print("\nPerformance Comparison:")
    print("-" * 40)
    
    avg_gnn_tracks = np.mean([len(t) for t in gnn_tracks])
    avg_jpda_tracks = np.mean([len(t) for t in jpda_tracks])
    
    print(f"Average tracks per frame:")
    print(f"  • GNN: {avg_gnn_tracks:.1f}")
    print(f"  • JPDA: {avg_jpda_tracks:.1f}")
    print(f"  • True targets (average): {np.mean([sum(1 for target in targets if target.birth_time <= t < target.death_time) for t in range(n_steps)]):.1f}")
    
    # Visualize results
    print("\nGenerating visualization...")
    plot_multi_target_results(
        targets, all_measurements,
        gnn_tracks, jpda_tracks,
        gnn_metrics, jpda_metrics
    )
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the demonstration
    run_multi_target_tracking_demo()