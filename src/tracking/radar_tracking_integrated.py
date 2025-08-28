#!/usr/bin/env python3
"""
Integrated Radar-to-Tracking Demo with Non-Cheating Radar

This demo shows the complete pipeline from radar signal processing to tracking,
using only realistic measurements derived from physics-based signal processing.
No ground truth positions are used in the detection or tracking process.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.gridspec import GridSpec
import sys
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import radar simulation components
from src.radar_simulation.detection import CFARDetector, CFARParameters, CFARType, MeasurementExtractor

# Import tracking components
from src.tracking.kalman_filters import initialize_constant_velocity_filter, initialize_constant_acceleration_filter
from src.tracking.motion_models import ConstantVelocityModel, ConstantAccelerationModel
from src.tracking.association import GlobalNearestNeighbor, Detection, Track as AssocTrack
from src.tracking.tracker_base import TrackState, TrackingMetrics


@dataclass
class RadarTarget:
    """Target for radar simulation"""
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    rcs: float  # Radar cross section in m²
    id: str  # Target identifier


class NonCheatingRadar:
    """
    Realistic radar that generates measurements through signal processing.
    No direct access to ground truth positions!
    """
    
    def __init__(self, 
                 center_frequency: float = 3e9,  # 3 GHz
                 bandwidth: float = 10e6,  # 10 MHz
                 max_range: float = 50000,  # 50 km
                 range_resolution: float = 15.0,  # 15 m
                 velocity_resolution: float = 1.0,  # 1 m/s
                 noise_figure: float = 3.0):  # 3 dB
        
        self.fc = center_frequency
        self.bw = bandwidth
        self.max_range = max_range
        self.range_res = range_resolution
        self.velocity_res = velocity_resolution
        self.noise_figure = noise_figure
        self.wavelength = 3e8 / center_frequency
        
        # CFAR detector setup
        self.cfar_params = CFARParameters(
            cfar_type=CFARType.CA_CFAR,
            num_training_cells=16,
            num_guard_cells=4,
            false_alarm_rate=1e-6,
            min_detection_snr=10.0  # 10 dB minimum SNR
        )
        
        self.detector = CFARDetector(
            self.cfar_params,
            range_resolution=self.range_res,
            velocity_resolution=self.velocity_res,
            wavelength=self.wavelength
        )
        
        # Range-Doppler map size
        self.n_range_bins = int(self.max_range / self.range_res)
        self.n_doppler_bins = 128
        
    def generate_range_doppler_map(self, targets: List[RadarTarget]) -> np.ndarray:
        """
        Generate range-Doppler map from targets using radar equation.
        This simulates the actual radar signal processing.
        """
        # Initialize with thermal noise
        noise_power = 10**(self.noise_figure/10) * 1.38e-23 * 290 * self.bw
        rd_map = np.sqrt(noise_power/2) * (np.random.randn(self.n_range_bins, self.n_doppler_bins) + 
                                           1j * np.random.randn(self.n_range_bins, self.n_doppler_bins))
        
        # Add target returns
        for target in targets:
            # Calculate range and radial velocity
            range_m = np.linalg.norm(target.position)
            
            # Radial velocity (positive = approaching)
            if range_m > 0:
                range_vec = target.position / range_m
                radial_velocity = -np.dot(target.velocity, range_vec)
            else:
                radial_velocity = 0
            
            # Skip if out of range
            if range_m > self.max_range:
                continue
            
            # Calculate SNR using simplified radar equation
            # SNR = (Pt * Gt * Gr * λ² * σ) / ((4π)³ * R⁴ * k * T * B * F * L)
            # Simplified: SNR ∝ σ / R⁴
            reference_range = 1000.0  # 1 km reference
            reference_snr = 40.0  # 40 dB at 1 km for 1 m² RCS
            
            snr_db = reference_snr + 10*np.log10(target.rcs) - 40*np.log10(range_m/reference_range)
            snr_linear = 10**(snr_db/10)
            
            # Convert to bin indices
            range_bin = int(range_m / self.range_res)
            max_doppler = self.wavelength * self.n_doppler_bins * self.velocity_res / 2
            doppler_bin = int(self.n_doppler_bins/2 + radial_velocity / self.velocity_res)
            
            # Add target return if within bounds
            if 0 <= range_bin < self.n_range_bins and 0 <= doppler_bin < self.n_doppler_bins:
                # Signal amplitude based on SNR
                signal_amplitude = np.sqrt(snr_linear * noise_power)
                signal_phase = np.random.uniform(0, 2*np.pi)
                
                # Add point spread due to windowing
                for dr in range(-1, 2):
                    for dd in range(-1, 2):
                        rb = range_bin + dr
                        db = doppler_bin + dd
                        if 0 <= rb < self.n_range_bins and 0 <= db < self.n_doppler_bins:
                            # Window weight (simple Gaussian)
                            weight = np.exp(-(dr**2 + dd**2)/2)
                            rd_map[rb, db] += weight * signal_amplitude * np.exp(1j * signal_phase)
        
        return rd_map
    
    def process(self, targets: List[RadarTarget], timestamp: float) -> List[Dict]:
        """
        Process targets and return measurements.
        This is the main interface - only returns what the radar can actually detect!
        """
        # Generate range-Doppler map
        rd_map = self.generate_range_doppler_map(targets)
        
        # CFAR detection
        detections = self.detector.detect(np.abs(rd_map)**2)
        
        # Convert to measurements
        measurements = []
        for det in detections:
            # Convert from polar to Cartesian (assuming single beam for simplicity)
            # In reality, you'd need azimuth information
            range_m = det.range_m
            radial_velocity = det.velocity_ms
            
            # For this demo, assume targets are along x-axis (boresight)
            # A real radar would have azimuth and elevation
            measurement = {
                'range': range_m,
                'radial_velocity': radial_velocity,
                'snr_db': det.snr_db,
                'range_std': det.range_std,
                'velocity_std': det.velocity_std,
                'timestamp': timestamp,
                # Simplified: assume detection along x-axis
                'x': range_m,
                'y': 0.0,
                'vx': -radial_velocity,  # Negative because positive radial = approaching
                'vy': 0.0
            }
            measurements.append(measurement)
        
        return measurements


class IntegratedTracker:
    """
    Tracker that uses only radar measurements (no cheating!)
    """
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.tracks = {}
        self.next_track_id = 0
        self.metrics = TrackingMetrics()
        
        # Association algorithm
        self.associator = GlobalNearestNeighbor(gate_threshold=9.21)  # 99% gate
        
        # Track confirmation/deletion thresholds
        self.confirmation_threshold = 3
        self.deletion_threshold = 5
    
    def update(self, measurements: List[Dict], timestamp: float):
        """Update tracks with radar measurements."""
        
        # Predict existing tracks
        for track_id, track_data in self.tracks.items():
            track_data['filter'].predict(self.dt)
            track_data['age'] += 1
        
        if len(self.tracks) > 0 and len(measurements) > 0:
            # Prepare for association
            track_list = []
            track_ids = []
            
            for track_id, track_data in self.tracks.items():
                if track_data['state'] == TrackState.CONFIRMED:
                    track = AssocTrack(
                        id=len(track_list),
                        state=track_data['filter'].x[:2],  # Position
                        covariance=track_data['filter'].P[:2, :2]
                    )
                    track_list.append(track)
                    track_ids.append(track_id)
            
            # Convert measurements for association
            detections = []
            for meas in measurements:
                det = Detection(
                    position=np.array([meas['x'], meas['y']]),
                    timestamp=timestamp,
                    measurement_noise=np.diag([meas['range_std']**2, meas['range_std']**2])
                )
                detections.append(det)
            
            # Perform association
            if len(track_list) > 0:
                result = self.associator.associate(track_list, detections)
                
                # Update associated tracks
                associated_measurements = set()
                for assoc_idx, meas_idx in result.associations.items():
                    if meas_idx is not None:
                        track_id = track_ids[assoc_idx]
                        track_data = self.tracks[track_id]
                        
                        # Update with measurement
                        meas = measurements[meas_idx]
                        z = np.array([meas['x'], meas['y']])
                        track_data['filter'].update(z)
                        
                        # Update track statistics
                        track_data['hits'] += 1
                        track_data['misses'] = 0
                        track_data['last_update'] = timestamp
                        track_data['measurements'].append(meas)
                        associated_measurements.add(meas_idx)
                        
                        # Confirm tentative tracks
                        if track_data['state'] == TrackState.TENTATIVE:
                            if track_data['hits'] >= self.confirmation_threshold:
                                track_data['state'] = TrackState.CONFIRMED
                                self.metrics.total_tracks_confirmed += 1
                
                # Update missed tracks
                for track_id in track_ids:
                    if track_id not in [track_ids[i] for i, m in result.associations.items() if m is not None]:
                        self.tracks[track_id]['misses'] += 1
                
                # Initiate new tracks from unassociated measurements
                for i, meas in enumerate(measurements):
                    if i not in associated_measurements:
                        self.initiate_track(meas, timestamp)
            else:
                # No confirmed tracks, initiate from all measurements
                for meas in measurements:
                    self.initiate_track(meas, timestamp)
        elif len(measurements) > 0:
            # No existing tracks
            for meas in measurements:
                self.initiate_track(meas, timestamp)
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
    
    def initiate_track(self, measurement: Dict, timestamp: float):
        """Initiate new track from measurement."""
        
        # Initialize Kalman filter
        kf = initialize_constant_velocity_filter(
            dim=2,
            dt=self.dt,
            process_noise_std=1.0,
            measurement_noise_std=measurement['range_std']
        )
        
        # Set initial state
        kf.x[:2] = [measurement['x'], measurement['y']]
        kf.x[2:] = [measurement['vx'], measurement['vy']]
        
        # Create track
        self.tracks[self.next_track_id] = {
            'filter': kf,
            'state': TrackState.TENTATIVE,
            'hits': 1,
            'misses': 0,
            'age': 0,
            'birth_time': timestamp,
            'last_update': timestamp,
            'measurements': [measurement],
            'id': self.next_track_id
        }
        
        self.metrics.total_tracks_initiated += 1
        self.next_track_id += 1
    
    def get_tracks(self) -> Dict:
        """Get current track states."""
        result = {}
        for track_id, track_data in self.tracks.items():
            result[track_id] = {
                'state': track_data['filter'].x.copy(),
                'covariance': track_data['filter'].P.copy(),
                'track_state': track_data['state'],
                'age': track_data['age'],
                'hits': track_data['hits'],
                'num_measurements': len(track_data['measurements'])
            }
        return result


def create_scenario() -> List[RadarTarget]:
    """Create a multi-target scenario."""
    targets = [
        # Target 1: Constant velocity
        RadarTarget(
            position=np.array([10000.0, 2000.0, 0.0]),
            velocity=np.array([-100.0, 10.0, 0.0]),
            rcs=50.0,
            id="CV_target"
        ),
        # Target 2: Another constant velocity
        RadarTarget(
            position=np.array([8000.0, -1000.0, 0.0]),
            velocity=np.array([-80.0, 20.0, 0.0]),
            rcs=20.0,
            id="CV_target2"
        ),
        # Target 3: Slow target
        RadarTarget(
            position=np.array([5000.0, 0.0, 0.0]),
            velocity=np.array([-30.0, 5.0, 0.0]),
            rcs=100.0,
            id="Slow_target"
        )
    ]
    return targets


def update_targets(targets: List[RadarTarget], dt: float):
    """Update target positions (true motion, not seen by radar)."""
    for target in targets:
        target.position += target.velocity * dt


def visualize_integrated_tracking(
    radar_measurements_history: List[List[Dict]],
    track_history: List[Dict],
    true_positions_history: List[List[np.ndarray]],
    time_history: List[float]
):
    """Visualize the integrated radar and tracking results."""
    
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # Plot 1: Radar measurements over time
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot all measurements
    for measurements in radar_measurements_history:
        if measurements:
            ranges = [m['range']/1000 for m in measurements]
            velocities = [m['radial_velocity'] for m in measurements]
            ax1.scatter(velocities, ranges, c='gray', s=20, alpha=0.3)
    
    ax1.set_xlabel('Radial Velocity (m/s)')
    ax1.set_ylabel('Range (km)')
    ax1.set_title('Radar Measurements (All Time)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-150, 50])
    ax1.set_ylim([0, 15])
    
    # Plot 2: Track trajectories
    ax2 = fig.add_subplot(gs[0, 1:])
    
    # Plot true trajectories (for validation only)
    for i in range(len(true_positions_history[0])):
        true_traj = np.array([pos[i] for pos in true_positions_history])
        ax2.plot(true_traj[:, 0]/1000, true_traj[:, 1]/1000, 'g--', 
                alpha=0.5, linewidth=2, label=f'True {i+1}' if len(true_traj) > 0 else '')
    
    # Plot tracked trajectories
    track_colors = {}
    for tracks in track_history:
        for track_id, track_info in tracks.items():
            if track_id not in track_colors:
                track_colors[track_id] = np.random.rand(3,)
            
            pos = track_info['state'][:2]
            ax2.plot(pos[0]/1000, pos[1]/1000, 'o', 
                    color=track_colors[track_id], markersize=4)
    
    # Add measurement points
    for measurements in radar_measurements_history[-10:]:  # Last 10 scans
        if measurements:
            x_meas = [m['x']/1000 for m in measurements]
            y_meas = [m['y']/1000 for m in measurements]
            ax2.scatter(x_meas, y_meas, c='red', s=10, alpha=0.2, marker='+')
    
    ax2.set_xlabel('X Position (km)')
    ax2.set_ylabel('Y Position (km)')
    ax2.set_title('Tracking Results (Solid=Tracked, Dashed=Truth, +=Measurements)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.legend(loc='upper right')
    
    # Plot 3: Number of tracks and measurements over time
    ax3 = fig.add_subplot(gs[1, 0])
    
    num_tracks = [len(tracks) for tracks in track_history]
    num_measurements = [len(meas) for meas in radar_measurements_history]
    num_true = len(true_positions_history[0]) if true_positions_history else 0
    
    ax3.plot(time_history, num_tracks, 'b-', label='Tracks')
    ax3.plot(time_history, num_measurements, 'r--', label='Measurements')
    ax3.axhline(y=num_true, color='g', linestyle=':', label='True Targets')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Count')
    ax3.set_title('Tracks and Measurements Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: SNR distribution
    ax4 = fig.add_subplot(gs[1, 1])
    
    all_snrs = []
    all_ranges = []
    for measurements in radar_measurements_history:
        for m in measurements:
            all_snrs.append(m['snr_db'])
            all_ranges.append(m['range']/1000)
    
    if all_snrs:
        scatter = ax4.scatter(all_ranges, all_snrs, c=all_snrs, 
                            cmap='viridis', s=20, alpha=0.6)
        plt.colorbar(scatter, ax=ax4, label='SNR (dB)')
        
        # Add theoretical curve
        ranges = np.linspace(1, 15, 100)
        theoretical_snr = 40 - 40*np.log10(ranges)
        ax4.plot(ranges, theoretical_snr, 'r--', label='Theoretical R⁻⁴')
    
    ax4.set_xlabel('Range (km)')
    ax4.set_ylabel('SNR (dB)')
    ax4.set_title('Detection SNR vs Range')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Performance metrics
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # Calculate metrics
    if track_history and radar_measurements_history:
        total_measurements = sum(len(m) for m in radar_measurements_history)
        avg_tracks = np.mean(num_tracks)
        max_tracks = max(num_tracks)
        
        # Latest tracks info
        latest_tracks = track_history[-1] if track_history else {}
        confirmed_tracks = sum(1 for t in latest_tracks.values() 
                              if t['track_state'] == TrackState.CONFIRMED)
        
        metrics_text = f"""Performance Metrics:
        
Radar Performance:
  • Total Detections: {total_measurements}
  • Avg Detections/Scan: {total_measurements/len(radar_measurements_history):.1f}
  • Avg SNR: {np.mean(all_snrs) if all_snrs else 0:.1f} dB
  • SNR Range: {min(all_snrs) if all_snrs else 0:.1f} - {max(all_snrs) if all_snrs else 0:.1f} dB
  
Tracking Performance:
  • Current Tracks: {len(latest_tracks)}
  • Confirmed Tracks: {confirmed_tracks}
  • Avg Tracks: {avg_tracks:.1f}
  • Max Tracks: {max_tracks}
  
Truth Data (Validation Only):
  • True Targets: {num_true}
  
System Configuration:
  • CFAR: CA-CFAR, Pfa=1e-6
  • Association: Global Nearest Neighbor
  • Min SNR: 10 dB
  
Key: All measurements derived from
radar signal processing - no cheating!"""
        
        ax5.text(0.05, 0.5, metrics_text, fontsize=9, family='monospace',
                verticalalignment='center')
    
    plt.suptitle('Integrated Non-Cheating Radar + Tracking System', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('integrated_radar_tracking.png', dpi=100, bbox_inches='tight')
    print("Saved visualization to 'integrated_radar_tracking.png'")
    plt.close()


def main():
    """Run integrated radar and tracking demonstration."""
    
    print("=" * 60)
    print("Integrated Non-Cheating Radar + Tracking Demo")
    print("=" * 60)
    
    # Initialize systems
    print("\n1. Initializing systems...")
    radar = NonCheatingRadar()
    tracker = IntegratedTracker(dt=0.1)
    
    # Create scenario
    print("2. Creating multi-target scenario...")
    targets = create_scenario()
    print(f"   • Created {len(targets)} targets")
    
    # Simulation parameters
    duration = 10.0  # seconds
    dt = 0.1  # seconds
    num_steps = int(duration / dt)
    
    # Storage for history
    radar_measurements_history = []
    track_history = []
    true_positions_history = []
    time_history = []
    
    print(f"\n3. Running simulation for {duration} seconds...")
    print("-" * 40)
    
    for step in range(num_steps):
        timestamp = step * dt
        
        # Update true target positions (ground truth - not seen by radar!)
        update_targets(targets, dt)
        true_positions = [t.position.copy() for t in targets]
        true_positions_history.append(true_positions)
        
        # Radar processing (no cheating - uses signal processing only!)
        measurements = radar.process(targets, timestamp)
        radar_measurements_history.append(measurements)
        
        # Update tracker with measurements
        tracker.update(measurements, timestamp)
        tracks = tracker.get_tracks()
        track_history.append(tracks)
        time_history.append(timestamp)
        
        # Print status every 1 second
        if step % 10 == 0:
            print(f"t={timestamp:5.1f}s: {len(measurements)} detections, "
                  f"{len(tracks)} tracks "
                  f"({sum(1 for t in tracks.values() if t['track_state']==TrackState.CONFIRMED)} confirmed)")
    
    print("-" * 40)
    print(f"\n4. Simulation complete!")
    print(f"   • Total detections: {sum(len(m) for m in radar_measurements_history)}")
    print(f"   • Tracks initiated: {tracker.metrics.total_tracks_initiated}")
    print(f"   • Tracks confirmed: {tracker.metrics.total_tracks_confirmed}")
    print(f"   • Tracks deleted: {tracker.metrics.total_tracks_deleted}")
    
    # Visualize results
    print("\n5. Generating visualization...")
    visualize_integrated_tracking(
        radar_measurements_history,
        track_history,
        true_positions_history,
        time_history
    )
    
    print("\n" + "=" * 60)
    print("Key Achievements:")
    print("✓ Radar measurements from signal processing only")
    print("✓ No direct access to target positions")
    print("✓ Realistic detection with SNR-based uncertainty")
    print("✓ Tracking operates on radar measurements only")
    print("✓ Complete non-cheating pipeline demonstrated")
    print("=" * 60)


if __name__ == "__main__":
    np.random.seed(42)
    main()