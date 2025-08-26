#!/usr/bin/env python3
"""
Realistic Radar-to-Tracking Demo Without Cheating

This demonstration shows a complete radar simulation that generates measurements
through proper signal processing without any access to ground truth positions.
The radar processes signals from transmission through detection, and only the
resulting measurements are provided to the tracking system.

Key Features:
- No ground truth access during signal processing
- Realistic measurement errors based on SNR
- Proper false alarms from thermal noise
- Range/Doppler ambiguities and blind zones
- Detection probability following radar equation physics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.gridspec import GridSpec
import sys
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import radar simulation components
from src.radar import Radar, RadarParameters
from src.target import Target, TargetType
from src.environment import Environment, WeatherType
from src.waveforms import WaveformGenerator, WaveformType, WaveformParameters
from src.iq_generator import IQDataGenerator, IQParameters
from src.signal import SignalProcessor

# Import our new non-cheating components
from src.radar_simulation.simulator import RadarSimulator, RadarSimulationParameters
from src.radar_simulation.detection import CFARDetector, CFARParameters, CFARType
from src.radar_simulation.processing import RangeDopplerProcessor, ProcessingParameters

# Import tracking components
from src.tracking.kalman_filters import initialize_constant_velocity_filter
from src.tracking.tracker_base import Track, TrackState, TrackingMetrics


def create_realistic_scenario() -> Tuple[List[Target], Environment]:
    """
    Create a realistic multi-target scenario for demonstration.
    Note: These targets are only used for signal generation, not for measurements!
    """
    targets = []
    
    # Target 1: Commercial aircraft at medium range
    target1 = Target(
        position=np.array([15000.0, 8000.0, 5000.0]),  # 15km range
        velocity=np.array([-200.0, 50.0, 0.0]),  # ~200 m/s
        rcs=100.0,  # Large RCS (100 m²)
        target_type=TargetType.AIRCRAFT
    )
    targets.append(target1)
    
    # Target 2: Small drone at close range
    target2 = Target(
        position=np.array([3000.0, 2000.0, 500.0]),  # 3km range
        velocity=np.array([20.0, -15.0, 5.0]),  # ~25 m/s
        rcs=0.01,  # Small RCS (0.01 m²)
        target_type=TargetType.DRONE
    )
    targets.append(target2)
    
    # Target 3: Fast missile at long range
    target3 = Target(
        position=np.array([35000.0, 10000.0, 8000.0]),  # 35km range
        velocity=np.array([-600.0, -100.0, -50.0]),  # ~600 m/s
        rcs=0.5,  # Medium RCS (0.5 m²)
        target_type=TargetType.MISSILE
    )
    targets.append(target3)
    
    # Create environment with some weather
    environment = Environment(
        temperature=15.0,  # Celsius
        pressure=1013.25,  # hPa
        humidity=60.0,  # %
        weather=WeatherType.LIGHT_RAIN,  # Add some attenuation
        wind_speed=10.0,  # m/s
        wind_direction=45.0  # degrees
    )
    
    return targets, environment


def setup_radar_system() -> Tuple[Radar, RadarParameters, WaveformGenerator]:
    """
    Configure a realistic radar system.
    """
    # S-band surveillance radar parameters
    radar_params = RadarParameters(
        frequency=3e9,  # 3 GHz (S-band)
        power=100e3,  # 100 kW peak power
        antenna_gain=35.0,  # 35 dB antenna gain
        pulse_width=1e-6,  # 1 μs pulse width
        prf=1000.0,  # 1 kHz PRF
        bandwidth=10e6,  # 10 MHz bandwidth
        noise_figure=3.0,  # 3 dB noise figure
        losses=5.0  # 5 dB system losses
    )
    
    radar = Radar(radar_params)
    
    # Configure waveform generator for LFM chirp
    waveform_params = WaveformParameters(
        waveform_type=WaveformType.LFM,
        pulse_width=1e-6,
        bandwidth=10e6,
        sample_rate=20e6  # 20 MHz sampling
    )
    
    waveform_gen = WaveformGenerator(waveform_params)
    
    return radar, radar_params, waveform_gen


def simulate_radar_returns(
    radar: Radar,
    targets: List[Target],
    environment: Environment,
    waveform_gen: WaveformGenerator,
    num_pulses: int = 64
) -> np.ndarray:
    """
    Simulate radar returns through proper signal processing.
    This is where the "non-cheating" happens - we generate signals based on physics.
    """
    # Create IQ data generator
    iq_params = IQParameters(
        center_frequency=radar.params.frequency,
        sample_rate=waveform_gen.params.sample_rate,
        pulse_width=radar.params.pulse_width,
        prf=radar.params.prf,
        num_pulses=num_pulses,
        adc_bits=14,  # 14-bit ADC
        iq_imbalance_gain=0.1,  # 10% gain imbalance
        iq_imbalance_phase=2.0,  # 2 degree phase imbalance
        phase_noise_level=-80.0  # -80 dBc/Hz phase noise
    )
    
    iq_generator = IQDataGenerator(iq_params)
    
    # Generate transmit waveform
    tx_waveform = waveform_gen.generate()
    
    # Generate CPI (Coherent Processing Interval) with targets
    # This uses physics-based radar equation, NOT direct target positions
    cpi_data = iq_generator.generate_cpi(
        waveform=tx_waveform,
        targets=targets,
        radar=radar,
        environment=environment,
        include_clutter=True,  # Add realistic clutter
        clutter_level=-30.0  # dB below noise
    )
    
    return cpi_data


def process_radar_data(
    cpi_data: np.ndarray,
    radar_params: RadarParameters,
    waveform_gen: WaveformGenerator
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Process radar data through range-Doppler processing and CFAR detection.
    Returns range-Doppler map and detections WITHOUT using any ground truth.
    """
    # Setup signal processor
    signal_processor = SignalProcessor(
        sample_rate=waveform_gen.params.sample_rate,
        bandwidth=radar_params.bandwidth
    )
    
    # Setup range-Doppler processor
    processing_params = ProcessingParameters(
        sample_rate=waveform_gen.params.sample_rate,
        pulse_repetition_frequency=radar_params.prf,
        center_frequency=radar_params.frequency,
        bandwidth=radar_params.bandwidth,
        range_fft_size=1024,
        doppler_fft_size=64,
        range_window='hann',
        doppler_window='hamming'
    )
    
    # Get reference waveform for matched filtering
    reference_waveform = waveform_gen.generate()
    
    processor = RangeDopplerProcessor(processing_params, reference_waveform)
    
    # Process CPI to get range-Doppler map
    processing_output = processor.process_pulse_train(cpi_data)
    range_doppler_map = processing_output['range_doppler_map']
    
    # Setup CFAR detector
    cfar_params = CFARParameters(
        cfar_type=CFARType.CA_CFAR,
        num_train_cells=16,
        num_guard_cells=4,
        false_alarm_rate=1e-6,
        min_snr_db=10.0  # Minimum 10 dB SNR for detection
    )
    
    # Calculate range and velocity resolutions
    c = 3e8
    range_resolution = c / (2 * radar_params.bandwidth)
    max_unambiguous_velocity = c * radar_params.prf / (4 * radar_params.frequency)
    velocity_resolution = 2 * max_unambiguous_velocity / processing_params.doppler_fft_size
    
    detector = CFARDetector(
        cfar_params,
        range_resolution=range_resolution,
        velocity_resolution=velocity_resolution,
        wavelength=c / radar_params.frequency
    )
    
    # Perform CFAR detection
    detections = detector.detect(np.abs(range_doppler_map)**2)
    
    # Convert to measurement format
    measurements = []
    for det in detections:
        # Convert range-Doppler indices to physical units
        range_m = det.range_bin * range_resolution
        velocity_mps = (det.doppler_bin - processing_params.doppler_fft_size//2) * velocity_resolution
        
        # Estimate measurement uncertainty based on SNR
        # Higher SNR = better accuracy
        range_std = range_resolution / np.sqrt(det.snr)
        velocity_std = velocity_resolution / np.sqrt(det.snr)
        
        measurements.append({
            'range': range_m,
            'velocity': velocity_mps,
            'snr': det.snr,
            'range_std': range_std,
            'velocity_std': velocity_std,
            'azimuth': 0.0,  # Assume boresight for this demo
            'elevation': 0.0
        })
    
    return range_doppler_map, measurements


class SimpleTracker:
    """
    Simple single-target tracker for demonstration.
    Uses only measurements, no ground truth!
    """
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.tracks = {}
        self.next_track_id = 0
        self.metrics = TrackingMetrics()
        
    def update(self, measurements: List[Dict], timestamp: float):
        """Update tracks with new measurements."""
        # For simplicity, create a track for each measurement
        # In reality, you'd use association algorithms
        
        for meas in measurements:
            # Convert polar to Cartesian
            x = meas['range'] * np.cos(meas['azimuth'])
            y = meas['range'] * np.sin(meas['azimuth'])
            
            # Find closest existing track or create new one
            min_dist = float('inf')
            closest_track_id = None
            
            for track_id, track in self.tracks.items():
                pred_pos = track['filter'].x[:2]
                dist = np.linalg.norm([x - pred_pos[0], y - pred_pos[1]])
                if dist < min_dist and dist < 500.0:  # 500m gate
                    min_dist = dist
                    closest_track_id = track_id
            
            if closest_track_id is not None:
                # Update existing track
                track = self.tracks[closest_track_id]
                track['filter'].predict(self.dt)
                track['filter'].update(np.array([x, y]))
                track['last_update'] = timestamp
                track['measurements'].append(meas)
            else:
                # Create new track
                kf = initialize_constant_velocity_filter(
                    dim=2,
                    dt=self.dt,
                    process_noise_std=1.0,
                    measurement_noise_std=meas['range_std']
                )
                kf.x[:2] = [x, y]
                kf.x[2:] = [meas['velocity'] * np.cos(meas['azimuth']),
                           meas['velocity'] * np.sin(meas['azimuth'])]
                
                self.tracks[self.next_track_id] = {
                    'filter': kf,
                    'last_update': timestamp,
                    'measurements': [meas],
                    'track_id': self.next_track_id
                }
                self.next_track_id += 1
                self.metrics.total_tracks_initiated += 1
        
        # Predict tracks without measurements
        for track in self.tracks.values():
            if track['last_update'] < timestamp:
                track['filter'].predict(self.dt)
    
    def get_track_states(self) -> Dict:
        """Get current track states."""
        states = {}
        for track_id, track in self.tracks.items():
            states[track_id] = {
                'position': track['filter'].x[:2].copy(),
                'velocity': track['filter'].x[2:].copy(),
                'covariance': track['filter'].P.copy(),
                'num_measurements': len(track['measurements'])
            }
        return states


def visualize_results(
    range_doppler_map: np.ndarray,
    measurements: List[Dict],
    track_states: Dict,
    true_targets: List[Target],
    radar_params: RadarParameters,
    timestamp: float
):
    """
    Visualize radar processing and tracking results.
    Shows what the radar actually sees vs ground truth.
    """
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # Range-Doppler map
    ax1 = fig.add_subplot(gs[0, 0])
    rd_map_db = 10 * np.log10(np.abs(range_doppler_map)**2 + 1e-10)
    im = ax1.imshow(rd_map_db, aspect='auto', origin='lower', cmap='jet',
                    vmin=np.max(rd_map_db) - 60, vmax=np.max(rd_map_db))
    ax1.set_xlabel('Doppler Bin')
    ax1.set_ylabel('Range Bin')
    ax1.set_title('Range-Doppler Map (dB)')
    plt.colorbar(im, ax=ax1, label='Power (dB)')
    
    # Mark detections
    for meas in measurements:
        c = 3e8
        range_resolution = c / (2 * radar_params.bandwidth)
        max_velocity = c * radar_params.prf / (4 * radar_params.frequency)
        velocity_resolution = 2 * max_velocity / 64
        
        range_bin = int(meas['range'] / range_resolution)
        doppler_bin = int(32 + meas['velocity'] / velocity_resolution)
        ax1.plot(doppler_bin, range_bin, 'r+', markersize=10, markeredgewidth=2)
    
    # Detection plot (Range vs Velocity)
    ax2 = fig.add_subplot(gs[0, 1])
    if measurements:
        ranges = [m['range'] / 1000 for m in measurements]  # Convert to km
        velocities = [m['velocity'] for m in measurements]
        snrs = [m['snr'] for m in measurements]
        
        scatter = ax2.scatter(velocities, ranges, c=snrs, s=100, cmap='viridis',
                            vmin=10, vmax=40, edgecolors='black', linewidth=1)
        plt.colorbar(scatter, ax=ax2, label='SNR (dB)')
        
        # Add error bars based on measurement uncertainty
        for i, meas in enumerate(measurements):
            ax2.errorbar(velocities[i], ranges[i], 
                        xerr=meas['velocity_std'], yerr=meas['range_std']/1000,
                        fmt='none', color='gray', alpha=0.5)
    
    ax2.set_xlabel('Radial Velocity (m/s)')
    ax2.set_ylabel('Range (km)')
    ax2.set_title(f'CFAR Detections (No Ground Truth Used!)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-800, 200])
    ax2.set_ylim([0, 50])
    
    # Ground truth comparison (for validation only)
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Plot true targets (these were NOT used for detection!)
    for i, target in enumerate(true_targets):
        true_range = np.linalg.norm(target.position) / 1000  # km
        true_velocity = -np.dot(target.velocity, target.position / np.linalg.norm(target.position))
        
        ax3.plot(true_velocity, true_range, 'gs', markersize=12, 
                label=f'True Target {i+1}')
    
    # Overlay detections
    if measurements:
        ax3.scatter(velocities, ranges, c='red', s=50, alpha=0.6,
                   label='Radar Detections')
    
    ax3.set_xlabel('Radial Velocity (m/s)')
    ax3.set_ylabel('Range (km)')
    ax3.set_title('Ground Truth vs Detections (Validation Only)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([-800, 200])
    ax3.set_ylim([0, 50])
    
    # Tracking results (Cartesian)
    ax4 = fig.add_subplot(gs[1, :2])
    
    # Plot tracks
    for track_id, state in track_states.items():
        pos = state['position']
        vel = state['velocity']
        
        # Plot track position
        ax4.plot(pos[0]/1000, pos[1]/1000, 'b^', markersize=10,
                label=f'Track {track_id}')
        
        # Plot velocity vector
        ax4.arrow(pos[0]/1000, pos[1]/1000, 
                 vel[0]/100, vel[1]/100,  # Scale velocity for visibility
                 head_width=0.5, head_length=0.3, fc='blue', ec='blue')
        
        # Plot uncertainty ellipse
        cov_2d = state['covariance'][:2, :2]
        eigenvalues, eigenvectors = np.linalg.eig(cov_2d)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(eigenvalues) / 1000  # 1-sigma, convert to km
        
        from matplotlib.patches import Ellipse
        ellipse = Ellipse((pos[0]/1000, pos[1]/1000), width, height, angle=angle,
                         facecolor='blue', alpha=0.2, edgecolor='blue')
        ax4.add_patch(ellipse)
    
    # Show radar at origin
    ax4.plot(0, 0, 'k*', markersize=20, label='Radar')
    
    # Add range rings
    for r in [10, 20, 30, 40]:
        circle = Circle((0, 0), r, fill=False, edgecolor='gray', 
                       linestyle='--', alpha=0.3)
        ax4.add_patch(circle)
    
    ax4.set_xlabel('X Position (km)')
    ax4.set_ylabel('Y Position (km)')
    ax4.set_title('Tracking Results (Using Only Radar Measurements)')
    ax4.set_xlim([-50, 50])
    ax4.set_ylim([-50, 50])
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Performance metrics
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    metrics_text = f"""Radar Performance (t={timestamp:.1f}s):
    
Detection Statistics:
  • Measurements: {len(measurements)}
  • Mean SNR: {np.mean([m['snr'] for m in measurements]) if measurements else 0:.1f} dB
  • Min SNR: {np.min([m['snr'] for m in measurements]) if measurements else 0:.1f} dB
  • Max SNR: {np.max([m['snr'] for m in measurements]) if measurements else 0:.1f} dB
  
Tracking Statistics:
  • Active Tracks: {len(track_states)}
  • Total Initiated: {len(track_states)}
  
Measurement Accuracy (estimated):
  • Range σ: {np.mean([m['range_std'] for m in measurements]) if measurements else 0:.1f} m
  • Velocity σ: {np.mean([m['velocity_std'] for m in measurements]) if measurements else 0:.1f} m/s
  
Radar Parameters:
  • Frequency: {radar_params.frequency/1e9:.1f} GHz
  • PRF: {radar_params.prf:.0f} Hz
  • Bandwidth: {radar_params.bandwidth/1e6:.1f} MHz
  • Range Res: {3e8/(2*radar_params.bandwidth):.1f} m
  
No Ground Truth Used in Processing!"""
    
    ax5.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.suptitle('Realistic Radar Simulation - No Cheating!', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('realistic_radar_tracking_demo.png', dpi=100, bbox_inches='tight')
    print("Saved visualization to 'realistic_radar_tracking_demo.png'")
    plt.close()


def main():
    """
    Main demonstration of realistic radar simulation without cheating.
    """
    print("=" * 60)
    print("Realistic Radar-to-Tracking Demo (No Cheating!)")
    print("=" * 60)
    
    # Setup scenario
    print("\n1. Creating realistic scenario...")
    targets, environment = create_realistic_scenario()
    print(f"   • Created {len(targets)} targets")
    print(f"   • Environment: {environment.weather.name}")
    
    # Setup radar
    print("\n2. Configuring radar system...")
    radar, radar_params, waveform_gen = setup_radar_system()
    print(f"   • Frequency: {radar_params.frequency/1e9:.1f} GHz")
    print(f"   • Power: {radar_params.power/1e3:.1f} kW")
    print(f"   • Max Range: {radar_params.max_unambiguous_range/1000:.1f} km")
    
    # Initialize tracker
    print("\n3. Initializing tracking system...")
    tracker = SimpleTracker(dt=1.0/radar_params.prf)
    
    # Simulation loop
    print("\n4. Running radar simulation...")
    num_scans = 10
    
    for scan in range(num_scans):
        timestamp = scan / radar_params.prf
        
        print(f"\n   Scan {scan+1}/{num_scans} (t={timestamp:.3f}s):")
        
        # Update target positions (for signal generation only!)
        for target in targets:
            target.update(1.0/radar_params.prf)
        
        # Generate radar returns (uses physics, not direct positions)
        print("      • Generating radar signals...")
        cpi_data = simulate_radar_returns(
            radar, targets, environment, waveform_gen, num_pulses=64
        )
        
        # Process radar data (no ground truth access!)
        print("      • Processing range-Doppler...")
        range_doppler_map, measurements = process_radar_data(
            cpi_data, radar_params, waveform_gen
        )
        
        print(f"      • Detected {len(measurements)} targets")
        
        if measurements:
            snrs = [m['snr'] for m in measurements]
            print(f"      • SNR range: {min(snrs):.1f} - {max(snrs):.1f} dB")
        
        # Update tracker with measurements only
        print("      • Updating tracks...")
        tracker.update(measurements, timestamp)
        track_states = tracker.get_track_states()
        print(f"      • Active tracks: {len(track_states)}")
        
        # Visualize last scan
        if scan == num_scans - 1:
            print("\n5. Generating visualization...")
            visualize_results(
                range_doppler_map, measurements, track_states,
                targets, radar_params, timestamp
            )
    
    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("\nKey Points:")
    print("✓ Radar measurements derived from signal processing only")
    print("✓ No direct access to target positions during detection")
    print("✓ Measurement errors scale with SNR (physics-based)")
    print("✓ Tracker operates on realistic measurements only")
    print("✓ Ground truth used only for validation comparison")
    print("=" * 60)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstration
    main()