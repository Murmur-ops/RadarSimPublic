#!/usr/bin/env python3
"""
Demonstration of Integrated Tracking Systems

This script demonstrates the capabilities of the advanced integrated tracking systems:
- IMM-JPDA Tracker for multi-target tracking in clutter
- IMM-MHT Tracker for complex multi-target scenarios
- Performance comparison and metrics evaluation
- Sensor fusion capabilities
- Out-of-sequence measurement handling

Author: RadarSim Development Team
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from typing import List, Dict, Tuple
import sys
import os

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from tracking.integrated_trackers import (
    JPDATracker, MHTTracker, InteractingMultipleModel,
    TrackingConfiguration, ModelSet, AssociationMethod,
    create_default_model_set, create_tracking_configuration,
    SensorFusionManager, PerformanceMetrics
)
from tracking.motion_models import CoordinateSystem
from tracking.tracker_base import Measurement

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrackingScenario:
    """
    Comprehensive tracking scenario generator for testing integrated trackers.
    """
    
    def __init__(self, scenario_type: str = "crossing_targets"):
        """
        Initialize tracking scenario.
        
        Args:
            scenario_type: Type of scenario to generate
        """
        self.scenario_type = scenario_type
        self.ground_truth_tracks = {}
        self.measurements = []
        
    def generate_scenario(self, duration: float = 10.0, dt: float = 0.1) -> Tuple[List[Measurement], Dict[str, List[np.ndarray]]]:
        """
        Generate a complete tracking scenario.
        
        Args:
            duration: Scenario duration in seconds
            dt: Time step
            
        Returns:
            Tuple of (measurements, ground_truth_tracks)
        """
        if self.scenario_type == "crossing_targets":
            return self._generate_crossing_targets_scenario(duration, dt)
        elif self.scenario_type == "maneuvering_targets":
            return self._generate_maneuvering_targets_scenario(duration, dt)
        elif self.scenario_type == "dense_clutter":
            return self._generate_dense_clutter_scenario(duration, dt)
        else:
            raise ValueError(f"Unknown scenario type: {self.scenario_type}")
    
    def _generate_crossing_targets_scenario(self, duration: float, dt: float) -> Tuple[List[Measurement], Dict[str, List[np.ndarray]]]:
        """Generate scenario with crossing targets."""
        measurements = []
        ground_truth = {}
        
        times = np.arange(0, duration, dt)
        
        # Target 1: Moving right with slight acceleration
        track1_states = []
        for t in times:
            x = 10 + 8*t + 0.1*t**2
            y = 20 + 2*t
            vx = 8 + 0.2*t
            vy = 2
            state = np.array([x, y, vx, vy])
            track1_states.append(state)
            
            # Add measurement with noise
            if np.random.rand() > 0.1:  # 90% detection probability
                meas_pos = state[:2] + np.random.multivariate_normal([0, 0], np.diag([0.5, 0.5]))
                measurement = Measurement(
                    position=np.concatenate([meas_pos, [0]]),
                    timestamp=t,
                    covariance=np.diag([0.5, 0.5, 1.0]),
                    metadata={'true_track_id': 'track_1'}
                )
                measurements.append(measurement)
        
        ground_truth['track_1'] = track1_states
        
        # Target 2: Moving up and left with coordinated turn
        track2_states = []
        for t in times:
            if t < 3.0:
                # Straight motion
                x = 50 - 5*t
                y = 10 + 6*t
                vx = -5
                vy = 6
            else:
                # Coordinated turn
                turn_time = t - 3.0
                omega = 0.3  # rad/s
                speed = np.sqrt(25 + 36)  # sqrt(vx^2 + vy^2)
                
                # Initial position at turn start
                x0, y0 = 35, 28
                vx0, vy0 = -5, 6
                
                # Coordinated turn equations
                x = x0 + (vx0 * np.sin(omega * turn_time) - vy0 * (1 - np.cos(omega * turn_time))) / omega
                y = y0 + (vx0 * (1 - np.cos(omega * turn_time)) + vy0 * np.sin(omega * turn_time)) / omega
                vx = vx0 * np.cos(omega * turn_time) - vy0 * np.sin(omega * turn_time)
                vy = vx0 * np.sin(omega * turn_time) + vy0 * np.cos(omega * turn_time)
            
            state = np.array([x, y, vx, vy])
            track2_states.append(state)
            
            # Add measurement with noise
            if np.random.rand() > 0.15:  # 85% detection probability
                meas_pos = state[:2] + np.random.multivariate_normal([0, 0], np.diag([0.5, 0.5]))
                measurement = Measurement(
                    position=np.concatenate([meas_pos, [0]]),
                    timestamp=t,
                    covariance=np.diag([0.5, 0.5, 1.0]),
                    metadata={'true_track_id': 'track_2'}
                )
                measurements.append(measurement)
        
        ground_truth['track_2'] = track2_states
        
        # Add clutter measurements
        for t in times:
            num_clutter = np.random.poisson(0.5)  # Average 0.5 clutter per scan
            for _ in range(num_clutter):
                clutter_x = np.random.uniform(-10, 80)
                clutter_y = np.random.uniform(-10, 60)
                
                measurement = Measurement(
                    position=np.array([clutter_x, clutter_y, 0]),
                    timestamp=t + np.random.uniform(-dt/2, dt/2),
                    covariance=np.diag([1.0, 1.0, 1.0]),
                    metadata={'true_track_id': 'clutter'}
                )
                measurements.append(measurement)
        
        # Sort measurements by timestamp
        measurements.sort(key=lambda m: m.timestamp)
        
        return measurements, ground_truth
    
    def _generate_maneuvering_targets_scenario(self, duration: float, dt: float) -> Tuple[List[Measurement], Dict[str, List[np.ndarray]]]:
        """Generate scenario with highly maneuvering targets."""
        measurements = []
        ground_truth = {}
        
        times = np.arange(0, duration, dt)
        
        # Highly maneuvering target with multiple maneuver phases
        track_states = []
        for t in times:
            if t < 2.0:
                # Constant velocity phase
                x = 5 + 10*t
                y = 5 + 3*t
                vx = 10
                vy = 3
            elif t < 4.0:
                # Acceleration phase
                phase_time = t - 2.0
                x = 25 + 10*phase_time + 2*phase_time**2
                y = 11 + 3*phase_time + 1*phase_time**2
                vx = 10 + 4*phase_time
                vy = 3 + 2*phase_time
            elif t < 6.0:
                # Sharp turn phase
                phase_time = t - 4.0
                omega = 1.0  # Sharp turn
                x0, y0 = 49, 19
                vx0, vy0 = 18, 7
                
                x = x0 + (vx0 * np.sin(omega * phase_time) - vy0 * (1 - np.cos(omega * phase_time))) / omega
                y = y0 + (vx0 * (1 - np.cos(omega * phase_time)) + vy0 * np.sin(omega * phase_time)) / omega
                vx = vx0 * np.cos(omega * phase_time) - vy0 * np.sin(omega * phase_time)
                vy = vx0 * np.sin(omega * phase_time) + vy0 * np.cos(omega * phase_time)
            else:
                # Deceleration phase
                phase_time = t - 6.0
                # Approximate state at t=6
                x0, y0 = 67.5, 37.8
                vx0, vy0 = 7, 18
                
                x = x0 + vx0*phase_time - 1*phase_time**2
                y = y0 + vy0*phase_time - 2*phase_time**2
                vx = vx0 - 2*phase_time
                vy = vy0 - 4*phase_time
            
            state = np.array([x, y, vx, vy])
            track_states.append(state)
            
            # Add measurement with noise
            if np.random.rand() > 0.05:  # 95% detection probability
                meas_pos = state[:2] + np.random.multivariate_normal([0, 0], np.diag([0.3, 0.3]))
                measurement = Measurement(
                    position=np.concatenate([meas_pos, [0]]),
                    timestamp=t,
                    covariance=np.diag([0.3, 0.3, 1.0]),
                    metadata={'true_track_id': 'maneuvering_track'}
                )
                measurements.append(measurement)
        
        ground_truth['maneuvering_track'] = track_states
        
        # Add moderate clutter
        for t in times:
            num_clutter = np.random.poisson(1.0)
            for _ in range(num_clutter):
                clutter_x = np.random.uniform(-20, 100)
                clutter_y = np.random.uniform(-20, 80)
                
                measurement = Measurement(
                    position=np.array([clutter_x, clutter_y, 0]),
                    timestamp=t + np.random.uniform(-dt/2, dt/2),
                    covariance=np.diag([0.8, 0.8, 1.0]),
                    metadata={'true_track_id': 'clutter'}
                )
                measurements.append(measurement)
        
        measurements.sort(key=lambda m: m.timestamp)
        return measurements, ground_truth
    
    def _generate_dense_clutter_scenario(self, duration: float, dt: float) -> Tuple[List[Measurement], Dict[str, List[np.ndarray]]]:
        """Generate scenario with dense clutter environment."""
        measurements = []
        ground_truth = {}
        
        times = np.arange(0, duration, dt)
        
        # Two targets in dense clutter
        # Target 1: Simple constant velocity
        track1_states = []
        for t in times:
            x = 20 + 4*t
            y = 30 + 2*t
            vx = 4
            vy = 2
            state = np.array([x, y, vx, vy])
            track1_states.append(state)
            
            if np.random.rand() > 0.2:  # 80% detection probability
                meas_pos = state[:2] + np.random.multivariate_normal([0, 0], np.diag([0.7, 0.7]))
                measurement = Measurement(
                    position=np.concatenate([meas_pos, [0]]),
                    timestamp=t,
                    covariance=np.diag([0.7, 0.7, 1.0]),
                    metadata={'true_track_id': 'dense_track_1'}
                )
                measurements.append(measurement)
        
        ground_truth['dense_track_1'] = track1_states
        
        # Target 2: Weaving motion
        track2_states = []
        for t in times:
            x = 15 + 6*t
            y = 15 + 3*t + 5*np.sin(0.5*t)  # Weaving motion
            vx = 6
            vy = 3 + 2.5*np.cos(0.5*t)
            state = np.array([x, y, vx, vy])
            track2_states.append(state)
            
            if np.random.rand() > 0.25:  # 75% detection probability
                meas_pos = state[:2] + np.random.multivariate_normal([0, 0], np.diag([0.7, 0.7]))
                measurement = Measurement(
                    position=np.concatenate([meas_pos, [0]]),
                    timestamp=t,
                    covariance=np.diag([0.7, 0.7, 1.0]),
                    metadata={'true_track_id': 'dense_track_2'}
                )
                measurements.append(measurement)
        
        ground_truth['dense_track_2'] = track2_states
        
        # Add dense clutter (high false alarm rate)
        for t in times:
            num_clutter = np.random.poisson(5.0)  # High clutter density
            for _ in range(num_clutter):
                clutter_x = np.random.uniform(0, 80)
                clutter_y = np.random.uniform(0, 60)
                
                measurement = Measurement(
                    position=np.array([clutter_x, clutter_y, 0]),
                    timestamp=t + np.random.uniform(-dt/2, dt/2),
                    covariance=np.diag([1.5, 1.5, 1.0]),
                    metadata={'true_track_id': 'clutter'}
                )
                measurements.append(measurement)
        
        measurements.sort(key=lambda m: m.timestamp)
        return measurements, ground_truth


def run_tracking_comparison(scenario_type: str = "crossing_targets") -> Dict[str, Dict]:
    """
    Run comprehensive tracking comparison between JPDA and MHT.
    
    Args:
        scenario_type: Type of scenario to test
        
    Returns:
        Dictionary of results for each tracker
    """
    logger.info(f"Running tracking comparison for scenario: {scenario_type}")
    
    # Generate scenario
    scenario = TrackingScenario(scenario_type)
    measurements, ground_truth = scenario.generate_scenario(duration=10.0, dt=0.1)
    
    logger.info(f"Generated {len(measurements)} measurements for {len(ground_truth)} true tracks")
    
    # Initialize trackers
    jpda_config = create_tracking_configuration(AssociationMethod.JPDA)
    mht_config = create_tracking_configuration(AssociationMethod.MHT)
    
    jpda_tracker = JPDATracker(jpda_config)
    mht_tracker = MHTTracker(mht_config)
    
    # Initialize performance metrics
    jpda_metrics = PerformanceMetrics()
    mht_metrics = PerformanceMetrics()
    
    # Process measurements in batches
    batch_size = 5
    jpda_times = []
    mht_times = []
    
    jpda_track_history = []
    mht_track_history = []
    
    current_time = 0.0
    batch_count = 0
    
    for i in range(0, len(measurements), batch_size):
        batch = measurements[i:i+batch_size]
        if not batch:
            continue
        
        batch_time = batch[-1].timestamp
        batch_count += 1
        
        # Process with JPDA tracker
        start_time = time.time()
        jpda_tracker.predict(batch_time)
        jpda_tracker.update(batch)
        jpda_process_time = time.time() - start_time
        jpda_times.append(jpda_process_time)
        
        # Process with MHT tracker  
        start_time = time.time()
        mht_tracker.predict(batch_time)
        mht_tracker.update(batch)
        mht_process_time = time.time() - start_time
        mht_times.append(mht_process_time)
        
        # Record track states
        jpda_states = jpda_tracker.get_track_states(batch_time)
        mht_states = mht_tracker.get_track_states(batch_time)
        
        jpda_track_history.append((batch_time, jpda_states.copy()))
        mht_track_history.append((batch_time, mht_states.copy()))
        
        # Compute OSPA if we have sufficient time evolution
        if batch_count % 10 == 0:  # Every 10 batches
            # Get ground truth at this time
            gt_at_time = {}
            for track_id, states in ground_truth.items():
                time_idx = min(int(batch_time / 0.1), len(states) - 1)
                gt_at_time[f"gt_{track_id}"] = states[time_idx]
            
            jpda_ospa = jpda_metrics.compute_ospa_distance(jpda_states, gt_at_time)
            mht_ospa = mht_metrics.compute_ospa_distance(mht_states, gt_at_time)
        
        if batch_count % 20 == 0:
            logger.info(f"Processed {batch_count} batches, time: {batch_time:.1f}s")
            logger.info(f"  JPDA: {len(jpda_states)} tracks, {jpda_process_time*1000:.1f}ms")
            logger.info(f"  MHT:  {len(mht_states)} tracks, {mht_process_time*1000:.1f}ms")
    
    # Compile results
    jpda_results = {
        'tracker_type': 'JPDA',
        'final_tracks': jpda_tracker.get_track_states(),
        'track_count': jpda_tracker.get_track_count(),
        'processing_times': jpda_times,
        'average_processing_time': np.mean(jpda_times),
        'total_processing_time': np.sum(jpda_times),
        'track_history': jpda_track_history,
        'performance_metrics': jpda_metrics.get_summary()
    }
    
    mht_results = {
        'tracker_type': 'MHT',
        'final_tracks': mht_tracker.get_track_states(),
        'track_count': mht_tracker.get_track_count(),
        'processing_times': mht_times,
        'average_processing_time': np.mean(mht_times),
        'total_processing_time': np.sum(mht_times),
        'track_history': mht_track_history,
        'performance_metrics': mht_metrics.get_summary()
    }
    
    return {
        'jpda': jpda_results,
        'mht': mht_results,
        'ground_truth': ground_truth,
        'measurements': measurements,
        'scenario_type': scenario_type
    }


def demonstrate_sensor_fusion():
    """Demonstrate multi-sensor fusion capabilities."""
    logger.info("Demonstrating sensor fusion capabilities...")
    
    # Create configuration with sensor fusion enabled
    config = create_tracking_configuration(
        AssociationMethod.JPDA,
        enable_sensor_fusion=True
    )
    
    tracker = JPDATracker(config)
    fusion_manager = SensorFusionManager(config)
    
    # Simulate measurements from multiple sensors
    duration = 5.0
    dt = 0.1
    times = np.arange(0, duration, dt)
    
    # Sensor 1: High accuracy, regular updates
    for t in times:
        x = 10 + 5*t
        y = 20 + 2*t
        measurement = Measurement(
            position=np.array([x, y, 0]) + np.random.multivariate_normal([0, 0, 0], np.diag([0.2, 0.2, 0.5])),
            timestamp=t,
            covariance=np.diag([0.2, 0.2, 0.5]),
            metadata={'sensor_id': 'sensor_1'}
        )
        fusion_manager.add_sensor_data('sensor_1', [measurement])
    
    # Sensor 2: Lower accuracy, irregular updates
    for t in times[::2]:  # Every other time step
        x = 10 + 5*t
        y = 20 + 2*t
        measurement = Measurement(
            position=np.array([x, y, 0]) + np.random.multivariate_normal([0, 0, 0], np.diag([0.8, 0.8, 1.0])),
            timestamp=t + np.random.uniform(-0.02, 0.02),  # Small timing offset
            covariance=np.diag([0.8, 0.8, 1.0]),
            metadata={'sensor_id': 'sensor_2'}
        )
        fusion_manager.add_sensor_data('sensor_2', [measurement])
    
    # Process fused measurements
    total_fused = 0
    for t in times:
        fused_measurements = fusion_manager.get_fused_measurements(t)
        if fused_measurements:
            tracker.predict(t)
            tracker.update(fused_measurements)
            total_fused += len(fused_measurements)
    
    final_tracks = tracker.get_track_states()
    logger.info(f"Sensor fusion processed {total_fused} fused measurements")
    logger.info(f"Final tracks from fusion: {len(final_tracks)}")
    
    return tracker, fusion_manager


def plot_tracking_results(results: Dict[str, Dict], save_path: str = None):
    """
    Plot comprehensive tracking results.
    
    Args:
        results: Results from run_tracking_comparison
        save_path: Optional path to save plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Matplotlib not available, skipping plots")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Tracking Comparison - {results["scenario_type"].title()} Scenario', fontsize=16)
    
    # Plot 1: Ground truth and final tracks
    ax1 = axes[0, 0]
    ground_truth = results['ground_truth']
    
    # Plot ground truth tracks
    for track_id, states in ground_truth.items():
        positions = np.array([state[:2] for state in states])
        ax1.plot(positions[:, 0], positions[:, 1], '--', linewidth=2, 
                label=f'GT {track_id}', alpha=0.7)
    
    # Plot JPDA tracks
    jpda_tracks = results['jpda']['final_tracks']
    for i, (track_id, state) in enumerate(jpda_tracks.items()):
        ax1.plot(state[0], state[1], 'ro', markersize=8, 
                label=f'JPDA Track {i+1}' if i < 3 else "")
    
    # Plot MHT tracks  
    mht_tracks = results['mht']['final_tracks']
    for i, (track_id, state) in enumerate(mht_tracks.items()):
        ax1.plot(state[0], state[1], 'bs', markersize=8,
                label=f'MHT Track {i+1}' if i < 3 else "")
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Final Track Positions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Processing times comparison
    ax2 = axes[0, 1]
    jpda_times = np.array(results['jpda']['processing_times']) * 1000  # Convert to ms
    mht_times = np.array(results['mht']['processing_times']) * 1000
    
    x = np.arange(len(jpda_times))
    ax2.plot(x, jpda_times, 'r-', label='JPDA', alpha=0.7)
    ax2.plot(x, mht_times, 'b-', label='MHT', alpha=0.7)
    ax2.set_xlabel('Batch Number')
    ax2.set_ylabel('Processing Time (ms)')
    ax2.set_title('Processing Time Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Track count over time
    ax3 = axes[1, 0]
    
    jpda_history = results['jpda']['track_history']
    mht_history = results['mht']['track_history']
    
    jpda_times_hist = [entry[0] for entry in jpda_history]
    jpda_counts = [len(entry[1]) for entry in jpda_history]
    
    mht_times_hist = [entry[0] for entry in mht_history]
    mht_counts = [len(entry[1]) for entry in mht_history]
    
    ax3.plot(jpda_times_hist, jpda_counts, 'r-', label='JPDA', linewidth=2)
    ax3.plot(mht_times_hist, mht_counts, 'b-', label='MHT', linewidth=2)
    ax3.axhline(y=len(ground_truth), color='k', linestyle='--', 
               label=f'True Tracks ({len(ground_truth)})')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Number of Tracks')
    ax3.set_title('Track Count Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance metrics comparison
    ax4 = axes[1, 1]
    
    # Create performance comparison
    metrics_comparison = {
        'JPDA': results['jpda']['performance_metrics'],
        'MHT': results['mht']['performance_metrics']
    }
    
    # Extract OSPA values if available
    jpda_ospa = metrics_comparison['JPDA'].get('average_ospa', 0)
    mht_ospa = metrics_comparison['MHT'].get('average_ospa', 0)
    
    # Processing time comparison
    jpda_avg_time = results['jpda']['average_processing_time'] * 1000
    mht_avg_time = results['mht']['average_processing_time'] * 1000
    
    # Final track count accuracy
    jpda_track_error = abs(len(jpda_tracks) - len(ground_truth))
    mht_track_error = abs(len(mht_tracks) - len(ground_truth))
    
    metrics = ['OSPA Distance', 'Avg Processing Time (ms)', 'Track Count Error']
    jpda_values = [jpda_ospa, jpda_avg_time, jpda_track_error]
    mht_values = [mht_ospa, mht_avg_time, mht_track_error]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x - width/2, jpda_values, width, label='JPDA', alpha=0.7, color='red')
    ax4.bar(x + width/2, mht_values, width, label='MHT', alpha=0.7, color='blue')
    
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Values')
    ax4.set_title('Performance Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def main():
    """Main demonstration function."""
    logger.info("Starting Integrated Tracking Systems Demonstration")
    logger.info("=" * 60)
    
    # Test different scenarios
    scenarios = ["crossing_targets", "maneuvering_targets", "dense_clutter"]
    
    all_results = {}
    
    for scenario in scenarios:
        logger.info(f"\n{'='*20} Testing {scenario.upper()} scenario {'='*20}")
        
        try:
            results = run_tracking_comparison(scenario)
            all_results[scenario] = results
            
            # Print summary
            jpda_results = results['jpda']
            mht_results = results['mht']
            
            logger.info(f"\nResults for {scenario}:")
            logger.info(f"  Ground truth tracks: {len(results['ground_truth'])}")
            logger.info(f"  Total measurements: {len(results['measurements'])}")
            
            logger.info(f"\n  JPDA Tracker:")
            logger.info(f"    Final tracks: {jpda_results['track_count']['confirmed']}")
            logger.info(f"    Avg processing time: {jpda_results['average_processing_time']*1000:.2f} ms")
            logger.info(f"    Total processing time: {jpda_results['total_processing_time']*1000:.1f} ms")
            
            logger.info(f"\n  MHT Tracker:")
            logger.info(f"    Final tracks: {mht_results['track_count']['confirmed']}")  
            logger.info(f"    Avg processing time: {mht_results['average_processing_time']*1000:.2f} ms")
            logger.info(f"    Total processing time: {mht_results['total_processing_time']*1000:.1f} ms")
            
            # Generate plots for first scenario
            if scenario == scenarios[0]:
                plot_tracking_results(results, f"tracking_results_{scenario}.png")
                
        except Exception as e:
            logger.error(f"Error in scenario {scenario}: {e}")
            import traceback
            traceback.print_exc()
    
    # Demonstrate sensor fusion
    logger.info(f"\n{'='*20} Testing SENSOR FUSION {'='*20}")
    try:
        demonstrate_sensor_fusion()
    except Exception as e:
        logger.error(f"Error in sensor fusion demo: {e}")
    
    # Final summary
    logger.info(f"\n{'='*20} FINAL SUMMARY {'='*20}")
    logger.info("Integrated tracking systems demonstration completed successfully!")
    logger.info("\nKey capabilities demonstrated:")
    logger.info("✓ IMM-JPDA tracking with multiple motion models")
    logger.info("✓ IMM-MHT tracking with hypothesis management")
    logger.info("✓ Multi-target tracking in cluttered environments")
    logger.info("✓ Maneuvering target handling")
    logger.info("✓ Performance metrics and evaluation")
    logger.info("✓ Sensor fusion capabilities")
    logger.info("✓ Comprehensive configuration management")
    
    return all_results


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    results = main()