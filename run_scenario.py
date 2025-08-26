#!/usr/bin/env python3
"""
Configurable scenario runner for radar simulations
Loads YAML configurations and executes simulations with visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config_loader import ConfigLoader, ScenarioConfig
from basic_jamming_demo import JammedRadarSimulator, Jammer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ScenarioRunner:
    """Run radar scenarios from configuration files"""
    
    def __init__(self, config: ScenarioConfig):
        """
        Initialize scenario runner
        
        Args:
            config: Scenario configuration object
        """
        self.config = config
        
        # Initialize radar simulator
        self.radar = JammedRadarSimulator(
            max_range=config.radar.max_range,
            range_resolution=config.radar.range_resolution,
            velocity_resolution=config.radar.velocity_resolution
        )
        
        # Set radar parameters
        self.radar.radar_power = config.radar.power
        self.radar.radar_gain = config.radar.antenna_gain
        self.radar.detection_threshold = config.radar.detection_threshold
        self.radar.n_doppler_bins = config.radar.n_doppler_bins
        
        # Storage for results
        self.results = {
            'detections': [],
            'tracks': [],
            'metrics': {}
        }
        
    def run(self) -> Dict:
        """
        Run the complete scenario
        
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Starting scenario: {self.config.name}")
        logger.info(f"Duration: {self.config.duration}s, Time step: {self.config.time_step}s")
        
        n_steps = int(self.config.duration / self.config.time_step)
        
        # Initialize targets and jammers
        targets = self._initialize_targets()
        jammers = self._initialize_jammers()
        
        # Storage for time series data
        all_detections = []
        target_histories = {t['name']: {'ranges': [], 'velocities': [], 'detected': []} 
                           for t in targets}
        
        # Run simulation loop
        for step in range(n_steps):
            current_time = step * self.config.time_step
            
            # Update target positions
            for target in targets:
                self._update_target_position(target, self.config.time_step)
                target_histories[target['name']]['ranges'].append(target['range'])
                target_histories[target['name']]['velocities'].append(target['velocity'])
            
            # Update jammer positions if attached to platforms
            for jammer in jammers:
                self._update_jammer_position(jammer, targets, self.config.time_step)
            
            # Generate range-Doppler map
            active_jammer = jammers[0] if jammers else None  # Use first jammer for now
            rd_map = self.radar.generate_range_doppler_map(targets, jammer=active_jammer)
            
            # Detect targets
            detections = self.radar.detect_targets(rd_map, jammer=active_jammer)
            
            # Store detections with timestamp
            for det in detections:
                det['time'] = current_time
                all_detections.append(det)
            
            # Check which targets were detected
            for target in targets:
                detected = False
                for det in detections:
                    if abs(det['range'] - target['range']) < 200 and \
                       abs(det['velocity'] - target['velocity']) < 5:
                        detected = True
                        break
                target_histories[target['name']]['detected'].append(detected)
            
            # Progress update
            if step % 50 == 0:
                logger.info(f"  Step {step}/{n_steps}: {len(detections)} detections")
        
        # Calculate metrics
        self.results['detections'] = all_detections
        self.results['target_histories'] = target_histories
        self.results['metrics'] = self._calculate_metrics(target_histories, all_detections)
        
        logger.info(f"Scenario complete: {len(all_detections)} total detections")
        
        return self.results
    
    def _initialize_targets(self) -> List[Dict]:
        """Initialize target dictionaries from configuration"""
        targets = []
        for target_cfg in self.config.targets:
            target = {
                'name': target_cfg.name,
                'type': target_cfg.type,
                'range': target_cfg.range,
                'velocity': target_cfg.velocity,
                'rcs': target_cfg.rcs,
                'azimuth': target_cfg.azimuth,
                'elevation': target_cfg.elevation,
                'maneuver': target_cfg.maneuver
            }
            targets.append(target)
        return targets
    
    def _initialize_jammers(self) -> List[Jammer]:
        """Initialize jammer objects from configuration"""
        jammers = []
        for jammer_cfg in self.config.jammers:
            # Determine jammer position
            if jammer_cfg.range is not None:
                # Standalone jammer with fixed position
                jammer = Jammer(
                    range_m=jammer_cfg.range,
                    velocity_ms=jammer_cfg.velocity or 0,
                    power_w=jammer_cfg.power,
                    jamming_type=jammer_cfg.type,
                    bandwidth_hz=jammer_cfg.bandwidth
                )
                jammer.antenna_gain = jammer_cfg.antenna_gain
                jammer.name = jammer_cfg.name
                jammer.platform = jammer_cfg.platform
                jammer.technique = jammer_cfg.technique
                jammers.append(jammer)
            else:
                # Jammer attached to platform - will get position from target
                jammer = Jammer(
                    range_m=10000,  # Default, will be updated
                    velocity_ms=0,
                    power_w=jammer_cfg.power,
                    jamming_type=jammer_cfg.type,
                    bandwidth_hz=jammer_cfg.bandwidth
                )
                jammer.antenna_gain = jammer_cfg.antenna_gain
                jammer.name = jammer_cfg.name
                jammer.platform = jammer_cfg.platform
                jammer.technique = jammer_cfg.technique
                jammer.activation_range = jammer_cfg.activation_range
                jammers.append(jammer)
        
        return jammers
    
    def _update_target_position(self, target: Dict, dt: float):
        """Update target position based on velocity and maneuvers"""
        # Simple constant velocity model for now
        target['range'] += target['velocity'] * dt
        
        # Apply maneuvers if specified
        if target.get('maneuver'):
            maneuver = target['maneuver']
            if maneuver['type'] == 'weaving':
                # Add sinusoidal variation
                t = self.results.get('current_time', 0)
                lateral_offset = maneuver.get('amplitude', 100) * np.sin(2 * np.pi * t / maneuver.get('period', 10))
                # This would affect azimuth in reality, simplified here
                target['range'] += lateral_offset * dt / maneuver.get('period', 10)
    
    def _update_jammer_position(self, jammer: Jammer, targets: List[Dict], dt: float):
        """Update jammer position if attached to platform"""
        if jammer.platform != 'standalone':
            # Find the platform
            for target in targets:
                if jammer.platform in target['name'] or target['name'] in jammer.platform:
                    jammer.range = target['range']
                    jammer.velocity = target['velocity']
                    break
    
    def _calculate_metrics(self, target_histories: Dict, detections: List[Dict]) -> Dict:
        """Calculate performance metrics"""
        metrics = {}
        
        # Detection probability for each target
        for name, history in target_histories.items():
            detected_frames = sum(history['detected'])
            total_frames = len(history['detected'])
            metrics[f'{name}_pd'] = detected_frames / total_frames if total_frames > 0 else 0
        
        # Overall detection rate
        total_targets = len(target_histories)
        avg_pd = sum(metrics[f'{name}_pd'] for name in target_histories) / total_targets if total_targets > 0 else 0
        metrics['average_pd'] = avg_pd
        
        # False alarm estimation (detections not matching any target)
        if detections:
            false_alarms = 0
            for det in detections:
                matched = False
                for name, history in target_histories.items():
                    time_idx = int(det['time'] / self.config.time_step)
                    if time_idx < len(history['ranges']):
                        if abs(det['range'] - history['ranges'][time_idx]) < 200:
                            matched = True
                            break
                if not matched:
                    false_alarms += 1
            metrics['false_alarm_rate'] = false_alarms / len(detections)
        else:
            metrics['false_alarm_rate'] = 0
        
        return metrics
    
    def visualize_results(self, save_path: Optional[str] = None):
        """Create visualization of scenario results"""
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Range vs Time for all targets
        ax1 = fig.add_subplot(gs[0, :])
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results['target_histories'])))
        
        for i, (name, history) in enumerate(self.results['target_histories'].items()):
            time_array = np.arange(len(history['ranges'])) * self.config.time_step
            ax1.plot(time_array, np.array(history['ranges'])/1000, 
                    label=name, color=colors[i], linewidth=2)
            
            # Mark detected points
            detected_times = [t for t, d in zip(time_array, history['detected']) if d]
            detected_ranges = [r/1000 for r, d in zip(history['ranges'], history['detected']) if d]
            ax1.scatter(detected_times, detected_ranges, color=colors[i], 
                       s=20, alpha=0.5, marker='o')
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Range (km)')
        ax1.set_title('Target Trajectories and Detections', fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. Detection Probability Bar Chart
        ax2 = fig.add_subplot(gs[1, 0])
        
        names = list(self.results['target_histories'].keys())
        pd_values = [self.results['metrics'][f'{name}_pd'] * 100 for name in names]
        
        bars = ax2.bar(range(len(names)), pd_values, color=colors[:len(names)])
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_ylabel('Detection Probability (%)')
        ax2.set_title('Target Detection Rates', fontweight='bold')
        ax2.set_ylim([0, 105])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, pd in zip(bars, pd_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{pd:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 3. Detection Timeline
        ax3 = fig.add_subplot(gs[1, 1:])
        
        if self.results['detections']:
            det_times = [d['time'] for d in self.results['detections']]
            det_ranges = [d['range']/1000 for d in self.results['detections']]
            det_snrs = [d['snr'] for d in self.results['detections']]
            
            scatter = ax3.scatter(det_times, det_ranges, c=det_snrs, 
                                cmap='viridis', s=10, alpha=0.6,
                                vmin=0, vmax=30)
            plt.colorbar(scatter, ax=ax3, label='SNR (dB)')
            
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Range (km)')
        ax3.set_title('All Detections (Color = SNR)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Velocity Distribution
        ax4 = fig.add_subplot(gs[2, 0])
        
        if self.results['detections']:
            velocities = [d['velocity'] for d in self.results['detections']]
            ax4.hist(velocities, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax4.set_xlabel('Velocity (m/s)')
            ax4.set_ylabel('Count')
            ax4.set_title('Detected Velocity Distribution', fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. SNR vs Range
        ax5 = fig.add_subplot(gs[2, 1])
        
        if self.results['detections']:
            ax5.scatter(det_ranges, det_snrs, alpha=0.3, s=5)
            
            # Add theoretical curve
            ranges_theory = np.linspace(1, self.config.radar.max_range/1000, 100)
            snr_theory = 40 - 40 * np.log10(ranges_theory)  # Simplified R^4 law
            ax5.plot(ranges_theory, snr_theory, 'r--', alpha=0.5, label='Theoretical')
            
            ax5.set_xlabel('Range (km)')
            ax5.set_ylabel('SNR (dB)')
            ax5.set_title('SNR vs Range', fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Metrics Summary
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        # Create metrics text
        metrics_text = f"Scenario: {self.config.name}\n"
        metrics_text += f"Duration: {self.config.duration}s\n"
        metrics_text += f"Targets: {len(self.config.targets)}\n"
        metrics_text += f"Jammers: {len(self.config.jammers)}\n\n"
        metrics_text += "Performance Metrics:\n"
        metrics_text += f"Avg Detection Rate: {self.results['metrics']['average_pd']*100:.1f}%\n"
        metrics_text += f"False Alarm Rate: {self.results['metrics']['false_alarm_rate']*100:.1f}%\n"
        metrics_text += f"Total Detections: {len(self.results['detections'])}\n"
        
        # Add jammer info if present
        if self.config.jammers:
            metrics_text += f"\nJamming:\n"
            for jammer in self.config.jammers:
                metrics_text += f"  {jammer.name}: {jammer.power}W, {jammer.type}\n"
        
        ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Main title
        fig.suptitle(f'Scenario Results: {self.config.description}', 
                    fontsize=14, fontweight='bold')
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        return fig


def main():
    """Main entry point for scenario runner"""
    
    parser = argparse.ArgumentParser(description='Run radar simulation scenarios from YAML configs')
    parser.add_argument('scenario', help='Scenario name (without .yaml extension)')
    parser.add_argument('--list', action='store_true', help='List available scenarios')
    parser.add_argument('--validate', action='store_true', help='Validate scenario without running')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    
    args = parser.parse_args()
    
    # Initialize config loader
    loader = ConfigLoader()
    
    # List scenarios if requested
    if args.list:
        print("Available scenarios:")
        for scenario in loader.list_scenarios():
            print(f"  - {scenario}")
        return
    
    # Load scenario
    try:
        config = loader.load_scenario(args.scenario)
        print(f"\nLoaded scenario: {config.name}")
        print(f"Description: {config.description}")
        
        # Validate
        warnings = loader.validate_scenario(config)
        if warnings:
            print("\nValidation warnings:")
            for warning in warnings:
                print(f"  ⚠️  {warning}")
        
        if args.validate:
            print("\nValidation complete.")
            return
        
        # Run scenario
        print("\n" + "="*60)
        print("Running Scenario")
        print("="*60)
        
        runner = ScenarioRunner(config)
        results = runner.run()
        
        # Print results summary
        print("\n" + "="*60)
        print("Results Summary")
        print("="*60)
        print(f"Total Detections: {len(results['detections'])}")
        print(f"Average Detection Rate: {results['metrics']['average_pd']*100:.1f}%")
        print(f"False Alarm Rate: {results['metrics']['false_alarm_rate']*100:.1f}%")
        
        print("\nPer-Target Detection Rates:")
        for target in config.targets:
            pd = results['metrics'].get(f'{target.name}_pd', 0)
            print(f"  {target.name}: {pd*100:.1f}%")
        
        # Create visualization
        if not args.no_viz:
            output_path = None
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                output_path = os.path.join(args.output, f"{args.scenario}_results.png")
            
            runner.visualize_results(save_path=output_path)
        
        print("\n✅ Scenario complete!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"Use --list to see available scenarios")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running scenario: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()