"""
YAML Integration for Advanced Jamming Systems
Parses YAML configurations and instantiates DRFM jammers with cooperative tactics
"""

import yaml
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from .drfm_jammer import DRFMJammer, DRFMTechnique
from .false_target_generator import FalseTargetGenerator, SwarmParams, ScreenParams
from .gate_pull_off import GatePullOff, RangeParams, VelocityParams, PullOffProfile
from .eccm_detector import ECCMDetector


@dataclass
class JammerConfig:
    """Configuration for a single jammer from YAML"""
    name: str
    type: str
    platform: str
    position: Dict[str, float]
    velocity: float
    drfm_params: Dict[str, Any]
    techniques: Dict[str, Any]
    power_management: Dict[str, Any]


@dataclass
class CooperativeConfig:
    """Configuration for cooperative jamming tactics"""
    coordination_mode: str
    datalink: Dict[str, Any]
    synchronized_ops: Dict[str, Any]
    technique_sequencing: List[Dict[str, Any]]


class YAMLJammingSystem:
    """
    Integrates YAML configuration with jamming system components
    """
    
    def __init__(self, config_path: str):
        """
        Initialize jamming system from YAML configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize components
        self.jammers = {}
        self.cooperative_config = None
        self.radar_params = None
        
        # Parse and initialize
        self._parse_configuration()
        self._initialize_jammers()
        self._setup_cooperative_tactics()
        
    def _load_config(self) -> Dict:
        """Load YAML configuration file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _parse_configuration(self):
        """Parse configuration sections"""
        # Radar configuration
        if 'radar' in self.config:
            self.radar_params = self.config['radar']
        
        # Jammer configurations
        if 'jammers' in self.config:
            for jammer_config in self.config['jammers']:
                self.jammers[jammer_config['name']] = self._parse_jammer(jammer_config)
        
        # Cooperative tactics
        if 'cooperative_tactics' in self.config:
            self.cooperative_config = self._parse_cooperative(
                self.config['cooperative_tactics']
            )
    
    def _parse_jammer(self, config: Dict) -> JammerConfig:
        """Parse individual jammer configuration"""
        # Extract position
        if 'initial_position' in config:
            pos = config['initial_position']
            # Convert spherical to Cartesian
            range_m = pos['range']
            azimuth_rad = np.deg2rad(pos['azimuth'])
            elevation_rad = np.deg2rad(pos['elevation'])
            
            x = range_m * np.cos(elevation_rad) * np.cos(azimuth_rad)
            y = range_m * np.cos(elevation_rad) * np.sin(azimuth_rad)
            z = range_m * np.sin(elevation_rad)
            
            position = {'x': x, 'y': y, 'z': z, 'range': range_m}
        else:
            position = {'x': 0, 'y': 0, 'z': 0, 'range': 0}
        
        return JammerConfig(
            name=config.get('name', 'Unknown'),
            type=config.get('type', 'drfm'),
            platform=config.get('platform', 'aircraft'),
            position=position,
            velocity=config.get('velocity', 0),
            drfm_params=config.get('drfm_parameters', {}),
            techniques=config.get('techniques', {}),
            power_management=config.get('power_management', {})
        )
    
    def _parse_cooperative(self, config: Dict) -> CooperativeConfig:
        """Parse cooperative tactics configuration"""
        return CooperativeConfig(
            coordination_mode=config.get('coordination', {}).get('mode', 'networked'),
            datalink=config.get('coordination', {}).get('datalink', {}),
            synchronized_ops=config.get('synchronized_operations', {}),
            technique_sequencing=config.get('technique_sequencing', [])
        )
    
    def _initialize_jammers(self):
        """Initialize jammer instances based on configuration"""
        for name, config in self.jammers.items():
            # Create DRFM jammer
            sample_rate = float(config.drfm_params.get('sample_rate', 1e9))
            memory_depth = int(config.drfm_params.get('memory_depth', 100000))
            
            drfm = DRFMJammer(
                sample_rate=sample_rate,
                memory_depth=memory_depth
            )
            
            # Create false target generator if enabled
            false_gen = None
            if config.techniques.get('false_targets', {}).get('enabled', False):
                # Get radar frequency if available
                carrier_freq = 10e9  # Default X-band
                if self.radar_params and 'parameters' in self.radar_params:
                    freq_val = self.radar_params['parameters'].get('frequency', 10e9)
                    # Handle string representation of scientific notation
                    if isinstance(freq_val, str):
                        carrier_freq = float(eval(freq_val))
                    else:
                        carrier_freq = float(freq_val)
                
                false_gen = FalseTargetGenerator(
                    sample_rate=sample_rate,  # Use the already converted float
                    carrier_frequency=carrier_freq
                )
            
            # Create gate pull-off if enabled
            gpo = None
            if config.techniques.get('gate_pull_off', {}).get('enabled', False):
                gpo = GatePullOff(
                    sample_rate=sample_rate  # Use the already converted float
                )
            
            # Store initialized components
            config.drfm_instance = drfm
            config.false_gen_instance = false_gen
            config.gpo_instance = gpo
    
    def _setup_cooperative_tactics(self):
        """Setup cooperative jamming tactics"""
        if not self.cooperative_config:
            return
        
        # Setup synchronized blinking patterns
        if self.cooperative_config.synchronized_ops.get('blinking', {}).get('enabled'):
            self._setup_blinking_patterns()
        
        # Setup false target coordination
        if self.cooperative_config.synchronized_ops.get('false_target_coordination', {}).get('enabled'):
            self._setup_coordinated_false_targets()
    
    def _setup_blinking_patterns(self):
        """Configure synchronized blinking patterns"""
        blinking = self.cooperative_config.synchronized_ops.get('blinking', {})
        if not blinking:
            return
            
        patterns = blinking.get('duty_cycles', {})
        
        for jammer_name, pattern in patterns.items():
            if jammer_name in self.jammers:
                # Store as attributes on the config object
                config = self.jammers[jammer_name]
                config.blinking_pattern = pattern
                config.blinking_frequency = blinking.get('frequency', 10)
    
    def _setup_coordinated_false_targets(self):
        """Configure coordinated false target generation"""
        coord = self.cooperative_config.synchronized_ops['false_target_coordination']
        
        if coord['mode'] == 'layered':
            # Assign roles to jammers
            params = coord['parameters']
            
            # Outer layer - long range false targets
            if params['outer_layer'] in self.jammers:
                jammer = self.jammers[params['outer_layer']]
                if jammer.false_gen_instance:
                    jammer.false_target_role = 'outer'
                    jammer.false_target_range = [30000, 50000]  # meters
            
            # Middle layer
            if params['middle_layer'] in self.jammers:
                jammer = self.jammers[params['middle_layer']]
                if jammer.false_gen_instance:
                    jammer.false_target_role = 'middle'
                    jammer.false_target_range = [15000, 30000]
            
            # Inner layer
            if params['inner_layer'] in self.jammers:
                jammer = self.jammers[params['inner_layer']]
                if jammer.false_gen_instance:
                    jammer.false_target_role = 'inner'
                    jammer.false_target_range = [5000, 15000]
    
    def execute_mission(self, duration: float, time_step: float) -> Dict[str, Any]:
        """
        Execute the jamming mission based on configuration
        
        Args:
            duration: Mission duration in seconds
            time_step: Simulation time step
            
        Returns:
            Dictionary containing mission results
        """
        results = {
            'time': [],
            'jammer_states': {},
            'false_targets': [],
            'js_ratios': [],
            'techniques_active': []
        }
        
        # Initialize jammer states
        for name in self.jammers:
            results['jammer_states'][name] = []
        
        # Time loop
        time = 0
        step = 0
        
        while time < duration:
            # Check technique sequencing
            active_techniques = self._get_active_techniques(time)
            results['techniques_active'].append(active_techniques)
            
            # Update each jammer
            for name, config in self.jammers.items():
                state = self._update_jammer(name, config, time, active_techniques)
                results['jammer_states'][name].append(state)
            
            # Calculate combined J/S ratio
            js_ratio = self._calculate_combined_js(time)
            results['js_ratios'].append(js_ratio)
            
            # Record time
            results['time'].append(time)
            
            # Update time
            time += time_step
            step += 1
        
        return results
    
    def _get_active_techniques(self, time: float) -> Dict[str, str]:
        """Get active techniques at current time"""
        active = {}
        
        if not self.cooperative_config:
            return active
        
        for sequence_item in self.cooperative_config.technique_sequencing:
            if sequence_item['time'] <= time:
                jammer = sequence_item['jammer']
                technique = sequence_item['technique']
                
                if jammer == 'all':
                    for name in self.jammers:
                        active[name] = technique
                else:
                    active[jammer] = technique
        
        return active
    
    def _update_jammer(self, name: str, config: JammerConfig, 
                      time: float, active_techniques: Dict) -> Dict:
        """Update jammer state"""
        state = {
            'active': False,
            'technique': None,
            'power': 0,
            'false_targets': 0
        }
        
        # Check if jammer has active technique
        if name in active_techniques:
            state['active'] = True
            state['technique'] = active_techniques[name]
            state['power'] = config.power_management.get('total_power', 100)
            
            # Generate false targets if applicable
            if state['technique'] == 'false_targets' and config.false_gen_instance:
                ft_params = config.techniques['false_targets']['parameters']
                state['false_targets'] = ft_params.get('num_targets', 0)
        
        return state
    
    def _calculate_combined_js(self, time: float) -> float:
        """Calculate combined jamming-to-signal ratio"""
        total_jamming_power = 0
        
        for name, config in self.jammers.items():
            if hasattr(config, 'power_management'):
                # Simple J/S calculation
                power = config.power_management.get('total_power', 0)
                gain = config.power_management.get('antenna_gain', 0)
                range_m = config.position['range']
                
                if range_m > 0:
                    # Simplified calculation
                    eff_power = power * 10**(gain/10)
                    path_loss = (range_m / 1000) ** 2
                    jam_power = eff_power / path_loss
                    total_jamming_power += jam_power
        
        # Convert to dB (simplified)
        if total_jamming_power > 0:
            return 10 * np.log10(total_jamming_power)
        return 0
    
    def generate_mission_report(self, results: Dict) -> str:
        """Generate mission effectiveness report"""
        report = []
        report.append("="*60)
        report.append("JAMMING MISSION REPORT")
        report.append("="*60)
        
        # Mission parameters
        report.append(f"\nScenario: {self.config['scenario']['name']}")
        report.append(f"Duration: {self.config['scenario']['duration']} seconds")
        report.append(f"Jammers: {len(self.jammers)}")
        
        # Jammer summary
        report.append("\nJAMMER CONFIGURATION:")
        for name, config in self.jammers.items():
            report.append(f"\n{name}:")
            report.append(f"  Type: {config.type}")
            report.append(f"  Platform: {config.platform}")
            report.append(f"  Range: {config.position['range']/1000:.1f} km")
            report.append(f"  Power: {config.power_management.get('total_power', 0)} W")
            
            # Active techniques
            techniques = []
            for tech, params in config.techniques.items():
                if params.get('enabled', False):
                    techniques.append(tech)
            report.append(f"  Techniques: {', '.join(techniques)}")
        
        # Cooperative tactics
        if self.cooperative_config:
            report.append("\nCOOPERATIVE TACTICS:")
            report.append(f"  Coordination: {self.cooperative_config.coordination_mode}")
            
            if self.cooperative_config.synchronized_ops.get('blinking', {}).get('enabled'):
                report.append("  Synchronized blinking: ENABLED")
            
            if self.cooperative_config.synchronized_ops.get('false_target_coordination', {}).get('enabled'):
                report.append("  Coordinated false targets: ENABLED")
        
        # Results analysis
        if results:
            report.append("\nMISSION RESULTS:")
            
            # Average J/S ratio
            avg_js = np.mean(results['js_ratios'])
            max_js = np.max(results['js_ratios'])
            report.append(f"  Average J/S: {avg_js:.1f} dB")
            report.append(f"  Peak J/S: {max_js:.1f} dB")
            
            # Technique usage
            techniques_used = set()
            for tech_dict in results['techniques_active']:
                techniques_used.update(tech_dict.values())
            report.append(f"  Techniques used: {', '.join(techniques_used)}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)


def load_and_execute_scenario(yaml_path: str) -> Dict[str, Any]:
    """
    Load and execute a jamming scenario from YAML
    
    Args:
        yaml_path: Path to YAML configuration
        
    Returns:
        Results dictionary
    """
    # Initialize system
    system = YAMLJammingSystem(yaml_path)
    
    # Get mission parameters
    duration = system.config['scenario']['duration']
    time_step = system.config['scenario']['time_step']
    
    # Execute mission
    results = system.execute_mission(duration, time_step)
    
    # Generate report
    report = system.generate_mission_report(results)
    print(report)
    
    return results


if __name__ == "__main__":
    # Example usage
    yaml_file = "configs/scenarios/cooperative_drfm_jamming.yaml"
    results = load_and_execute_scenario(yaml_file)