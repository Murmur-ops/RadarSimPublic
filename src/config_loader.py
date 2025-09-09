#!/usr/bin/env python3
"""
Configuration loader for radar simulation scenarios
Handles YAML parsing, validation, and scenario setup
"""

import yaml
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

# Import classification mappings if available
try:
    from classification import (
        get_target_class_from_config,
        get_radar_mission_from_config
    )
    CLASSIFICATION_AVAILABLE = True
except ImportError:
    CLASSIFICATION_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RadarConfig:
    """Radar system configuration"""
    type: str
    frequency: float
    power: float
    antenna_gain: float
    pulse_width: float
    prf: float
    bandwidth: float
    noise_figure: float
    losses: float
    range_resolution: float
    max_range: float
    velocity_resolution: float
    n_doppler_bins: int
    detection_threshold: float
    cfar_guard_cells: int = 3
    cfar_training_cells: int = 10
    tracking_config: Optional[Dict] = None


@dataclass
class TargetConfig:
    """Target configuration"""
    name: str
    type: str
    range: float
    azimuth: float
    elevation: float
    velocity: float
    rcs: float
    maneuver: Optional[Dict] = None
    
    def to_dict(self):
        """Convert to dictionary for simulation"""
        return {
            'name': self.name,
            'type': self.type,
            'range': self.range,
            'velocity': self.velocity,
            'rcs': self.rcs,
            'azimuth': self.azimuth,
            'elevation': self.elevation
        }
    
    def get_target_class(self):
        """Get TargetClass enum for this target"""
        if CLASSIFICATION_AVAILABLE:
            return get_target_class_from_config(self.type)
        return None


@dataclass
class JammerConfig:
    """Jammer configuration"""
    name: str
    type: str
    platform: str
    power: float
    antenna_gain: float
    bandwidth: float = 1e6
    technique: str = "noise"
    range: Optional[float] = None
    azimuth: Optional[float] = None
    elevation: Optional[float] = None
    velocity: Optional[float] = None
    activation_range: Optional[float] = None
    additional_params: Dict = field(default_factory=dict)


@dataclass
class EnvironmentConfig:
    """Environmental configuration"""
    temperature: float = 15.0
    pressure: float = 1013.25
    humidity: float = 50.0
    weather: str = "clear"
    terrain: str = "flat"
    rain_rate: float = 0.0
    sea_state: Optional[int] = None
    wind_speed: Optional[float] = None


@dataclass
class ScenarioConfig:
    """Complete scenario configuration"""
    name: str
    description: str
    duration: float
    time_step: float
    radar: RadarConfig
    targets: List[TargetConfig]
    jammers: List[JammerConfig]
    environment: EnvironmentConfig
    clutter: Optional[Dict] = None
    output: Optional[Dict] = None


class ConfigLoader:
    """Load and validate simulation configurations"""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize configuration loader
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.scenarios_dir = self.config_dir / "scenarios"
        self.templates_dir = self.config_dir / "templates"
        
        # Create directories if they don't exist
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
    def load_scenario(self, scenario_name: str) -> ScenarioConfig:
        """
        Load a scenario configuration from YAML
        
        Args:
            scenario_name: Name of scenario file (with or without .yaml)
            
        Returns:
            ScenarioConfig object
        """
        # Check if this is already a path
        if '/' in scenario_name or scenario_name.startswith('configs'):
            # User provided a path, use it directly
            filepath = Path(scenario_name)
            if not filepath.exists():
                # Try without the redundant configs/scenarios prefix
                if scenario_name.startswith('configs/scenarios/'):
                    scenario_name = scenario_name.replace('configs/scenarios/', '')
                    filepath = self.scenarios_dir / scenario_name
        else:
            # Just the scenario name, add .yaml if needed
            if not scenario_name.endswith('.yaml'):
                scenario_name += '.yaml'
            filepath = self.scenarios_dir / scenario_name
        
        if not filepath.exists():
            raise FileNotFoundError(f"Scenario file not found: {filepath}")
        
        logger.info(f"Loading scenario: {filepath}")
        
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return self._parse_scenario(config_dict)
    
    def _parse_scenario(self, config_dict: Dict) -> ScenarioConfig:
        """Parse scenario dictionary into configuration objects"""
        
        scenario = config_dict['scenario']
        
        # Parse radar configuration
        radar_cfg = config_dict['radar']
        radar = RadarConfig(
            type=radar_cfg['type'],
            frequency=radar_cfg['parameters']['frequency'],
            power=radar_cfg['parameters']['power'],
            antenna_gain=radar_cfg['parameters']['antenna_gain'],
            pulse_width=radar_cfg['parameters']['pulse_width'],
            prf=radar_cfg['parameters']['prf'],
            bandwidth=radar_cfg['parameters']['bandwidth'],
            noise_figure=radar_cfg['parameters']['noise_figure'],
            losses=radar_cfg['parameters']['losses'],
            range_resolution=radar_cfg['processing']['range_resolution'],
            max_range=radar_cfg['processing']['max_range'],
            velocity_resolution=radar_cfg['processing']['velocity_resolution'],
            n_doppler_bins=radar_cfg['processing']['n_doppler_bins'],
            detection_threshold=radar_cfg['processing']['detection_threshold'],
            cfar_guard_cells=radar_cfg['processing'].get('cfar_guard_cells', 3),
            cfar_training_cells=radar_cfg['processing'].get('cfar_training_cells', 10),
            tracking_config=radar_cfg.get('tracking')
        )
        
        # Parse targets
        targets = []
        for target_cfg in config_dict['targets']:
            # Handle both position formats
            initial_pos = target_cfg['initial_position']
            
            if isinstance(initial_pos, list):
                # Convert from Cartesian [x, y] km to spherical coordinates
                x_km, y_km = initial_pos[0], initial_pos[1] if len(initial_pos) > 1 else 0
                z_km = initial_pos[2] if len(initial_pos) > 2 else 0
                
                # Convert to meters
                x_m, y_m, z_m = x_km * 1000, y_km * 1000, z_km * 1000
                
                # Calculate spherical coordinates
                range_m = np.sqrt(x_m**2 + y_m**2 + z_m**2)
                azimuth_rad = np.arctan2(y_m, x_m)
                azimuth_deg = np.degrees(azimuth_rad)
                
                if range_m > 0:
                    elevation_rad = np.arcsin(z_m / range_m)
                    elevation_deg = np.degrees(elevation_rad)
                else:
                    elevation_deg = 0
                
                target_range = range_m
                target_azimuth = azimuth_deg
                target_elevation = elevation_deg
            else:
                # Dict format with range, azimuth, elevation
                target_range = initial_pos['range']
                target_azimuth = initial_pos.get('azimuth', 0)
                target_elevation = initial_pos.get('elevation', 0)
            
            # Handle velocity format (can be list [vx, vy] or dict)
            velocity = target_cfg['velocity']
            if isinstance(velocity, dict):
                velocity = [velocity.get('x', 0), velocity.get('y', 0), velocity.get('z', 0)]
            elif isinstance(velocity, (int, float)):
                velocity = [velocity, 0, 0]
            elif len(velocity) == 2:
                velocity = [velocity[0], velocity[1], 0]
            
            target = TargetConfig(
                name=target_cfg['name'],
                type=target_cfg['type'],
                range=target_range,
                azimuth=target_azimuth,
                elevation=target_elevation,
                velocity=velocity,
                rcs=target_cfg['rcs'],
                maneuver=target_cfg.get('maneuver')
            )
            targets.append(target)
        
        # Parse jammers
        jammers = []
        for jammer_cfg in config_dict.get('jammers', []):
            params = jammer_cfg.get('parameters', {})
            
            # Handle position - either from initial_position or attached to platform
            if 'initial_position' in jammer_cfg:
                pos = jammer_cfg['initial_position']
                jammer = JammerConfig(
                    name=jammer_cfg['name'],
                    type=jammer_cfg['type'],
                    platform=jammer_cfg.get('platform', 'standalone'),
                    power=params['power'],
                    antenna_gain=params['antenna_gain'],
                    bandwidth=params.get('bandwidth', 1e6),
                    technique=params.get('technique', 'noise'),
                    range=pos.get('range'),
                    azimuth=pos.get('azimuth'),
                    elevation=pos.get('elevation'),
                    velocity=jammer_cfg.get('velocity', 0),
                    activation_range=params.get('activation_range'),
                    additional_params={k: v for k, v in params.items() 
                                     if k not in ['power', 'antenna_gain', 'bandwidth', 'technique', 'activation_range']}
                )
            else:
                # Jammer attached to platform
                jammer = JammerConfig(
                    name=jammer_cfg['name'],
                    type=jammer_cfg['type'],
                    platform=jammer_cfg['platform'],
                    power=params['power'],
                    antenna_gain=params['antenna_gain'],
                    bandwidth=params.get('bandwidth', 1e6),
                    technique=params.get('technique', 'noise'),
                    activation_range=params.get('activation_range'),
                    additional_params={k: v for k, v in params.items() 
                                     if k not in ['power', 'antenna_gain', 'bandwidth', 'technique', 'activation_range']}
                )
            jammers.append(jammer)
        
        # Parse environment
        env_cfg = config_dict.get('environment', {})
        environment = EnvironmentConfig(
            temperature=env_cfg.get('temperature', 15.0),
            pressure=env_cfg.get('pressure', 1013.25),
            humidity=env_cfg.get('humidity', 50.0),
            weather=env_cfg.get('weather', 'clear'),
            terrain=env_cfg.get('terrain', 'flat'),
            rain_rate=env_cfg.get('rain_rate', 0.0),
            sea_state=env_cfg.get('sea_state'),
            wind_speed=env_cfg.get('wind_speed')
        )
        
        # Create scenario configuration
        return ScenarioConfig(
            name=scenario['name'],
            description=scenario['description'],
            duration=scenario['duration'],
            time_step=scenario['time_step'],
            radar=radar,
            targets=targets,
            jammers=jammers,
            environment=environment,
            clutter=config_dict.get('clutter'),
            output=config_dict.get('output')
        )
    
    def list_scenarios(self) -> List[str]:
        """List available scenario files"""
        scenarios = []
        for file in self.scenarios_dir.glob("*.yaml"):
            scenarios.append(file.stem)
        return sorted(scenarios)
    
    def validate_scenario(self, scenario: ScenarioConfig) -> List[str]:
        """
        Validate scenario configuration
        
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        # Check radar parameters
        if scenario.radar.max_range > 3e8 / (2 * scenario.radar.prf):
            warnings.append(f"Max range exceeds unambiguous range for PRF {scenario.radar.prf}")
        
        # Check target ranges
        for target in scenario.targets:
            if target.range > scenario.radar.max_range:
                warnings.append(f"Target {target.name} is beyond max radar range")
            
            # Calculate velocity magnitude if it's a list
            if isinstance(target.velocity, list):
                velocity_mag = np.linalg.norm(target.velocity)
            else:
                velocity_mag = abs(target.velocity)
            
            if velocity_mag > scenario.radar.n_doppler_bins * scenario.radar.velocity_resolution / 2:
                warnings.append(f"Target {target.name} velocity may cause Doppler ambiguity")
        
        # Check jammer parameters
        for jammer in scenario.jammers:
            if jammer.platform != 'standalone' and jammer.platform not in [t.name for t in scenario.targets]:
                if not jammer.platform.startswith('fighter') and not jammer.platform.startswith('ASM'):
                    warnings.append(f"Jammer {jammer.name} attached to unknown platform: {jammer.platform}")
        
        # Check simulation timing
        if scenario.time_step > 0.5:
            warnings.append("Time step > 0.5s may cause tracking issues")
        
        return warnings
    
    def save_scenario(self, scenario: ScenarioConfig, filename: str):
        """Save scenario configuration to YAML file"""
        
        if not filename.endswith('.yaml'):
            filename += '.yaml'
        
        filepath = self.scenarios_dir / filename
        
        # Convert to dictionary
        config_dict = {
            'scenario': {
                'name': scenario.name,
                'description': scenario.description,
                'duration': scenario.duration,
                'time_step': scenario.time_step
            },
            'radar': self._radar_to_dict(scenario.radar),
            'targets': [self._target_to_dict(t) for t in scenario.targets],
            'jammers': [self._jammer_to_dict(j) for j in scenario.jammers],
            'environment': self._environment_to_dict(scenario.environment)
        }
        
        if scenario.clutter:
            config_dict['clutter'] = scenario.clutter
        if scenario.output:
            config_dict['output'] = scenario.output
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved scenario to {filepath}")
    
    def _radar_to_dict(self, radar: RadarConfig) -> Dict:
        """Convert radar config to dictionary"""
        return {
            'type': radar.type,
            'parameters': {
                'frequency': radar.frequency,
                'power': radar.power,
                'antenna_gain': radar.antenna_gain,
                'pulse_width': radar.pulse_width,
                'prf': radar.prf,
                'bandwidth': radar.bandwidth,
                'noise_figure': radar.noise_figure,
                'losses': radar.losses
            },
            'processing': {
                'range_resolution': radar.range_resolution,
                'max_range': radar.max_range,
                'velocity_resolution': radar.velocity_resolution,
                'n_doppler_bins': radar.n_doppler_bins,
                'detection_threshold': radar.detection_threshold,
                'cfar_guard_cells': radar.cfar_guard_cells,
                'cfar_training_cells': radar.cfar_training_cells
            }
        }
    
    def _target_to_dict(self, target: TargetConfig) -> Dict:
        """Convert target config to dictionary"""
        return {
            'name': target.name,
            'type': target.type,
            'initial_position': {
                'range': target.range,
                'azimuth': target.azimuth,
                'elevation': target.elevation
            },
            'velocity': target.velocity,
            'rcs': target.rcs,
            'maneuver': target.maneuver
        }
    
    def _jammer_to_dict(self, jammer: JammerConfig) -> Dict:
        """Convert jammer config to dictionary"""
        result = {
            'name': jammer.name,
            'type': jammer.type,
            'platform': jammer.platform,
            'parameters': {
                'power': jammer.power,
                'antenna_gain': jammer.antenna_gain,
                'bandwidth': jammer.bandwidth,
                'technique': jammer.technique
            }
        }
        
        if jammer.range is not None:
            result['initial_position'] = {
                'range': jammer.range,
                'azimuth': jammer.azimuth,
                'elevation': jammer.elevation
            }
            
        if jammer.velocity is not None:
            result['velocity'] = jammer.velocity
            
        if jammer.activation_range is not None:
            result['parameters']['activation_range'] = jammer.activation_range
            
        result['parameters'].update(jammer.additional_params)
        
        return result
    
    def _environment_to_dict(self, env: EnvironmentConfig) -> Dict:
        """Convert environment config to dictionary"""
        return {
            'temperature': env.temperature,
            'pressure': env.pressure,
            'humidity': env.humidity,
            'weather': env.weather,
            'terrain': env.terrain,
            'rain_rate': env.rain_rate,
            'sea_state': env.sea_state,
            'wind_speed': env.wind_speed
        }


if __name__ == "__main__":
    # Test configuration loader
    loader = ConfigLoader()
    
    print("Available scenarios:")
    for scenario in loader.list_scenarios():
        print(f"  - {scenario}")
    
    # Try loading a scenario
    try:
        config = loader.load_scenario("air_defense")
        print(f"\nLoaded scenario: {config.name}")
        print(f"Description: {config.description}")
        print(f"Duration: {config.duration}s")
        print(f"Targets: {len(config.targets)}")
        print(f"Jammers: {len(config.jammers)}")
        
        # Validate
        warnings = loader.validate_scenario(config)
        if warnings:
            print("\nValidation warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("\nScenario validation passed!")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")