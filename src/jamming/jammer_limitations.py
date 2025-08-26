"""
High-Fidelity Jammer Limitations and Array Operations Model
Realistic constraints on jamming systems including hardware, thermal, and physics limitations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import scipy.signal as signal


class JammerType(Enum):
    """Types of jamming platforms with different capabilities"""
    STANDOFF = "standoff"  # EA-18G Growler, high power
    POD = "pod"  # ALQ-99 pod, medium power
    SELF_PROTECT = "self_protect"  # Internal suite, low power
    EXPENDABLE = "expendable"  # MALD-J decoy, very limited


@dataclass
class JammerHardwareSpecs:
    """Realistic hardware specifications for jammers"""
    # Power constraints
    peak_power: float  # Watts
    average_power: float  # Watts (thermal limited)
    duty_cycle_max: float  # Maximum duty cycle (0-1)
    
    # Bandwidth constraints
    instantaneous_bandwidth: float  # Hz
    tunable_bandwidth: float  # Hz (total frequency coverage)
    
    # DRFM constraints
    memory_depth: int  # samples
    bit_resolution: int  # bits (affects fidelity)
    sampling_rate: float  # Hz
    processing_delay: float  # seconds (DRFM loop delay)
    
    # Antenna constraints
    antenna_gain: float  # dBi
    beamwidth_azimuth: float  # degrees
    beamwidth_elevation: float  # degrees
    scan_rate_max: float  # degrees/second
    simultaneous_beams: int  # number of beams
    
    # Thermal constraints
    max_continuous_time: float  # seconds before thermal shutdown
    cooldown_time: float  # seconds required between operations
    thermal_capacity: float  # Joules
    
    # Digital constraints
    false_targets_max: int  # Maximum simultaneous false targets
    techniques_simultaneous: int  # Concurrent techniques


class JammerArrayModel:
    """
    High-fidelity model of jammer antenna array operations
    Including beam forming, steering, and pattern limitations
    """
    
    def __init__(self, jammer_type: JammerType):
        self.jammer_type = jammer_type
        self.specs = self._get_hardware_specs(jammer_type)
        
        # Array configuration
        self.array_config = self._configure_array(jammer_type)
        
        # Thermal state
        self.thermal_energy = 0.0  # Current thermal load
        self.operational_time = 0.0
        self.is_cooling = False
        self.cooldown_remaining = 0.0
        
        # Resource allocation
        self.power_allocated = {}  # Power per technique/target
        self.beams_allocated = {}  # Beam assignments
        
    def _get_hardware_specs(self, jammer_type: JammerType) -> JammerHardwareSpecs:
        """Get realistic hardware specs based on jammer type"""
        
        specs_db = {
            JammerType.STANDOFF: JammerHardwareSpecs(
                peak_power=10000,  # 10 kW peak
                average_power=1000,  # 1 kW average
                duty_cycle_max=0.2,  # 20% duty cycle
                instantaneous_bandwidth=500e6,  # 500 MHz
                tunable_bandwidth=18e9,  # 0.5-18 GHz
                memory_depth=10000000,  # 10M samples
                bit_resolution=12,  # 12-bit ADC/DAC
                sampling_rate=2e9,  # 2 GS/s
                processing_delay=0.5e-6,  # 500 ns
                antenna_gain=20,  # 20 dBi
                beamwidth_azimuth=15,  # degrees
                beamwidth_elevation=15,
                scan_rate_max=60,  # deg/s
                simultaneous_beams=4,  # 4 beams
                max_continuous_time=30,  # 30 seconds
                cooldown_time=10,  # 10 seconds
                thermal_capacity=30000,  # 30 kJ
                false_targets_max=20,
                techniques_simultaneous=3
            ),
            
            JammerType.POD: JammerHardwareSpecs(
                peak_power=5000,  # 5 kW
                average_power=500,  # 500 W
                duty_cycle_max=0.15,
                instantaneous_bandwidth=200e6,  # 200 MHz
                tunable_bandwidth=10e9,  # 2-12 GHz
                memory_depth=5000000,
                bit_resolution=10,
                sampling_rate=1e9,
                processing_delay=1e-6,
                antenna_gain=15,
                beamwidth_azimuth=20,
                beamwidth_elevation=20,
                scan_rate_max=45,
                simultaneous_beams=2,
                max_continuous_time=20,
                cooldown_time=15,
                thermal_capacity=15000,
                false_targets_max=10,
                techniques_simultaneous=2
            ),
            
            JammerType.SELF_PROTECT: JammerHardwareSpecs(
                peak_power=1000,  # 1 kW
                average_power=100,  # 100 W
                duty_cycle_max=0.1,
                instantaneous_bandwidth=100e6,
                tunable_bandwidth=8e9,
                memory_depth=1000000,
                bit_resolution=8,
                sampling_rate=500e6,
                processing_delay=2e-6,
                antenna_gain=10,
                beamwidth_azimuth=30,
                beamwidth_elevation=30,
                scan_rate_max=30,
                simultaneous_beams=1,
                max_continuous_time=10,
                cooldown_time=20,
                thermal_capacity=5000,
                false_targets_max=5,
                techniques_simultaneous=1
            ),
            
            JammerType.EXPENDABLE: JammerHardwareSpecs(
                peak_power=100,  # 100 W
                average_power=20,  # 20 W
                duty_cycle_max=0.3,  # Can run higher duty cycle
                instantaneous_bandwidth=50e6,
                tunable_bandwidth=4e9,
                memory_depth=100000,
                bit_resolution=6,
                sampling_rate=200e6,
                processing_delay=5e-6,
                antenna_gain=6,
                beamwidth_azimuth=60,
                beamwidth_elevation=60,
                scan_rate_max=20,
                simultaneous_beams=1,
                max_continuous_time=300,  # Can run until fuel exhausted
                cooldown_time=0,  # No cooldown (expendable)
                thermal_capacity=1000,
                false_targets_max=3,
                techniques_simultaneous=1
            )
        }
        
        return specs_db[jammer_type]
    
    def _configure_array(self, jammer_type: JammerType) -> Dict:
        """Configure antenna array based on jammer type"""
        
        if jammer_type == JammerType.STANDOFF:
            # Phased array with multiple panels
            return {
                'type': 'phased_array',
                'elements': 64,  # 8x8 array
                'spacing': 0.5,  # wavelengths
                'panels': 2,  # Forward and aft
                'polarization': 'dual',
                'steering_range_az': [-60, 60],  # degrees
                'steering_range_el': [-30, 30]
            }
        
        elif jammer_type == JammerType.POD:
            # Linear array in pod
            return {
                'type': 'linear_array',
                'elements': 16,
                'spacing': 0.5,
                'panels': 1,
                'polarization': 'vertical',
                'steering_range_az': [-45, 45],
                'steering_range_el': [-20, 20]
            }
        
        elif jammer_type == JammerType.SELF_PROTECT:
            # Small conformal array
            return {
                'type': 'conformal',
                'elements': 4,
                'spacing': 0.7,
                'panels': 1,
                'polarization': 'vertical',
                'steering_range_az': [-30, 30],
                'steering_range_el': [-15, 15]
            }
        
        else:  # EXPENDABLE
            # Simple horn antenna
            return {
                'type': 'horn',
                'elements': 1,
                'spacing': 0,
                'panels': 1,
                'polarization': 'circular',
                'steering_range_az': [0, 0],  # No steering
                'steering_range_el': [0, 0]
            }
    
    def calculate_antenna_pattern(self, azimuth: float, elevation: float,
                                 frequency: float) -> float:
        """
        Calculate antenna gain at given angles
        
        Args:
            azimuth: Azimuth angle in degrees
            elevation: Elevation angle in degrees
            frequency: Operating frequency in Hz
            
        Returns:
            Gain in dBi
        """
        # Check if within steering limits
        az_range = self.array_config['steering_range_az']
        el_range = self.array_config['steering_range_el']
        
        if azimuth < az_range[0] or azimuth > az_range[1]:
            return -30  # Very low gain outside steering range
        if elevation < el_range[0] or elevation > el_range[1]:
            return -30
        
        # Calculate array factor
        if self.array_config['type'] == 'phased_array':
            # 2D array pattern
            wavelength = 3e8 / frequency
            d = self.array_config['spacing'] * wavelength
            N = int(np.sqrt(self.array_config['elements']))
            
            # Simplified array factor
            theta_az = np.radians(azimuth)
            theta_el = np.radians(elevation)
            
            # Array factor in azimuth
            psi_az = 2 * np.pi * d * np.sin(theta_az) / wavelength
            AF_az = np.sin(N * psi_az / 2) / (N * np.sin(psi_az / 2)) if psi_az != 0 else 1
            
            # Array factor in elevation
            psi_el = 2 * np.pi * d * np.sin(theta_el) / wavelength
            AF_el = np.sin(N * psi_el / 2) / (N * np.sin(psi_el / 2)) if psi_el != 0 else 1
            
            # Combined pattern
            pattern = 20 * np.log10(abs(AF_az * AF_el) + 1e-10)
            
        elif self.array_config['type'] == 'linear_array':
            # 1D array pattern
            wavelength = 3e8 / frequency
            d = self.array_config['spacing'] * wavelength
            N = self.array_config['elements']
            
            theta = np.radians(azimuth)
            psi = 2 * np.pi * d * np.sin(theta) / wavelength
            AF = np.sin(N * psi / 2) / (N * np.sin(psi / 2)) if psi != 0 else 1
            
            pattern = 20 * np.log10(abs(AF) + 1e-10)
            
        else:
            # Simple beam pattern
            theta_3db_az = self.specs.beamwidth_azimuth
            theta_3db_el = self.specs.beamwidth_elevation
            
            # Gaussian beam approximation
            pattern = -12 * ((azimuth/theta_3db_az)**2 + (elevation/theta_3db_el)**2)
        
        # Add element gain
        total_gain = self.specs.antenna_gain + pattern
        
        # Account for scan loss
        scan_loss = 3 * np.cos(np.radians(azimuth)) * np.cos(np.radians(elevation))
        total_gain += scan_loss
        
        return total_gain
    
    def calculate_erp(self, power: float, azimuth: float, 
                     elevation: float, frequency: float) -> float:
        """
        Calculate Effective Radiated Power
        
        Args:
            power: Transmit power in Watts
            azimuth: Target azimuth
            elevation: Target elevation
            frequency: Operating frequency
            
        Returns:
            ERP in Watts
        """
        gain_dbi = self.calculate_antenna_pattern(azimuth, elevation, frequency)
        gain_linear = 10 ** (gain_dbi / 10)
        
        return power * gain_linear
    
    def check_thermal_constraints(self, power: float, duration: float) -> bool:
        """
        Check if operation is thermally feasible
        
        Args:
            power: Power level in Watts
            duration: Duration in seconds
            
        Returns:
            True if operation is feasible
        """
        energy_required = power * duration
        
        # Check thermal capacity
        if self.thermal_energy + energy_required > self.specs.thermal_capacity:
            return False
        
        # Check continuous operation time
        if self.operational_time + duration > self.specs.max_continuous_time:
            return False
        
        # Check duty cycle
        average_power = (self.thermal_energy + energy_required) / (self.operational_time + duration)
        if average_power > self.specs.average_power:
            return False
        
        return True
    
    def update_thermal_state(self, power: float, duration: float):
        """Update thermal state after operation"""
        energy = power * duration
        self.thermal_energy += energy
        self.operational_time += duration
        
        # Check if cooling needed
        if self.thermal_energy > 0.8 * self.specs.thermal_capacity:
            self.is_cooling = True
            self.cooldown_remaining = self.specs.cooldown_time
    
    def cool_down(self, time_delta: float):
        """Process cooling over time"""
        if self.is_cooling:
            self.cooldown_remaining -= time_delta
            
            # Dissipate heat
            cooling_rate = self.specs.thermal_capacity / self.specs.cooldown_time
            self.thermal_energy = max(0, self.thermal_energy - cooling_rate * time_delta)
            
            if self.cooldown_remaining <= 0:
                self.is_cooling = False
                self.operational_time = 0
    
    def calculate_jamming_effectiveness(self, target_range: float,
                                       target_rcs: float,
                                       radar_power: float,
                                       radar_gain: float,
                                       frequency: float,
                                       azimuth: float,
                                       elevation: float) -> Dict:
        """
        Calculate J/S ratio and effectiveness
        
        Returns:
            Dictionary with J/S ratio and limitations
        """
        # Check if we can engage
        if self.is_cooling:
            return {
                'js_ratio_db': -np.inf,
                'effective': False,
                'limitation': 'thermal_cooling',
                'time_remaining': self.cooldown_remaining
            }
        
        # Check angle limits
        az_range = self.array_config['steering_range_az']
        el_range = self.array_config['steering_range_el']
        
        if azimuth < az_range[0] or azimuth > az_range[1]:
            return {
                'js_ratio_db': -np.inf,
                'effective': False,
                'limitation': 'azimuth_limit',
                'angle': azimuth
            }
        
        if elevation < el_range[0] or elevation > el_range[1]:
            return {
                'js_ratio_db': -np.inf,
                'effective': False,
                'limitation': 'elevation_limit',
                'angle': elevation
            }
        
        # Calculate jamming power at radar
        jammer_power = self.specs.peak_power * self.specs.duty_cycle_max
        jammer_erp = self.calculate_erp(jammer_power, azimuth, elevation, frequency)
        
        # One-way path loss for jamming
        wavelength = 3e8 / frequency
        path_loss_jam = (4 * np.pi * target_range / wavelength) ** 2
        
        jamming_power_at_radar = jammer_erp / path_loss_jam
        
        # Calculate signal power (radar return)
        radar_erp = radar_power * (10 ** (radar_gain / 10))
        path_loss_radar = (4 * np.pi * target_range / wavelength) ** 4  # Two-way
        
        signal_power = (radar_erp * target_rcs * wavelength**2) / ((4*np.pi)**3 * path_loss_radar)
        
        # J/S ratio
        js_ratio = jamming_power_at_radar / signal_power if signal_power > 0 else np.inf
        js_ratio_db = 10 * np.log10(js_ratio) if js_ratio > 0 else -np.inf
        
        # Determine effectiveness
        effective = js_ratio_db > 10  # Need 10 dB for effective jamming
        
        # Check bandwidth limitations
        if frequency < (10e9 - self.specs.tunable_bandwidth/2) or \
           frequency > (10e9 + self.specs.tunable_bandwidth/2):
            effective = False
            limitation = 'frequency_coverage'
        else:
            limitation = None
        
        return {
            'js_ratio_db': js_ratio_db,
            'effective': effective,
            'limitation': limitation,
            'jammer_erp_w': jammer_erp,
            'jamming_power_at_radar_w': jamming_power_at_radar,
            'signal_power_w': signal_power
        }
    
    def allocate_resources(self, targets: List[Dict]) -> Dict:
        """
        Allocate jammer resources to multiple targets
        
        Args:
            targets: List of target dictionaries with range, angle, priority
            
        Returns:
            Resource allocation plan
        """
        allocation = {
            'targets_engaged': [],
            'targets_dropped': [],
            'power_distribution': {},
            'beam_assignment': {},
            'techniques': {}
        }
        
        # Sort by priority
        sorted_targets = sorted(targets, key=lambda x: x['priority'], reverse=True)
        
        # Allocate beams
        beams_used = 0
        total_power_allocated = 0
        
        for target in sorted_targets:
            if beams_used >= self.specs.simultaneous_beams:
                allocation['targets_dropped'].append(target)
                continue
            
            # Check if we have power budget
            power_required = self.specs.peak_power / self.specs.simultaneous_beams
            if total_power_allocated + power_required > self.specs.peak_power:
                allocation['targets_dropped'].append(target)
                continue
            
            # Check angular coverage
            if target['azimuth'] < self.array_config['steering_range_az'][0] or \
               target['azimuth'] > self.array_config['steering_range_az'][1]:
                allocation['targets_dropped'].append(target)
                continue
            
            # Allocate
            allocation['targets_engaged'].append(target)
            allocation['power_distribution'][target['id']] = power_required
            allocation['beam_assignment'][target['id']] = beams_used
            
            beams_used += 1
            total_power_allocated += power_required
        
        return allocation
    
    def calculate_drfm_fidelity(self, bandwidth: float, 
                               memory_used: int) -> float:
        """
        Calculate DRFM signal fidelity based on limitations
        
        Returns:
            Fidelity factor (0-1)
        """
        # Bandwidth limitation
        if bandwidth > self.specs.instantaneous_bandwidth:
            bandwidth_fidelity = self.specs.instantaneous_bandwidth / bandwidth
        else:
            bandwidth_fidelity = 1.0
        
        # Bit resolution impact
        bit_fidelity = (2 ** self.specs.bit_resolution - 1) / (2 ** 12 - 1)  # Normalized to 12-bit
        
        # Memory limitation
        if memory_used > self.specs.memory_depth:
            memory_fidelity = self.specs.memory_depth / memory_used
        else:
            memory_fidelity = 1.0
        
        # Sampling rate limitation (Nyquist)
        required_sampling = 2 * bandwidth
        if required_sampling > self.specs.sampling_rate:
            sampling_fidelity = self.specs.sampling_rate / required_sampling
        else:
            sampling_fidelity = 1.0
        
        # Combined fidelity
        total_fidelity = (bandwidth_fidelity * bit_fidelity * 
                         memory_fidelity * sampling_fidelity)
        
        return total_fidelity
    
    def calculate_false_target_quality(self, num_targets: int,
                                      target_complexity: str = 'medium') -> float:
        """
        Calculate quality degradation with multiple false targets
        
        Args:
            num_targets: Number of false targets
            target_complexity: 'simple', 'medium', 'complex'
            
        Returns:
            Quality factor (0-1)
        """
        # Memory per target
        memory_per_target = {
            'simple': 10000,  # Simple delay/doppler
            'medium': 50000,  # With modulation
            'complex': 200000  # Full kinematics
        }
        
        memory_required = num_targets * memory_per_target[target_complexity]
        
        # Check limits
        if num_targets > self.specs.false_targets_max:
            return 0.0  # Can't generate this many
        
        # Quality degradation
        memory_factor = min(1.0, self.specs.memory_depth / memory_required)
        processing_factor = min(1.0, self.specs.false_targets_max / num_targets)
        
        # Power sharing impact
        power_factor = min(1.0, 1.0 / np.sqrt(num_targets))
        
        return memory_factor * processing_factor * power_factor


def analyze_jammer_limitations():
    """Analyze limitations across different jammer types"""
    
    print("\n" + "="*70)
    print(" JAMMER LIMITATIONS ANALYSIS")
    print("="*70)
    
    jammers = {
        'EA-18G Growler': JammerArrayModel(JammerType.STANDOFF),
        'ALQ-99 Pod': JammerArrayModel(JammerType.POD),
        'F-35 ASQ-239': JammerArrayModel(JammerType.SELF_PROTECT),
        'MALD-J': JammerArrayModel(JammerType.EXPENDABLE)
    }
    
    for name, jammer in jammers.items():
        print(f"\n{name} ({jammer.jammer_type.value}):")
        print(f"  Peak Power: {jammer.specs.peak_power} W")
        print(f"  Average Power: {jammer.specs.average_power} W")
        print(f"  Duty Cycle: {jammer.specs.duty_cycle_max*100:.0f}%")
        print(f"  Bandwidth: {jammer.specs.instantaneous_bandwidth/1e6:.0f} MHz")
        print(f"  Beams: {jammer.specs.simultaneous_beams}")
        print(f"  Max False Targets: {jammer.specs.false_targets_max}")
        print(f"  Continuous Operation: {jammer.specs.max_continuous_time} seconds")
        print(f"  Cooldown Required: {jammer.specs.cooldown_time} seconds")
    
    # Test scenarios
    print("\n" + "="*70)
    print(" OPERATIONAL SCENARIOS")
    print("="*70)
    
    scenarios = [
        {'name': 'Close Range', 'range': 20000, 'angle': 0},
        {'name': 'Medium Range', 'range': 50000, 'angle': 30},
        {'name': 'Long Range', 'range': 100000, 'angle': 45},
        {'name': 'Off-Axis', 'range': 50000, 'angle': 70}
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']} (Range: {scenario['range']/1000:.0f}km, Angle: {scenario['angle']}°):")
        
        for name, jammer in jammers.items():
            result = jammer.calculate_jamming_effectiveness(
                target_range=scenario['range'],
                target_rcs=5.0,  # Fighter RCS
                radar_power=1e6,  # 1 MW radar
                radar_gain=40,  # 40 dB
                frequency=10e9,  # X-band
                azimuth=scenario['angle'],
                elevation=0
            )
            
            if result['effective']:
                print(f"  {name:15} J/S: {result['js_ratio_db']:.1f} dB ✓")
            else:
                print(f"  {name:15} INEFFECTIVE - {result['limitation']}")
    
    # False target quality analysis
    print("\n" + "="*70)
    print(" FALSE TARGET GENERATION LIMITS")
    print("="*70)
    
    target_counts = [5, 10, 20, 40, 80]
    
    for count in target_counts:
        print(f"\n{count} False Targets:")
        for name, jammer in jammers.items():
            quality = jammer.calculate_false_target_quality(count, 'medium')
            if quality > 0:
                print(f"  {name:15} Quality: {quality*100:.0f}%")
            else:
                print(f"  {name:15} CANNOT GENERATE")
    
    return jammers


if __name__ == "__main__":
    analyze_jammer_limitations()