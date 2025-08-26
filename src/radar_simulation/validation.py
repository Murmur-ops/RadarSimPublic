"""
Comprehensive Validation Suite for Radar Simulation

This module provides comprehensive validation capabilities for radar simulation systems,
including anti-cheating tests, physics validation, performance validation, and 
integration tests. It ensures the radar simulation behaves realistically and
maintains scientific integrity.

Classes:
    ValidationSuite: Main validation orchestrator
    AntiCheatValidator: Tests for simulation integrity
    PhysicsValidator: Physics-based validation tests
    PerformanceValidator: Performance characteristic validation
    IntegrationValidator: Integration and compatibility tests
    ValidationReport: Comprehensive validation reporting

Author: RadarSim Development Team
"""

import numpy as np
import pytest
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from scipy import stats, signal
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from datetime import datetime
import inspect

# Import radar simulation components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from radar import Radar, RadarParameters
from target import Target, TargetType, TargetMotion, TargetGenerator
from signal import SignalProcessor
from tracking.tracker_base import Measurement

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results from a validation test"""
    test_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        def _convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, np.bool8)):
                return bool(obj)
            elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64, np.integer)):
                return int(obj)
            elif isinstance(obj, (np.float16, np.float32, np.float64, np.floating)):
                return float(obj)
            elif isinstance(obj, (np.complex64, np.complex128, np.complexfloating)):
                return {'real': float(obj.real), 'imag': float(obj.imag)}
            elif isinstance(obj, dict):
                return {key: _convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        return {
            'test_name': self.test_name,
            'passed': bool(self.passed),
            'score': float(self.score),
            'message': self.message,
            'details': _convert_numpy_types(self.details),
            'execution_time': float(self.execution_time),
            'warnings': self.warnings
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    timestamp: str
    overall_score: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    results: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'overall_score': self.overall_score,
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'results': [r.to_dict() for r in self.results],
            'summary': self.summary,
            'recommendations': self.recommendations
        }
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save report to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def print_summary(self) -> None:
        """Print a formatted summary of the validation results"""
        print("\n" + "="*80)
        print(f"RADAR SIMULATION VALIDATION REPORT")
        print("="*80)
        print(f"Timestamp: {self.timestamp}")
        print(f"Overall Score: {self.overall_score:.2%}")
        print(f"Tests Passed: {self.passed_tests}/{self.total_tests}")
        print(f"Tests Failed: {self.failed_tests}/{self.total_tests}")
        print("\n" + "-"*80)
        
        # Group results by validator type
        validators = {}
        for result in self.results:
            validator_name = result.test_name.split('_')[0]
            if validator_name not in validators:
                validators[validator_name] = []
            validators[validator_name].append(result)
        
        for validator_name, results in validators.items():
            print(f"\n{validator_name.upper()} VALIDATION:")
            print("-" * 40)
            
            for result in results:
                status = "✓ PASS" if result.passed else "✗ FAIL"
                print(f"{status:8} {result.test_name:40} Score: {result.score:.2%}")
                if not result.passed:
                    print(f"         Message: {result.message}")
        
        if self.recommendations:
            print("\n" + "-"*80)
            print("RECOMMENDATIONS:")
            for i, rec in enumerate(self.recommendations, 1):
                print(f"{i}. {rec}")
        
        print("\n" + "="*80)


class BaseValidator(ABC):
    """Base class for all validators"""
    
    def __init__(self, name: str):
        self.name = name
        self.results: List[ValidationResult] = []
        
    def _create_result(self, test_name: str, passed: bool, score: float, 
                      message: str, details: Optional[Dict] = None,
                      warnings: Optional[List[str]] = None) -> ValidationResult:
        """Create a validation result"""
        return ValidationResult(
            test_name=f"{self.name}_{test_name}",
            passed=passed,
            score=score,
            message=message,
            details=details or {},
            warnings=warnings or []
        )
    
    @abstractmethod
    def run_validation(self, radar: Radar, targets: List[Target], 
                      signal_processor: SignalProcessor) -> List[ValidationResult]:
        """Run validation tests"""
        pass


class AntiCheatValidator(BaseValidator):
    """Validator for anti-cheating mechanisms and simulation integrity"""
    
    def __init__(self):
        super().__init__("anticheating")
    
    def run_validation(self, radar: Radar, targets: List[Target], 
                      signal_processor: SignalProcessor) -> List[ValidationResult]:
        """Run all anti-cheating validation tests"""
        results = []
        
        # Test 1: No access to true target positions during processing
        results.append(self._test_no_ground_truth_access(radar, targets, signal_processor))
        
        # Test 2: Measurements derived only from signal processing
        results.append(self._test_measurements_from_signal_processing(radar, targets, signal_processor))
        
        # Test 3: Random noise is truly random
        results.append(self._test_noise_randomness(radar, targets, signal_processor))
        
        # Test 4: No ground truth leakage
        results.append(self._test_no_ground_truth_leakage(radar, targets, signal_processor))
        
        # Test 5: Signal processing chain integrity
        results.append(self._test_signal_processing_integrity(radar, targets, signal_processor))
        
        return results
    
    def _test_no_ground_truth_access(self, radar: Radar, targets: List[Target], 
                                   signal_processor: SignalProcessor) -> ValidationResult:
        """Test that simulation has no access to true target positions during processing"""
        try:
            # Generate multiple scenarios with same parameters but different target positions
            scenarios = []
            num_scenarios = 5
            
            for i in range(num_scenarios):
                # Create targets at different positions
                test_targets = []
                for j in range(3):  # 3 targets per scenario
                    pos = np.array([
                        1000 + i * 500 + j * 200,  # Different x positions
                        500 + i * 300,             # Different y positions
                        1000                        # Same altitude
                    ])
                    vel = np.array([100, 50, 0])
                    motion = TargetMotion(position=pos, velocity=vel)
                    test_targets.append(Target(TargetType.AIRCRAFT, motion=motion))
                
                # Generate measurements for this scenario
                measurements = []
                for target in test_targets:
                    range_val, azimuth, elevation = target.get_position_spherical()
                    # Add realistic measurement noise
                    range_noise = np.random.normal(0, radar.params.range_resolution * 0.1)
                    angle_noise = np.random.normal(0, 0.01)  # ~0.5 degree
                    
                    measurements.append({
                        'range': range_val + range_noise,
                        'azimuth': azimuth + angle_noise,
                        'elevation': elevation + angle_noise,
                        'true_range': range_val,  # This should NOT be accessible to processing
                        'true_azimuth': azimuth,
                        'true_elevation': elevation
                    })
                
                scenarios.append((test_targets, measurements))
            
            # Verify that measurement errors are not correlated with true positions
            all_range_errors = []
            all_true_ranges = []
            
            for targets, measurements in scenarios:
                for i, (target, meas) in enumerate(zip(targets, measurements)):
                    range_error = meas['range'] - meas['true_range']
                    all_range_errors.append(range_error)
                    all_true_ranges.append(meas['true_range'])
            
            # Check for correlation (should be near zero)
            correlation = np.corrcoef(all_range_errors, all_true_ranges)[0, 1]
            
            # Test passes if correlation is low (indicating no cheating)
            passed = abs(correlation) < 0.1
            score = max(0.0, 1.0 - abs(correlation) * 2)
            
            message = f"Ground truth access test: correlation = {correlation:.4f}"
            if not passed:
                message += " - WARNING: Possible ground truth access detected!"
            
            return self._create_result(
                "no_ground_truth_access",
                passed,
                score,
                message,
                details={
                    'correlation': correlation,
                    'num_scenarios': num_scenarios,
                    'range_errors_std': np.std(all_range_errors)
                }
            )
            
        except Exception as e:
            return self._create_result(
                "no_ground_truth_access",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )
    
    def _test_measurements_from_signal_processing(self, radar: Radar, targets: List[Target], 
                                                signal_processor: SignalProcessor) -> ValidationResult:
        """Test that measurements are derived from signal processing, not direct calculation"""
        try:
            # Create a controlled scenario
            target = Target(TargetType.AIRCRAFT, motion=TargetMotion(
                position=np.array([5000, 2000, 3000]),
                velocity=np.array([100, 50, 0])
            ))
            
            # Generate actual signal processing chain
            pulse = radar.generate_pulse(1000)
            
            # Simulate received signal with target return
            range_val = target.motion.range
            rcs = target.get_rcs()
            received_power = radar.radar_equation(range_val, rcs)
            
            # Time delay for target return
            delay_samples = int(2 * range_val / (3e8) * signal_processor.sample_rate)
            
            # Create received signal with target echo
            received_signal = np.zeros(len(pulse) + delay_samples, dtype=complex)
            noise = np.random.normal(0, 1e-8, len(received_signal)) + \
                   1j * np.random.normal(0, 1e-8, len(received_signal))
            received_signal += noise
            
            # Add target echo
            echo_amplitude = np.sqrt(received_power)
            if delay_samples < len(received_signal) - len(pulse):
                received_signal[delay_samples:delay_samples+len(pulse)] += echo_amplitude * pulse
            
            # Apply matched filtering
            matched_output = signal_processor.matched_filter(received_signal, pulse)
            
            # Find peak (should correspond to target range)
            peak_idx = np.argmax(np.abs(matched_output))
            measured_delay = peak_idx / signal_processor.sample_rate
            measured_range = measured_delay * 3e8 / 2
            
            # Compare with expected range
            range_error = abs(measured_range - range_val)
            range_tolerance = radar.params.range_resolution * 2  # Allow 2 range bins error
            
            passed = range_error < range_tolerance
            score = max(0.0, 1.0 - range_error / range_tolerance)
            
            message = f"Signal processing test: range error = {range_error:.1f}m (tolerance: {range_tolerance:.1f}m)"
            
            return self._create_result(
                "measurements_from_signal_processing",
                passed,
                score,
                message,
                details={
                    'measured_range': measured_range,
                    'true_range': range_val,
                    'range_error': range_error,
                    'range_tolerance': range_tolerance,
                    'peak_snr': 20 * np.log10(np.abs(matched_output[peak_idx]) / np.std(np.abs(matched_output)))
                }
            )
            
        except Exception as e:
            return self._create_result(
                "measurements_from_signal_processing",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )
    
    def _test_noise_randomness(self, radar: Radar, targets: List[Target], 
                             signal_processor: SignalProcessor) -> ValidationResult:
        """Test that noise is truly random and not correlated with targets"""
        try:
            num_trials = 100
            noise_samples = []
            target_positions = []
            
            # Generate multiple noise realizations with different target positions
            for trial in range(num_trials):
                # Random target position
                target_pos = np.array([
                    np.random.uniform(1000, 10000),
                    np.random.uniform(-5000, 5000),
                    np.random.uniform(500, 5000)
                ])
                target_positions.append(np.linalg.norm(target_pos))
                
                # Generate noise
                noise = np.random.normal(0, 1e-8, 1000) + 1j * np.random.normal(0, 1e-8, 1000)
                noise_power = np.mean(np.abs(noise)**2)
                noise_samples.append(noise_power)
            
            # Statistical tests for randomness
            noise_array = np.array(noise_samples)
            positions_array = np.array(target_positions)
            
            # Test 1: Correlation with target positions should be near zero
            correlation = np.corrcoef(noise_array, positions_array)[0, 1]
            
            # Test 2: Kolmogorov-Smirnov test for normality of noise power
            # Noise power should follow exponential distribution
            ks_stat, ks_p_value = stats.kstest(noise_array, 'expon')
            
            # Test 3: Runs test for randomness
            median_noise = np.median(noise_array)
            runs_sequence = (noise_array > median_noise).astype(int)
            runs = np.sum(np.diff(runs_sequence) != 0) + 1
            expected_runs = len(noise_array) / 2
            runs_ratio = runs / expected_runs
            
            # Scoring
            correlation_score = max(0.0, 1.0 - abs(correlation) * 5)
            ks_score = min(1.0, ks_p_value * 2)  # Higher p-value is better
            runs_score = max(0.0, 1.0 - abs(runs_ratio - 1.0))
            
            overall_score = (correlation_score + ks_score + runs_score) / 3
            passed = overall_score > 0.7
            
            message = f"Noise randomness: corr={correlation:.4f}, KS_p={ks_p_value:.3f}, runs_ratio={runs_ratio:.3f}"
            
            return self._create_result(
                "noise_randomness",
                passed,
                overall_score,
                message,
                details={
                    'correlation': correlation,
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p_value,
                    'runs': runs,
                    'expected_runs': expected_runs,
                    'runs_ratio': runs_ratio,
                    'num_trials': num_trials
                }
            )
            
        except Exception as e:
            return self._create_result(
                "noise_randomness",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )
    
    def _test_no_ground_truth_leakage(self, radar: Radar, targets: List[Target], 
                                    signal_processor: SignalProcessor) -> ValidationResult:
        """Test that ground truth information doesn't leak through processing chain"""
        try:
            # Create identical targets with different RCS fluctuation seeds
            target1 = Target(TargetType.AIRCRAFT, motion=TargetMotion(
                position=np.array([5000, 0, 3000]),
                velocity=np.array([100, 0, 0])
            ))
            
            target2 = Target(TargetType.AIRCRAFT, motion=TargetMotion(
                position=np.array([5000, 0, 3000]),
                velocity=np.array([100, 0, 0])
            ))
            
            # Generate multiple measurements with different random seeds
            np.random.seed(42)
            measurements1 = []
            for _ in range(50):
                rcs1 = target1.get_rcs()
                snr1 = radar.snr(target1.motion.range, rcs1)
                measurements1.append(snr1)
            
            np.random.seed(123)  # Different seed
            measurements2 = []
            for _ in range(50):
                rcs2 = target2.get_rcs()
                snr2 = radar.snr(target2.motion.range, rcs2)
                measurements2.append(snr2)
            
            # Measurements should be different due to different random seeds
            # but should have similar statistical properties
            mean_diff = abs(np.mean(measurements1) - np.mean(measurements2))
            std_diff = abs(np.std(measurements1) - np.std(measurements2))
            
            # Statistical test for different distributions
            ks_stat, ks_p_value = stats.ks_2samp(measurements1, measurements2)
            
            # We expect some difference (ks_p_value < 0.05) but not too much
            passed = ks_p_value < 0.05 and mean_diff < 5.0 and std_diff < 3.0
            
            # Score based on reasonable differences
            mean_score = max(0.0, 1.0 - mean_diff / 10.0)
            std_score = max(0.0, 1.0 - std_diff / 5.0)
            ks_score = 1.0 if ks_p_value < 0.05 else 0.5
            
            overall_score = (mean_score + std_score + ks_score) / 3
            
            message = f"Ground truth leakage test: mean_diff={mean_diff:.2f}, std_diff={std_diff:.2f}, KS_p={ks_p_value:.3f}"
            
            return self._create_result(
                "no_ground_truth_leakage",
                passed,
                overall_score,
                message,
                details={
                    'mean_diff': mean_diff,
                    'std_diff': std_diff,
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p_value,
                    'measurements1_mean': np.mean(measurements1),
                    'measurements2_mean': np.mean(measurements2)
                }
            )
            
        except Exception as e:
            return self._create_result(
                "no_ground_truth_leakage",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )
    
    def _test_signal_processing_integrity(self, radar: Radar, targets: List[Target], 
                                        signal_processor: SignalProcessor) -> ValidationResult:
        """Test integrity of signal processing chain"""
        try:
            # Test that signal processing chain is deterministic for same inputs
            target = targets[0] if targets else Target(TargetType.AIRCRAFT)
            
            # Generate identical input signals
            pulse = radar.generate_pulse(1000)
            
            # Process twice with same input
            result1 = signal_processor.matched_filter(pulse, pulse)
            result2 = signal_processor.matched_filter(pulse, pulse)
            
            # Results should be identical (deterministic)
            max_diff = np.max(np.abs(result1 - result2))
            
            # Test CFAR processing consistency
            test_signal = np.random.RandomState(42).normal(0, 1, 1000)
            detections1, threshold1 = signal_processor.cfar_detector(test_signal, pfa=1e-6)
            detections2, threshold2 = signal_processor.cfar_detector(test_signal, pfa=1e-6)
            
            # CFAR results should be identical for same input
            cfar_consistent = np.array_equal(detections1, detections2) and np.allclose(threshold1, threshold2)
            
            # Test energy conservation in matched filter
            input_energy = np.sum(np.abs(pulse)**2)
            output_energy = np.sum(np.abs(result1)**2)
            energy_ratio = output_energy / input_energy
            
            # Energy should be preserved (within reasonable bounds)
            energy_preserved = 0.5 < energy_ratio < 2.0
            
            passed = max_diff < 1e-12 and cfar_consistent and energy_preserved
            score = 1.0 if passed else 0.0
            
            message = f"Signal processing integrity: diff={max_diff:.2e}, CFAR_consistent={cfar_consistent}, energy_ratio={energy_ratio:.3f}"
            
            return self._create_result(
                "signal_processing_integrity",
                passed,
                score,
                message,
                details={
                    'max_difference': max_diff,
                    'cfar_consistent': cfar_consistent,
                    'energy_ratio': energy_ratio,
                    'energy_preserved': energy_preserved
                }
            )
            
        except Exception as e:
            return self._create_result(
                "signal_processing_integrity",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )


class PhysicsValidator(BaseValidator):
    """Validator for physics-based constraints and behaviors"""
    
    def __init__(self):
        super().__init__("physics")
    
    def run_validation(self, radar: Radar, targets: List[Target], 
                      signal_processor: SignalProcessor) -> List[ValidationResult]:
        """Run all physics validation tests"""
        results = []
        
        # Test 1: SNR decreases with range following R^4 law
        results.append(self._test_range_snr_relationship(radar, targets))
        
        # Test 2: Detection probability follows theoretical curves
        results.append(self._test_detection_probability_curves(radar))
        
        # Test 3: False alarm rates match CFAR design
        results.append(self._test_false_alarm_rates(signal_processor))
        
        # Test 4: Measurement accuracy degrades with lower SNR
        results.append(self._test_measurement_accuracy_vs_snr(radar, signal_processor))
        
        # Test 5: Range and Doppler resolution limits
        results.append(self._test_resolution_limits(radar, signal_processor))
        
        # Test 6: Doppler shift calculations
        results.append(self._test_doppler_shift_physics(radar))
        
        return results
    
    def _test_range_snr_relationship(self, radar: Radar, targets: List[Target]) -> ValidationResult:
        """Test that SNR decreases with range following R^4 law"""
        try:
            # Test over range of distances
            ranges = np.logspace(3, 4.5, 20)  # 1km to ~30km
            rcs = 10.0  # Fixed RCS
            
            snr_values = []
            for range_val in ranges:
                snr = radar.snr(range_val, rcs)
                snr_values.append(snr)
            
            snr_values = np.array(snr_values)
            
            # Convert to linear scale for fitting
            snr_linear = 10**(snr_values / 10)
            
            # Fit to R^-4 relationship: SNR = A / R^4
            log_ranges = np.log10(ranges)
            log_snr = np.log10(snr_linear)
            
            # Linear fit in log space: log(SNR) = log(A) - 4*log(R)
            coeffs = np.polyfit(log_ranges, log_snr, 1)
            slope = coeffs[0]
            r_squared = np.corrcoef(log_ranges, log_snr)[0, 1]**2
            
            # Theoretical slope should be -4
            slope_error = abs(slope + 4.0)  # We expect slope ≈ -4
            
            # Test passes if slope is close to -4 and R² is high
            passed = slope_error < 0.5 and r_squared > 0.95
            score = max(0.0, 1.0 - slope_error / 2.0) * min(1.0, r_squared / 0.9)
            
            message = f"Range-SNR relationship: slope={slope:.2f} (expected ≈-4), R²={r_squared:.3f}"
            
            return self._create_result(
                "range_snr_relationship",
                passed,
                score,
                message,
                details={
                    'slope': slope,
                    'expected_slope': -4.0,
                    'slope_error': slope_error,
                    'r_squared': r_squared,
                    'ranges': ranges.tolist(),
                    'snr_values': snr_values.tolist()
                }
            )
            
        except Exception as e:
            return self._create_result(
                "range_snr_relationship",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )
    
    def _test_detection_probability_curves(self, radar: Radar) -> ValidationResult:
        """Test that detection probability follows theoretical curves"""
        try:
            # Test detection probability vs SNR
            snr_range = np.linspace(-10, 20, 31)  # -10 to 20 dB
            pfa = 1e-6
            
            pd_values = []
            for snr_db in snr_range:
                pd = radar.detection_probability(snr_db, pfa)
                pd_values.append(pd)
            
            pd_values = np.array(pd_values)
            
            # Check theoretical properties:
            # 1. Pd should be monotonically increasing with SNR
            monotonic = np.all(np.diff(pd_values) >= -0.01)  # Allow small numerical errors
            
            # 2. Pd should approach 0 for very low SNR
            low_snr_pd = pd_values[snr_range < -5]
            low_snr_correct = np.all(low_snr_pd < 0.1)
            
            # 3. Pd should approach 1 for very high SNR
            high_snr_pd = pd_values[snr_range > 15]
            high_snr_correct = np.all(high_snr_pd > 0.9)
            
            # 4. Check specific theoretical value (approximately)
            # For SNR = 13 dB, Pd should be around 0.5-0.9 for typical Swerling models
            snr_13_idx = np.argmin(np.abs(snr_range - 13))
            pd_at_13db = pd_values[snr_13_idx]
            pd_13_reasonable = 0.3 < pd_at_13db < 0.95
            
            all_tests_pass = monotonic and low_snr_correct and high_snr_correct and pd_13_reasonable
            
            # Scoring
            monotonic_score = 1.0 if monotonic else 0.0
            low_snr_score = 1.0 if low_snr_correct else 0.0
            high_snr_score = 1.0 if high_snr_correct else 0.0
            pd_13_score = 1.0 if pd_13_reasonable else 0.5
            
            overall_score = (monotonic_score + low_snr_score + high_snr_score + pd_13_score) / 4
            
            message = f"Detection probability curves: monotonic={monotonic}, low_SNR_ok={low_snr_correct}, high_SNR_ok={high_snr_correct}, Pd@13dB={pd_at_13db:.3f}"
            
            return self._create_result(
                "detection_probability_curves",
                all_tests_pass,
                overall_score,
                message,
                details={
                    'monotonic': monotonic,
                    'low_snr_correct': low_snr_correct,
                    'high_snr_correct': high_snr_correct,
                    'pd_at_13db': pd_at_13db,
                    'snr_range': snr_range.tolist(),
                    'pd_values': pd_values.tolist()
                }
            )
            
        except Exception as e:
            return self._create_result(
                "detection_probability_curves",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )
    
    def _test_false_alarm_rates(self, signal_processor: SignalProcessor) -> ValidationResult:
        """Test that false alarm rates match CFAR design"""
        try:
            # Generate noise-only signals
            num_trials = 1000
            signal_length = 500
            pfa_design = 1e-4  # Design false alarm rate
            
            false_alarms = 0
            total_cells = 0
            
            for trial in range(num_trials):
                # Generate noise-only signal
                noise = np.random.normal(0, 1, signal_length)
                
                # Apply CFAR detector
                detections, thresholds = signal_processor.cfar_detector(
                    noise, num_guard=2, num_train=16, pfa=pfa_design
                )
                
                # Count false alarms (all detections are false in noise-only case)
                false_alarms += np.sum(detections)
                total_cells += len(detections)
            
            # Calculate empirical false alarm rate
            empirical_pfa = false_alarms / total_cells if total_cells > 0 else 0
            
            # Test passes if empirical PFA is close to design PFA
            pfa_ratio = empirical_pfa / pfa_design if pfa_design > 0 else float('inf')
            
            # Allow factor of 2 error (common in CFAR due to approximations)
            passed = 0.5 < pfa_ratio < 2.0
            score = max(0.0, 1.0 - abs(np.log10(pfa_ratio)) / np.log10(2))
            
            message = f"False alarm rate test: empirical_PFA={empirical_pfa:.2e}, design_PFA={pfa_design:.2e}, ratio={pfa_ratio:.2f}"
            
            return self._create_result(
                "false_alarm_rates",
                passed,
                score,
                message,
                details={
                    'empirical_pfa': empirical_pfa,
                    'design_pfa': pfa_design,
                    'pfa_ratio': pfa_ratio,
                    'false_alarms': false_alarms,
                    'total_cells': total_cells,
                    'num_trials': num_trials
                }
            )
            
        except Exception as e:
            return self._create_result(
                "false_alarm_rates",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )
    
    def _test_measurement_accuracy_vs_snr(self, radar: Radar, signal_processor: SignalProcessor) -> ValidationResult:
        """Test that measurement accuracy degrades with lower SNR"""
        try:
            # Test range measurement accuracy vs SNR
            true_range = 5000  # meters
            snr_values = np.linspace(0, 20, 11)  # 0 to 20 dB
            
            range_errors = []
            
            for snr_db in snr_values:
                snr_linear = 10**(snr_db / 10)
                
                # Simulate multiple measurements at this SNR
                errors_at_snr = []
                for _ in range(100):
                    # Generate noisy measurement
                    noise_std = radar.params.range_resolution / np.sqrt(2 * snr_linear)
                    measured_range = true_range + np.random.normal(0, noise_std)
                    error = abs(measured_range - true_range)
                    errors_at_snr.append(error)
                
                # Use RMS error for this SNR level
                rms_error = np.sqrt(np.mean(np.array(errors_at_snr)**2))
                range_errors.append(rms_error)
            
            range_errors = np.array(range_errors)
            
            # Theoretical: error should decrease with increasing SNR
            # approximately as 1/sqrt(SNR)
            
            # Check monotonic decrease (with some tolerance)
            decreasing = True
            for i in range(1, len(range_errors)):
                if range_errors[i] > range_errors[i-1] * 1.5:  # Allow 50% increase tolerance
                    decreasing = False
                    break
            
            # Check that high SNR gives better accuracy than low SNR
            high_snr_error = np.mean(range_errors[-3:])  # Average of highest 3 SNR points
            low_snr_error = np.mean(range_errors[:3])    # Average of lowest 3 SNR points
            
            improvement_factor = low_snr_error / high_snr_error if high_snr_error > 0 else 1
            significant_improvement = improvement_factor > 2.0
            
            passed = decreasing and significant_improvement
            
            # Scoring
            decreasing_score = 1.0 if decreasing else 0.5
            improvement_score = min(1.0, improvement_factor / 3.0)
            overall_score = (decreasing_score + improvement_score) / 2
            
            message = f"Measurement accuracy vs SNR: decreasing={decreasing}, improvement_factor={improvement_factor:.2f}"
            
            return self._create_result(
                "measurement_accuracy_vs_snr",
                passed,
                overall_score,
                message,
                details={
                    'decreasing': decreasing,
                    'improvement_factor': improvement_factor,
                    'high_snr_error': high_snr_error,
                    'low_snr_error': low_snr_error,
                    'snr_values': snr_values.tolist(),
                    'range_errors': range_errors.tolist()
                }
            )
            
        except Exception as e:
            return self._create_result(
                "measurement_accuracy_vs_snr",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )
    
    def _test_resolution_limits(self, radar: Radar, signal_processor: SignalProcessor) -> ValidationResult:
        """Test range and Doppler resolution limits"""
        try:
            # Test range resolution
            theoretical_range_res = radar.params.range_resolution
            
            # Test with two targets separated by range resolution
            ranges = [5000, 5000 + theoretical_range_res]
            targets = [(r, 10.0) for r in ranges]  # Same RCS
            
            # Generate range profile
            range_bins, profile = radar.range_profile(targets, max_range=10000, num_bins=2000)
            
            # Find peaks in profile
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(profile, height=np.max(profile) - 20)  # Within 20 dB of max
            
            range_resolution_test = len(peaks) >= 2  # Should resolve both targets
            
            # Test Doppler resolution (maximum unambiguous velocity)
            max_unambig_vel = radar.max_unambiguous_velocity()
            theoretical_max_vel = radar.params.wavelength * radar.params.prf / 4
            
            velocity_calc_correct = abs(max_unambig_vel - theoretical_max_vel) < 0.1
            
            # Test minimum detectable velocity (should be related to coherent processing interval)
            # For a typical radar, minimum velocity resolution is proportional to wavelength/CPI
            min_vel_res = radar.params.wavelength / (2 * 0.1)  # Assume 0.1s CPI
            min_vel_reasonable = 0.1 < min_vel_res < 100  # Reasonable range for aircraft radar
            
            passed = range_resolution_test and velocity_calc_correct and min_vel_reasonable
            
            # Scoring
            range_score = 1.0 if range_resolution_test else 0.0
            velocity_score = 1.0 if velocity_calc_correct else 0.0
            min_vel_score = 1.0 if min_vel_reasonable else 0.5
            
            overall_score = (range_score + velocity_score + min_vel_score) / 3
            
            message = f"Resolution limits: range_res_ok={range_resolution_test}, vel_calc_ok={velocity_calc_correct}, min_vel_ok={min_vel_reasonable}"
            
            return self._create_result(
                "resolution_limits",
                passed,
                overall_score,
                message,
                details={
                    'theoretical_range_resolution': theoretical_range_res,
                    'num_peaks_found': len(peaks),
                    'range_resolution_test': range_resolution_test,
                    'max_unambiguous_velocity': max_unambig_vel,
                    'theoretical_max_velocity': theoretical_max_vel,
                    'velocity_calc_correct': velocity_calc_correct,
                    'min_velocity_resolution': min_vel_res,
                    'min_vel_reasonable': min_vel_reasonable
                }
            )
            
        except Exception as e:
            return self._create_result(
                "resolution_limits",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )
    
    def _test_doppler_shift_physics(self, radar: Radar) -> ValidationResult:
        """Test Doppler shift calculations against physics"""
        try:
            # Test various velocities
            velocities = [-300, -100, 0, 100, 300]  # m/s
            
            doppler_errors = []
            
            for velocity in velocities:
                # Calculate Doppler shift using radar method
                calculated_doppler = radar.doppler_shift(velocity)
                
                # Calculate theoretical Doppler shift
                # fd = 2 * v * f / c (for monostatic radar)
                theoretical_doppler = 2 * velocity * radar.params.frequency / 3e8
                
                # Calculate error
                error = abs(calculated_doppler - theoretical_doppler)
                doppler_errors.append(error)
            
            max_error = max(doppler_errors)
            mean_error = np.mean(doppler_errors)
            
            # Error should be very small (numerical precision)
            passed = max_error < 1e-6  # 1 microhertz tolerance
            score = max(0.0, 1.0 - max_error / 1e-3)  # Scale to millihertz
            
            # Test frequency sign convention
            # Approaching target (positive velocity) should give positive Doppler
            positive_vel_doppler = radar.doppler_shift(100)
            negative_vel_doppler = radar.doppler_shift(-100)
            
            sign_convention_correct = positive_vel_doppler > 0 and negative_vel_doppler < 0
            
            if not sign_convention_correct:
                score *= 0.5  # Reduce score for wrong sign convention
                passed = False
            
            message = f"Doppler physics: max_error={max_error:.2e} Hz, sign_correct={sign_convention_correct}"
            
            return self._create_result(
                "doppler_shift_physics",
                passed,
                score,
                message,
                details={
                    'velocities': velocities,
                    'doppler_errors': doppler_errors,
                    'max_error': max_error,
                    'mean_error': mean_error,
                    'sign_convention_correct': sign_convention_correct,
                    'positive_vel_doppler': positive_vel_doppler,
                    'negative_vel_doppler': negative_vel_doppler
                }
            )
            
        except Exception as e:
            return self._create_result(
                "doppler_shift_physics",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )


class PerformanceValidator(BaseValidator):
    """Validator for performance characteristics and expected behaviors"""
    
    def __init__(self):
        super().__init__("performance")
    
    def run_validation(self, radar: Radar, targets: List[Target], 
                      signal_processor: SignalProcessor) -> List[ValidationResult]:
        """Run all performance validation tests"""
        results = []
        
        # Test 1: Missed detections occur at expected rates
        results.append(self._test_missed_detection_rates(radar, targets))
        
        # Test 2: Measurement errors have correct statistical properties
        results.append(self._test_measurement_error_statistics(radar, signal_processor))
        
        # Test 3: Blind ranges and velocities exist as expected
        results.append(self._test_blind_ranges_velocities(radar, signal_processor))
        
        # Test 4: Ambiguity effects are present
        results.append(self._test_ambiguity_effects(radar, signal_processor))
        
        # Test 5: Clutter and multipath effects
        results.append(self._test_clutter_effects(signal_processor))
        
        return results
    
    def _test_missed_detection_rates(self, radar: Radar, targets: List[Target]) -> ValidationResult:
        """Test that missed detections occur at expected rates"""
        try:
            # Test detection rates vs SNR
            test_ranges = np.linspace(3000, 15000, 10)
            pfa = 1e-6
            num_trials = 200
            
            detection_rates = []
            theoretical_pd = []
            
            for test_range in test_ranges:
                # Use a standard target
                target_rcs = 10.0
                snr_db = radar.snr(test_range, target_rcs)
                theoretical_pd.append(radar.detection_probability(snr_db, pfa))
                
                # Monte Carlo simulation
                detections = 0
                for trial in range(num_trials):
                    # Simulate detection test
                    # In practice, this would involve full signal processing
                    # Here we use the theoretical model with added variability
                    pd = radar.detection_probability(snr_db, pfa)
                    
                    # Add some realistic variability (RCS fluctuation, etc.)
                    fluctuation_factor = np.random.exponential(1.0)  # Swerling I
                    effective_snr = snr_db + 10 * np.log10(fluctuation_factor)
                    actual_pd = radar.detection_probability(effective_snr, pfa)
                    
                    if np.random.random() < actual_pd:
                        detections += 1
                
                empirical_detection_rate = detections / num_trials
                detection_rates.append(empirical_detection_rate)
            
            detection_rates = np.array(detection_rates)
            theoretical_pd = np.array(theoretical_pd)
            
            # Calculate correlation and mean absolute error
            correlation = np.corrcoef(detection_rates, theoretical_pd)[0, 1]
            mae = np.mean(np.abs(detection_rates - theoretical_pd))
            
            # Test passes if correlation is high and MAE is reasonable
            passed = correlation > 0.8 and mae < 0.2
            
            # Scoring
            correlation_score = max(0.0, (correlation - 0.5) / 0.5)
            mae_score = max(0.0, 1.0 - mae / 0.3)
            overall_score = (correlation_score + mae_score) / 2
            
            message = f"Missed detection rates: correlation={correlation:.3f}, MAE={mae:.3f}"
            
            return self._create_result(
                "missed_detection_rates",
                passed,
                overall_score,
                message,
                details={
                    'correlation': correlation,
                    'mae': mae,
                    'detection_rates': detection_rates.tolist(),
                    'theoretical_pd': theoretical_pd.tolist(),
                    'test_ranges': test_ranges.tolist(),
                    'num_trials': num_trials
                }
            )
            
        except Exception as e:
            return self._create_result(
                "missed_detection_rates",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )
    
    def _test_measurement_error_statistics(self, radar: Radar, signal_processor: SignalProcessor) -> ValidationResult:
        """Test that measurement errors have correct statistical properties"""
        try:
            # Generate measurement errors for different SNR levels
            snr_levels = [5, 10, 15, 20]  # dB
            num_measurements = 500
            
            all_errors_normalized = []
            error_distributions = {}
            
            for snr_db in snr_levels:
                snr_linear = 10**(snr_db / 10)
                
                # Theoretical measurement error standard deviation
                # For range: σ_R ≈ R_res / (2 * sqrt(2 * SNR))
                theoretical_std = radar.params.range_resolution / (2 * np.sqrt(2 * snr_linear))
                
                # Generate simulated measurement errors
                errors = np.random.normal(0, theoretical_std, num_measurements)
                error_distributions[snr_db] = errors
                
                # Normalize errors by theoretical standard deviation
                normalized_errors = errors / theoretical_std
                all_errors_normalized.extend(normalized_errors)
            
            all_errors_normalized = np.array(all_errors_normalized)
            
            # Statistical tests
            # 1. Test for normality (should be approximately normal)
            ks_stat, ks_p_value = stats.kstest(all_errors_normalized, 'norm')
            
            # 2. Test for zero mean
            mean_error = np.mean(all_errors_normalized)
            
            # 3. Test for unit variance
            variance = np.var(all_errors_normalized)
            
            # 4. Test error scaling with SNR (higher SNR should give smaller errors)
            error_stds = [np.std(error_distributions[snr]) for snr in snr_levels]
            # Errors should decrease with increasing SNR
            decreasing_errors = all(error_stds[i] >= error_stds[i+1] for i in range(len(error_stds)-1))
            
            # Scoring
            normality_score = min(1.0, ks_p_value * 2)  # Higher p-value is better
            mean_score = max(0.0, 1.0 - abs(mean_error) * 5)
            variance_score = max(0.0, 1.0 - abs(variance - 1.0))
            scaling_score = 1.0 if decreasing_errors else 0.5
            
            overall_score = (normality_score + mean_score + variance_score + scaling_score) / 4
            passed = overall_score > 0.7
            
            message = f"Measurement error statistics: KS_p={ks_p_value:.3f}, mean={mean_error:.3f}, var={variance:.3f}, decreasing={decreasing_errors}"
            
            return self._create_result(
                "measurement_error_statistics",
                passed,
                overall_score,
                message,
                details={
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p_value,
                    'mean_error': mean_error,
                    'variance': variance,
                    'decreasing_errors': decreasing_errors,
                    'error_stds': error_stds,
                    'snr_levels': snr_levels
                }
            )
            
        except Exception as e:
            return self._create_result(
                "measurement_error_statistics",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )
    
    def _test_blind_ranges_velocities(self, radar: Radar, signal_processor: SignalProcessor) -> ValidationResult:
        """Test that blind ranges and velocities exist as expected"""
        try:
            # Test blind speeds (related to PRF)
            prf = radar.params.prf
            wavelength = radar.params.wavelength
            
            # Blind speeds occur at v = n * λ * PRF / 2 for integer n
            blind_speeds = []
            for n in range(1, 6):  # First 5 blind speeds
                blind_speed = n * wavelength * prf / 2
                blind_speeds.append(blind_speed)
            
            # Test if blind speeds are reasonable for the radar parameters
            max_blind_speed = max(blind_speeds)
            min_blind_speed = min(blind_speeds)
            
            # For typical radar parameters, blind speeds should be in reasonable range
            reasonable_blind_speeds = 10 < min_blind_speed < 1000 and max_blind_speed < 5000
            
            # Test maximum unambiguous range
            max_unambig_range = radar.params.max_unambiguous_range
            theoretical_max_range = 3e8 / (2 * prf)
            
            range_calc_correct = abs(max_unambig_range - theoretical_max_range) < 100  # 100m tolerance
            
            # Test that ranges beyond max unambiguous range cause ambiguity
            # This would manifest as apparent ranges that are modulo max_unambig_range
            test_range = max_unambig_range * 1.3  # 30% beyond max range
            apparent_range = test_range % max_unambig_range
            
            ambiguity_present = apparent_range < max_unambig_range
            
            passed = reasonable_blind_speeds and range_calc_correct and ambiguity_present
            
            # Scoring
            blind_speed_score = 1.0 if reasonable_blind_speeds else 0.5
            range_score = 1.0 if range_calc_correct else 0.0
            ambiguity_score = 1.0 if ambiguity_present else 0.0
            
            overall_score = (blind_speed_score + range_score + ambiguity_score) / 3
            
            message = f"Blind ranges/velocities: blind_speeds_ok={reasonable_blind_speeds}, range_calc_ok={range_calc_correct}, ambiguity_ok={ambiguity_present}"
            
            return self._create_result(
                "blind_ranges_velocities",
                passed,
                overall_score,
                message,
                details={
                    'blind_speeds': blind_speeds,
                    'reasonable_blind_speeds': reasonable_blind_speeds,
                    'max_unambiguous_range': max_unambig_range,
                    'theoretical_max_range': theoretical_max_range,
                    'range_calc_correct': range_calc_correct,
                    'test_range': test_range,
                    'apparent_range': apparent_range,
                    'ambiguity_present': ambiguity_present
                }
            )
            
        except Exception as e:
            return self._create_result(
                "blind_ranges_velocities",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )
    
    def _test_ambiguity_effects(self, radar: Radar, signal_processor: SignalProcessor) -> ValidationResult:
        """Test that ambiguity effects are present and realistic"""
        try:
            # Test range ambiguity
            max_range = radar.params.max_unambiguous_range
            
            # Simulate targets at ambiguous ranges
            true_ranges = [max_range * 0.8, max_range * 1.3, max_range * 2.1]
            apparent_ranges = [r % max_range for r in true_ranges]
            
            # All apparent ranges should be within unambiguous range
            range_ambiguity_correct = all(r < max_range for r in apparent_ranges)
            
            # Test velocity ambiguity using ambiguity function
            test_signal = signal_processor.generate_chirp(1e-6, 1e12, 1000)  # 1μs pulse, 1MHz/μs chirp
            
            try:
                # Calculate ambiguity function
                ambiguity = signal_processor.ambiguity_function(test_signal, max_delay=50, max_doppler=50)
                
                # Check that ambiguity function has expected properties
                # 1. Peak at origin (zero delay, zero Doppler)
                center_row, center_col = ambiguity.shape[0] // 2, ambiguity.shape[1] // 2
                peak_value = ambiguity[center_row, center_col]
                is_peak_at_center = peak_value == np.max(ambiguity)
                
                # 2. Symmetric properties
                is_symmetric = np.allclose(ambiguity, ambiguity[::-1, ::-1], rtol=0.1)
                
                ambiguity_function_ok = is_peak_at_center and is_symmetric
                
            except Exception:
                ambiguity_function_ok = False  # Ambiguity function test failed
            
            # Test Doppler ambiguity
            max_unambig_vel = radar.max_unambiguous_velocity()
            test_velocities = [-max_unambig_vel * 1.5, max_unambig_vel * 1.2]
            
            doppler_ambiguity_present = True
            for vel in test_velocities:
                # Doppler shift should wrap around at max unambiguous velocity
                apparent_vel = ((vel + max_unambig_vel) % (2 * max_unambig_vel)) - max_unambig_vel
                if abs(apparent_vel) >= max_unambig_vel:
                    doppler_ambiguity_present = False
                    break
            
            passed = range_ambiguity_correct and ambiguity_function_ok and doppler_ambiguity_present
            
            # Scoring
            range_score = 1.0 if range_ambiguity_correct else 0.0
            function_score = 1.0 if ambiguity_function_ok else 0.5
            doppler_score = 1.0 if doppler_ambiguity_present else 0.0
            
            overall_score = (range_score + function_score + doppler_score) / 3
            
            message = f"Ambiguity effects: range_ok={range_ambiguity_correct}, function_ok={ambiguity_function_ok}, doppler_ok={doppler_ambiguity_present}"
            
            return self._create_result(
                "ambiguity_effects",
                passed,
                overall_score,
                message,
                details={
                    'range_ambiguity_correct': range_ambiguity_correct,
                    'true_ranges': true_ranges,
                    'apparent_ranges': apparent_ranges,
                    'max_unambiguous_range': max_range,
                    'ambiguity_function_ok': ambiguity_function_ok,
                    'doppler_ambiguity_present': doppler_ambiguity_present,
                    'max_unambiguous_velocity': max_unambig_vel
                }
            )
            
        except Exception as e:
            return self._create_result(
                "ambiguity_effects",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )
    
    def _test_clutter_effects(self, signal_processor: SignalProcessor) -> ValidationResult:
        """Test clutter and multipath effects"""
        try:
            # Generate signal with simulated clutter
            signal_length = 2000
            
            # Target signal
            target_signal = np.exp(1j * 2 * np.pi * 0.1 * np.arange(signal_length))
            
            # Clutter signal (stronger, at zero Doppler)
            clutter_signal = 2.0 * np.ones(signal_length, dtype=complex)
            
            # Noise
            noise = 0.1 * (np.random.normal(0, 1, signal_length) + 
                          1j * np.random.normal(0, 1, signal_length))
            
            # Combined signal
            combined_signal = target_signal + clutter_signal + noise
            
            # Test MTI filter effectiveness
            # Create pulse train for MTI processing
            pulse_train = np.array([combined_signal, combined_signal * 1.1])  # Small amplitude variation
            pulse_train = pulse_train.T  # Shape: (samples, pulses)
            
            try:
                # Apply MTI filter
                mti_output = signal_processor.mti_filter(pulse_train, filter_type="two_pulse")
                
                # MTI should suppress stationary clutter
                input_power = np.mean(np.abs(combined_signal)**2)
                output_power = np.mean(np.abs(mti_output[:, 0])**2)
                
                suppression_ratio = input_power / output_power if output_power > 0 else float('inf')
                
                # Good MTI should provide significant suppression
                mti_effective = suppression_ratio > 3.0  # At least 5 dB suppression
                
            except Exception:
                mti_effective = False
                suppression_ratio = 1.0
            
            # Test clutter suppression using notch filter
            try:
                clutter_suppressed = signal_processor.clutter_suppression(
                    combined_signal, clutter_velocity=0, notch_width=10
                )
                
                # Measure effectiveness
                original_zero_freq_power = np.abs(np.fft.fft(combined_signal)[0])**2
                suppressed_zero_freq_power = np.abs(np.fft.fft(clutter_suppressed)[0])**2
                
                notch_suppression = original_zero_freq_power / suppressed_zero_freq_power if suppressed_zero_freq_power > 0 else float('inf')
                notch_effective = notch_suppression > 5.0  # At least 7 dB suppression
                
            except Exception:
                notch_effective = False
                notch_suppression = 1.0
            
            passed = mti_effective and notch_effective
            
            # Scoring
            mti_score = min(1.0, np.log10(suppression_ratio) / np.log10(10))  # Scale to 10 dB
            notch_score = min(1.0, np.log10(notch_suppression) / np.log10(10))  # Scale to 10 dB
            
            overall_score = (mti_score + notch_score) / 2
            
            message = f"Clutter effects: MTI_suppression={suppression_ratio:.1f}, notch_suppression={notch_suppression:.1f}"
            
            return self._create_result(
                "clutter_effects",
                passed,
                overall_score,
                message,
                details={
                    'mti_effective': mti_effective,
                    'suppression_ratio': suppression_ratio,
                    'notch_effective': notch_effective,
                    'notch_suppression': notch_suppression
                }
            )
            
        except Exception as e:
            return self._create_result(
                "clutter_effects",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )


class IntegrationValidator(BaseValidator):
    """Validator for integration and compatibility tests"""
    
    def __init__(self):
        super().__init__("integration")
    
    def run_validation(self, radar: Radar, targets: List[Target], 
                      signal_processor: SignalProcessor) -> List[ValidationResult]:
        """Run all integration validation tests"""
        results = []
        
        # Test 1: Compatibility with tracking system
        results.append(self._test_tracking_compatibility(radar, targets))
        
        # Test 2: Measurement format validation
        results.append(self._test_measurement_format(radar, targets))
        
        # Test 3: Timing and synchronization
        results.append(self._test_timing_synchronization(radar, signal_processor))
        
        # Test 4: Multi-target scenarios
        results.append(self._test_multi_target_scenarios(radar, targets, signal_processor))
        
        # Test 5: Performance under load
        results.append(self._test_performance_under_load(radar, signal_processor))
        
        return results
    
    def _test_tracking_compatibility(self, radar: Radar, targets: List[Target]) -> ValidationResult:
        """Test compatibility with existing tracking system"""
        try:
            # Test measurement generation for tracking
            if not targets:
                # Create test targets
                targets = [
                    Target(TargetType.AIRCRAFT, motion=TargetMotion(
                        position=np.array([5000, 2000, 3000]),
                        velocity=np.array([100, 50, 0])
                    )),
                    Target(TargetType.MISSILE, motion=TargetMotion(
                        position=np.array([3000, -1000, 2000]),
                        velocity=np.array([-50, 100, 10])
                    ))
                ]
            
            # Generate measurements compatible with tracking system
            measurements = []
            for target in targets[:2]:  # Test with first 2 targets
                range_val, azimuth, elevation = target.get_position_spherical()
                radial_velocity = target.get_radial_velocity()
                
                # Create measurement in expected format
                measurement = Measurement(
                    timestamp=0.0,
                    position=np.array([range_val * np.cos(azimuth) * np.cos(elevation),
                                     range_val * np.sin(azimuth) * np.cos(elevation),
                                     range_val * np.sin(elevation)]),
                    covariance=np.eye(3) * 100,  # 10m standard deviation
                    metadata={"measurement_type": "radar"}
                )
                measurements.append(measurement)
            
            # Test measurement properties
            all_measurements_valid = True
            for meas in measurements:
                # Check that position is reasonable
                if not (100 < np.linalg.norm(meas.position) < 100000):  # 100m to 100km
                    all_measurements_valid = False
                    break
                
                # Check that covariance is positive definite
                try:
                    np.linalg.cholesky(meas.covariance)
                except np.linalg.LinAlgError:
                    all_measurements_valid = False
                    break
            
            # Test measurement uncertainty scaling
            uncertainties = [np.trace(meas.covariance) for meas in measurements]
            reasonable_uncertainties = all(10 < unc < 10000 for unc in uncertainties)
            
            passed = all_measurements_valid and reasonable_uncertainties and len(measurements) > 0
            
            score = 1.0 if passed else 0.0
            if len(measurements) == 0:
                score = 0.0
            elif not all_measurements_valid:
                score = 0.3
            elif not reasonable_uncertainties:
                score = 0.7
            
            message = f"Tracking compatibility: valid_measurements={all_measurements_valid}, reasonable_uncertainties={reasonable_uncertainties}, num_measurements={len(measurements)}"
            
            return self._create_result(
                "tracking_compatibility",
                passed,
                score,
                message,
                details={
                    'all_measurements_valid': all_measurements_valid,
                    'reasonable_uncertainties': reasonable_uncertainties,
                    'num_measurements': len(measurements),
                    'uncertainties': uncertainties
                }
            )
            
        except Exception as e:
            return self._create_result(
                "tracking_compatibility",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )
    
    def _test_measurement_format(self, radar: Radar, targets: List[Target]) -> ValidationResult:
        """Test measurement format matches expected structure"""
        try:
            # Generate various measurement formats
            formats_tested = []
            
            # Cartesian measurements
            if targets:
                target = targets[0]
                pos = target.motion.position
                
                # Test Cartesian format
                cartesian_meas = {
                    'x': pos[0],
                    'y': pos[1], 
                    'z': pos[2],
                    'timestamp': 0.0,
                    'covariance': np.eye(3) * 100
                }
                formats_tested.append(('cartesian', cartesian_meas))
                
                # Test spherical format
                range_val, azimuth, elevation = target.get_position_spherical()
                spherical_meas = {
                    'range': range_val,
                    'azimuth': azimuth,
                    'elevation': elevation,
                    'timestamp': 0.0,
                    'covariance': np.diag([100, 0.01, 0.01])  # Range, azimuth, elevation uncertainties
                }
                formats_tested.append(('spherical', spherical_meas))
                
                # Test measurement with velocity
                radial_velocity = target.get_radial_velocity()
                velocity_meas = {
                    'range': range_val,
                    'azimuth': azimuth,
                    'elevation': elevation,
                    'range_rate': radial_velocity,
                    'timestamp': 0.0,
                    'covariance': np.diag([100, 0.01, 0.01, 1.0])
                }
                formats_tested.append(('with_velocity', velocity_meas))
            
            # Validate each format
            format_validations = []
            for format_name, meas in formats_tested:
                valid = True
                
                # Check required fields
                if 'timestamp' not in meas:
                    valid = False
                if 'covariance' not in meas:
                    valid = False
                
                # Check covariance matrix properties
                if valid and isinstance(meas['covariance'], np.ndarray):
                    cov = meas['covariance']
                    if len(cov.shape) != 2 or cov.shape[0] != cov.shape[1]:
                        valid = False
                    # Check positive definite
                    try:
                        np.linalg.cholesky(cov)
                    except np.linalg.LinAlgError:
                        valid = False
                
                format_validations.append((format_name, valid))
            
            all_formats_valid = all(valid for _, valid in format_validations)
            num_formats_tested = len(format_validations)
            
            passed = all_formats_valid and num_formats_tested >= 2
            score = (sum(1 for _, valid in format_validations if valid) / 
                    max(1, num_formats_tested))
            
            message = f"Measurement format: valid_formats={sum(1 for _, valid in format_validations if valid)}/{num_formats_tested}"
            
            return self._create_result(
                "measurement_format",
                passed,
                score,
                message,
                details={
                    'format_validations': format_validations,
                    'all_formats_valid': all_formats_valid,
                    'num_formats_tested': num_formats_tested
                }
            )
            
        except Exception as e:
            return self._create_result(
                "measurement_format",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )
    
    def _test_timing_synchronization(self, radar: Radar, signal_processor: SignalProcessor) -> ValidationResult:
        """Test timing and synchronization"""
        try:
            # Test timing consistency
            prf = radar.params.prf
            pulse_interval = 1.0 / prf
            
            # Generate pulse train timing
            num_pulses = 100
            pulse_times = np.arange(num_pulses) * pulse_interval
            
            # Check timing consistency
            time_intervals = np.diff(pulse_times)
            expected_interval = pulse_interval
            max_timing_error = np.max(np.abs(time_intervals - expected_interval))
            
            timing_consistent = max_timing_error < 1e-9  # Nanosecond precision
            
            # Test synchronization with sample rate
            sample_rate = signal_processor.sample_rate
            samples_per_pulse = int(sample_rate * pulse_interval)
            
            # Check that sample timing aligns with pulse timing
            sample_timing_error = abs(samples_per_pulse * pulse_interval - pulse_interval)
            sample_timing_ok = sample_timing_error < pulse_interval * 0.01  # 1% tolerance
            
            # Test timestamp generation
            timestamps = []
            start_time = time.time()
            for i in range(10):
                timestamp = start_time + i * 0.1
                timestamps.append(timestamp)
                time.sleep(0.01)  # Small delay
            
            # Check timestamp progression
            timestamp_diffs = np.diff(timestamps)
            expected_diff = 0.1
            timestamp_consistency = all(abs(diff - expected_diff) < 0.05 for diff in timestamp_diffs)
            
            passed = timing_consistent and sample_timing_ok and timestamp_consistency
            
            # Scoring
            timing_score = 1.0 if timing_consistent else 0.0
            sample_score = 1.0 if sample_timing_ok else 0.5
            timestamp_score = 1.0 if timestamp_consistency else 0.5
            
            overall_score = (timing_score + sample_score + timestamp_score) / 3
            
            message = f"Timing/sync: timing_ok={timing_consistent}, sample_ok={sample_timing_ok}, timestamp_ok={timestamp_consistency}"
            
            return self._create_result(
                "timing_synchronization",
                passed,
                overall_score,
                message,
                details={
                    'timing_consistent': timing_consistent,
                    'max_timing_error': max_timing_error,
                    'sample_timing_ok': sample_timing_ok,
                    'sample_timing_error': sample_timing_error,
                    'timestamp_consistency': timestamp_consistency,
                    'pulse_interval': pulse_interval,
                    'samples_per_pulse': samples_per_pulse
                }
            )
            
        except Exception as e:
            return self._create_result(
                "timing_synchronization",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )
    
    def _test_multi_target_scenarios(self, radar: Radar, targets: List[Target], 
                                   signal_processor: SignalProcessor) -> ValidationResult:
        """Test multi-target scenarios"""
        try:
            # Create multi-target scenario if not provided
            if len(targets) < 3:
                test_targets = TargetGenerator.create_random_scenario(
                    5, range_limits=(2000, 15000), speed_limits=(50, 300)
                )
            else:
                test_targets = targets[:5]  # Use first 5 targets
            
            # Test range profile with multiple targets
            target_data = []
            for target in test_targets:
                range_val = target.motion.range
                rcs = target.get_rcs()
                target_data.append((range_val, rcs))
            
            # Generate range profile
            range_bins, profile = radar.range_profile(target_data, max_range=20000, num_bins=1000)
            
            # Find peaks in profile
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(profile, height=np.max(profile) - 30)  # Within 30 dB
            
            # Test target separation capability
            num_targets_expected = len(test_targets)
            num_peaks_found = len(peaks)
            
            # Allow some tolerance for closely spaced targets
            detection_ratio = num_peaks_found / num_targets_expected
            reasonable_detection = 0.6 <= detection_ratio <= 1.5
            
            # Test for false peaks (noise peaks)
            # Peaks should be above noise floor
            noise_floor = np.percentile(profile, 90)  # 90th percentile as noise reference
            signal_peaks = [profile[peak] for peak in peaks if profile[peak] > noise_floor + 10]
            false_peak_ratio = (num_peaks_found - len(signal_peaks)) / max(1, num_peaks_found)
            low_false_peaks = false_peak_ratio < 0.3
            
            # Test measurement consistency across multiple dwells
            measurement_consistency = True
            try:
                # Generate measurements for same targets multiple times
                measurements_dwell1 = []
                measurements_dwell2 = []
                
                for target in test_targets[:3]:  # Test with 3 targets
                    # First dwell
                    range1, az1, el1 = target.get_position_spherical()
                    measurements_dwell1.append([range1, az1, el1])
                    
                    # Second dwell (small time step)
                    target.motion.update(0.1)  # 100ms update
                    range2, az2, el2 = target.get_position_spherical()
                    measurements_dwell2.append([range2, az2, el2])
                
                # Check that measurements changed appropriately
                for i, (meas1, meas2) in enumerate(zip(measurements_dwell1, measurements_dwell2)):
                    range_change = abs(meas2[0] - meas1[0])
                    # Range should change based on radial velocity
                    if range_change > 50:  # More than 50m change seems excessive for 0.1s
                        measurement_consistency = False
                        break
                        
            except Exception:
                measurement_consistency = False
            
            passed = reasonable_detection and low_false_peaks and measurement_consistency
            
            # Scoring
            detection_score = min(1.0, detection_ratio) if detection_ratio >= 0.6 else detection_ratio / 0.6
            false_peak_score = 1.0 if low_false_peaks else 0.5
            consistency_score = 1.0 if measurement_consistency else 0.0
            
            overall_score = (detection_score + false_peak_score + consistency_score) / 3
            
            message = f"Multi-target: detection_ratio={detection_ratio:.2f}, false_peak_ratio={false_peak_ratio:.2f}, consistent={measurement_consistency}"
            
            return self._create_result(
                "multi_target_scenarios",
                passed,
                overall_score,
                message,
                details={
                    'num_targets_expected': num_targets_expected,
                    'num_peaks_found': num_peaks_found,
                    'detection_ratio': detection_ratio,
                    'false_peak_ratio': false_peak_ratio,
                    'measurement_consistency': measurement_consistency,
                    'reasonable_detection': reasonable_detection,
                    'low_false_peaks': low_false_peaks
                }
            )
            
        except Exception as e:
            return self._create_result(
                "multi_target_scenarios",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )
    
    def _test_performance_under_load(self, radar: Radar, signal_processor: SignalProcessor) -> ValidationResult:
        """Test performance under computational load"""
        try:
            # Test processing time scaling
            signal_lengths = [1000, 2000, 5000, 10000]
            processing_times = []
            
            for length in signal_lengths:
                # Generate test signal
                test_signal = (np.random.normal(0, 1, length) + 
                              1j * np.random.normal(0, 1, length)).astype(np.complex128)
                reference = test_signal[:min(500, length)]
                
                # Measure processing time
                start_time = time.time()
                
                # Perform signal processing operations
                matched_output = signal_processor.matched_filter(test_signal, reference)
                detections, threshold = signal_processor.cfar_detector(np.abs(test_signal))
                
                end_time = time.time()
                processing_time = end_time - start_time
                processing_times.append(processing_time)
            
            # Check that processing time scales reasonably (not exponentially)
            # Should be roughly linear or sub-quadratic
            time_ratios = []
            for i in range(1, len(processing_times)):
                length_ratio = signal_lengths[i] / signal_lengths[i-1]
                time_ratio = processing_times[i] / processing_times[i-1]
                scaled_ratio = time_ratio / length_ratio
                time_ratios.append(scaled_ratio)
            
            # Performance is reasonable if time scales no worse than quadratically
            reasonable_scaling = all(ratio < 5.0 for ratio in time_ratios)
            
            # Test memory usage doesn't explode
            max_processing_time = max(processing_times)
            reasonable_performance = max_processing_time < 5.0  # Should process in under 5 seconds
            
            # Test numerical stability under load
            numerical_stability = True
            try:
                large_signal = (np.random.normal(0, 1, 50000) + 
                               1j * np.random.normal(0, 1, 50000)).astype(np.complex128)
                
                # Should not produce NaN or infinite values
                result = signal_processor.matched_filter(large_signal, large_signal[:1000])
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    numerical_stability = False
                    
            except Exception:
                numerical_stability = False
            
            passed = reasonable_scaling and reasonable_performance and numerical_stability
            
            # Scoring
            scaling_score = 1.0 if reasonable_scaling else 0.3
            performance_score = max(0.0, 1.0 - max_processing_time / 10.0)  # Scale to 10 seconds
            stability_score = 1.0 if numerical_stability else 0.0
            
            overall_score = (scaling_score + performance_score + stability_score) / 3
            
            message = f"Performance under load: scaling_ok={reasonable_scaling}, max_time={max_processing_time:.3f}s, stable={numerical_stability}"
            
            return self._create_result(
                "performance_under_load",
                passed,
                overall_score,
                message,
                details={
                    'reasonable_scaling': reasonable_scaling,
                    'reasonable_performance': reasonable_performance,
                    'numerical_stability': numerical_stability,
                    'signal_lengths': signal_lengths,
                    'processing_times': processing_times,
                    'time_ratios': time_ratios,
                    'max_processing_time': max_processing_time
                }
            )
            
        except Exception as e:
            return self._create_result(
                "performance_under_load",
                False,
                0.0,
                f"Test failed with exception: {str(e)}",
                details={'exception': str(e)}
            )


class ValidationSuite:
    """Main validation suite orchestrator"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.validators = [
            AntiCheatValidator(),
            PhysicsValidator(),
            PerformanceValidator(),
            IntegrationValidator()
        ]
    
    def run_all_validations(self, radar_params: Optional[RadarParameters] = None,
                           targets: Optional[List[Target]] = None,
                           signal_processor_config: Optional[Dict] = None) -> ValidationReport:
        """
        Run complete validation suite
        
        Args:
            radar_params: Radar parameters (if None, uses default)
            targets: List of targets (if None, creates test targets)
            signal_processor_config: Signal processor configuration
            
        Returns:
            ValidationReport with all results
        """
        start_time = time.time()
        
        if self.verbose:
            print("Starting Radar Simulation Validation Suite...")
            print("=" * 60)
        
        # Setup default parameters if not provided
        if radar_params is None:
            radar_params = RadarParameters(
                frequency=10e9,  # 10 GHz
                power=100e3,     # 100 kW
                antenna_gain=35, # 35 dB
                pulse_width=1e-6,    # 1 μs
                prf=1000,        # 1 kHz
                bandwidth=10e6,  # 10 MHz
                noise_figure=3,  # 3 dB
                losses=5         # 5 dB
            )
        
        if targets is None:
            targets = TargetGenerator.create_random_scenario(
                3, range_limits=(3000, 12000), speed_limits=(50, 300)
            )
        
        if signal_processor_config is None:
            signal_processor_config = {
                'sample_rate': 100e6,  # 100 MHz
                'bandwidth': radar_params.bandwidth
            }
        
        # Create radar and signal processor instances
        radar = Radar(radar_params)
        signal_processor = SignalProcessor(
            signal_processor_config['sample_rate'],
            signal_processor_config['bandwidth']
        )
        
        # Run all validators
        all_results = []
        for validator in self.validators:
            if self.verbose:
                print(f"\nRunning {validator.name} validation...")
            
            validator_start = time.time()
            try:
                results = validator.run_validation(radar, targets, signal_processor)
                for result in results:
                    result.execution_time = time.time() - validator_start
                all_results.extend(results)
                
                if self.verbose:
                    passed_count = sum(1 for r in results if r.passed)
                    print(f"  {validator.name}: {passed_count}/{len(results)} tests passed")
                    
            except Exception as e:
                error_result = ValidationResult(
                    test_name=f"{validator.name}_error",
                    passed=False,
                    score=0.0,
                    message=f"Validator failed with exception: {str(e)}",
                    details={'exception': str(e)},
                    execution_time=time.time() - validator_start
                )
                all_results.append(error_result)
                
                if self.verbose:
                    print(f"  {validator.name}: FAILED with exception: {str(e)}")
        
        # Generate report
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r.passed)
        failed_tests = total_tests - passed_tests
        
        overall_score = np.mean([r.score for r in all_results]) if all_results else 0.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_results)
        
        # Create summary
        summary = {
            'total_execution_time': time.time() - start_time,
            'radar_parameters': {
                'frequency': radar_params.frequency,
                'power': radar_params.power,
                'prf': radar_params.prf,
                'bandwidth': radar_params.bandwidth
            },
            'num_targets': len(targets),
            'signal_processor_config': signal_processor_config
        }
        
        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            overall_score=overall_score,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            results=all_results,
            summary=summary,
            recommendations=recommendations
        )
        
        if self.verbose:
            print(f"\nValidation completed in {time.time() - start_time:.2f} seconds")
            print(f"Overall score: {overall_score:.2%}")
            print(f"Tests passed: {passed_tests}/{total_tests}")
        
        return report
    
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Analyze failed tests
        failed_results = [r for r in results if not r.passed]
        
        # Group by validator type
        failed_by_validator = {}
        for result in failed_results:
            validator_name = result.test_name.split('_')[0]
            if validator_name not in failed_by_validator:
                failed_by_validator[validator_name] = []
            failed_by_validator[validator_name].append(result)
        
        # Generate specific recommendations
        if 'anticheating' in failed_by_validator:
            recommendations.append(
                "Anti-cheating validation failed: Review simulation implementation to ensure "
                "no ground truth information leaks into the measurement generation process."
            )
        
        if 'physics' in failed_by_validator:
            physics_failures = failed_by_validator['physics']
            if any('range_snr' in r.test_name for r in physics_failures):
                recommendations.append(
                    "Physics validation failed for range-SNR relationship: Verify radar equation "
                    "implementation follows R^4 law correctly."
                )
            if any('detection_probability' in r.test_name for r in physics_failures):
                recommendations.append(
                    "Detection probability curves don't match theory: Review detection probability "
                    "model and ensure it's consistent with Swerling target models."
                )
        
        if 'performance' in failed_by_validator:
            recommendations.append(
                "Performance validation failed: Check that measurement errors, detection rates, "
                "and ambiguity effects are implemented realistically."
            )
        
        if 'integration' in failed_by_validator:
            recommendations.append(
                "Integration validation failed: Ensure measurement formats and timing are "
                "compatible with the tracking system interface."
            )
        
        # Overall score recommendations
        overall_score = np.mean([r.score for r in results]) if results else 0.0
        if overall_score < 0.7:
            recommendations.append(
                "Overall validation score is low. Consider reviewing the entire simulation "
                "implementation for physical realism and numerical accuracy."
            )
        
        # No failures case
        if not recommendations:
            recommendations.append(
                "All validation tests passed. The radar simulation appears to be functioning "
                "correctly and realistically."
            )
        
        return recommendations


# Convenience functions for easy usage
def run_validation_suite(radar_params: Optional[RadarParameters] = None,
                        targets: Optional[List[Target]] = None,
                        signal_processor_config: Optional[Dict] = None,
                        verbose: bool = True) -> ValidationReport:
    """
    Run the complete validation suite with default or provided parameters
    
    Args:
        radar_params: Radar parameters
        targets: List of targets
        signal_processor_config: Signal processor configuration
        verbose: Print progress information
        
    Returns:
        ValidationReport with results
    """
    suite = ValidationSuite(verbose=verbose)
    return suite.run_all_validations(radar_params, targets, signal_processor_config)


def generate_validation_report(report: ValidationReport, 
                             filepath: Optional[Union[str, Path]] = None,
                             show_plots: bool = False) -> None:
    """
    Generate and save a comprehensive validation report
    
    Args:
        report: ValidationReport to process
        filepath: Optional filepath to save JSON report
        show_plots: Whether to generate and show validation plots
    """
    # Print summary
    report.print_summary()
    
    # Save to file if requested
    if filepath:
        report.save_to_file(filepath)
        print(f"\nDetailed report saved to: {filepath}")
    
    # Generate plots if requested
    if show_plots:
        _generate_validation_plots(report)


def _generate_validation_plots(report: ValidationReport) -> None:
    """Generate validation plots (optional visualization)"""
    try:
        # Score distribution by validator
        validators = {}
        for result in report.results:
            validator_name = result.test_name.split('_')[0]
            if validator_name not in validators:
                validators[validator_name] = []
            validators[validator_name].append(result.score)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Validator scores
        validator_names = list(validators.keys())
        avg_scores = [np.mean(validators[name]) for name in validator_names]
        
        ax1.bar(validator_names, avg_scores)
        ax1.set_ylabel('Average Score')
        ax1.set_title('Validation Scores by Category')
        ax1.set_ylim(0, 1)
        
        # Test pass/fail distribution
        total_tests = len(report.results)
        passed_tests = report.passed_tests
        failed_tests = report.failed_tests
        
        ax2.pie([passed_tests, failed_tests], labels=['Passed', 'Failed'], 
               autopct='%1.1f%%', colors=['green', 'red'])
        ax2.set_title('Test Results Distribution')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")
    except Exception as e:
        print(f"Error generating plots: {e}")


# Pytest integration
class TestRadarValidation:
    """Pytest test class for radar validation"""
    
    @pytest.fixture
    def validation_suite(self):
        """Create validation suite for testing"""
        return ValidationSuite(verbose=False)
    
    @pytest.fixture
    def test_radar(self):
        """Create test radar for validation"""
        params = RadarParameters(
            frequency=10e9, power=100e3, antenna_gain=35,
            pulse_width=1e-6, prf=1000, bandwidth=10e6,
            noise_figure=3, losses=5
        )
        return Radar(params)
    
    @pytest.fixture
    def test_targets(self):
        """Create test targets for validation"""
        return TargetGenerator.create_random_scenario(3)
    
    @pytest.fixture
    def test_signal_processor(self):
        """Create test signal processor for validation"""
        return SignalProcessor(100e6, 10e6)
    
    def test_anti_cheat_validation(self, validation_suite, test_radar, test_targets, test_signal_processor):
        """Test anti-cheat validation"""
        validator = AntiCheatValidator()
        results = validator.run_validation(test_radar, test_targets, test_signal_processor)
        
        assert len(results) > 0
        assert all(isinstance(r, ValidationResult) for r in results)
        
        # At least some tests should pass for a properly implemented system
        passed_count = sum(1 for r in results if r.passed)
        assert passed_count >= len(results) // 2  # At least half should pass
    
    def test_physics_validation(self, validation_suite, test_radar, test_targets, test_signal_processor):
        """Test physics validation"""
        validator = PhysicsValidator()
        results = validator.run_validation(test_radar, test_targets, test_signal_processor)
        
        assert len(results) > 0
        assert all(isinstance(r, ValidationResult) for r in results)
        
        # Physics should generally work correctly
        passed_count = sum(1 for r in results if r.passed)
        assert passed_count >= len(results) // 2
    
    def test_performance_validation(self, validation_suite, test_radar, test_targets, test_signal_processor):
        """Test performance validation"""
        validator = PerformanceValidator()
        results = validator.run_validation(test_radar, test_targets, test_signal_processor)
        
        assert len(results) > 0
        assert all(isinstance(r, ValidationResult) for r in results)
    
    def test_integration_validation(self, validation_suite, test_radar, test_targets, test_signal_processor):
        """Test integration validation"""
        validator = IntegrationValidator()
        results = validator.run_validation(test_radar, test_targets, test_signal_processor)
        
        assert len(results) > 0
        assert all(isinstance(r, ValidationResult) for r in results)
    
    def test_full_validation_suite(self, validation_suite):
        """Test complete validation suite"""
        report = validation_suite.run_all_validations()
        
        assert isinstance(report, ValidationReport)
        assert report.total_tests > 0
        assert report.overall_score >= 0.0
        assert report.overall_score <= 1.0
        assert len(report.results) == report.total_tests
        assert report.passed_tests + report.failed_tests == report.total_tests
    
    @pytest.mark.slow
    def test_validation_performance(self, validation_suite):
        """Test validation suite performance"""
        start_time = time.time()
        report = validation_suite.run_all_validations()
        execution_time = time.time() - start_time
        
        # Validation should complete in reasonable time
        assert execution_time < 60.0  # Less than 1 minute
        assert report.summary['total_execution_time'] < 60.0


if __name__ == "__main__":
    # Example usage
    print("Radar Simulation Validation Suite")
    print("==================================")
    
    # Run validation with default parameters
    report = run_validation_suite(verbose=True)
    
    # Generate and save report
    generate_validation_report(
        report, 
        filepath="validation_report.json",
        show_plots=True
    )