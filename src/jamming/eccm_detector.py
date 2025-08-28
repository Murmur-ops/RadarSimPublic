"""
Electronic Counter-Countermeasures (ECCM) Detection Module

This module implements detection algorithms to identify and counter DRFM jamming,
false targets, and other electronic warfare techniques.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import scipy.signal as signal
import scipy.stats as stats


class JammingType(Enum):
    """Types of jamming that can be detected"""
    NONE = "none"
    NOISE = "noise"
    DRFM = "drfm"
    FALSE_TARGETS = "false_targets"
    GATE_PULL_OFF = "gate_pull_off"
    BARRAGE = "barrage"
    SPOT = "spot"
    REPEATER = "repeater"
    DECEPTION = "deception"


@dataclass
class ECCMDetectionResult:
    """Result of ECCM detection analysis"""
    jamming_detected: bool
    jamming_type: JammingType
    confidence: float  # 0-1 confidence level
    parameters: Dict[str, Any] = field(default_factory=dict)
    recommended_countermeasures: List[str] = field(default_factory=list)
    detection_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PulseConsistencyMetrics:
    """Metrics for pulse-to-pulse consistency analysis"""
    phase_consistency: float
    amplitude_consistency: float
    frequency_consistency: float
    timing_consistency: float
    modulation_consistency: float
    
    @property
    def overall_consistency(self) -> float:
        """Calculate overall consistency score"""
        return np.mean([
            self.phase_consistency,
            self.amplitude_consistency,
            self.frequency_consistency,
            self.timing_consistency,
            self.modulation_consistency
        ])


class ECCMDetector:
    """
    Electronic Counter-Countermeasures detector for identifying jamming techniques
    """
    
    def __init__(self,
                 sample_rate: float = 1e9,
                 detection_threshold: float = 0.7,
                 analysis_window: int = 10):
        """
        Initialize ECCM detector
        
        Args:
            sample_rate: Sampling rate (Hz)
            detection_threshold: Detection confidence threshold (0-1)
            analysis_window: Number of pulses to analyze
        """
        self.sample_rate = sample_rate
        self.detection_threshold = detection_threshold
        self.analysis_window = analysis_window
        
        # Storage for pulse history
        self.pulse_history = []
        self.detection_history = []
        
        # Detection parameters
        self.drfm_phase_tolerance = 0.1  # radians
        self.false_target_correlation_threshold = 0.9
        self.noise_jamming_snr_threshold = 10  # dB
        
    def detect_jamming(self, 
                       signal_data: np.ndarray,
                       reference_signal: Optional[np.ndarray] = None) -> ECCMDetectionResult:
        """
        Main detection method - analyzes signal for jamming
        
        Args:
            signal_data: Received signal to analyze
            reference_signal: Expected signal (if available)
            
        Returns:
            Detection result with jamming type and confidence
        """
        # Store signal in history
        self.pulse_history.append(signal_data)
        if len(self.pulse_history) > self.analysis_window:
            self.pulse_history.pop(0)
        
        # Run detection algorithms
        results = []
        
        # Check for DRFM
        drfm_result = self._detect_drfm(signal_data, reference_signal)
        results.append(drfm_result)
        
        # Check for false targets
        false_target_result = self._detect_false_targets(signal_data)
        results.append(false_target_result)
        
        # Check for noise jamming
        noise_result = self._detect_noise_jamming(signal_data)
        results.append(noise_result)
        
        # Check for gate pull-off
        if len(self.pulse_history) >= 3:
            gpo_result = self._detect_gate_pull_off()
            results.append(gpo_result)
        
        # Find highest confidence detection
        best_result = max(results, key=lambda x: x.confidence)
        
        # Store detection result
        self.detection_history.append(best_result)
        
        return best_result
    
    def _detect_drfm(self, 
                     signal_data: np.ndarray,
                     reference_signal: Optional[np.ndarray]) -> ECCMDetectionResult:
        """
        Detect DRFM jamming through phase consistency analysis
        
        DRFM signals have unnaturally consistent phase relationships
        """
        if len(self.pulse_history) < 2:
            return ECCMDetectionResult(
                jamming_detected=False,
                jamming_type=JammingType.NONE,
                confidence=0.0
            )
        
        # Analyze pulse-to-pulse consistency
        consistency = self._analyze_pulse_consistency()
        
        # DRFM detection logic
        drfm_detected = False
        confidence = 0.0
        
        # Check for unnatural phase consistency
        if consistency.phase_consistency > 0.95:
            drfm_detected = True
            confidence = consistency.phase_consistency
            
        # Check for digital quantization artifacts
        quantization_score = self._detect_quantization_artifacts(signal_data)
        if quantization_score > 0.7:
            drfm_detected = True
            confidence = max(confidence, quantization_score)
        
        # Check for processing delay
        if reference_signal is not None:
            delay_score = self._detect_processing_delay(signal_data, reference_signal)
            if delay_score > 0.6:
                drfm_detected = True
                confidence = max(confidence, delay_score)
        
        return ECCMDetectionResult(
            jamming_detected=drfm_detected,
            jamming_type=JammingType.DRFM if drfm_detected else JammingType.NONE,
            confidence=confidence,
            parameters={'consistency': consistency.overall_consistency},
            recommended_countermeasures=['frequency_agility', 'waveform_diversity'] if drfm_detected else [],
            detection_metrics={
                'phase_consistency': consistency.phase_consistency,
                'quantization_score': quantization_score
            }
        )
    
    def _detect_false_targets(self, signal_data: np.ndarray) -> ECCMDetectionResult:
        """
        Detect false targets through correlation analysis
        
        False targets often have high correlation with each other
        """
        if len(self.pulse_history) < 2:
            return ECCMDetectionResult(
                jamming_detected=False,
                jamming_type=JammingType.NONE,
                confidence=0.0
            )
        
        # Find peaks in signal
        peaks = self._find_signal_peaks(signal_data)
        
        if len(peaks) < 2:
            return ECCMDetectionResult(
                jamming_detected=False,
                jamming_type=JammingType.NONE,
                confidence=0.0
            )
        
        # Calculate correlation between peaks
        correlations = []
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                corr = self._calculate_peak_correlation(signal_data, peaks[i], peaks[j])
                correlations.append(corr)
        
        # High correlation indicates false targets
        avg_correlation = np.mean(correlations) if correlations else 0
        false_targets_detected = avg_correlation > self.false_target_correlation_threshold
        
        # Check for unrealistic target spacing
        spacing_score = self._analyze_target_spacing(peaks)
        
        confidence = max(avg_correlation, spacing_score)
        
        return ECCMDetectionResult(
            jamming_detected=false_targets_detected,
            jamming_type=JammingType.FALSE_TARGETS if false_targets_detected else JammingType.NONE,
            confidence=confidence,
            parameters={'num_targets': len(peaks), 'correlation': avg_correlation},
            recommended_countermeasures=['doppler_processing', 'track_validation'] if false_targets_detected else [],
            detection_metrics={
                'avg_correlation': avg_correlation,
                'spacing_score': spacing_score
            }
        )
    
    def _detect_noise_jamming(self, signal_data: np.ndarray) -> ECCMDetectionResult:
        """
        Detect noise jamming through SNR analysis and spectral characteristics
        """
        # Calculate noise statistics
        noise_power = np.median(np.abs(signal_data)**2)
        peak_power = np.max(np.abs(signal_data)**2)
        
        if peak_power > 0:
            snr_db = 10 * np.log10(peak_power / noise_power)
        else:
            snr_db = 0
        
        # Check for elevated noise floor
        noise_jamming_detected = snr_db < self.noise_jamming_snr_threshold
        
        # Analyze spectral flatness
        spectral_flatness = self._calculate_spectral_flatness(signal_data)
        
        # Determine jamming type
        if noise_jamming_detected:
            if spectral_flatness > 0.8:
                jamming_type = JammingType.BARRAGE
            else:
                jamming_type = JammingType.SPOT
        else:
            jamming_type = JammingType.NONE
        
        confidence = 1.0 - (snr_db / 30.0) if snr_db < 30 else 0.0
        
        return ECCMDetectionResult(
            jamming_detected=noise_jamming_detected,
            jamming_type=jamming_type,
            confidence=confidence,
            parameters={'snr_db': snr_db, 'spectral_flatness': spectral_flatness},
            recommended_countermeasures=['increase_power', 'frequency_hopping'] if noise_jamming_detected else [],
            detection_metrics={
                'snr_db': snr_db,
                'spectral_flatness': spectral_flatness
            }
        )
    
    def _detect_gate_pull_off(self) -> ECCMDetectionResult:
        """
        Detect gate pull-off through tracking parameter evolution
        """
        # Analyze range/velocity progression
        range_progression = []
        velocity_progression = []
        
        for pulse in self.pulse_history[-3:]:
            peak_idx = np.argmax(np.abs(pulse))
            range_progression.append(peak_idx)
            
            # Simple velocity estimation from phase
            phase = np.angle(pulse[peak_idx])
            velocity_progression.append(phase)
        
        # Check for monotonic progression (indication of pull-off)
        range_drift = np.diff(range_progression)
        velocity_drift = np.diff(velocity_progression)
        
        range_pull_detected = np.all(range_drift > 0) or np.all(range_drift < 0)
        velocity_pull_detected = np.all(velocity_drift > 0) or np.all(velocity_drift < 0)
        
        gpo_detected = range_pull_detected or velocity_pull_detected
        
        confidence = 0.0
        if range_pull_detected:
            confidence = max(confidence, 0.8)
        if velocity_pull_detected:
            confidence = max(confidence, 0.7)
        
        return ECCMDetectionResult(
            jamming_detected=gpo_detected,
            jamming_type=JammingType.GATE_PULL_OFF if gpo_detected else JammingType.NONE,
            confidence=confidence,
            parameters={
                'range_pull': range_pull_detected,
                'velocity_pull': velocity_pull_detected
            },
            recommended_countermeasures=['leading_edge_tracking', 'multi_hypothesis_tracking'] if gpo_detected else [],
            detection_metrics={
                'range_drift': float(np.mean(np.abs(range_drift))),
                'velocity_drift': float(np.mean(np.abs(velocity_drift)))
            }
        )
    
    def _analyze_pulse_consistency(self) -> PulseConsistencyMetrics:
        """Analyze consistency between pulses"""
        if len(self.pulse_history) < 2:
            return PulseConsistencyMetrics(1.0, 1.0, 1.0, 1.0, 1.0)
        
        # Compare last two pulses
        pulse1 = self.pulse_history[-2]
        pulse2 = self.pulse_history[-1]
        
        # Phase consistency
        phase1 = np.angle(pulse1)
        phase2 = np.angle(pulse2)
        phase_diff = np.abs(phase1 - phase2)
        phase_consistency = 1.0 - np.mean(phase_diff) / np.pi
        
        # Amplitude consistency
        amp1 = np.abs(pulse1)
        amp2 = np.abs(pulse2)
        amp_correlation = np.corrcoef(amp1, amp2)[0, 1]
        amplitude_consistency = max(0, amp_correlation)
        
        # Frequency consistency (via FFT)
        fft1 = np.fft.fft(pulse1)
        fft2 = np.fft.fft(pulse2)
        freq_correlation = np.corrcoef(np.abs(fft1), np.abs(fft2))[0, 1]
        frequency_consistency = max(0, freq_correlation)
        
        # Timing consistency (placeholder)
        timing_consistency = 0.9
        
        # Modulation consistency (placeholder)
        modulation_consistency = 0.85
        
        return PulseConsistencyMetrics(
            phase_consistency=phase_consistency,
            amplitude_consistency=amplitude_consistency,
            frequency_consistency=frequency_consistency,
            timing_consistency=timing_consistency,
            modulation_consistency=modulation_consistency
        )
    
    def _detect_quantization_artifacts(self, signal_data: np.ndarray) -> float:
        """Detect digital quantization artifacts in signal"""
        # Look for discrete amplitude levels
        amplitudes = np.abs(signal_data)
        unique_levels = len(np.unique(np.round(amplitudes * 1000) / 1000))
        total_samples = len(amplitudes)
        
        # Fewer unique levels indicates quantization
        quantization_score = 1.0 - (unique_levels / total_samples)
        return np.clip(quantization_score, 0, 1)
    
    def _detect_processing_delay(self, 
                                 signal_data: np.ndarray,
                                 reference_signal: np.ndarray) -> float:
        """Detect processing delay between signals"""
        # Cross-correlation to find delay
        correlation = signal.correlate(signal_data, reference_signal, mode='same')
        peak_idx = np.argmax(np.abs(correlation))
        expected_idx = len(correlation) // 2
        
        # Delay from expected position
        delay_samples = abs(peak_idx - expected_idx)
        delay_time = delay_samples / self.sample_rate
        
        # DRFM typically has 0.1-1 microsecond delay
        if 0.1e-6 < delay_time < 1e-6:
            return 0.8
        elif delay_time > 1e-6:
            return 0.6
        else:
            return 0.2
    
    def _find_signal_peaks(self, signal_data: np.ndarray) -> List[int]:
        """Find peaks in signal"""
        magnitude = np.abs(signal_data)
        threshold = np.mean(magnitude) + 2 * np.std(magnitude)
        peaks, _ = signal.find_peaks(magnitude, height=threshold, distance=10)
        return peaks.tolist()
    
    def _calculate_peak_correlation(self, 
                                   signal_data: np.ndarray,
                                   peak1_idx: int,
                                   peak2_idx: int) -> float:
        """Calculate correlation between two peaks"""
        window_size = 50
        
        # Extract windows around peaks
        window1 = signal_data[max(0, peak1_idx-window_size):peak1_idx+window_size]
        window2 = signal_data[max(0, peak2_idx-window_size):peak2_idx+window_size]
        
        # Pad to same size
        min_len = min(len(window1), len(window2))
        if min_len > 0:
            correlation = np.corrcoef(
                np.abs(window1[:min_len]),
                np.abs(window2[:min_len])
            )[0, 1]
            return abs(correlation)
        return 0.0
    
    def _analyze_target_spacing(self, peaks: List[int]) -> float:
        """Analyze spacing between targets for unrealistic patterns"""
        if len(peaks) < 2:
            return 0.0
        
        spacings = np.diff(peaks)
        
        # Check for perfectly regular spacing (suspicious)
        std_spacing = np.std(spacings)
        mean_spacing = np.mean(spacings)
        
        if mean_spacing > 0:
            regularity = 1.0 - (std_spacing / mean_spacing)
            return regularity if regularity > 0.8 else 0.0
        return 0.0
    
    def _calculate_spectral_flatness(self, signal_data: np.ndarray) -> float:
        """Calculate spectral flatness (Wiener entropy)"""
        # Compute power spectrum
        fft = np.fft.fft(signal_data)
        power_spectrum = np.abs(fft[:len(fft)//2])**2
        
        # Avoid log(0)
        power_spectrum = power_spectrum[power_spectrum > 0]
        
        if len(power_spectrum) == 0:
            return 0.0
        
        # Geometric mean
        log_spectrum = np.log(power_spectrum)
        geometric_mean = np.exp(np.mean(log_spectrum))
        
        # Arithmetic mean
        arithmetic_mean = np.mean(power_spectrum)
        
        # Spectral flatness
        if arithmetic_mean > 0:
            flatness = geometric_mean / arithmetic_mean
            return np.clip(flatness, 0, 1)
        return 0.0
    
    def get_countermeasure_recommendations(self, 
                                          detection_result: ECCMDetectionResult) -> List[str]:
        """
        Get detailed countermeasure recommendations based on detection
        
        Args:
            detection_result: Detection result from analyze
            
        Returns:
            List of recommended countermeasures
        """
        recommendations = []
        
        if detection_result.jamming_type == JammingType.DRFM:
            recommendations.extend([
                "Enable frequency agility",
                "Use non-repeating waveforms",
                "Implement waveform diversity",
                "Use leading-edge tracking",
                "Enable pulse compression sidelobe blanking"
            ])
        elif detection_result.jamming_type == JammingType.FALSE_TARGETS:
            recommendations.extend([
                "Implement multi-hypothesis tracking",
                "Use Doppler processing for validation",
                "Enable track history analysis",
                "Apply kinematic filtering",
                "Use amplitude discrimination"
            ])
        elif detection_result.jamming_type in [JammingType.BARRAGE, JammingType.SPOT]:
            recommendations.extend([
                "Increase transmit power",
                "Enable frequency hopping",
                "Use pulse integration",
                "Implement adaptive filtering",
                "Switch to backup frequency"
            ])
        elif detection_result.jamming_type == JammingType.GATE_PULL_OFF:
            recommendations.extend([
                "Switch to leading-edge tracking",
                "Enable multi-range gate tracking",
                "Use velocity gate memory",
                "Implement track coast mode",
                "Apply predictive filtering"
            ])
        
        return recommendations
    
    def reset(self):
        """Reset detector state"""
        self.pulse_history.clear()
        self.detection_history.clear()


def create_adaptive_eccm_suite() -> Dict[str, Any]:
    """
    Create a comprehensive ECCM suite with multiple detection algorithms
    
    Returns:
        Dictionary of ECCM components
    """
    return {
        'primary_detector': ECCMDetector(detection_threshold=0.7),
        'sensitive_detector': ECCMDetector(detection_threshold=0.5),
        'fast_detector': ECCMDetector(analysis_window=3),
        'parameters': {
            'frequency_agility_enabled': True,
            'waveform_diversity_enabled': True,
            'adaptive_processing_enabled': True
        }
    }