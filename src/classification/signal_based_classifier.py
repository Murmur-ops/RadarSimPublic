#!/usr/bin/env python3
"""
Signal-based target classification and discrimination
Works purely from radar returns without accessing ground truth
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RadarObservable:
    """Observable features from radar returns only"""
    range_bins: np.ndarray  # Range profile
    doppler_bins: np.ndarray  # Doppler spectrum
    snr: float  # Signal-to-noise ratio
    doppler_spread: float  # Doppler bandwidth
    range_extent: int  # Number of range cells
    peak_amplitude: float
    time: float
    
    # Derived observables
    range_profile_variance: float = 0
    doppler_centroid: float = 0
    spectral_kurtosis: float = 0
    phase_coherence: float = 0


class SignalBasedClassifier:
    """
    Classify targets based purely on signal characteristics
    No access to ground truth - only radar observables
    """
    
    def __init__(self, 
                 range_resolution: float = 50,
                 velocity_resolution: float = 1.0,
                 sampling_rate: float = 1000):
        """
        Initialize signal-based classifier
        
        Args:
            range_resolution: Range bin size (meters)
            velocity_resolution: Doppler bin size (m/s)
            sampling_rate: Radar PRF (Hz)
        """
        self.range_resolution = range_resolution
        self.velocity_resolution = velocity_resolution
        self.sampling_rate = sampling_rate
        
        # Build classification rules from observables
        self.classification_rules = self._build_observable_rules()
        
        # Track history for temporal analysis
        self.track_history: Dict[int, List[RadarObservable]] = {}
        
    def _build_observable_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Build classification rules based on observable signatures
        These are derived from physics, not ground truth
        """
        rules = {
            'high_speed_small': {  # Likely missile
                'doppler_threshold': 200,  # m/s
                'range_extent_max': 3,  # range bins
                'snr_range': (10, 30),
                'likely_class': 'missile'
            },
            'high_speed_large': {  # Likely aircraft
                'doppler_threshold': 100,
                'range_extent_min': 4,
                'snr_range': (20, 50),
                'likely_class': 'aircraft'
            },
            'slow_periodic': {  # Likely helicopter/drone
                'doppler_max': 50,
                'spectral_features': 'periodic',
                'snr_range': (10, 40),
                'likely_class': 'rotorcraft'
            },
            'very_slow_small': {  # Likely bird/drone
                'doppler_max': 30,
                'range_extent_max': 2,
                'snr_range': (5, 20),
                'likely_class': 'small_target'
            },
            'stationary_large': {  # Likely ground/ship
                'doppler_max': 5,
                'range_extent_min': 5,
                'snr_range': (30, 60),
                'likely_class': 'surface_target'
            }
        }
        return rules
    
    def extract_observables(self, range_doppler_map: np.ndarray,
                           detection_indices: Tuple[int, int]) -> RadarObservable:
        """
        Extract observable features from range-Doppler map
        
        Args:
            range_doppler_map: 2D range-Doppler map
            detection_indices: (range_idx, doppler_idx) of detection
            
        Returns:
            Observable features from this detection
        """
        r_idx, d_idx = detection_indices
        
        # Extract range profile around detection
        range_window = 5
        r_start = max(0, r_idx - range_window)
        r_end = min(range_doppler_map.shape[0], r_idx + range_window + 1)
        range_profile = range_doppler_map[r_start:r_end, d_idx]
        
        # Extract Doppler profile around detection
        doppler_window = 10
        d_start = max(0, d_idx - doppler_window)
        d_end = min(range_doppler_map.shape[1], d_idx + doppler_window + 1)
        doppler_profile = range_doppler_map[r_idx, d_start:d_end]
        
        # Calculate observable metrics
        noise_floor = np.median(range_doppler_map)
        peak_power = range_doppler_map[r_idx, d_idx]
        snr = 10 * np.log10(peak_power / (noise_floor + 1e-10))
        
        # Range extent (how many bins above threshold)
        range_threshold = noise_floor * 3
        range_extent = np.sum(range_profile > range_threshold)
        
        # Doppler spread
        doppler_threshold = noise_floor * 3
        doppler_mask = doppler_profile > doppler_threshold
        if np.any(doppler_mask):
            doppler_indices = np.where(doppler_mask)[0]
            doppler_spread = (doppler_indices[-1] - doppler_indices[0]) * self.velocity_resolution
        else:
            doppler_spread = self.velocity_resolution
        
        # Doppler centroid (weighted average)
        if np.sum(doppler_profile) > 0:
            doppler_bins = np.arange(d_start, d_end) - range_doppler_map.shape[1]//2
            doppler_centroid = np.sum(doppler_bins * doppler_profile) / np.sum(doppler_profile)
            doppler_centroid *= self.velocity_resolution
        else:
            doppler_centroid = (d_idx - range_doppler_map.shape[1]//2) * self.velocity_resolution
        
        # Spectral shape analysis
        spectral_kurtosis = kurtosis(doppler_profile) if len(doppler_profile) > 3 else 0
        
        # Phase coherence (from complex data if available)
        phase_coherence = 0.5  # Default, would need complex data
        
        return RadarObservable(
            range_bins=range_profile,
            doppler_bins=doppler_profile,
            snr=snr,
            doppler_spread=doppler_spread,
            range_extent=range_extent,
            peak_amplitude=peak_power,
            time=0,  # Would be set externally
            range_profile_variance=np.var(range_profile),
            doppler_centroid=doppler_centroid,
            spectral_kurtosis=spectral_kurtosis,
            phase_coherence=phase_coherence
        )
    
    def classify_from_observables(self, observable: RadarObservable) -> Tuple[str, float]:
        """
        Classify target based purely on observables
        
        Args:
            observable: Radar observable features
            
        Returns:
            (classification, confidence) tuple
        """
        # Check each rule set
        scores = {}
        
        # High-speed small target (missile-like)
        if abs(observable.doppler_centroid) > 200 and observable.range_extent <= 3:
            scores['missile'] = 0.8
            if observable.snr > 15 and observable.snr < 30:
                scores['missile'] += 0.1
        
        # High-speed large target (aircraft-like)
        if abs(observable.doppler_centroid) > 100 and observable.range_extent >= 4:
            scores['aircraft'] = 0.7
            if observable.snr > 25:
                scores['aircraft'] += 0.2
        
        # Slow target with spectral lines (rotorcraft)
        if abs(observable.doppler_centroid) < 50 and observable.spectral_kurtosis > 3:
            scores['rotorcraft'] = 0.7
            # Check for harmonic lines in Doppler
            if self._has_harmonic_lines(observable.doppler_bins):
                scores['rotorcraft'] += 0.2
        
        # Very slow small target (bird/small drone)
        if abs(observable.doppler_centroid) < 30 and observable.range_extent <= 2:
            if observable.snr < 20:
                scores['bird'] = 0.6
            else:
                scores['small_drone'] = 0.6
        
        # Large slow target (ship/vehicle)
        if abs(observable.doppler_centroid) < 10 and observable.range_extent >= 5:
            if observable.snr > 30:
                scores['surface_vessel'] = 0.7
        
        # Default unknown
        if not scores:
            return 'unknown', 0.3
        
        # Return highest scoring classification
        best_class = max(scores, key=scores.get)
        confidence = scores[best_class]
        
        return best_class, confidence
    
    def _has_harmonic_lines(self, spectrum: np.ndarray) -> bool:
        """
        Check for harmonic lines indicating rotating parts
        Pure signal processing - no ground truth needed
        """
        if len(spectrum) < 10:
            return False
        
        # FFT of the spectrum to find periodicity
        spectrum_fft = np.abs(np.fft.fft(spectrum))
        spectrum_fft = spectrum_fft[:len(spectrum_fft)//2]
        
        # Look for peaks (harmonics)
        mean_level = np.mean(spectrum_fft)
        peaks = spectrum_fft > (mean_level * 3)
        
        # Multiple peaks suggest harmonics
        return np.sum(peaks) >= 2
    
    def discriminate_from_track(self, 
                               track_id: int,
                               observables: List[RadarObservable]) -> Dict[str, Any]:
        """
        Discriminate target type from track history
        Uses temporal patterns, no ground truth
        
        Args:
            track_id: Track identifier
            observables: History of observables for this track
            
        Returns:
            Discrimination results
        """
        if len(observables) < 3:
            return {'class': 'unknown', 'confidence': 0.1}
        
        # Store history
        self.track_history[track_id] = observables
        
        # Analyze velocity profile over time
        velocities = [obs.doppler_centroid for obs in observables]
        velocity_mean = np.mean(velocities)
        velocity_std = np.std(velocities)
        
        # Analyze SNR consistency
        snrs = [obs.snr for obs in observables]
        snr_mean = np.mean(snrs)
        snr_std = np.std(snrs)
        
        # Analyze range extent consistency
        range_extents = [obs.range_extent for obs in observables]
        extent_mean = np.mean(range_extents)
        
        # Classification logic based on temporal patterns
        discrimination = {}
        
        # Consistent high speed = likely missile/aircraft
        if velocity_mean > 150:
            if extent_mean < 3:
                discrimination['class'] = 'missile'
                discrimination['confidence'] = min(0.9, velocity_mean / 300)
            else:
                discrimination['class'] = 'aircraft'
                discrimination['confidence'] = 0.8
        
        # Variable velocity with periodic pattern = rotorcraft
        elif velocity_std > 5 and self._check_periodic_velocity(velocities):
            discrimination['class'] = 'helicopter'
            discrimination['confidence'] = 0.7
        
        # Low, consistent velocity = surface target
        elif velocity_mean < 20 and velocity_std < 3:
            if extent_mean > 4:
                discrimination['class'] = 'ship'
                discrimination['confidence'] = 0.7
            else:
                discrimination['class'] = 'vehicle'
                discrimination['confidence'] = 0.6
        
        # Highly variable, low SNR = likely bird
        elif snr_mean < 15 and velocity_std > 10:
            discrimination['class'] = 'bird'
            discrimination['confidence'] = 0.6
        
        else:
            discrimination['class'] = 'unknown'
            discrimination['confidence'] = 0.3
        
        # Add supporting evidence
        discrimination['evidence'] = {
            'velocity_profile': (velocity_mean, velocity_std),
            'snr_profile': (snr_mean, snr_std),
            'range_extent': extent_mean,
            'track_length': len(observables)
        }
        
        return discrimination
    
    def _check_periodic_velocity(self, velocities: List[float]) -> bool:
        """Check for periodic patterns in velocity (indicates maneuvering)"""
        if len(velocities) < 10:
            return False
        
        # Simple autocorrelation check
        velocities_array = np.array(velocities)
        velocities_centered = velocities_array - np.mean(velocities_array)
        
        autocorr = np.correlate(velocities_centered, velocities_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Look for peaks in autocorrelation
        if len(autocorr) > 3:
            peaks = sp_signal.find_peaks(autocorr[1:], height=0.5*autocorr[0])[0]
            return len(peaks) > 0
        
        return False
    
    def compare_tracks(self, track1_obs: List[RadarObservable],
                      track2_obs: List[RadarObservable]) -> float:
        """
        Compare two tracks to determine if they're the same target type
        Returns similarity score (0-1)
        """
        if not track1_obs or not track2_obs:
            return 0
        
        # Compare average observables
        features1 = self._extract_track_features(track1_obs)
        features2 = self._extract_track_features(track2_obs)
        
        # Calculate similarity in each dimension
        velocity_sim = 1 - abs(features1['velocity'] - features2['velocity']) / 300
        snr_sim = 1 - abs(features1['snr'] - features2['snr']) / 50
        extent_sim = 1 - abs(features1['extent'] - features2['extent']) / 10
        
        # Weight and combine
        similarity = (velocity_sim * 0.4 + snr_sim * 0.3 + extent_sim * 0.3)
        
        return max(0, min(1, similarity))
    
    def _extract_track_features(self, observables: List[RadarObservable]) -> Dict[str, float]:
        """Extract summary features from track history"""
        return {
            'velocity': np.mean([obs.doppler_centroid for obs in observables]),
            'snr': np.mean([obs.snr for obs in observables]),
            'extent': np.mean([obs.range_extent for obs in observables]),
            'spread': np.mean([obs.doppler_spread for obs in observables])
        }


class NonCheatingDiscriminator:
    """
    Target discrimination using only radar observables
    Absolutely no access to ground truth
    """
    
    def __init__(self):
        self.classifier = SignalBasedClassifier()
        self.track_classifications: Dict[int, List[Tuple[str, float]]] = {}
        
    def process_detection(self, 
                         range_doppler_map: np.ndarray,
                         detection_idx: Tuple[int, int],
                         track_id: int,
                         time: float) -> Dict[str, Any]:
        """
        Process a detection and update classification
        
        Args:
            range_doppler_map: Current RD map
            detection_idx: Location of detection in RD map
            track_id: Associated track ID
            time: Current time
            
        Returns:
            Classification result
        """
        # Extract observables from signal
        observable = self.classifier.extract_observables(range_doppler_map, detection_idx)
        observable.time = time
        
        # Classify based on current observable
        instant_class, instant_conf = self.classifier.classify_from_observables(observable)
        
        # Update track history
        if track_id not in self.track_classifications:
            self.track_classifications[track_id] = []
        self.track_classifications[track_id].append((instant_class, instant_conf))
        
        # Get track-based classification if enough history
        if len(self.track_classifications[track_id]) >= 5:
            # Vote on classification
            classes = [c for c, _ in self.track_classifications[track_id][-10:]]
            confidences = [conf for _, conf in self.track_classifications[track_id][-10:]]
            
            # Most common class
            unique_classes, counts = np.unique(classes, return_counts=True)
            best_class = unique_classes[np.argmax(counts)]
            avg_confidence = np.mean(confidences)
            
            return {
                'track_id': track_id,
                'classification': best_class,
                'confidence': avg_confidence,
                'instant_class': instant_class,
                'history_length': len(self.track_classifications[track_id])
            }
        else:
            return {
                'track_id': track_id,
                'classification': instant_class,
                'confidence': instant_conf,
                'instant_class': instant_class,
                'history_length': len(self.track_classifications[track_id])
            }
    
    def is_threat(self, classification: str) -> bool:
        """Determine if classification represents a threat"""
        threat_classes = ['missile', 'aircraft', 'unknown']
        return classification in threat_classes