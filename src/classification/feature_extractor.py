#!/usr/bin/env python3
"""
Feature extraction for radar target classification
Extracts micro-Doppler, RCS, and kinematic features from radar returns
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TargetFeatures:
    """Container for extracted target features"""
    # Kinematic features
    velocity: float
    acceleration: float
    altitude: float
    closing_velocity: float
    turn_rate: float
    
    # RCS features
    rcs_mean: float
    rcs_variance: float
    rcs_max: float
    rcs_min: float
    rcs_fluctuation_rate: float
    
    # Micro-Doppler features
    doppler_spread: float
    doppler_centroid: float
    num_doppler_components: int
    max_doppler_shift: float
    doppler_periodicity: Optional[float] = None
    
    # Spectral features
    spectral_entropy: float = 0.0
    spectral_flux: float = 0.0
    peak_frequency: float = 0.0
    bandwidth: float = 0.0
    
    # Time-domain features
    signal_duration: float = 0.0
    duty_cycle: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert features to numerical vector for ML classification"""
        return np.array([
            self.velocity,
            self.acceleration,
            self.altitude,
            self.closing_velocity,
            self.turn_rate,
            self.rcs_mean,
            self.rcs_variance,
            self.rcs_max,
            self.rcs_min,
            self.rcs_fluctuation_rate,
            self.doppler_spread,
            self.doppler_centroid,
            self.num_doppler_components,
            self.max_doppler_shift,
            self.doppler_periodicity or 0,
            self.spectral_entropy,
            self.spectral_flux,
            self.peak_frequency,
            self.bandwidth
        ])


class FeatureExtractor:
    """Extract classification features from radar data"""
    
    def __init__(self, 
                 sampling_rate: float = 1000,
                 velocity_resolution: float = 1.0,
                 range_resolution: float = 50):
        """
        Initialize feature extractor
        
        Args:
            sampling_rate: Radar sampling rate (Hz)
            velocity_resolution: Doppler velocity resolution (m/s)
            range_resolution: Range resolution (m)
        """
        self.sampling_rate = sampling_rate
        self.velocity_resolution = velocity_resolution
        self.range_resolution = range_resolution
        
    def extract_features(self, 
                        detections: List[Dict],
                        time_window: float = 5.0) -> TargetFeatures:
        """
        Extract features from a sequence of detections
        
        Args:
            detections: List of detection dictionaries with 'range', 'velocity', 'snr', 'time'
            time_window: Time window for feature extraction (seconds)
            
        Returns:
            TargetFeatures object
        """
        if not detections:
            return self._empty_features()
        
        # Extract time series
        times = np.array([d['time'] for d in detections])
        ranges = np.array([d['range'] for d in detections])
        velocities = np.array([d['velocity'] for d in detections])
        snrs = np.array([d.get('snr', 10) for d in detections])
        
        # Calculate kinematic features
        kinematic = self._extract_kinematic_features(times, ranges, velocities)
        
        # Calculate RCS features (using SNR as proxy)
        rcs = self._extract_rcs_features(snrs, times)
        
        # Calculate Doppler features
        doppler = self._extract_doppler_features(velocities, times)
        
        # Calculate spectral features
        spectral = self._extract_spectral_features(velocities, self.sampling_rate)
        
        # Combine all features
        features = TargetFeatures(
            velocity=kinematic['velocity'],
            acceleration=kinematic['acceleration'],
            altitude=kinematic['altitude'],
            closing_velocity=kinematic['closing_velocity'],
            turn_rate=kinematic['turn_rate'],
            rcs_mean=rcs['mean'],
            rcs_variance=rcs['variance'],
            rcs_max=rcs['max'],
            rcs_min=rcs['min'],
            rcs_fluctuation_rate=rcs['fluctuation_rate'],
            doppler_spread=doppler['spread'],
            doppler_centroid=doppler['centroid'],
            num_doppler_components=doppler['num_components'],
            max_doppler_shift=doppler['max_shift'],
            doppler_periodicity=doppler.get('periodicity'),
            spectral_entropy=spectral['entropy'],
            spectral_flux=spectral['flux'],
            peak_frequency=spectral['peak_freq'],
            bandwidth=spectral['bandwidth']
        )
        
        return features
    
    def _extract_kinematic_features(self, times: np.ndarray, 
                                   ranges: np.ndarray,
                                   velocities: np.ndarray) -> Dict:
        """Extract kinematic features from track data"""
        
        # Mean velocity
        mean_velocity = np.mean(np.abs(velocities))
        
        # Acceleration (velocity change rate)
        if len(velocities) > 1:
            dt = np.diff(times)
            dv = np.diff(velocities)
            accelerations = dv / (dt + 1e-10)
            mean_acceleration = np.mean(np.abs(accelerations))
        else:
            mean_acceleration = 0
        
        # Closing velocity (range rate)
        if len(ranges) > 1:
            dr = np.diff(ranges)
            dt = np.diff(times)
            closing_velocity = -np.mean(dr / (dt + 1e-10))
        else:
            closing_velocity = velocities[0] if len(velocities) > 0 else 0
        
        # Turn rate estimation (from velocity changes)
        if len(velocities) > 2:
            velocity_changes = np.diff(velocities)
            turn_rate = np.std(velocity_changes)
        else:
            turn_rate = 0
        
        # Altitude estimation (simplified - would need elevation angle)
        altitude = 1000  # Default assumption
        
        return {
            'velocity': mean_velocity,
            'acceleration': mean_acceleration,
            'altitude': altitude,
            'closing_velocity': closing_velocity,
            'turn_rate': turn_rate
        }
    
    def _extract_rcs_features(self, snrs: np.ndarray, times: np.ndarray) -> Dict:
        """Extract RCS-related features from SNR measurements"""
        
        # Convert SNR to linear scale (proxy for RCS)
        rcs_proxy = 10 ** (snrs / 10)
        
        # Statistical features
        rcs_mean = np.mean(rcs_proxy)
        rcs_variance = np.var(rcs_proxy)
        rcs_max = np.max(rcs_proxy)
        rcs_min = np.min(rcs_proxy)
        
        # Fluctuation rate (zero-crossing rate of detrended signal)
        if len(rcs_proxy) > 3:
            detrended = rcs_proxy - np.mean(rcs_proxy)
            zero_crossings = np.sum(np.diff(np.sign(detrended)) != 0)
            time_span = times[-1] - times[0] if len(times) > 1 else 1
            fluctuation_rate = zero_crossings / time_span
        else:
            fluctuation_rate = 0
        
        return {
            'mean': rcs_mean,
            'variance': rcs_variance,
            'max': rcs_max,
            'min': rcs_min,
            'fluctuation_rate': fluctuation_rate
        }
    
    def _extract_doppler_features(self, velocities: np.ndarray, 
                                 times: np.ndarray) -> Dict:
        """Extract Doppler-related features"""
        
        # Doppler spread
        doppler_spread = np.std(velocities) if len(velocities) > 1 else 0
        
        # Doppler centroid
        doppler_centroid = np.mean(velocities)
        
        # Number of significant Doppler components (using FFT)
        if len(velocities) > 10:
            fft = np.fft.fft(velocities)
            magnitude = np.abs(fft)
            threshold = 0.1 * np.max(magnitude)
            num_components = np.sum(magnitude > threshold)
        else:
            num_components = 1
        
        # Maximum Doppler shift
        max_shift = np.max(np.abs(velocities)) if len(velocities) > 0 else 0
        
        # Check for periodicity (for rotating parts)
        periodicity = self._detect_periodicity(velocities, times)
        
        return {
            'spread': doppler_spread,
            'centroid': doppler_centroid,
            'num_components': num_components,
            'max_shift': max_shift,
            'periodicity': periodicity
        }
    
    def _detect_periodicity(self, signal_data: np.ndarray, 
                           times: np.ndarray) -> Optional[float]:
        """Detect periodicity in signal (for blade rotation, wing beats, etc.)"""
        
        if len(signal_data) < 20:
            return None
        
        # Remove DC component
        signal_centered = signal_data - np.mean(signal_data)
        
        # Compute autocorrelation
        autocorr = np.correlate(signal_centered, signal_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation
        peaks, properties = signal.find_peaks(autocorr, height=0.3*np.max(autocorr))
        
        if len(peaks) > 1:
            # Period is the distance between peaks
            dt = (times[-1] - times[0]) / len(times) if len(times) > 1 else 1
            period = (peaks[1] - peaks[0]) * dt
            frequency = 1 / period if period > 0 else None
            return frequency
        
        return None
    
    def _extract_spectral_features(self, velocities: np.ndarray, 
                                  sampling_rate: float) -> Dict:
        """Extract spectral features from velocity/Doppler data"""
        
        if len(velocities) < 4:
            return {
                'entropy': 0,
                'flux': 0,
                'peak_freq': 0,
                'bandwidth': 0
            }
        
        # Compute power spectrum
        freqs, psd = signal.welch(velocities, fs=sampling_rate, nperseg=min(len(velocities), 64))
        
        # Normalize PSD
        psd_norm = psd / (np.sum(psd) + 1e-10)
        
        # Spectral entropy
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        
        # Spectral flux (rate of change)
        if len(velocities) > 8:
            _, psd2 = signal.welch(velocities[len(velocities)//2:], 
                                  fs=sampling_rate, 
                                  nperseg=min(len(velocities)//2, 32))
            flux = np.sum(np.abs(psd[:len(psd2)] - psd2))
        else:
            flux = 0
        
        # Peak frequency
        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx] if len(freqs) > 0 else 0
        
        # Bandwidth (frequency spread)
        if np.sum(psd) > 0:
            freq_mean = np.sum(freqs * psd) / np.sum(psd)
            bandwidth = np.sqrt(np.sum((freqs - freq_mean)**2 * psd) / np.sum(psd))
        else:
            bandwidth = 0
        
        return {
            'entropy': entropy,
            'flux': flux,
            'peak_freq': peak_freq,
            'bandwidth': bandwidth
        }
    
    def _empty_features(self) -> TargetFeatures:
        """Return empty feature set"""
        return TargetFeatures(
            velocity=0, acceleration=0, altitude=0, closing_velocity=0, turn_rate=0,
            rcs_mean=0, rcs_variance=0, rcs_max=0, rcs_min=0, rcs_fluctuation_rate=0,
            doppler_spread=0, doppler_centroid=0, num_doppler_components=0,
            max_doppler_shift=0, doppler_periodicity=None,
            spectral_entropy=0, spectral_flux=0, peak_frequency=0, bandwidth=0
        )
    
    def extract_micro_doppler_signature(self, 
                                       range_doppler_map: np.ndarray,
                                       target_range_bin: int,
                                       time_sequence: List[np.ndarray]) -> np.ndarray:
        """
        Extract micro-Doppler signature from time series of range-Doppler maps
        
        Args:
            range_doppler_map: Current range-Doppler map
            target_range_bin: Range bin containing target
            time_sequence: List of previous range-Doppler maps
            
        Returns:
            Micro-Doppler spectrogram
        """
        if not time_sequence:
            return np.zeros((32, 32))
        
        # Extract Doppler profile at target range over time
        doppler_profiles = []
        for rd_map in time_sequence[-32:]:  # Last 32 frames
            if target_range_bin < rd_map.shape[0]:
                doppler_profiles.append(rd_map[target_range_bin, :])
        
        if not doppler_profiles:
            return np.zeros((32, 32))
        
        # Stack to create time-Doppler matrix
        micro_doppler = np.array(doppler_profiles).T
        
        # Apply STFT to get micro-Doppler spectrogram
        if micro_doppler.shape[1] > 8:
            spectrograms = []
            for i in range(micro_doppler.shape[0]):
                f, t, Sxx = signal.spectrogram(micro_doppler[i, :], 
                                              fs=self.sampling_rate,
                                              nperseg=min(8, micro_doppler.shape[1]//2))
                spectrograms.append(Sxx)
            
            # Average across Doppler bins
            spectrogram = np.mean(spectrograms, axis=0)
            
            # Resize to standard size
            if spectrogram.shape != (32, 32):
                from scipy.ndimage import zoom
                zoom_factors = (32/spectrogram.shape[0], 32/spectrogram.shape[1])
                spectrogram = zoom(spectrogram, zoom_factors)
        else:
            spectrogram = np.zeros((32, 32))
        
        return spectrogram