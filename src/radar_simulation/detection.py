"""
CFAR detection and measurement extraction for radar simulation.

This module implements multiple CFAR (Constant False Alarm Rate) algorithms for 
detecting targets in range-Doppler maps and converting detections to tracking-compatible 
measurements. Critical constraint: Only uses detection statistics, no access to true positions.
"""

import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings

from src.tracking.tracker_base import Measurement


class CFARType(Enum):
    """CFAR detector types"""
    CA_CFAR = "ca_cfar"      # Cell Averaging CFAR
    OS_CFAR = "os_cfar"      # Ordered Statistics CFAR 
    GO_CFAR = "go_cfar"      # Greatest Of CFAR
    SO_CFAR = "so_cfar"      # Smallest Of CFAR
    ACCA_CFAR = "acca_cfar"  # Adaptive Cell CFAR


@dataclass
class CFARParameters:
    """Parameters for CFAR detection"""
    num_training_cells: int = 16    # Number of training cells (one side)
    num_guard_cells: int = 2        # Number of guard cells (one side)
    false_alarm_rate: float = 1e-6  # Probability of false alarm
    cfar_type: CFARType = CFARType.CA_CFAR
    os_index: Optional[int] = None  # Index for OS-CFAR (None = auto-calculate)
    min_detection_snr: float = 10.0  # Minimum SNR for detection (dB)
    max_detections_per_cell: int = 1  # Maximum detections per range-Doppler cell


@dataclass
class DetectionResult:
    """Single detection result with measurement uncertainties"""
    range_bin: int                    # Range bin index
    doppler_bin: int                  # Doppler bin index
    range_m: float                    # Estimated range in meters
    velocity_ms: float                # Estimated radial velocity in m/s
    snr_db: float                     # Signal-to-noise ratio in dB
    magnitude: float                  # Detection magnitude
    range_std: float                  # Range uncertainty (standard deviation)
    velocity_std: float               # Velocity uncertainty (standard deviation)
    azimuth_rad: Optional[float] = None     # Azimuth estimate (if available)
    elevation_rad: Optional[float] = None   # Elevation estimate (if available)
    azimuth_std: Optional[float] = None     # Azimuth uncertainty
    elevation_std: Optional[float] = None   # Elevation uncertainty
    metadata: Dict[str, Any] = None   # Additional detection metadata
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CFARDetector:
    """
    CFAR detector for radar range-Doppler maps.
    
    Implements multiple CFAR algorithms with measurement uncertainty estimation
    based on detection SNR and system parameters.
    """
    
    def __init__(self, 
                 cfar_params: CFARParameters,
                 range_resolution: float,
                 velocity_resolution: float,
                 wavelength: float):
        """
        Initialize CFAR detector.
        
        Args:
            cfar_params: CFAR detection parameters
            range_resolution: Range resolution in meters
            velocity_resolution: Velocity resolution in m/s
            wavelength: Radar wavelength in meters
        """
        self.cfar_params = cfar_params
        self.range_resolution = range_resolution
        self.velocity_resolution = velocity_resolution
        self.wavelength = wavelength
        
        # Pre-compute CFAR threshold scale factor
        self._compute_cfar_threshold()
        
        # Initialize detection statistics
        self.detection_stats = {
            'total_detections': 0,
            'false_alarms_estimated': 0,
            'detection_snr_mean': 0.0,
            'detection_snr_std': 0.0
        }
    
    def _compute_cfar_threshold(self) -> None:
        """Compute CFAR threshold scale factor based on false alarm rate."""
        pfa = self.cfar_params.false_alarm_rate
        n_train = self.cfar_params.num_training_cells * 2  # Both sides
        
        if self.cfar_params.cfar_type == CFARType.CA_CFAR:
            # CA-CFAR threshold for complex data
            self.threshold_scale = n_train * (pfa**(-1/n_train) - 1)
            
        elif self.cfar_params.cfar_type == CFARType.OS_CFAR:
            # OS-CFAR requires iterative solution
            if self.cfar_params.os_index is None:
                # Use 3/4 of training cells as default
                self.os_index = int(3 * n_train / 4)
            else:
                self.os_index = self.cfar_params.os_index
            
            # Simplified approximation for OS-CFAR threshold
            self.threshold_scale = self._compute_os_cfar_threshold(n_train, self.os_index, pfa)
            
        elif self.cfar_params.cfar_type == CFARType.GO_CFAR:
            # GO-CFAR uses maximum of two sides
            n_side = n_train // 2
            threshold_side = n_side * (pfa**(-1/n_side) - 1)
            # Scale for taking maximum of two sides
            self.threshold_scale = threshold_side * 1.5
            
        elif self.cfar_params.cfar_type == CFARType.SO_CFAR:
            # SO-CFAR uses minimum of two sides  
            n_side = n_train // 2
            threshold_side = n_side * (pfa**(-1/n_side) - 1)
            # Scale for taking minimum of two sides
            self.threshold_scale = threshold_side * 0.7
            
        else:
            # Default to CA-CFAR
            self.threshold_scale = n_train * (pfa**(-1/n_train) - 1)
    
    def _compute_os_cfar_threshold(self, n_train: int, k: int, pfa: float) -> float:
        """
        Compute OS-CFAR threshold using approximation.
        
        Args:
            n_train: Number of training cells
            k: Order statistic index
            pfa: False alarm rate
            
        Returns:
            Threshold scale factor
        """
        # Approximation for OS-CFAR threshold
        # More accurate methods would require numerical integration
        beta = np.math.factorial(n_train) / (np.math.factorial(k-1) * np.math.factorial(n_train-k))
        alpha = (1 - pfa**(1/beta))
        return k / alpha if alpha > 0 else n_train * 10  # Fallback
    
    def detect(self, 
               range_doppler_map: npt.NDArray[np.float64],
               range_bins: Optional[npt.NDArray[np.float64]] = None,
               doppler_bins: Optional[npt.NDArray[np.float64]] = None) -> List[DetectionResult]:
        """
        Perform CFAR detection on range-Doppler map.
        
        Args:
            range_doppler_map: 2D range-Doppler magnitude map (range x Doppler)
            range_bins: Range bin centers in meters (optional)
            doppler_bins: Doppler bin centers in m/s (optional)
            
        Returns:
            List of detection results with measurement uncertainties
        """
        # Input validation
        if range_doppler_map.ndim != 2:
            raise ValueError("Range-Doppler map must be 2D")
        
        n_range, n_doppler = range_doppler_map.shape
        
        # Generate default bin arrays if not provided
        if range_bins is None:
            range_bins = np.arange(n_range) * self.range_resolution
        if doppler_bins is None:
            doppler_bins = (np.arange(n_doppler) - n_doppler//2) * self.velocity_resolution
        
        # Apply CFAR detection
        if self.cfar_params.cfar_type == CFARType.CA_CFAR:
            detections = self._ca_cfar_2d(range_doppler_map, range_bins, doppler_bins)
        elif self.cfar_params.cfar_type == CFARType.OS_CFAR:
            detections = self._os_cfar_2d(range_doppler_map, range_bins, doppler_bins)
        elif self.cfar_params.cfar_type == CFARType.GO_CFAR:
            detections = self._go_cfar_2d(range_doppler_map, range_bins, doppler_bins)
        elif self.cfar_params.cfar_type == CFARType.SO_CFAR:
            detections = self._so_cfar_2d(range_doppler_map, range_bins, doppler_bins)
        else:
            # Default to CA-CFAR
            detections = self._ca_cfar_2d(range_doppler_map, range_bins, doppler_bins)
        
        # Apply additional filtering
        detections = self._filter_detections(detections)
        
        # Update detection statistics
        self._update_detection_stats(detections)
        
        return detections
    
    def _ca_cfar_2d(self, 
                    rd_map: npt.NDArray[np.float64],
                    range_bins: npt.NDArray[np.float64],
                    doppler_bins: npt.NDArray[np.float64]) -> List[DetectionResult]:
        """Cell Averaging CFAR in 2D."""
        n_range, n_doppler = rd_map.shape
        detections = []
        
        # CFAR window parameters
        n_train = self.cfar_params.num_training_cells
        n_guard = self.cfar_params.num_guard_cells
        
        # Iterate through cells (avoiding edges)
        for r_idx in range(n_guard + n_train, n_range - n_guard - n_train):
            for d_idx in range(n_guard + n_train, n_doppler - n_guard - n_train):
                
                # Extract cell under test
                cut_value = rd_map[r_idx, d_idx]
                
                # Extract training cells (excluding guard cells and CUT)
                training_cells = []
                
                # Range training cells
                for r_offset in range(-n_train-n_guard, -n_guard):
                    training_cells.append(rd_map[r_idx + r_offset, d_idx])
                for r_offset in range(n_guard + 1, n_train + n_guard + 1):
                    training_cells.append(rd_map[r_idx + r_offset, d_idx])
                
                # Doppler training cells  
                for d_offset in range(-n_train-n_guard, -n_guard):
                    training_cells.append(rd_map[r_idx, d_idx + d_offset])
                for d_offset in range(n_guard + 1, n_train + n_guard + 1):
                    training_cells.append(rd_map[r_idx, d_idx + d_offset])
                
                training_cells = np.array(training_cells)
                
                # Calculate noise level (average of training cells)
                noise_level = np.mean(training_cells)
                
                # Calculate threshold
                threshold = noise_level * self.threshold_scale
                
                # Check for detection
                if cut_value > threshold and noise_level > 0:
                    # Calculate SNR
                    snr_linear = cut_value / noise_level
                    snr_db = 10 * np.log10(snr_linear)
                    
                    if snr_db >= self.cfar_params.min_detection_snr:
                        # Create detection
                        detection = self._create_detection(
                            r_idx, d_idx, range_bins[r_idx], doppler_bins[d_idx],
                            snr_db, cut_value, noise_level
                        )
                        detections.append(detection)
        
        return detections
    
    def _os_cfar_2d(self, 
                    rd_map: npt.NDArray[np.float64],
                    range_bins: npt.NDArray[np.float64], 
                    doppler_bins: npt.NDArray[np.float64]) -> List[DetectionResult]:
        """Ordered Statistics CFAR in 2D."""
        n_range, n_doppler = rd_map.shape
        detections = []
        
        n_train = self.cfar_params.num_training_cells
        n_guard = self.cfar_params.num_guard_cells
        
        for r_idx in range(n_guard + n_train, n_range - n_guard - n_train):
            for d_idx in range(n_guard + n_train, n_doppler - n_guard - n_train):
                
                cut_value = rd_map[r_idx, d_idx]
                
                # Extract training cells
                training_cells = []
                
                # Range training cells
                for r_offset in range(-n_train-n_guard, -n_guard):
                    training_cells.append(rd_map[r_idx + r_offset, d_idx])
                for r_offset in range(n_guard + 1, n_train + n_guard + 1):
                    training_cells.append(rd_map[r_idx + r_offset, d_idx])
                
                # Doppler training cells
                for d_offset in range(-n_train-n_guard, -n_guard):
                    training_cells.append(rd_map[r_idx, d_idx + d_offset])
                for d_offset in range(n_guard + 1, n_train + n_guard + 1):
                    training_cells.append(rd_map[r_idx, d_idx + d_offset])
                
                training_cells = np.array(training_cells)
                
                # Sort training cells and select order statistic
                sorted_cells = np.sort(training_cells)
                noise_level = sorted_cells[self.os_index] if self.os_index < len(sorted_cells) else sorted_cells[-1]
                
                # Calculate threshold
                threshold = noise_level * self.threshold_scale
                
                if cut_value > threshold and noise_level > 0:
                    snr_linear = cut_value / noise_level
                    snr_db = 10 * np.log10(snr_linear)
                    
                    if snr_db >= self.cfar_params.min_detection_snr:
                        detection = self._create_detection(
                            r_idx, d_idx, range_bins[r_idx], doppler_bins[d_idx],
                            snr_db, cut_value, noise_level
                        )
                        detections.append(detection)
        
        return detections
    
    def _go_cfar_2d(self, 
                    rd_map: npt.NDArray[np.float64],
                    range_bins: npt.NDArray[np.float64],
                    doppler_bins: npt.NDArray[np.float64]) -> List[DetectionResult]:
        """Greatest Of CFAR in 2D."""
        n_range, n_doppler = rd_map.shape
        detections = []
        
        n_train = self.cfar_params.num_training_cells
        n_guard = self.cfar_params.num_guard_cells
        
        for r_idx in range(n_guard + n_train, n_range - n_guard - n_train):
            for d_idx in range(n_guard + n_train, n_doppler - n_guard - n_train):
                
                cut_value = rd_map[r_idx, d_idx]
                
                # Extract training cells for leading and lagging windows
                leading_cells = []
                lagging_cells = []
                
                # Range dimension
                for r_offset in range(-n_train-n_guard, -n_guard):
                    leading_cells.append(rd_map[r_idx + r_offset, d_idx])
                for r_offset in range(n_guard + 1, n_train + n_guard + 1):
                    lagging_cells.append(rd_map[r_idx + r_offset, d_idx])
                
                # Doppler dimension  
                for d_offset in range(-n_train-n_guard, -n_guard):
                    leading_cells.append(rd_map[r_idx, d_idx + d_offset])
                for d_offset in range(n_guard + 1, n_train + n_guard + 1):
                    lagging_cells.append(rd_map[r_idx, d_idx + d_offset])
                
                # Calculate noise levels for each side
                noise_leading = np.mean(leading_cells) if leading_cells else 0
                noise_lagging = np.mean(lagging_cells) if lagging_cells else 0
                
                # Take maximum (GO-CFAR)
                noise_level = max(noise_leading, noise_lagging)
                
                threshold = noise_level * self.threshold_scale
                
                if cut_value > threshold and noise_level > 0:
                    snr_linear = cut_value / noise_level
                    snr_db = 10 * np.log10(snr_linear)
                    
                    if snr_db >= self.cfar_params.min_detection_snr:
                        detection = self._create_detection(
                            r_idx, d_idx, range_bins[r_idx], doppler_bins[d_idx],
                            snr_db, cut_value, noise_level
                        )
                        detections.append(detection)
        
        return detections
    
    def _so_cfar_2d(self, 
                    rd_map: npt.NDArray[np.float64],
                    range_bins: npt.NDArray[np.float64],
                    doppler_bins: npt.NDArray[np.float64]) -> List[DetectionResult]:
        """Smallest Of CFAR in 2D."""
        n_range, n_doppler = rd_map.shape
        detections = []
        
        n_train = self.cfar_params.num_training_cells
        n_guard = self.cfar_params.num_guard_cells
        
        for r_idx in range(n_guard + n_train, n_range - n_guard - n_train):
            for d_idx in range(n_guard + n_train, n_doppler - n_guard - n_train):
                
                cut_value = rd_map[r_idx, d_idx]
                
                # Extract training cells for leading and lagging windows
                leading_cells = []
                lagging_cells = []
                
                # Range dimension
                for r_offset in range(-n_train-n_guard, -n_guard):
                    leading_cells.append(rd_map[r_idx + r_offset, d_idx])
                for r_offset in range(n_guard + 1, n_train + n_guard + 1):
                    lagging_cells.append(rd_map[r_idx + r_offset, d_idx])
                
                # Doppler dimension
                for d_offset in range(-n_train-n_guard, -n_guard):
                    leading_cells.append(rd_map[r_idx, d_idx + d_offset])
                for d_offset in range(n_guard + 1, n_train + n_guard + 1):
                    lagging_cells.append(rd_map[r_idx, d_idx + d_offset])
                
                # Calculate noise levels for each side
                noise_leading = np.mean(leading_cells) if leading_cells else 0
                noise_lagging = np.mean(lagging_cells) if lagging_cells else 0
                
                # Take minimum (SO-CFAR)
                noise_level = min(noise_leading, noise_lagging) if min(noise_leading, noise_lagging) > 0 else max(noise_leading, noise_lagging)
                
                threshold = noise_level * self.threshold_scale
                
                if cut_value > threshold and noise_level > 0:
                    snr_linear = cut_value / noise_level
                    snr_db = 10 * np.log10(snr_linear)
                    
                    if snr_db >= self.cfar_params.min_detection_snr:
                        detection = self._create_detection(
                            r_idx, d_idx, range_bins[r_idx], doppler_bins[d_idx],
                            snr_db, cut_value, noise_level
                        )
                        detections.append(detection)
        
        return detections
    
    def _create_detection(self,
                         range_bin: int,
                         doppler_bin: int, 
                         range_m: float,
                         velocity_ms: float,
                         snr_db: float,
                         magnitude: float,
                         noise_level: float) -> DetectionResult:
        """
        Create detection result with uncertainty estimates.
        
        Args:
            range_bin: Range bin index
            doppler_bin: Doppler bin index
            range_m: Range estimate in meters
            velocity_ms: Velocity estimate in m/s
            snr_db: SNR in dB
            magnitude: Detection magnitude
            noise_level: Estimated noise level
            
        Returns:
            Detection result with uncertainties
        """
        # Calculate measurement uncertainties based on SNR
        # These are theoretical estimates - could be refined with empirical data
        snr_linear = 10**(snr_db / 10)
        
        # Range uncertainty (dominated by thermal noise and quantization)
        # Cramer-Rao lower bound approximation
        range_std = self.range_resolution / (2 * np.sqrt(2 * snr_linear))
        
        # Velocity uncertainty 
        velocity_std = self.velocity_resolution / (2 * np.sqrt(2 * snr_linear))
        
        # Additional metadata
        metadata = {
            'cfar_type': self.cfar_params.cfar_type.value,
            'noise_level': noise_level,
            'threshold_scale': self.threshold_scale,
            'range_resolution': self.range_resolution,
            'velocity_resolution': self.velocity_resolution
        }
        
        return DetectionResult(
            range_bin=range_bin,
            doppler_bin=doppler_bin,
            range_m=range_m,
            velocity_ms=velocity_ms,
            snr_db=snr_db,
            magnitude=magnitude,
            range_std=range_std,
            velocity_std=velocity_std,
            metadata=metadata
        )
    
    def _filter_detections(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Apply additional filtering to detections."""
        if not detections:
            return detections
        
        # Remove duplicate detections in same cell
        cell_detections = {}
        for detection in detections:
            cell_key = (detection.range_bin, detection.doppler_bin)
            if cell_key not in cell_detections:
                cell_detections[cell_key] = []
            cell_detections[cell_key].append(detection)
        
        # Keep only strongest detection per cell
        filtered_detections = []
        for cell_dets in cell_detections.values():
            if len(cell_dets) <= self.cfar_params.max_detections_per_cell:
                filtered_detections.extend(cell_dets)
            else:
                # Sort by SNR and keep strongest
                cell_dets.sort(key=lambda x: x.snr_db, reverse=True)
                filtered_detections.extend(cell_dets[:self.cfar_params.max_detections_per_cell])
        
        return filtered_detections
    
    def _update_detection_stats(self, detections: List[DetectionResult]) -> None:
        """Update detection statistics."""
        self.detection_stats['total_detections'] += len(detections)
        
        if detections:
            snr_values = [det.snr_db for det in detections]
            self.detection_stats['detection_snr_mean'] = np.mean(snr_values)
            self.detection_stats['detection_snr_std'] = np.std(snr_values)
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return self.detection_stats.copy()


class MeasurementExtractor:
    """
    Extract tracking-compatible measurements from radar detections.
    
    Converts DetectionResult objects to Measurement objects compatible with
    the tracking system, including proper coordinate transformations and
    covariance matrix construction.
    """
    
    def __init__(self,
                 radar_position: npt.NDArray[np.float64] = None,
                 coordinate_system: str = "cartesian"):
        """
        Initialize measurement extractor.
        
        Args:
            radar_position: Radar position [x, y, z] in meters
            coordinate_system: Output coordinate system ("cartesian", "polar")
        """
        self.radar_position = radar_position if radar_position is not None else np.zeros(3)
        self.coordinate_system = coordinate_system
        
    def extract_measurements(self,
                           detections: List[DetectionResult],
                           timestamp: float,
                           beam_azimuth: Optional[float] = None,
                           beam_elevation: Optional[float] = None,
                           antenna_pattern_width: Optional[Tuple[float, float]] = None) -> List[Measurement]:
        """
        Convert detections to tracking measurements.
        
        Args:
            detections: List of detection results
            timestamp: Measurement timestamp
            beam_azimuth: Beam center azimuth in radians (for beamforming radars)
            beam_elevation: Beam center elevation in radians
            antenna_pattern_width: (azimuth_width, elevation_width) in radians
            
        Returns:
            List of tracking-compatible measurements
        """
        measurements = []
        
        for detection in detections:
            try:
                measurement = self._convert_detection_to_measurement(
                    detection, timestamp, beam_azimuth, beam_elevation, antenna_pattern_width
                )
                measurements.append(measurement)
            except Exception as e:
                warnings.warn(f"Failed to convert detection to measurement: {e}")
                continue
        
        return measurements
    
    def _convert_detection_to_measurement(self,
                                        detection: DetectionResult,
                                        timestamp: float,
                                        beam_azimuth: Optional[float],
                                        beam_elevation: Optional[float],
                                        antenna_pattern_width: Optional[Tuple[float, float]]) -> Measurement:
        """Convert single detection to measurement."""
        
        # Extract range and range-rate
        range_m = detection.range_m
        range_rate_ms = detection.velocity_ms
        
        # Estimate azimuth and elevation if not provided
        if detection.azimuth_rad is not None:
            azimuth = detection.azimuth_rad
            azimuth_std = detection.azimuth_std or 0.1  # Default 0.1 rad uncertainty
        else:
            # Use beam center or default
            azimuth = beam_azimuth if beam_azimuth is not None else 0.0
            # Angular uncertainty based on antenna beamwidth
            if antenna_pattern_width is not None:
                azimuth_std = antenna_pattern_width[0] / 4.0  # Conservative estimate
            else:
                azimuth_std = np.radians(5.0)  # Default 5 degree uncertainty
        
        if detection.elevation_rad is not None:
            elevation = detection.elevation_rad  
            elevation_std = detection.elevation_std or 0.1
        else:
            elevation = beam_elevation if beam_elevation is not None else 0.0
            if antenna_pattern_width is not None:
                elevation_std = antenna_pattern_width[1] / 4.0
            else:
                elevation_std = np.radians(5.0)
        
        # Convert to Cartesian coordinates
        if self.coordinate_system == "cartesian":
            # Spherical to Cartesian conversion
            x = range_m * np.cos(elevation) * np.cos(azimuth)
            y = range_m * np.cos(elevation) * np.sin(azimuth)
            z = range_m * np.sin(elevation)
            
            # Velocity components (assuming radial velocity only)
            vx = range_rate_ms * np.cos(elevation) * np.cos(azimuth)
            vy = range_rate_ms * np.cos(elevation) * np.sin(azimuth)
            vz = range_rate_ms * np.sin(elevation)
            
            position = np.array([x, y, z]) + self.radar_position
            velocity = np.array([vx, vy, vz])
            
            # Construct covariance matrix in Cartesian coordinates
            covariance = self._construct_cartesian_covariance(
                range_m, azimuth, elevation,
                detection.range_std, azimuth_std, elevation_std
            )
            
        else:
            # Keep in polar coordinates
            position = np.array([range_m, azimuth, elevation])
            velocity = np.array([range_rate_ms, 0.0, 0.0])  # Only radial velocity available
            
            # Covariance in polar coordinates
            covariance = np.diag([detection.range_std**2, azimuth_std**2, elevation_std**2])
        
        # Create measurement object
        measurement = Measurement(
            position=position,
            timestamp=timestamp,
            covariance=covariance,
            velocity=velocity,
            snr=detection.snr_db,
            range_rate=range_rate_ms,
            azimuth=azimuth,
            elevation=elevation,
            metadata={
                'detection_bin': (detection.range_bin, detection.doppler_bin),
                'detection_magnitude': detection.magnitude,
                'cfar_type': detection.metadata.get('cfar_type', 'unknown'),
                'coordinate_system': self.coordinate_system,
                **detection.metadata
            }
        )
        
        return measurement
    
    def _construct_cartesian_covariance(self,
                                      range_m: float,
                                      azimuth: float,
                                      elevation: float,
                                      range_std: float,
                                      azimuth_std: float,
                                      elevation_std: float) -> npt.NDArray[np.float64]:
        """
        Construct covariance matrix in Cartesian coordinates from polar uncertainties.
        
        Uses Jacobian transformation of polar to Cartesian coordinates.
        """
        # Jacobian matrix for spherical to Cartesian transformation
        cos_el, sin_el = np.cos(elevation), np.sin(elevation)
        cos_az, sin_az = np.cos(azimuth), np.sin(azimuth)
        
        # ∂(x,y,z)/∂(r,az,el)
        jacobian = np.array([
            [cos_el * cos_az, -range_m * cos_el * sin_az, -range_m * sin_el * cos_az],  # ∂x/∂(r,az,el)
            [cos_el * sin_az,  range_m * cos_el * cos_az, -range_m * sin_el * sin_az],  # ∂y/∂(r,az,el)
            [sin_el,           0,                          range_m * cos_el]            # ∂z/∂(r,az,el)
        ])
        
        # Polar covariance matrix
        polar_cov = np.diag([range_std**2, azimuth_std**2, elevation_std**2])
        
        # Transform to Cartesian: C_cart = J * C_polar * J^T
        cartesian_cov = jacobian @ polar_cov @ jacobian.T
        
        return cartesian_cov
    
    def estimate_angle_from_monopulse(self,
                                    sum_channel: complex,
                                    diff_channel: complex,
                                    beam_center: float,
                                    k_monopulse: float = 1.6) -> Tuple[float, float]:
        """
        Estimate angle using monopulse technique.
        
        Args:
            sum_channel: Sum channel complex amplitude
            diff_channel: Difference channel complex amplitude  
            beam_center: Beam center angle in radians
            k_monopulse: Monopulse slope constant
            
        Returns:
            Tuple of (estimated_angle, angle_uncertainty)
        """
        if abs(sum_channel) < 1e-10:
            return beam_center, np.radians(10.0)  # Large uncertainty if no signal
        
        # Monopulse ratio
        monopulse_ratio = diff_channel / sum_channel
        
        # Angle estimate
        angle_offset = np.real(monopulse_ratio) / k_monopulse
        estimated_angle = beam_center + angle_offset
        
        # Uncertainty estimate (simplified)
        snr_linear = abs(sum_channel)**2 / (1e-10)  # Simplified SNR
        angle_uncertainty = 1.0 / (k_monopulse * np.sqrt(2 * snr_linear))
        
        return estimated_angle, angle_uncertainty