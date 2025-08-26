"""
Main simulation orchestrator for radar tracking integration.

This module coordinates all radar simulation components in proper sequence,
manages timing and synchronization, processes complete CPIs, and ensures
no ground truth leakage to the tracking system.
"""

import numpy as np
import numpy.typing as npt
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import time
import warnings
from pathlib import Path
import json

from radar import Radar, RadarParameters
from target import Target
from environment import Environment
from waveforms import WaveformGenerator, WaveformParameters, WaveformType
from iq_generator import IQDataGenerator, IQParameters
from tracking.tracker_base import Measurement, BaseTracker
from .detection import CFARDetector, CFARParameters, MeasurementExtractor, DetectionResult


@dataclass
class RadarSimulationParameters:
    """Parameters for radar simulation"""
    # Timing parameters
    coherent_processing_interval: float = 0.1  # CPI duration in seconds
    pulse_repetition_frequency: float = 1000   # PRF in Hz
    simulation_duration: float = 10.0          # Total simulation time in seconds
    
    # Processing parameters
    range_fft_size: int = 1024                 # Range FFT size
    doppler_fft_size: int = 128                # Doppler FFT size
    range_window: str = "hamming"              # Range processing window
    doppler_window: str = "hann"               # Doppler processing window
    
    # Detection parameters
    enable_detection: bool = True              # Enable CFAR detection
    enable_tracking: bool = True               # Enable tracking integration
    
    # Data storage
    save_intermediate_products: bool = False   # Save range-Doppler maps, etc.
    output_directory: Optional[str] = None     # Output directory for data
    
    # Performance parameters
    max_targets_per_cpi: int = 100            # Maximum targets to process per CPI
    processing_gain_db: float = 20.0          # Processing gain from pulse integration


@dataclass
class SimulationState:
    """Current state of the simulation"""
    current_time: float = 0.0
    cpi_count: int = 0
    total_detections: int = 0
    total_measurements: int = 0
    processing_times: List[float] = field(default_factory=list)
    range_doppler_maps: List[npt.NDArray] = field(default_factory=list)
    detection_lists: List[List[DetectionResult]] = field(default_factory=list)
    measurement_lists: List[List[Measurement]] = field(default_factory=list)


class RadarSimulator:
    """
    Main radar simulation orchestrator.
    
    Coordinates all components: waveform generation, IQ data generation,
    signal processing, CFAR detection, and measurement extraction.
    Ensures proper timing and no ground truth leakage.
    """
    
    def __init__(self,
                 radar_params: RadarParameters,
                 waveform_params: WaveformParameters,
                 sim_params: RadarSimulationParameters,
                 cfar_params: Optional[CFARParameters] = None,
                 iq_params: Optional[IQParameters] = None):
        """
        Initialize radar simulator.
        
        Args:
            radar_params: Radar system parameters
            waveform_params: Waveform generation parameters
            sim_params: Simulation parameters
            cfar_params: CFAR detection parameters (optional)
            iq_params: IQ generation parameters (optional)
        """
        self.radar_params = radar_params
        self.waveform_params = waveform_params
        self.sim_params = sim_params
        self.cfar_params = cfar_params or CFARParameters()
        self.iq_params = iq_params or IQParameters()
        
        # Initialize components
        self._initialize_components()
        
        # Simulation state
        self.state = SimulationState()
        
        # Performance metrics
        self.metrics = {
            'total_cpi_processed': 0,
            'average_processing_time': 0.0,
            'detection_rate': 0.0,
            'false_alarm_rate': 0.0,
            'measurement_extraction_rate': 0.0
        }
        
        # Callbacks for external integration
        self.measurement_callback: Optional[Callable[[List[Measurement], float], None]] = None
        self.cpi_callback: Optional[Callable[[npt.NDArray, float], None]] = None
        
    def _initialize_components(self) -> None:
        """Initialize all simulation components."""
        # Core radar
        self.radar = Radar(self.radar_params)
        
        # Waveform generator
        self.waveform_generator = WaveformGenerator(self.waveform_params)
        
        # IQ data generator
        self.iq_generator = IQDataGenerator(
            self.radar, 
            self.waveform_generator,
            self.iq_params
        )
        
        # CFAR detector
        if self.sim_params.enable_detection:
            range_resolution = 3e8 / (2 * self.waveform_params.bandwidth)
            max_velocity = self.radar_params.wavelength * self.radar_params.prf / 4
            velocity_resolution = 2 * max_velocity / self.sim_params.doppler_fft_size
            
            self.cfar_detector = CFARDetector(
                self.cfar_params,
                range_resolution,
                velocity_resolution,
                self.radar_params.wavelength
            )
        
        # Measurement extractor
        if self.sim_params.enable_tracking:
            self.measurement_extractor = MeasurementExtractor()
            
        # Setup output directory
        if self.sim_params.save_intermediate_products and self.sim_params.output_directory:
            self.output_path = Path(self.sim_params.output_directory)
            self.output_path.mkdir(parents=True, exist_ok=True)
    
    def run_simulation(self,
                      targets: List[Target],
                      environment: Optional[Environment] = None,
                      tracker: Optional[BaseTracker] = None,
                      waveform_type: WaveformType = WaveformType.LFM,
                      **waveform_kwargs) -> Dict[str, Any]:
        """
        Run complete radar simulation.
        
        Args:
            targets: List of targets to simulate
            environment: Environmental conditions
            tracker: Optional tracker for real-time tracking
            waveform_type: Waveform type to use
            **waveform_kwargs: Additional waveform parameters
            
        Returns:
            Dictionary containing simulation results and metrics
        """
        print(f"Starting radar simulation...")
        print(f"Duration: {self.sim_params.simulation_duration}s")
        print(f"CPI: {self.sim_params.coherent_processing_interval}s")
        print(f"Targets: {len(targets)}")
        
        # Reset simulation state
        self.state = SimulationState()
        start_time = time.time()
        
        # Calculate simulation parameters
        cpi_duration = self.sim_params.coherent_processing_interval
        total_cpis = int(self.sim_params.simulation_duration / cpi_duration)
        
        # Main simulation loop
        for cpi_idx in range(total_cpis):
            cpi_start_time = time.time()
            
            # Update simulation time
            self.state.current_time = cpi_idx * cpi_duration
            self.state.cpi_count = cpi_idx
            
            try:
                # Process one CPI
                measurements = self._process_cpi(
                    targets, environment, waveform_type, **waveform_kwargs
                )
                
                # Update tracker if provided
                if tracker is not None and measurements:
                    tracker.predict(self.state.current_time)
                    tracker.update(measurements)
                    tracker.manage_tracks()
                
                # Call measurement callback if registered
                if self.measurement_callback is not None:
                    self.measurement_callback(measurements, self.state.current_time)
                
                # Update metrics
                self.state.total_measurements += len(measurements)
                
            except Exception as e:
                warnings.warn(f"Error processing CPI {cpi_idx}: {e}")
                continue
            
            # Record processing time
            cpi_processing_time = time.time() - cpi_start_time
            self.state.processing_times.append(cpi_processing_time)
            
            # Progress reporting
            if (cpi_idx + 1) % max(1, total_cpis // 10) == 0:
                progress = (cpi_idx + 1) / total_cpis * 100
                print(f"Progress: {progress:.1f}% (CPI {cpi_idx + 1}/{total_cpis})")
        
        # Finalize simulation
        total_time = time.time() - start_time
        results = self._finalize_simulation(total_time)
        
        print(f"Simulation completed in {total_time:.2f}s")
        print(f"Total detections: {self.state.total_detections}")
        print(f"Total measurements: {self.state.total_measurements}")
        
        return results
    
    def _process_cpi(self,
                    targets: List[Target],
                    environment: Optional[Environment],
                    waveform_type: WaveformType,
                    **waveform_kwargs) -> List[Measurement]:
        """
        Process one Coherent Processing Interval.
        
        Args:
            targets: List of targets
            environment: Environmental conditions
            waveform_type: Waveform type
            **waveform_kwargs: Waveform parameters
            
        Returns:
            List of measurements extracted from this CPI
        """
        # Calculate number of pulses in CPI
        pri = 1.0 / self.radar_params.prf
        num_pulses = int(self.sim_params.coherent_processing_interval / pri)
        
        # Generate IQ data for CPI
        iq_data = self.iq_generator.generate_cpi(
            targets=targets,
            num_pulses=num_pulses,
            pri=pri,
            waveform_type=waveform_type,
            environment=environment,
            add_clutter=True,
            **waveform_kwargs
        )
        
        # Signal processing: Range-Doppler processing
        range_doppler_map = self._range_doppler_processing(iq_data)
        
        # Store intermediate products if requested
        if self.sim_params.save_intermediate_products:
            self.state.range_doppler_maps.append(range_doppler_map)
            
        # Call CPI callback if registered
        if self.cpi_callback is not None:
            self.cpi_callback(range_doppler_map, self.state.current_time)
        
        # CFAR detection
        measurements = []
        if self.sim_params.enable_detection:
            detections = self._perform_cfar_detection(range_doppler_map)
            self.state.total_detections += len(detections)
            
            if self.sim_params.save_intermediate_products:
                self.state.detection_lists.append(detections)
            
            # Extract measurements for tracking
            if self.sim_params.enable_tracking and detections:
                measurements = self._extract_measurements(detections)
                
                if self.sim_params.save_intermediate_products:
                    self.state.measurement_lists.append(measurements)
        
        return measurements
    
    def _range_doppler_processing(self, iq_data: npt.NDArray[np.complex64]) -> npt.NDArray[np.float64]:
        """
        Perform range-Doppler processing on IQ data.
        
        Args:
            iq_data: Complex IQ data (fast_time x slow_time)
            
        Returns:
            Range-Doppler magnitude map
        """
        # Apply windowing
        if self.sim_params.range_window:
            range_window = self._get_window(self.sim_params.range_window, iq_data.shape[0])
            iq_data = iq_data * range_window[:, np.newaxis]
        
        # Range FFT (fast-time)
        range_fft = np.fft.fft(iq_data, n=self.sim_params.range_fft_size, axis=0)
        
        # Apply Doppler windowing
        if self.sim_params.doppler_window:
            doppler_window = self._get_window(self.sim_params.doppler_window, iq_data.shape[1])
            range_fft = range_fft * doppler_window[np.newaxis, :]
        
        # Doppler FFT (slow-time)
        range_doppler = np.fft.fftshift(
            np.fft.fft(range_fft, n=self.sim_params.doppler_fft_size, axis=1),
            axes=1
        )
        
        # Calculate magnitude with processing gain
        rd_magnitude = np.abs(range_doppler)
        
        # Apply processing gain
        processing_gain_linear = 10**(self.sim_params.processing_gain_db / 10)
        rd_magnitude *= np.sqrt(processing_gain_linear)
        
        return rd_magnitude
    
    def _get_window(self, window_type: str, length: int) -> npt.NDArray[np.float64]:
        """Get window function."""
        if window_type == "hamming":
            return np.hamming(length)
        elif window_type == "hann":
            return np.hann(length)
        elif window_type == "blackman":
            return np.blackman(length)
        elif window_type == "kaiser":
            return np.kaiser(length, 5)
        else:
            return np.ones(length)  # Rectangular window
    
    def _perform_cfar_detection(self, range_doppler_map: npt.NDArray[np.float64]) -> List[DetectionResult]:
        """Perform CFAR detection on range-Doppler map."""
        # Generate range and velocity bins
        c = 3e8
        max_range = c * self.sim_params.range_fft_size / (2 * self.waveform_params.sample_rate)
        range_bins = np.linspace(0, max_range, self.sim_params.range_fft_size)
        
        max_velocity = self.radar_params.wavelength * self.radar_params.prf / 4
        velocity_bins = np.linspace(-max_velocity, max_velocity, self.sim_params.doppler_fft_size)
        
        # Perform CFAR detection
        detections = self.cfar_detector.detect(
            range_doppler_map[:len(range_bins), :len(velocity_bins)],
            range_bins[:len(range_bins)],
            velocity_bins
        )
        
        return detections
    
    def _extract_measurements(self, detections: List[DetectionResult]) -> List[Measurement]:
        """Extract tracking measurements from detections."""
        measurements = self.measurement_extractor.extract_measurements(
            detections,
            self.state.current_time,
            beam_azimuth=0.0,  # Assume boresight pointing
            beam_elevation=0.0,
            antenna_pattern_width=(np.radians(3.0), np.radians(3.0))  # 3-degree beamwidth
        )
        
        return measurements
    
    def _finalize_simulation(self, total_time: float) -> Dict[str, Any]:
        """Finalize simulation and compile results."""
        # Calculate metrics
        if self.state.processing_times:
            self.metrics['average_processing_time'] = np.mean(self.state.processing_times)
        
        self.metrics['total_cpi_processed'] = self.state.cpi_count
        
        # Detection statistics
        if hasattr(self.cfar_detector, 'get_detection_statistics'):
            detection_stats = self.cfar_detector.get_detection_statistics()
            self.metrics.update(detection_stats)
        
        # Compile results
        results = {
            'simulation_parameters': {
                'duration': self.sim_params.simulation_duration,
                'cpi_duration': self.sim_params.coherent_processing_interval,
                'total_cpis': self.state.cpi_count,
                'waveform_type': 'radar_waveform'
            },
            'performance_metrics': self.metrics,
            'simulation_state': {
                'total_detections': self.state.total_detections,
                'total_measurements': self.state.total_measurements,
                'processing_times': self.state.processing_times,
                'simulation_time': total_time
            }
        }
        
        # Add intermediate products if saved
        if self.sim_params.save_intermediate_products:
            results['intermediate_products'] = {
                'range_doppler_maps': self.state.range_doppler_maps,
                'detection_lists': self.state.detection_lists,
                'measurement_lists': self.state.measurement_lists
            }
            
            # Save to file if output directory specified
            if self.sim_params.output_directory:
                self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save simulation results to files."""
        if not self.output_path:
            return
        
        # Save summary as JSON (excluding large arrays)
        summary = {
            'simulation_parameters': results['simulation_parameters'],
            'performance_metrics': results['performance_metrics'],
            'simulation_state': {
                k: v for k, v in results['simulation_state'].items() 
                if not isinstance(v, list) or len(v) < 100
            }
        }
        
        summary_file = self.output_path / 'simulation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save large arrays as numpy files
        if 'intermediate_products' in results:
            products = results['intermediate_products']
            
            if products['range_doppler_maps']:
                rd_file = self.output_path / 'range_doppler_maps.npz'
                np.savez_compressed(rd_file, *products['range_doppler_maps'])
            
            # Save processing times
            if self.state.processing_times:
                times_file = self.output_path / 'processing_times.npy'
                np.save(times_file, np.array(self.state.processing_times))
        
        print(f"Results saved to {self.output_path}")
    
    def register_measurement_callback(self, callback: Callable[[List[Measurement], float], None]) -> None:
        """Register callback for real-time measurement processing."""
        self.measurement_callback = callback
    
    def register_cpi_callback(self, callback: Callable[[npt.NDArray, float], None]) -> None:
        """Register callback for real-time CPI processing."""
        self.cpi_callback = callback
    
    def get_simulation_state(self) -> SimulationState:
        """Get current simulation state."""
        return self.state
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.metrics.copy()
    
    def process_single_cpi(self,
                          targets: List[Target],
                          environment: Optional[Environment] = None,
                          waveform_type: WaveformType = WaveformType.LFM,
                          **waveform_kwargs) -> Tuple[npt.NDArray, List[DetectionResult], List[Measurement]]:
        """
        Process a single CPI and return all intermediate products.
        
        Useful for testing and debugging.
        
        Args:
            targets: List of targets
            environment: Environmental conditions
            waveform_type: Waveform type
            **waveform_kwargs: Waveform parameters
            
        Returns:
            Tuple of (range_doppler_map, detections, measurements)
        """
        # Generate IQ data
        pri = 1.0 / self.radar_params.prf
        num_pulses = int(self.sim_params.coherent_processing_interval / pri)
        
        iq_data = self.iq_generator.generate_cpi(
            targets=targets,
            num_pulses=num_pulses,
            pri=pri,
            waveform_type=waveform_type,
            environment=environment,
            **waveform_kwargs
        )
        
        # Range-Doppler processing
        rd_map = self._range_doppler_processing(iq_data)
        
        # CFAR detection
        detections = []
        if self.sim_params.enable_detection:
            detections = self._perform_cfar_detection(rd_map)
        
        # Measurement extraction
        measurements = []
        if self.sim_params.enable_tracking and detections:
            measurements = self._extract_measurements(detections)
        
        return rd_map, detections, measurements
    
    def estimate_processing_requirements(self,
                                       num_targets: int,
                                       simulation_duration: float) -> Dict[str, Any]:
        """
        Estimate computational requirements for simulation.
        
        Args:
            num_targets: Number of targets to simulate
            simulation_duration: Duration in seconds
            
        Returns:
            Dictionary with estimated requirements
        """
        # Calculate basic parameters
        total_cpis = int(simulation_duration / self.sim_params.coherent_processing_interval)
        pri = 1.0 / self.radar_params.prf
        pulses_per_cpi = int(self.sim_params.coherent_processing_interval / pri)
        
        # Estimate memory requirements
        samples_per_pulse = self.waveform_params.num_samples
        iq_data_size = samples_per_pulse * pulses_per_cpi * 8  # Complex64 = 8 bytes
        rd_map_size = self.sim_params.range_fft_size * self.sim_params.doppler_fft_size * 8
        
        # Estimate computation
        range_ffts = total_cpis * pulses_per_cpi * np.log2(self.sim_params.range_fft_size)
        doppler_ffts = total_cpis * self.sim_params.range_fft_size * np.log2(self.sim_params.doppler_fft_size)
        cfar_operations = total_cpis * self.sim_params.range_fft_size * self.sim_params.doppler_fft_size
        
        return {
            'total_cpis': total_cpis,
            'pulses_per_cpi': pulses_per_cpi,
            'memory_per_cpi': {
                'iq_data_mb': iq_data_size / 1e6,
                'rd_map_mb': rd_map_size / 1e6
            },
            'computational_complexity': {
                'range_ffts': range_ffts,
                'doppler_ffts': doppler_ffts,
                'cfar_operations': cfar_operations,
                'total_operations': range_ffts + doppler_ffts + cfar_operations
            },
            'estimated_detections': total_cpis * num_targets * 0.8,  # Assume 80% detection rate
            'estimated_processing_time': total_cpis * 0.01  # Rough estimate
        }