"""
Radar Simulation Package for Tracking Integration.

This package provides comprehensive radar simulation capabilities specifically designed
for integration with tracking systems. It includes realistic signal processing,
CFAR detection, and measurement extraction while ensuring no ground truth leakage.

Main Components:
- CFARDetector: Multiple CFAR algorithms for target detection
- MeasurementExtractor: Convert detections to tracking-compatible measurements  
- RadarSimulator: Main orchestrator for complete simulation pipeline

Key Features:
- Realistic radar signal processing chain
- Multiple CFAR detection algorithms (CA, OS, GO, SO)
- Measurement uncertainty estimation based on SNR
- No ground truth leakage to tracking system
- Integration with existing tracking components
- Configurable simulation parameters
- Performance metrics and intermediate product storage
"""

from .detection import (
    CFARDetector,
    CFARParameters, 
    CFARType,
    DetectionResult,
    MeasurementExtractor
)

# Import simulator components separately to avoid dependency issues
try:
    from .simulator import (
        RadarSimulator,
        RadarSimulationParameters,
        SimulationState
    )
    _SIMULATOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Simulator not available due to dependency issue: {e}")
    _SIMULATOR_AVAILABLE = False

# Version information
__version__ = "1.0.0"
__author__ = "Radar Simulation Team"

# Default configuration values
DEFAULT_CFAR_PARAMS = CFARParameters(
    num_training_cells=16,
    num_guard_cells=2,
    false_alarm_rate=1e-6,
    cfar_type=CFARType.CA_CFAR,
    min_detection_snr=10.0
)

# Default simulation parameters (only if simulator is available)
if _SIMULATOR_AVAILABLE:
    DEFAULT_SIM_PARAMS = RadarSimulationParameters(
        coherent_processing_interval=0.1,
        pulse_repetition_frequency=1000,
        simulation_duration=10.0,
        range_fft_size=1024,
        doppler_fft_size=128,
        enable_detection=True,
        enable_tracking=True
    )
else:
    DEFAULT_SIM_PARAMS = None

# Convenience functions
def create_default_cfar_detector(range_resolution: float, 
                                velocity_resolution: float,
                                wavelength: float,
                                cfar_type: CFARType = CFARType.CA_CFAR) -> CFARDetector:
    """
    Create CFAR detector with default parameters.
    
    Args:
        range_resolution: Range resolution in meters
        velocity_resolution: Velocity resolution in m/s  
        wavelength: Radar wavelength in meters
        cfar_type: Type of CFAR algorithm
        
    Returns:
        Configured CFAR detector
    """
    params = CFARParameters(cfar_type=cfar_type)
    return CFARDetector(params, range_resolution, velocity_resolution, wavelength)

def create_default_simulator(radar_params,
                           waveform_params,
                           cfar_type: CFARType = CFARType.CA_CFAR):
    """
    Create radar simulator with default parameters.
    
    Args:
        radar_params: Radar system parameters
        waveform_params: Waveform generation parameters
        cfar_type: Type of CFAR algorithm
        
    Returns:
        Configured radar simulator or None if not available
    """
    if not _SIMULATOR_AVAILABLE:
        raise ImportError("RadarSimulator not available due to dependency issues")
        
    sim_params = DEFAULT_SIM_PARAMS
    cfar_params = CFARParameters(cfar_type=cfar_type)
    
    return RadarSimulator(
        radar_params=radar_params,
        waveform_params=waveform_params,
        sim_params=sim_params,
        cfar_params=cfar_params
    )

# Utility functions for measurement processing
def measurements_to_dict(measurements):
    """
    Convert list of measurements to dictionary format.
    
    Args:
        measurements: List of Measurement objects
        
    Returns:
        Dictionary with measurement data
    """
    if not measurements:
        return {
            'positions': [],
            'velocities': [],
            'timestamps': [],
            'snr_values': [],
            'covariances': []
        }
    
    return {
        'positions': [m.position.tolist() for m in measurements],
        'velocities': [m.velocity.tolist() if m.velocity is not None else None for m in measurements],
        'timestamps': [m.timestamp for m in measurements],
        'snr_values': [m.snr for m in measurements if m.snr is not None],
        'covariances': [m.covariance.tolist() for m in measurements],
        'metadata': [m.metadata for m in measurements]
    }

def filter_measurements_by_snr(measurements, min_snr_db: float):
    """
    Filter measurements by minimum SNR.
    
    Args:
        measurements: List of Measurement objects
        min_snr_db: Minimum SNR in dB
        
    Returns:
        Filtered list of measurements
    """
    return [m for m in measurements if m.snr is not None and m.snr >= min_snr_db]

def cluster_detections_by_range(detections, range_threshold: float = 100.0):
    """
    Cluster detections by range proximity.
    
    Args:
        detections: List of DetectionResult objects
        range_threshold: Range clustering threshold in meters
        
    Returns:
        List of detection clusters
    """
    if not detections:
        return []
    
    # Sort by range
    sorted_detections = sorted(detections, key=lambda d: d.range_m)
    
    clusters = []
    current_cluster = [sorted_detections[0]]
    
    for detection in sorted_detections[1:]:
        if detection.range_m - current_cluster[-1].range_m <= range_threshold:
            current_cluster.append(detection)
        else:
            clusters.append(current_cluster)
            current_cluster = [detection]
    
    clusters.append(current_cluster)
    return clusters

# Export all public components
__all__ = [
    # Main classes
    'CFARDetector',
    'MeasurementExtractor', 
    'RadarSimulator',
    
    # Parameter classes
    'CFARParameters',
    'RadarSimulationParameters',
    'SimulationState',
    
    # Data classes
    'DetectionResult',
    
    # Enums
    'CFARType',
    
    # Default configurations
    'DEFAULT_CFAR_PARAMS',
    'DEFAULT_SIM_PARAMS',
    
    # Convenience functions
    'create_default_cfar_detector',
    'create_default_simulator',
    
    # Utility functions
    'measurements_to_dict',
    'filter_measurements_by_snr',
    'cluster_detections_by_range',
    
    # Version info
    '__version__',
    '__author__'
]

# Package-level configuration
import logging

# Set up logging for the package
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Performance and compatibility checks
def _check_dependencies():
    """Check for required dependencies."""
    try:
        import scipy
        import numpy
        logger.debug("All required dependencies are available")
    except ImportError as e:
        logger.warning(f"Missing dependency: {e}")

def _validate_numpy_version():
    """Validate NumPy version for typing compatibility."""
    import numpy as np
    try:
        # Check if numpy typing is available (NumPy >= 1.20)
        import numpy.typing
        logger.debug(f"NumPy {np.__version__} with typing support")
    except ImportError:
        logger.warning(f"NumPy {np.__version__} lacks typing support. Consider upgrading.")

# Run compatibility checks on import
_check_dependencies()
_validate_numpy_version()

# Package initialization message
logger.info(f"Radar Simulation Package v{__version__} initialized")
logger.info("Components: CFAR Detection, Measurement Extraction, Simulation Orchestration")
logger.info("Tracking Integration: Enabled with measurement uncertainty estimation")