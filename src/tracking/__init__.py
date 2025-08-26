"""
Radar target tracking module

This module provides comprehensive target tracking capabilities for radar systems,
including various motion models, coordinate transformations, utility functions
for Extended Kalman Filter (EKF) implementations, and advanced integrated tracking systems.

Available motion models:
- Constant Velocity (CV) - Linear motion with constant velocity
- Constant Acceleration (CA) - Motion with constant acceleration
- Coordinated Turn (CT) - Turning motion with constant turn rate
- Singer Acceleration - Adaptive acceleration model with maneuver detection

Coordinate systems supported:
- Cartesian 2D/3D
- Polar (range, azimuth)
- Spherical (range, azimuth, elevation)

Advanced tracking systems:
- IMM-JPDA - Interacting Multiple Model with Joint Probabilistic Data Association
- IMM-MHT - Interacting Multiple Model with Multiple Hypothesis Tracking
- Sensor fusion and out-of-sequence measurement handling
- Comprehensive performance metrics and evaluation tools
"""

try:
    from .motion_models import (
        # Base classes and enums
        MotionModel,
        CoordinateSystem,
        ModelParameters,
        
        # Motion model implementations
        ConstantVelocityModel,
        ConstantAccelerationModel,
        CoordinatedTurnModel,
        SingerAccelerationModel,
        
        # Utility functions
        cartesian_to_polar,
        polar_to_cartesian,
        cartesian_to_spherical,
        spherical_to_cartesian,
        compute_turn_rate,
        estimate_adaptive_process_noise,
        get_measurement_jacobian,
        create_motion_model,
    )
    _motion_models_available = True
except ImportError:
    _motion_models_available = False

from .kalman_filters import (
    BaseKalmanFilter,
    KalmanFilter,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    initialize_constant_velocity_filter,
    initialize_constant_acceleration_filter,
    adaptive_noise_estimation,
    predict_multiple_steps,
    compute_nees,
    compute_nis
)

from .imm_filter import (
    IMMFilter,
    IMMParameters,
    ModelState,
    create_imm_filter,
    create_standard_imm_cv_ca_ct,
    analyze_model_switching,
    compute_model_likelihood_ratios,
    demonstrate_imm_maneuvering_target
)

try:
    from .tracker_base import (
        BaseTracker,
        Track,
        Measurement,
        TrackState,
        TrackingMetrics
    )
    _tracker_base_available = True
except ImportError:
    _tracker_base_available = False

try:
    from .track_initiation import (
        TrackInitiator,
        TrackBeforeDetect,
        SensorType,
        TrackStatus,
        create_default_track_initiator
    )
    _track_initiation_available = True
except ImportError:
    _track_initiation_available = False

try:
    from .integrated_trackers import (
        # Core tracking classes
        JPDATracker,
        MHTTracker,
        InteractingMultipleModel,
        
        # Configuration and data structures
        TrackingConfiguration,
        ModelSet,
        AssociationMethod,
        Association,
        Hypothesis,
        HypothesisStatus,
        
        # Utility functions
        create_default_model_set,
        create_tracking_configuration,
        create_sample_scenario,
        
        # Advanced features
        SensorFusionManager,
        PerformanceMetrics
    )
    _integrated_trackers_available = True
except ImportError:
    _integrated_trackers_available = False

try:
    from .resource_aware_tracker import (
        # Resource-aware tracking classes
        ResourceAwareTracker,
        TrackQuality,
        ResourceRequest,
        TrackResourceState,
        QualityDegradationModel,
        
        # Utility functions
        create_resource_aware_tracker
    )
    _resource_aware_available = True
except ImportError:
    _resource_aware_available = False

# Build __all__ list dynamically based on available modules
__all__ = [
    # Kalman Filter implementations (always available)
    'BaseKalmanFilter',
    'KalmanFilter',
    'ExtendedKalmanFilter',
    'UnscentedKalmanFilter',
    'initialize_constant_velocity_filter',
    'initialize_constant_acceleration_filter',
    'adaptive_noise_estimation',
    'predict_multiple_steps',
    'compute_nees',
    'compute_nis',
    
    # IMM Filter implementations (always available)
    'IMMFilter',
    'IMMParameters',
    'ModelState',
    'create_imm_filter',
    'create_standard_imm_cv_ca_ct',
    'analyze_model_switching',
    'compute_model_likelihood_ratios',
    'demonstrate_imm_maneuvering_target',
]

# Add motion model exports if available
if _motion_models_available:
    __all__.extend([
        # Base classes and enums
        'MotionModel',
        'CoordinateSystem',
        'ModelParameters',
        
        # Motion model implementations
        'ConstantVelocityModel',
        'ConstantAccelerationModel',
        'CoordinatedTurnModel', 
        'SingerAccelerationModel',
        
        # Utility functions
        'cartesian_to_polar',
        'polar_to_cartesian',
        'cartesian_to_spherical',
        'spherical_to_cartesian',
        'compute_turn_rate',
        'estimate_adaptive_process_noise',
        'get_measurement_jacobian',
        'create_motion_model',
    ])

# Add tracker base exports if available
if _tracker_base_available:
    __all__.extend([
        'BaseTracker',
        'Track',
        'Measurement',
        'TrackState',
        'TrackingMetrics',
    ])

# Add track initiation exports if available  
if _track_initiation_available:
    __all__.extend([
        'TrackInitiator',
        'TrackBeforeDetect',
        'SensorType',
        'TrackStatus',
        'create_default_track_initiator',
    ])

# Add integrated tracker exports if available
if _integrated_trackers_available:
    __all__.extend([
        # Core tracking classes
        'JPDATracker',
        'MHTTracker',
        'InteractingMultipleModel',
        
        # Configuration and data structures
        'TrackingConfiguration',
        'ModelSet',
        'AssociationMethod',
        'Association',
        'Hypothesis',
        'HypothesisStatus',
        
        # Utility functions
        'create_default_model_set',
        'create_tracking_configuration',
        'create_sample_scenario',
        
        # Advanced features
        'SensorFusionManager',
        'PerformanceMetrics',
    ])

# Add resource-aware tracker exports if available
if _resource_aware_available:
    __all__.extend([
        # Resource-aware tracking classes
        'ResourceAwareTracker',
        'TrackQuality',
        'ResourceRequest',
        'TrackResourceState',
        'QualityDegradationModel',
        
        # Utility functions
        'create_resource_aware_tracker',
    ])

__version__ = "1.0.0"