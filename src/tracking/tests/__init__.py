"""
Test suite for the tracking system.

This package contains comprehensive tests for all tracking components:
- Kalman filter implementations (KF, EKF, UKF)
- Motion models (CV, CA, CT, Singer)
- Data association algorithms (GNN, JPDA, MHT)
- IMM filter implementation
- Integrated tracking systems

Test Structure:
- test_kalman_filters.py: Tests for all Kalman filter variants
- test_motion_models.py: Tests for motion models and coordinate transforms
- test_association.py: Tests for data association algorithms
- test_imm_filter.py: Tests for IMM filter implementation
- test_integrated_trackers.py: Integration tests for complete systems

To run all tests:
    pytest src/tracking/tests/

To run specific test modules:
    pytest src/tracking/tests/test_kalman_filters.py
    pytest src/tracking/tests/test_motion_models.py -v

To run with coverage:
    pytest src/tracking/tests/ --cov=src.tracking --cov-report=html

Author: RadarSim Project
"""

__all__ = [
    'test_kalman_filters',
    'test_motion_models', 
    'test_association',
    'test_imm_filter',
    'test_integrated_trackers'
]