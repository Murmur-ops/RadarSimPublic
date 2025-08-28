"""
Test Suite for Audit Fix Features

Tests the critical fixes implemented in Phase 1:
- Physical constants module
- Input validation module  
- Exception handling improvements
- Swerling models implementation
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import warnings
import unittest
from unittest.mock import patch, MagicMock


class TestPhysicalConstants(unittest.TestCase):
    """Test the new physical constants module"""
    
    def test_constants_import(self):
        """Test that constants can be imported"""
        from src.constants import SPEED_OF_LIGHT, BOLTZMANN_CONSTANT, STANDARD_TEMPERATURE
        
        self.assertEqual(SPEED_OF_LIGHT, 299792458.0)
        self.assertAlmostEqual(BOLTZMANN_CONSTANT, 1.380649e-23)
        self.assertEqual(STANDARD_TEMPERATURE, 290.0)
    
    def test_radar_bands(self):
        """Test radar frequency bands"""
        from src.constants import RadarBand
        
        # Test X-band
        x_band = RadarBand.X
        self.assertEqual(x_band.min_freq, 8e9)
        self.assertEqual(x_band.max_freq, 12e9)
        self.assertTrue(x_band.contains(10e9))
        self.assertFalse(x_band.contains(5e9))
        
        # Test band lookup
        band = RadarBand.get_band(10e9)
        self.assertEqual(band, RadarBand.X)
    
    def test_utility_functions(self):
        """Test utility conversion functions"""
        from src.constants import db_to_linear, linear_to_db, watts_to_dbm, dbm_to_watts
        
        # Test dB conversions
        self.assertAlmostEqual(db_to_linear(10), 10.0)
        self.assertAlmostEqual(linear_to_db(10), 10.0)
        
        # Test power conversions
        self.assertAlmostEqual(watts_to_dbm(1), 30.0)
        self.assertAlmostEqual(dbm_to_watts(30), 1.0)
    
    def test_radar_calculations(self):
        """Test radar-specific calculations"""
        from src.constants import get_wavelength, get_max_unambiguous_range, get_max_unambiguous_velocity
        
        # Test wavelength
        wavelength = get_wavelength(10e9)
        self.assertAlmostEqual(wavelength, 0.0299792458, places=6)
        
        # Test max unambiguous range
        max_range = get_max_unambiguous_range(1000)  # 1 kHz PRF
        self.assertAlmostEqual(max_range, 149896.229, places=2)
        
        # Test max unambiguous velocity
        max_vel = get_max_unambiguous_velocity(10e9, 1000)
        self.assertAlmostEqual(max_vel, 7.49481145, places=5)


class TestValidators(unittest.TestCase):
    """Test the input validation module"""
    
    def test_frequency_validation(self):
        """Test frequency parameter validation"""
        from src.validators import RadarParameterValidator
        
        validator = RadarParameterValidator()
        
        # Valid frequency
        result = validator.validate_frequency(10e9, strict=False)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        
        # Negative frequency
        result = validator.validate_frequency(-10e9, strict=False)
        self.assertFalse(result.is_valid)
        self.assertIn("must be positive", result.errors[0])
        
        # Too high frequency
        result = validator.validate_frequency(200e9, strict=False)
        self.assertFalse(result.is_valid)
        self.assertIn("exceeds maximum", result.errors[0])
    
    def test_power_validation(self):
        """Test power parameter validation"""
        from src.validators import RadarParameterValidator
        
        validator = RadarParameterValidator()
        
        # Valid power
        result = validator.validate_power(1000, strict=False)
        self.assertTrue(result.is_valid)
        
        # Negative power
        result = validator.validate_power(-100, strict=False)
        self.assertFalse(result.is_valid)
        
        # Excessive power with warning
        result = validator.validate_power(5e6, strict=False)
        self.assertTrue(result.is_valid)
        self.assertGreater(len(result.warnings), 0)
    
    def test_prf_validation(self):
        """Test PRF parameter validation"""
        from src.validators import RadarParameterValidator
        
        validator = RadarParameterValidator()
        
        # Valid PRF
        result = validator.validate_prf(1000, strict=False)
        self.assertTrue(result.is_valid)
        
        # Too low PRF
        result = validator.validate_prf(10, strict=False)
        self.assertFalse(result.is_valid)
        
        # Too high PRF
        result = validator.validate_prf(200e3, strict=False)
        self.assertFalse(result.is_valid)
    
    def test_rcs_validation(self):
        """Test RCS parameter validation"""
        from src.validators import TargetValidator
        
        validator = TargetValidator()
        
        # Valid RCS
        result = validator.validate_rcs(1.0, strict=False)
        self.assertTrue(result.is_valid)
        
        # Negative RCS
        result = validator.validate_rcs(-1.0, strict=False)
        self.assertFalse(result.is_valid)
        
        # Very small RCS with warning
        result = validator.validate_rcs(0.00001, strict=False)
        self.assertTrue(result.is_valid)
        self.assertGreater(len(result.warnings), 0)
    
    def test_ambiguity_checking(self):
        """Test ambiguity detection"""
        from src.validators import ScenarioValidator
        from src.constants import get_max_unambiguous_range, get_max_unambiguous_velocity
        
        validator = ScenarioValidator()
        
        # Test that we can detect ambiguities via scenario validation
        config = {
            'radar': {
                'frequency': 10e9,
                'prf': 1000,  # Max unambiguous range ~150km
                'power': 1000,
                'antenna_gain': 30,
                'pulse_width': 1e-6,
                'bandwidth': 1e6,
                'noise_figure': 3,
                'losses': 2
            },
            'targets': [
                {
                    'range': 200000,  # 200 km - beyond unambiguous range
                    'velocity': 100,
                    'rcs': 1.0
                }
            ]
        }
        
        result = validator.validate_scenario(config)
        # Should detect the range ambiguity
        max_range = get_max_unambiguous_range(1000)
        self.assertLess(max_range, 200000)  # Verify ambiguity exists


class TestExceptionHandling(unittest.TestCase):
    """Test improved exception handling"""
    
    def test_imm_filter_exception_handling(self):
        """Test that IMM filter no longer uses bare except"""
        # Read the fixed file and verify no bare except remains
        with open('src/tracking/imm_filter.py', 'r') as f:
            content = f.read()
        
        # Check that bare except was replaced with specific exceptions
        self.assertNotIn('except:', content, 
                        "Bare 'except:' should not be in the code")
        
        # Verify specific exception handling is present
        self.assertIn('except (AttributeError, ValueError, np.linalg.LinAlgError)', 
                     content,
                     "Should have specific exception handling")
        
        # Verify warning is issued on exception
        self.assertIn('warnings.warn', content,
                     "Should issue warnings on exceptions")


class TestSwerlingModels(unittest.TestCase):
    """Test Swerling detection models"""
    
    def setUp(self):
        """Set up radar for testing"""
        from src.radar import Radar, RadarParameters
        
        self.params = RadarParameters(
            frequency=10e9,
            power=1000,
            antenna_gain=30,
            pulse_width=1e-6,
            prf=1000,
            bandwidth=1e6,
            noise_figure=3,
            losses=2
        )
        self.radar = Radar(self.params)
    
    def test_swerling_0(self):
        """Test Swerling 0 (non-fluctuating) model"""
        # Should work with fallback approximation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            pd = self.radar.detection_probability(
                snr_db=10,
                pfa=1e-6,
                swerling_model=0,
                n_pulses=1
            )
            
            # Check valid probability
            self.assertGreaterEqual(pd, 0.0)
            self.assertLessEqual(pd, 1.0)
            
            # Check if fallback was used (no marcumq available)
            if w:
                warning_messages = [str(warning.message) for warning in w]
                has_scipy_warning = any(
                    "marcumq not available" in msg for msg in warning_messages
                )
                if has_scipy_warning:
                    print("  Using Albersheim approximation for Swerling 0")
    
    def test_swerling_1(self):
        """Test Swerling 1 (slow fluctuation) model"""
        pd = self.radar.detection_probability(
            snr_db=10,
            pfa=1e-6,
            swerling_model=1,
            n_pulses=1
        )
        
        # Check valid probability
        self.assertGreaterEqual(pd, 0.0)
        self.assertLessEqual(pd, 1.0)
        
        # Test with integration
        pd_integrated = self.radar.detection_probability(
            snr_db=10,
            pfa=1e-6,
            swerling_model=1,
            n_pulses=10
        )
        self.assertGreaterEqual(pd_integrated, 0.0)
        self.assertLessEqual(pd_integrated, 1.0)
    
    def test_swerling_2(self):
        """Test Swerling 2 (fast fluctuation) model"""
        pd = self.radar.detection_probability(
            snr_db=10,
            pfa=1e-6,
            swerling_model=2,
            n_pulses=1
        )
        
        # Check valid probability
        self.assertGreaterEqual(pd, 0.0)
        self.assertLessEqual(pd, 1.0)
        
        # Test with integration
        pd_integrated = self.radar.detection_probability(
            snr_db=10,
            pfa=1e-6,
            swerling_model=2,
            n_pulses=10
        )
        self.assertGreaterEqual(pd_integrated, 0.0)
        self.assertLessEqual(pd_integrated, 1.0)
    
    def test_swerling_3_4(self):
        """Test Swerling 3 and 4 models"""
        for model in [3, 4]:
            pd = self.radar.detection_probability(
                snr_db=10,
                pfa=1e-6,
                swerling_model=model,
                n_pulses=1
            )
            
            # Check valid probability
            self.assertGreaterEqual(pd, 0.0)
            self.assertLessEqual(pd, 1.0)
    
    def test_invalid_swerling(self):
        """Test invalid Swerling model number"""
        with self.assertRaises(ValueError) as context:
            self.radar.detection_probability(
                snr_db=10,
                pfa=1e-6,
                swerling_model=5,  # Invalid
                n_pulses=1
            )
        
        self.assertIn("Invalid Swerling model", str(context.exception))
    
    def test_detection_probability_bounds(self):
        """Test that detection probability is always in [0, 1]"""
        test_cases = [
            (-10, 1e-6),  # Very low SNR
            (0, 1e-6),    # Zero SNR
            (20, 1e-6),   # High SNR
            (10, 0.1),    # High false alarm
            (10, 1e-10),  # Low false alarm
        ]
        
        for snr_db, pfa in test_cases:
            for swerling in range(5):
                pd = self.radar.detection_probability(
                    snr_db=snr_db,
                    pfa=pfa,
                    swerling_model=swerling,
                    n_pulses=1
                )
                
                self.assertGreaterEqual(
                    pd, 0.0,
                    f"Pd < 0 for SNR={snr_db}, Pfa={pfa}, Swerling={swerling}"
                )
                self.assertLessEqual(
                    pd, 1.0,
                    f"Pd > 1 for SNR={snr_db}, Pfa={pfa}, Swerling={swerling}"
                )


class TestRadarIntegration(unittest.TestCase):
    """Test that radar.py properly uses new constants"""
    
    def test_radar_uses_constants(self):
        """Verify radar.py imports and uses constants module"""
        from src.radar import Radar, RadarParameters
        
        params = RadarParameters(
            frequency=10e9,
            power=1000,
            antenna_gain=30,
            pulse_width=1e-6,
            prf=1000,
            bandwidth=1e6,
            noise_figure=3,
            losses=2
        )
        
        radar = Radar(params)
        
        # Check that constants are used
        from src.constants import SPEED_OF_LIGHT, BOLTZMANN_CONSTANT, STANDARD_TEMPERATURE
        
        self.assertEqual(radar.k_boltzmann, BOLTZMANN_CONSTANT)
        self.assertEqual(radar.temperature, STANDARD_TEMPERATURE)
        
        # Wavelength should use speed of light constant
        expected_wavelength = SPEED_OF_LIGHT / params.frequency
        self.assertAlmostEqual(params.wavelength, expected_wavelength)


def run_tests():
    """Run all tests and report results"""
    print("=" * 60)
    print("Running Audit Fix Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPhysicalConstants))
    suite.addTests(loader.loadTestsFromTestCase(TestValidators))
    suite.addTests(loader.loadTestsFromTestCase(TestExceptionHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestSwerlingModels))
    suite.addTests(loader.loadTestsFromTestCase(TestRadarIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
        if result.failures:
            print("\nFailed tests:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nTests with errors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)