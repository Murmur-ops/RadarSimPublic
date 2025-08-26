#!/usr/bin/env python3
"""
Simple demonstration of ML threat classification
Run this to see the machine learning system in action
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ml_threat_priority import MLThreatClassifier, PDWSequence, ThreatPriority

def main():
    print("="*60)
    print("SIMPLE ML THREAT CLASSIFICATION DEMO")
    print("="*60)
    
    # Initialize classifier
    classifier = MLThreatClassifier()
    
    # Simulate different threat types
    scenarios = [
        {
            "name": "Anti-Ship Missile",
            "pdw": PDWSequence(
                timestamps=np.arange(0, 0.1, 0.0001),  # 1000 pulses in 100ms
                frequencies=9.5e9 + np.random.randn(1000) * 1e6,  # X-band
                pulse_widths=np.ones(1000) * 0.5e-6,  # 0.5 microsecond pulses
                amplitudes=-20 + np.random.randn(1000) * 2,
                pri_values=np.ones(999) * 100e-6,  # 100 microsecond PRI
                aoa_values=45 + np.random.randn(1000) * 0.5,
                track_id=1001
            )
        },
        {
            "name": "Fighter Aircraft",
            "pdw": PDWSequence(
                timestamps=np.arange(0, 0.2, 0.001),
                frequencies=9e9 + np.random.choice([0, 100e6, 200e6], 200),  # Frequency hopping
                pulse_widths=np.ones(200) * 10e-6,  # 10 microsecond pulses
                amplitudes=-30 + np.random.randn(200) * 3,
                pri_values=np.ones(199) * 1e-3,
                aoa_values=60 + np.random.randn(200) * 1,
                track_id=1002
            )
        },
        {
            "name": "Commercial Navigation Radar",
            "pdw": PDWSequence(
                timestamps=np.arange(0, 2.0, 0.05),  # Slow rotation
                frequencies=np.ones(40) * 9.4e9,  # Fixed frequency
                pulse_widths=np.ones(40) * 1e-3,  # 1ms pulses
                amplitudes=-60 + np.random.randn(40) * 5,
                pri_values=np.ones(39) * 50e-3,  # 50ms PRI
                aoa_values=np.linspace(0, 360, 40),  # Rotating antenna
                track_id=1003
            )
        }
    ]
    
    print("\nClassifying threats...\n")
    
    for scenario in scenarios:
        priority, confidence, details = classifier.classify(scenario["pdw"])
        
        print(f"{scenario['name']}:")
        print(f"  Classification: {priority.name}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Inference time: {details['inference_time_ms']:.2f}ms")
        print()
    
    # Show performance summary
    stats = classifier.get_performance_stats()
    print("="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Mean inference time: {stats['mean_inference_ms']:.2f}ms")
    print(f"Max inference time: {stats['max_inference_ms']:.2f}ms")
    
    if stats['mean_inference_ms'] < 10:
        print("\n✓ Real-time capable (<10ms average)")
    else:
        print("\n⚠ Performance optimization may be needed")

if __name__ == "__main__":
    main()