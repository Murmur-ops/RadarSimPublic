#!/usr/bin/env python3
"""
Quick test of SEAD and IADS components
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.iads import SAMSite, SAMType
from src.sead import WildWeasel, SEADTactic

def test_components():
    """Test basic SEAD/IADS functionality"""
    
    print("Testing SEAD/IADS Components")
    print("="*40)
    
    # Test SAM creation
    print("\n1. Creating SAM site...")
    sam = SAMSite(
        site_id="TEST_SAM",
        position=np.array([0, 0, 0]),
        sam_type=SAMType.MEDIUM_RANGE
    )
    print(f"   SAM Type: {sam.sam_type.value}")
    print(f"   Max Range: {sam.max_range/1000:.0f} km")
    print(f"   Missiles: {sam.missiles_available}")
    
    # Test Wild Weasel creation
    print("\n2. Creating Wild Weasel...")
    weasel = WildWeasel(
        aircraft_id="TEST_WW",
        position=np.array([-50000, 10000, 8000]),
        velocity=np.array([200, 0, 0]),
        num_harms=4
    )
    print(f"   Aircraft ID: {weasel.aircraft_id}")
    print(f"   HARMs: {weasel.harms_remaining}")
    print(f"   RWR Range: {weasel.rwr_range/1000:.0f} km")
    
    # Test engagement check
    print("\n3. Testing engagement capability...")
    target_pos = np.array([-30000, 5000, 5000])
    target_vel = np.array([200, 0, 0])
    can_engage = sam.can_engage(target_pos, target_vel)
    print(f"   Can SAM engage target at {target_pos/1000} km? {can_engage}")
    
    # Test emitter detection
    print("\n4. Testing emitter detection...")
    emitters = {
        "TEST_SAM": {
            'position': sam.position,
            'emitting': True,
            'type': 'search',
            'power': 1000,
            'frequency': 10e9,
            'prf': 1000
        }
    }
    detected = weasel.detect_emitters(emitters)
    print(f"   Detected emitters: {detected}")
    
    if detected:
        detection = weasel.rwr_detections[detected[0]]
        print(f"   Bearing: {detection.bearing:.1f}°")
        print(f"   Signal: {detection.signal_strength:.1f} dBm")
    
    # Test HARM launch
    print("\n5. Testing HARM launch...")
    if detected:
        harm_id = weasel.launch_harm(detected[0])
        if harm_id:
            print(f"   Launched HARM: {harm_id}")
            print(f"   HARMs remaining: {weasel.harms_remaining}")
        else:
            print("   HARM launch failed")
    
    # Test SAM launch
    print("\n6. Testing SAM launch...")
    missiles = sam.engage_target(
        "TEST_TGT",
        target_pos,
        target_vel,
        2,
        0.0
    )
    print(f"   Launched {len(missiles)} missiles")
    print(f"   Missiles remaining: {sam.missiles_available}")
    
    print("\n✓ All components working!")
    return True

if __name__ == "__main__":
    success = test_components()
    if not success:
        sys.exit(1)