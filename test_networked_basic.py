#!/usr/bin/env python3
"""
Basic test of networked radar functionality
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.networked_radar import (
    NetworkedRadar, NetworkNode, RadarNodeType,
    CommunicationLink, NetworkArchitecture
)
from src.networked_radar import (
    DataFusionCenter, FusionArchitecture,
    NetworkTrack, DistributedTrackFusion
)
from src.radar import RadarParameters

def test_basic_setup():
    """Test basic networked radar setup."""
    print("Testing basic networked radar setup...")
    
    # Create radar parameters
    params = RadarParameters(
        frequency=10e9,
        power=1000,
        antenna_gain=30,
        pulse_width=1e-6,
        prf=1000,
        bandwidth=10e6,
        noise_figure=3,
        losses=2
    )
    
    # Create nodes
    node1 = NetworkNode(
        node_id="radar_1",
        position=np.array([0, 0, 0]),
        node_type=RadarNodeType.MONOSTATIC
    )
    
    node2 = NetworkNode(
        node_id="radar_2",
        position=np.array([10000, 0, 0]),
        node_type=RadarNodeType.MONOSTATIC
    )
    
    # Create networked radars
    radar1 = NetworkedRadar(params, node1)
    radar2 = NetworkedRadar(params, node2)
    
    print(f"Created radar 1 at position {node1.position}")
    print(f"Created radar 2 at position {node2.position}")
    
    # Test bistatic calculations
    target_pos = np.array([5000, 5000, 1000])
    bistatic_range, tx_range, rx_range = radar2.calculate_bistatic_range(
        target_pos, node1.position
    )
    
    print(f"\nBistatic calculation for target at {target_pos}:")
    print(f"  TX range: {tx_range:.1f} m")
    print(f"  RX range: {rx_range:.1f} m")
    print(f"  Bistatic range: {bistatic_range:.1f} m")
    
    return True

def test_track_fusion():
    """Test track fusion capabilities."""
    print("\nTesting track fusion...")
    
    # Create sample tracks - ensure float arrays
    track1 = NetworkTrack(
        track_id="track_1",
        local_id="local_1",
        source_node="radar_1",
        state=np.array([1000.0, 2000.0, 100.0, 10.0, 5.0, 0.0]),
        covariance=np.diag([100.0, 100.0, 50.0, 10.0, 10.0, 5.0]),
        timestamp=0.0,
        quality=0.9
    )
    
    track2 = NetworkTrack(
        track_id="track_2", 
        local_id="local_2",
        source_node="radar_2",
        state=np.array([1050.0, 1980.0, 110.0, 12.0, 4.0, 0.0]),
        covariance=np.diag([80.0, 80.0, 40.0, 8.0, 8.0, 4.0]),
        timestamp=0.0,
        quality=0.85
    )
    
    # Test Covariance Intersection fusion
    fusion = DistributedTrackFusion()
    fused_track = fusion.covariance_intersection([track1, track2])
    
    print(f"Track 1 position: {track1.state[:3]}")
    print(f"Track 2 position: {track2.state[:3]}")
    print(f"Fused position: {fused_track.state[:3]}")
    print(f"Fused covariance trace: {np.trace(fused_track.covariance):.1f}")
    
    return True

def test_fusion_center():
    """Test fusion center functionality."""
    print("\nTesting fusion center...")
    
    # Create fusion center
    fusion_center = DataFusionCenter(
        center_id="central",
        architecture=FusionArchitecture.CENTRALIZED
    )
    
    # Create tracks from different nodes - ensure float arrays
    tracks_node1 = [
        NetworkTrack(
            track_id="n1_t1",
            local_id="t1",
            source_node="radar_1",
            state=np.array([1000.0, 2000.0, 100.0, 10.0, 5.0, 0.0]),
            covariance=np.diag([100.0, 100.0, 50.0, 10.0, 10.0, 5.0]),
            timestamp=0.0,
            quality=0.9
        )
    ]
    
    tracks_node2 = [
        NetworkTrack(
            track_id="n2_t1",
            local_id="t1",
            source_node="radar_2",
            state=np.array([1050.0, 1980.0, 110.0, 12.0, 4.0, 0.0]),
            covariance=np.diag([80.0, 80.0, 40.0, 8.0, 8.0, 4.0]),
            timestamp=0.0,
            quality=0.85
        )
    ]
    
    # Send tracks to fusion center
    fusion_center.receive_local_tracks("radar_1", tracks_node1)
    fusion_center.receive_local_tracks("radar_2", tracks_node2)
    
    # Perform fusion
    fused_tracks = fusion_center.perform_fusion()
    
    print(f"Number of nodes: {len(fusion_center.local_tracks)}")
    print(f"Number of fused tracks: {len(fused_tracks)}")
    
    # Get metrics
    metrics = fusion_center.get_fusion_metrics()
    print(f"Fusion metrics:")
    print(f"  Tracks received: {metrics['tracks_received']}")
    print(f"  Tracks fused: {metrics['tracks_fused']}")
    print(f"  Architecture: {metrics['architecture']}")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("NETWORKED RADAR BASIC TESTS")
    print("=" * 60)
    
    tests = [
        ("Basic Setup", test_basic_setup),
        ("Track Fusion", test_track_fusion),
        ("Fusion Center", test_fusion_center)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"Error in {test_name}: {e}")
            results.append((test_name, "ERROR"))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    for test_name, result in results:
        print(f"{test_name}: {result}")

if __name__ == "__main__":
    main()