#!/usr/bin/env python3
"""
Quick test script to verify all YAML configurations load and run
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.config_loader import ConfigLoader
import yaml

def test_yaml_loading():
    """Test that all YAML files load correctly"""
    
    scenarios_dir = Path('configs/scenarios')
    example_dir = Path('example_configs')
    
    all_yamls = list(scenarios_dir.glob('*.yaml')) + list(example_dir.glob('*.yaml'))
    
    print(f"Found {len(all_yamls)} YAML files to test")
    print("=" * 60)
    
    results = []
    
    for yaml_file in all_yamls:
        try:
            # Load YAML
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check structure
            has_scenario = 'scenario' in config
            has_radar = 'radar' in config
            has_targets = 'targets' in config and len(config['targets']) > 0
            
            # Check target format
            target_formats = []
            for target in config.get('targets', []):
                pos = target.get('initial_position')
                if isinstance(pos, list):
                    target_formats.append('list')
                elif isinstance(pos, dict):
                    target_formats.append('dict')
                else:
                    target_formats.append('unknown')
            
            status = "✅" if (has_scenario and has_radar and has_targets) else "⚠️"
            
            results.append({
                'file': yaml_file.name,
                'status': status,
                'targets': len(config.get('targets', [])),
                'format': target_formats[0] if target_formats else 'N/A'
            })
            
            print(f"{status} {yaml_file.name:35} | {len(config.get('targets', []))} targets | pos format: {target_formats[0] if target_formats else 'N/A'}")
            
        except Exception as e:
            results.append({
                'file': yaml_file.name,
                'status': '❌',
                'error': str(e)
            })
            print(f"❌ {yaml_file.name:35} | Error: {e}")
    
    print("=" * 60)
    
    # Summary
    success = sum(1 for r in results if r['status'] == '✅')
    warning = sum(1 for r in results if r['status'] == '⚠️')
    failed = sum(1 for r in results if r['status'] == '❌')
    
    print(f"\nSummary:")
    print(f"  ✅ Success: {success}")
    print(f"  ⚠️  Warning: {warning}")
    print(f"  ❌ Failed:  {failed}")
    
    # Test config loader
    print("\n" + "=" * 60)
    print("Testing ConfigLoader with each scenario...")
    print("=" * 60)
    
    loader = ConfigLoader('configs')
    
    test_scenarios = ['simple_tracking', 'air_defense', 'naval_engagement']
    
    for scenario_name in test_scenarios:
        try:
            config = loader.load_scenario(scenario_name)
            print(f"✅ {scenario_name:30} loaded successfully")
            print(f"   - {len(config.targets)} targets")
            print(f"   - Duration: {config.duration}s")
            
            # Check if targets have proper position data
            for target in config.targets:
                if hasattr(target, 'range') and hasattr(target, 'azimuth'):
                    print(f"   - {target.name}: range={target.range:.0f}m, az={target.azimuth:.1f}°")
                    
        except Exception as e:
            print(f"❌ {scenario_name:30} failed: {e}")
    
    return success == len(all_yamls)

if __name__ == "__main__":
    success = test_yaml_loading()
    sys.exit(0 if success else 1)