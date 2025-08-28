#!/usr/bin/env python3
"""
SEAD vs IADS Demonstration

Simulates a Wild Weasel SEAD mission against a layered IADS,
showing the cat-and-mouse game between HARM-equipped aircraft
and SAM sites using emission control (EMCON) tactics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge
import matplotlib.animation as animation
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.iads import IADSNetwork, SAMSite, SAMType, EngagementStatus, ThreatLevel
from src.sead import WildWeasel, SEADTactic, HARMMode


def setup_iads():
    """Create a layered IADS network"""
    iads = IADSNetwork("IADS_DEMO")
    
    # Long-range SAM (S-400 equivalent)
    sam_lr = SAMSite(
        site_id="SAM_LR_1",
        position=np.array([0, 0, 0]),
        sam_type=SAMType.LONG_RANGE,
        num_launchers=2,
        missiles_per_launcher=4
    )
    iads.add_sam_site("SAM_LR_1", sam_lr)
    
    # Medium-range SAMs (Buk equivalent)
    sam_mr1 = SAMSite(
        site_id="SAM_MR_1",
        position=np.array([-30000, 20000, 0]),
        sam_type=SAMType.MEDIUM_RANGE,
        num_launchers=1,
        missiles_per_launcher=4
    )
    iads.add_sam_site("SAM_MR_1", sam_mr1)
    
    sam_mr2 = SAMSite(
        site_id="SAM_MR_2",
        position=np.array([30000, 15000, 0]),
        sam_type=SAMType.MEDIUM_RANGE,
        num_launchers=1,
        missiles_per_launcher=4
    )
    iads.add_sam_site("SAM_MR_2", sam_mr2)
    
    # Short-range/SHORAD (Pantsir equivalent)
    sam_sr = SAMSite(
        site_id="SAM_SR_1",
        position=np.array([0, -10000, 0]),
        sam_type=SAMType.SHORT_RANGE,
        num_launchers=1,
        missiles_per_launcher=8
    )
    iads.add_sam_site("SAM_SR_1", sam_sr)
    
    return iads


def setup_sead_package():
    """Create SEAD strike package"""
    aircraft = []
    
    # Lead Wild Weasel
    weasel1 = WildWeasel(
        aircraft_id="VIPER_1",
        position=np.array([-100000, 50000, 8000]),
        velocity=np.array([200, -50, 0]),
        num_harms=4
    )
    weasel1.tactic = SEADTactic.REACTIVE
    aircraft.append(weasel1)
    
    # Wingman Wild Weasel
    weasel2 = WildWeasel(
        aircraft_id="VIPER_2",
        position=np.array([-95000, 48000, 8000]),
        velocity=np.array([200, -50, 0]),
        num_harms=4
    )
    weasel2.tactic = SEADTactic.PREEMPTIVE
    aircraft.append(weasel2)
    
    return aircraft


def simulate_engagement(duration=120, dt=0.5):
    """
    Run SEAD vs IADS simulation
    
    Args:
        duration: Simulation duration in seconds
        dt: Time step
    """
    # Initialize forces
    iads = setup_iads()
    sead_aircraft = setup_sead_package()
    
    # Simulation data storage
    timeline = []
    time_steps = np.arange(0, duration, dt)
    
    # EMCON management for SAMs
    emcon_schedule = {
        "SAM_LR_1": {'on': [0, 30, 60, 90], 'off': [20, 50, 80]},
        "SAM_MR_1": {'on': [10, 40, 70], 'off': [25, 55, 85]},
        "SAM_MR_2": {'on': [5, 35, 65], 'off': [15, 45, 75]},
        "SAM_SR_1": {'on': [0], 'off': []}  # Always on (point defense)
    }
    
    print("\n" + "="*60)
    print("SEAD vs IADS ENGAGEMENT SIMULATION")
    print("="*60)
    print(f"\nForces:")
    print(f"  BLUE (SEAD): 2x F-16CJ Wild Weasel, 8x AGM-88 HARM total")
    print(f"  RED (IADS): 1x Long-range SAM, 2x Medium-range SAM, 1x SHORAD")
    print("\nStarting simulation...\n")
    
    for t in time_steps:
        frame_data = {
            'time': t,
            'aircraft': [],
            'sams': [],
            'harms': [],
            'missiles': [],
            'detections': [],
            'engagements': []
        }
        
        # Update EMCON for SAMs
        emitters = {}
        for site_id, sam_site in iads.sam_sites.items():
            # Check EMCON schedule
            schedule = emcon_schedule.get(site_id, {'on': [0], 'off': []})
            emitting = False
            
            for on_time in schedule['on']:
                if on_time <= t < on_time + 10:  # 10 second emission windows
                    emitting = True
                    break
            
            sam_site.set_emcon(emitting)
            
            if emitting:
                emitters[site_id] = {
                    'position': sam_site.position,
                    'emitting': True,
                    'type': 'fcr' if t > 40 else 'search',  # Switch to FCR mode when threatened
                    'power': 1000 if sam_site.sam_type == SAMType.LONG_RANGE else 500,
                    'frequency': 10e9,
                    'prf': 1000
                }
        
        # Update SEAD aircraft
        for aircraft in sead_aircraft:
            # Update position
            aircraft.update_position(dt)
            
            # Execute SEAD tactics
            action = aircraft.execute_sead_tactic(emitters)
            
            if action['type'] == 'harm_launch':
                print(f"[T+{t:05.1f}] {aircraft.aircraft_id} launched HARM at {action['details']['target']}")
            
            # Update HARMs
            impacts = aircraft.update_harms(dt, emitters)
            for impact in impacts:
                if impact['result'] == 'hit':
                    # Destroy SAM site
                    if impact['target_id'] in iads.sam_sites:
                        iads.sam_sites[impact['target_id']].operational = False
                        print(f"[T+{t:05.1f}] {impact['target_id']} DESTROYED by {impact['harm_id']}")
            
            # Store aircraft data
            frame_data['aircraft'].append({
                'id': aircraft.aircraft_id,
                'position': aircraft.position.copy(),
                'harms_remaining': aircraft.harms_remaining,
                'detections': list(aircraft.rwr_detections.keys())
            })
            
            # Store HARM data
            for harm in aircraft.harms_launched.values():
                frame_data['harms'].append({
                    'id': harm.missile_id,
                    'position': harm.position.copy(),
                    'target': harm.target_emitter_id
                })
        
        # IADS detection and engagement
        # Create synthetic detections for demo
        for aircraft in sead_aircraft:
            for site_id, sam_site in iads.sam_sites.items():
                if not sam_site.operational or not sam_site.radar_active:
                    continue
                
                range_to_aircraft = np.linalg.norm(aircraft.position - sam_site.position)
                
                if range_to_aircraft < sam_site.radar_range:
                    # Detection
                    detection = {
                        'position': aircraft.position + np.random.randn(3) * 100,
                        'velocity': aircraft.velocity,
                        'timestamp': t,
                        'rcs': 5.0
                    }
                    
                    # Process detection
                    track_ids = iads.process_sensor_data(site_id, [detection])
                    
                    # Engagement decision
                    if range_to_aircraft < sam_site.max_range * 0.8:
                        for track_id in track_ids:
                            if track_id not in iads.active_engagements:
                                assignment = iads.assign_weapons(track_id)
                                if assignment:
                                    # Launch SAMs
                                    num_missiles = assignment['num_missiles']
                                    launched = sam_site.engage_target(
                                        track_id,
                                        aircraft.position,
                                        aircraft.velocity,
                                        num_missiles,
                                        t
                                    )
                                    if launched:
                                        print(f"[T+{t:05.1f}] {site_id} launched {len(launched)} SAMs at {aircraft.aircraft_id}")
        
        # Update SAM missiles
        for site_id, sam_site in iads.sam_sites.items():
            if not sam_site.operational:
                continue
            
            # Create target updates for missile guidance
            target_updates = {}
            for aircraft in sead_aircraft:
                # Simple track ID mapping
                target_updates[f"TRK_{sead_aircraft.index(aircraft):04d}"] = {
                    'position': aircraft.position,
                    'velocity': aircraft.velocity
                }
            
            # Update missiles
            intercepts = sam_site.update_missiles(dt, target_updates, t)
            
            for intercept in intercepts:
                if intercept['result'] == 'hit':
                    print(f"[T+{t:05.1f}] Aircraft hit by {intercept['missile_id']}")
            
            # Store SAM data
            frame_data['sams'].append({
                'id': site_id,
                'position': sam_site.position.copy(),
                'type': sam_site.sam_type.value,
                'operational': sam_site.operational,
                'emitting': sam_site.radar_active,
                'missiles_available': sam_site.missiles_available
            })
            
            # Store missile data
            for missile in sam_site.missiles_in_flight.values():
                frame_data['missiles'].append({
                    'id': missile.missile_id,
                    'position': missile.position.copy(),
                    'target': missile.target_id
                })
        
        timeline.append(frame_data)
    
    return timeline


def visualize_engagement(timeline):
    """Create visualization of SEAD vs IADS engagement"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Main tactical picture
    ax1 = axes[0, 0]
    ax1.set_title('SEAD vs IADS Tactical Picture', fontweight='bold')
    ax1.set_xlabel('East-West (km)')
    ax1.set_ylabel('North-South (km)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-120, 60])
    ax1.set_ylim([-40, 80])
    
    # Select mid-engagement frame
    frame_idx = len(timeline) // 2
    frame = timeline[frame_idx]
    
    # Plot SAM sites
    sam_colors = {
        'long_range': 'red',
        'medium_range': 'orange',
        'short_range': 'yellow'
    }
    
    for sam in frame['sams']:
        pos = sam['position'] / 1000
        color = sam_colors.get(sam['type'], 'gray')
        marker = '^' if sam['operational'] else 'x'
        
        ax1.scatter(pos[0], pos[1], s=150, c=color, marker=marker, 
                   edgecolor='black', linewidth=2, zorder=5)
        
        # Engagement envelope
        if sam['operational']:
            if sam['type'] == 'long_range':
                radius = 400
            elif sam['type'] == 'medium_range':
                radius = 50
            else:
                radius = 20
            
            circle = Circle((pos[0], pos[1]), radius, fill=False,
                          edgecolor=color, linestyle='--', alpha=0.3)
            ax1.add_patch(circle)
        
        # Radar emission indicator
        if sam['emitting']:
            wedge = Wedge((pos[0], pos[1]), radius, 0, 360,
                        facecolor=color, alpha=0.1)
            ax1.add_patch(wedge)
        
        ax1.text(pos[0]+2, pos[1]+2, sam['id'], fontsize=8)
    
    # Plot aircraft
    for aircraft in frame['aircraft']:
        pos = aircraft['position'] / 1000
        ax1.scatter(pos[0], pos[1], s=100, c='blue', marker='>', zorder=6)
        ax1.text(pos[0]+2, pos[1]+2, 
                f"{aircraft['id']}\nHARMs: {aircraft['harms_remaining']}", 
                fontsize=8, color='blue')
    
    # Plot HARMs
    for harm in frame['harms']:
        pos = harm['position'] / 1000
        ax1.scatter(pos[0], pos[1], s=30, c='green', marker='v', zorder=7)
        # Draw line to target
        for sam in frame['sams']:
            if sam['id'] == harm['target']:
                target_pos = sam['position'] / 1000
                ax1.plot([pos[0], target_pos[0]], [pos[1], target_pos[1]],
                        'g--', alpha=0.5, linewidth=1)
    
    # Plot SAMs
    for missile in frame['missiles']:
        pos = missile['position'] / 1000
        ax1.scatter(pos[0], pos[1], s=20, c='red', marker='^', zorder=7)
    
    # Timeline plot
    ax2 = axes[0, 1]
    ax2.set_title('Engagement Timeline', fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Active Systems')
    ax2.grid(True, alpha=0.3)
    
    times = [frame['time'] for frame in timeline]
    
    # Track emissions
    sam_emissions = {sam_id: [] for sam_id in ["SAM_LR_1", "SAM_MR_1", "SAM_MR_2", "SAM_SR_1"]}
    for frame in timeline:
        for sam in frame['sams']:
            sam_emissions[sam['id']].append(1 if sam['emitting'] else 0)
    
    offset = 0
    for sam_id, emissions in sam_emissions.items():
        ax2.fill_between(times, offset + np.array(emissions), offset, 
                        alpha=0.5, label=sam_id)
        offset += 1.2
    
    # HARM launches
    harm_launches = []
    for i, frame in enumerate(timeline):
        if i > 0 and len(frame['harms']) > len(timeline[i-1]['harms']):
            harm_launches.append((frame['time'], len(frame['harms']) - len(timeline[i-1]['harms'])))
    
    for launch_time, num in harm_launches:
        ax2.axvline(launch_time, color='green', linestyle='--', alpha=0.5)
        ax2.text(launch_time, offset-0.5, f"{num} HARM", rotation=90, fontsize=8)
    
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_ylim([-0.5, offset])
    
    # Force balance
    ax3 = axes[1, 0]
    ax3.set_title('Force Balance Over Time', fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3)
    
    operational_sams = []
    harms_in_flight = []
    sams_in_flight = []
    
    for frame in timeline:
        operational_sams.append(sum(1 for sam in frame['sams'] if sam['operational']))
        harms_in_flight.append(len(frame['harms']))
        sams_in_flight.append(len(frame['missiles']))
    
    ax3.plot(times, operational_sams, 'r-', label='Operational SAMs', linewidth=2)
    ax3.plot(times, harms_in_flight, 'g-', label='HARMs in flight', linewidth=2)
    ax3.plot(times, sams_in_flight, 'b-', label='SAMs in flight', linewidth=2)
    ax3.legend(loc='upper right')
    
    # Summary statistics
    ax4 = axes[1, 1]
    ax4.set_title('Engagement Summary', fontweight='bold')
    ax4.axis('off')
    
    # Calculate statistics
    total_harms = 8
    harms_fired = total_harms - min(aircraft['harms_remaining'] 
                                    for frame in timeline[-5:] 
                                    for aircraft in frame['aircraft'])
    
    sams_destroyed = 4 - operational_sams[-1]
    
    summary_text = f"""
    BLUE FORCE (SEAD):
    • Aircraft: 2x F-16CJ Wild Weasel
    • HARMs carried: {total_harms}
    • HARMs fired: {harms_fired}
    • SAMs destroyed: {sams_destroyed}
    
    RED FORCE (IADS):
    • Initial SAMs: 4 sites
    • Operational: {operational_sams[-1]} sites
    • Missiles fired: {sum(1 for frame in timeline for _ in frame['missiles'])}
    
    TACTICS EMPLOYED:
    • EMCON cycling by SAMs
    • Reactive/Preemptive HARM shots
    • Layered defense coordination
    • Memory mode for shutdown counters
    
    OUTCOME: {'SEAD Success' if sams_destroyed >= 2 else 'IADS Holds'}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'SEAD vs IADS Engagement - T+{frame["time"]:.1f}s', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def main():
    """Run SEAD vs IADS demonstration"""
    
    # Run simulation
    print("Running SEAD vs IADS simulation...")
    timeline = simulate_engagement(duration=90, dt=0.5)
    
    # Create visualization
    print("\nGenerating visualization...")
    fig = visualize_engagement(timeline)
    
    # Save figure
    output_dir = "results/sead_iads"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "sead_vs_iads_demo.png")
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()