#!/usr/bin/env python3
"""
Simple SEAD vs IADS Demonstration

A streamlined simulation showing Wild Weasel tactics against SAM sites.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.iads import SAMSite, SAMType
from src.sead import WildWeasel, SEADTactic


def run_simple_engagement():
    """Run a simple SEAD vs SAM engagement"""
    
    print("\n" + "="*60)
    print("SIMPLE SEAD vs IADS ENGAGEMENT")
    print("="*60)
    
    # Create SAM site
    sam = SAMSite(
        site_id="SAM_1",
        position=np.array([0, 0, 0]),
        sam_type=SAMType.MEDIUM_RANGE,
        num_launchers=1,
        missiles_per_launcher=4
    )
    
    # Create Wild Weasel
    weasel = WildWeasel(
        aircraft_id="VIPER_1",
        position=np.array([-60000, 20000, 8000]),
        velocity=np.array([200, -50, 0]),
        num_harms=4
    )
    weasel.tactic = SEADTactic.REACTIVE
    
    print(f"\nInitial Setup:")
    print(f"  SAM Site: {sam.site_id} at origin")
    print(f"    - Type: {sam.sam_type.value}")
    print(f"    - Max range: {sam.max_range/1000:.0f} km")
    print(f"    - Missiles: {sam.missiles_available}")
    print(f"  Wild Weasel: {weasel.aircraft_id}")
    print(f"    - Position: {weasel.position/1000} km")
    print(f"    - HARMs: {weasel.harms_remaining}")
    
    # Simulation parameters
    duration = 60  # seconds
    dt = 0.5
    time_steps = np.arange(0, duration, dt)
    
    # Data storage
    timeline = {
        'time': [],
        'aircraft_pos': [],
        'sam_emitting': [],
        'harm_positions': [],
        'sam_missiles': [],
        'events': []
    }
    
    print(f"\nStarting {duration}s engagement simulation...")
    print("-" * 40)
    
    # EMCON schedule for SAM
    emcon_times = [0, 10, 20, 30, 40, 50]  # Toggle every 10 seconds
    
    for i, t in enumerate(time_steps):
        # Update aircraft position
        weasel.update_position(dt)
        
        # SAM EMCON management
        sam_emitting = int(t / 10) % 2 == 0  # On for 10s, off for 10s
        sam.set_emcon(sam_emitting)
        
        # Prepare emitter data for RWR
        emitters = {}
        if sam_emitting and sam.operational:
            emitters[sam.site_id] = {
                'position': sam.position,
                'emitting': True,
                'type': 'fcr' if t > 20 else 'search',
                'power': 500,
                'frequency': 10e9,
                'prf': 1000
            }
        
        # Wild Weasel detects and engages
        detected = weasel.detect_emitters(emitters)
        
        if detected and weasel.harms_remaining > 0:
            # Check if we should launch
            range_to_sam = np.linalg.norm(weasel.position - sam.position)
            if 20000 < range_to_sam < 80000:  # Engagement window
                target = weasel.select_target()
                if target:
                    harm_id = weasel.launch_harm(target)
                    if harm_id:
                        timeline['events'].append({
                            'time': t,
                            'type': 'harm_launch',
                            'details': f"HARM launched at {target} from {range_to_sam/1000:.1f} km"
                        })
                        print(f"[T+{t:04.1f}] HARM launched at {target}")
        
        # Update HARMs in flight
        impacts = weasel.update_harms(dt, emitters)
        for impact in impacts:
            if impact['result'] == 'hit':
                sam.operational = False
                timeline['events'].append({
                    'time': t,
                    'type': 'sam_destroyed',
                    'details': f"{impact['target_id']} destroyed"
                })
                print(f"[T+{t:04.1f}] SAM DESTROYED!")
                break
        
        # SAM engagement (if operational)
        if sam.operational and sam_emitting:
            range_to_aircraft = np.linalg.norm(weasel.position - sam.position)
            if range_to_aircraft < sam.max_range * 0.7 and sam.missiles_available > 0:
                # Launch SAMs
                launched = sam.engage_target(
                    "WW_1",
                    weasel.position,
                    weasel.velocity,
                    2,  # 2 missiles
                    t
                )
                if launched:
                    timeline['events'].append({
                        'time': t,
                        'type': 'sam_launch',
                        'details': f"{len(launched)} SAMs launched"
                    })
                    print(f"[T+{t:04.1f}] SAM launched {len(launched)} missiles")
        
        # Update SAM missiles
        if sam.missiles_in_flight:
            target_updates = {
                "WW_1": {
                    'position': weasel.position,
                    'velocity': weasel.velocity
                }
            }
            intercepts = sam.update_missiles(dt, target_updates, t)
            for intercept in intercepts:
                if intercept['result'] == 'hit':
                    timeline['events'].append({
                        'time': t,
                        'type': 'aircraft_hit',
                        'details': "Aircraft hit"
                    })
                    print(f"[T+{t:04.1f}] AIRCRAFT HIT!")
        
        # Store state
        timeline['time'].append(t)
        timeline['aircraft_pos'].append(weasel.position.copy())
        timeline['sam_emitting'].append(sam_emitting)
        timeline['harm_positions'].append([h.position.copy() for h in weasel.harms_launched.values()])
        timeline['sam_missiles'].append([m.position.copy() for m in sam.missiles_in_flight.values()])
    
    return timeline, sam, weasel


def visualize_simple_engagement(timeline, sam, weasel):
    """Create visualization of the engagement"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Tactical picture at mid-point
    ax1 = axes[0]
    ax1.set_title('Tactical Picture (T+30s)', fontweight='bold')
    ax1.set_xlabel('East-West (km)')
    ax1.set_ylabel('North-South (km)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Select frame at T+30s
    frame_idx = 60  # 30 seconds at 0.5s dt
    if frame_idx >= len(timeline['time']):
        frame_idx = len(timeline['time']) // 2
    
    # Plot SAM site
    sam_pos = sam.position / 1000
    ax1.scatter(sam_pos[0], sam_pos[1], s=200, c='red', marker='^',
               label='SAM Site', zorder=5)
    
    # SAM engagement envelope
    circle = Circle((sam_pos[0], sam_pos[1]), sam.max_range/1000,
                   fill=False, edgecolor='red', linestyle='--', alpha=0.3)
    ax1.add_patch(circle)
    
    # Radar emission
    if timeline['sam_emitting'][frame_idx]:
        wedge = Wedge((sam_pos[0], sam_pos[1]), sam.max_range/1000,
                     0, 360, facecolor='red', alpha=0.1)
        ax1.add_patch(wedge)
        ax1.text(sam_pos[0]+2, sam_pos[1]+2, "EMITTING", fontsize=8, color='red')
    
    # Plot aircraft
    ac_pos = timeline['aircraft_pos'][frame_idx] / 1000
    ax1.scatter(ac_pos[0], ac_pos[1], s=100, c='blue', marker='>',
               label='Wild Weasel', zorder=6)
    
    # Plot HARMs
    for harm_pos in timeline['harm_positions'][frame_idx]:
        h_pos = harm_pos / 1000
        ax1.scatter(h_pos[0], h_pos[1], s=30, c='green', marker='v', zorder=7)
        # Line to target
        ax1.plot([h_pos[0], sam_pos[0]], [h_pos[1], sam_pos[1]],
                'g--', alpha=0.3, linewidth=1)
    
    # Plot SAM missiles
    for missile_pos in timeline['sam_missiles'][frame_idx]:
        m_pos = missile_pos / 1000
        ax1.scatter(m_pos[0], m_pos[1], s=20, c='orange', marker='^', zorder=7)
    
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_xlim([-80, 20])
    ax1.set_ylim([-20, 40])
    
    # 2. Engagement timeline
    ax2 = axes[1]
    ax2.set_title('Engagement Timeline', fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Status')
    ax2.grid(True, alpha=0.3)
    
    times = timeline['time']
    
    # SAM emissions
    emissions = [1 if e else 0 for e in timeline['sam_emitting']]
    ax2.fill_between(times, emissions, 0, alpha=0.3, color='red', label='SAM Emitting')
    
    # HARMs in flight
    harms_count = [len(h) for h in timeline['harm_positions']]
    ax2.plot(times, np.array(harms_count) / 4 + 1.2, 'g-', label='HARMs in Flight', linewidth=2)
    
    # SAMs in flight
    sams_count = [len(m) for m in timeline['sam_missiles']]
    ax2.plot(times, np.array(sams_count) / 4 + 2.4, 'orange', label='SAMs in Flight', linewidth=2)
    
    # Mark events
    for event in timeline['events']:
        if event['type'] == 'harm_launch':
            ax2.axvline(event['time'], color='green', linestyle='--', alpha=0.5)
        elif event['type'] == 'sam_launch':
            ax2.axvline(event['time'], color='orange', linestyle='--', alpha=0.5)
        elif event['type'] == 'sam_destroyed':
            ax2.axvline(event['time'], color='red', linestyle='-', alpha=0.8, linewidth=2)
            ax2.text(event['time'], 3.5, 'SAM KILLED', rotation=90, fontsize=8)
    
    ax2.set_ylim([-0.2, 4])
    ax2.set_yticks([0.5, 1.7, 2.9])
    ax2.set_yticklabels(['SAM Radar', 'HARMs', 'SAMs'])
    ax2.legend(loc='upper right', fontsize=8)
    
    # 3. Engagement summary
    ax3 = axes[2]
    ax3.set_title('Engagement Summary', fontweight='bold')
    ax3.axis('off')
    
    # Calculate statistics
    total_harm_launches = sum(1 for e in timeline['events'] if e['type'] == 'harm_launch')
    total_sam_launches = sum(1 for e in timeline['events'] if e['type'] == 'sam_launch')
    sam_destroyed = any(e['type'] == 'sam_destroyed' for e in timeline['events'])
    aircraft_hit = any(e['type'] == 'aircraft_hit' for e in timeline['events'])
    
    summary_text = f"""
    SEAD AIRCRAFT (BLUE):
    • Type: F-16CJ Wild Weasel
    • HARMs carried: 4
    • HARMs fired: {total_harm_launches}
    • Status: {'HIT' if aircraft_hit else 'OPERATIONAL'}
    
    SAM SITE (RED):
    • Type: {sam.sam_type.value}
    • Missiles available: 4
    • Missiles fired: {total_sam_launches * 2}
    • Status: {'DESTROYED' if sam_destroyed else 'OPERATIONAL'}
    
    TACTICS:
    • SAM used EMCON (10s on/off)
    • SEAD used reactive targeting
    • HARM memory mode vs shutdown
    
    OUTCOME: {'SEAD SUCCESS' if sam_destroyed else 'SAM SURVIVES'}
    """
    
    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('SEAD vs SAM Engagement Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def main():
    """Run simple SEAD vs IADS demonstration"""
    
    # Run engagement
    timeline, sam, weasel = run_simple_engagement()
    
    print("\n" + "="*60)
    print("ENGAGEMENT COMPLETE")
    print("="*60)
    
    # Final status
    print(f"\nFinal Status:")
    print(f"  SAM: {'DESTROYED' if not sam.operational else 'OPERATIONAL'}")
    print(f"  SAM missiles remaining: {sam.missiles_available}")
    print(f"  Wild Weasel HARMs remaining: {weasel.harms_remaining}")
    print(f"  Threats destroyed: {weasel.threats_destroyed}")
    
    # Create visualization
    print("\nGenerating visualization...")
    fig = visualize_simple_engagement(timeline, sam, weasel)
    
    # Save figure
    output_dir = "results/sead_iads"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "sead_iads_simple.png")
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_file}")
    
    plt.show()


if __name__ == "__main__":
    main()