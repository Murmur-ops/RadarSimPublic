#!/usr/bin/env python3
"""
Growler False Target Deception Visualization
Shows how EA-18G generates false missile targets to saturate radar tracking
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Wedge
from matplotlib.gridspec import GridSpec
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def create_growler_deception_visualization(config_file='configs/scenarios/growler_false_target_deception.yaml'):
    """
    Create visualization showing Growler false target deception attack
    """
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Growler Electronic Warfare - False Target Deception Attack\nEA-18G Generates 4 False Missiles to Saturate Radar', 
                 fontsize=14, fontweight='bold')
    
    # Create grid for subplots
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[2, 1, 1])
    
    # ========== LEFT: Radar Tracking Picture with False Targets ==========
    ax1 = fig.add_subplot(gs[:, 0])  # Span both rows
    ax1.set_title('Radar Picture - False Target Deception Scenario', fontweight='bold', pad=10)
    ax1.set_xlabel('Range (km)')
    ax1.set_ylabel('Cross-Range (km)')
    ax1.set_xlim(-15, 115)
    ax1.set_ylim(-55, 55)
    ax1.grid(True, alpha=0.3)
    
    # Draw radar at origin
    radar = Circle((0, 0), 2, color='blue', zorder=5)
    ax1.add_patch(radar)
    ax1.text(0, 0, 'RADAR', ha='center', va='center', color='white', fontweight='bold', fontsize=8)
    
    # Draw range rings
    for r in [20, 40, 60, 80, 100]:
        circle = Circle((0, 0), r, fill=False, edgecolor='gray', alpha=0.2, linestyle='--')
        ax1.add_patch(circle)
        # Only label some rings to reduce clutter
        if r in [40, 80]:
            ax1.text(r, -5, f'{r}km', fontsize=7, alpha=0.4, ha='center')
    
    # Draw Growler EA-18G platform (standoff jamming position)
    growler_x = 85 * np.cos(np.radians(10))
    growler_y = 85 * np.sin(np.radians(10))
    
    # Growler aircraft symbol (larger triangle)
    ax1.scatter(growler_x, growler_y, s=300, c='purple', marker='^', 
               edgecolor='black', linewidth=2, zorder=4)
    ax1.text(growler_x, growler_y-5, 'EA-18G\nGrowler', fontsize=9, 
            ha='center', color='purple', fontweight='bold')
    
    # Draw jamming coverage arc (showing area of influence)
    # Calculate angle from Growler to radar (at origin)
    angle_to_radar = np.degrees(np.arctan2(-growler_y, -growler_x))
    # Create wedge pointing toward radar with 60° coverage - shorter to avoid overlap
    jamming_arc = Wedge((growler_x, growler_y), 25, 
                        theta1=angle_to_radar-20, theta2=angle_to_radar+20,
                        facecolor='purple', alpha=0.1, edgecolor='purple', linewidth=2)
    ax1.add_patch(jamming_arc)
    
    # Draw false targets (Phantoms) - from config
    false_targets = config['scenario']['events'][0]['parameters']['false_target_profiles']
    
    for i, phantom in enumerate(false_targets):
        x = phantom['position'][0]
        y = phantom['position'][1]
        
        # False target with distinct styling (hollow red triangles)
        ax1.scatter(x, y, s=100, facecolors='none', edgecolors='red', 
                   marker='^', linewidth=1.5, linestyle='--', alpha=0.7)
        
        # Add phantom label - only show a few to avoid clutter
        if i in [0, 2, 4, 6]:  # Show every other label
            name = phantom['name'].replace('Phantom-ASM-', 'P-')
            ax1.text(x+2, y, name, fontsize=6, ha='left', color='red', 
                    style='italic', alpha=0.7)
        
        # Draw false velocity vectors
        vx, vy = phantom['velocity'][0]/20, phantom['velocity'][1]/20  # Scale for display
        ax1.arrow(x, y, vx, vy, head_width=1.5, head_length=2,
                 fc='red', ec='darkred', alpha=0.3, linewidth=1, linestyle='--')
        
        # Add threat characteristics annotation
        if 'threat_characteristics' in phantom:
            if 'hypersonic_speed' in phantom['threat_characteristics']:
                ax1.text(x+5, y, '(HYPER)', fontsize=6, color='orange', fontweight='bold')
            elif 'ballistic_profile' in phantom['threat_characteristics']:
                ax1.text(x+5, y, '(BALLISTIC)', fontsize=6, color='orange', fontweight='bold')
    
    # Draw real missile (hidden among false targets)
    real_missile = config['scenario']['events'][1]['parameters']['missile']
    real_x = real_missile['launch_position'][0]
    real_y = real_missile['launch_position'][1]
    
    # Real missile with solid fill
    ax1.scatter(real_x, real_y, s=140, c='darkred', marker='^', 
               edgecolor='yellow', linewidth=2.5, zorder=3)
    ax1.text(real_x-5, real_y, 'REAL\nMissile', fontsize=7, 
            ha='right', color='darkred', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    # Draw real missile vector
    ax1.arrow(real_x, real_y, -real_x*0.4, -real_y*0.4, head_width=2, head_length=3,
             fc='darkred', ec='black', alpha=0.8, linewidth=2)
    
    # False target corridor (within jamming beam)
    # Create a more accurate representation of where false targets actually appear
    # This is a narrow corridor along the jamming beam path
    from matplotlib.patches import Polygon
    
    # Calculate the corridor bounds based on jamming cone
    corridor_near_x = 45 * np.cos(np.radians(10))
    corridor_near_y_low = 45 * np.sin(np.radians(5))
    corridor_near_y_high = 45 * np.sin(np.radians(15))
    
    corridor_far_x = 80 * np.cos(np.radians(10))
    corridor_far_y_low = 80 * np.sin(np.radians(5))
    corridor_far_y_high = 80 * np.sin(np.radians(15))
    
    # Create polygon for false target corridor
    corridor_points = [
        [corridor_near_x, corridor_near_y_low],
        [corridor_near_x, corridor_near_y_high],
        [corridor_far_x, corridor_far_y_high],
        [corridor_far_x, corridor_far_y_low]
    ]
    
    corridor_zone = Polygon(corridor_points, facecolor='red', alpha=0.1,
                           edgecolor='red', linewidth=1.5, linestyle='--')
    ax1.add_patch(corridor_zone)
    # Move text further up to avoid any overlap
    ax1.text(70, 22, 'False Target\nCorridor', ha='center', va='center', 
             fontsize=9, color='red', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))
    
    # Plot initial real targets (that will be dropped)
    targets = config['targets']
    for target in targets:
        # Convert range/azimuth to x/y
        r = target['initial_position']['range'] / 1000  # Convert to km
        az = np.radians(target['initial_position']['azimuth'])
        x = r * np.cos(az)
        y = r * np.sin(az)
        
        # Determine marker based on expected outcome
        if target['expected_outcome'] == 'LOST':
            marker = 'X'
            color = 'red'
            alpha = 0.4
        elif target['expected_outcome'] == 'DEGRADED':
            marker = 'v'
            color = 'yellow'
            alpha = 0.6
        else:  # MAINTAINED
            marker = 'o'
            color = 'green'
            alpha = 0.8
            
        # Plot target
        if 'commercial' in target['type'].lower():
            marker_symbol = 'P'
            size = 80
        elif 'fighter' in target['type'].lower():
            marker_symbol = 'v'
            size = 70
        else:
            marker_symbol = 'o'
            size = 60
            
        ax1.scatter(x, y, s=size, c=color, marker=marker_symbol, alpha=alpha, 
                   edgecolor='black', linewidth=0.5)
        
        # Add label
        name_short = target['name'].split('-')[1]
        ax1.text(x, y-3, name_short, fontsize=6, ha='center', alpha=0.7)
    
    # Add EW attack marker - moved to top right corner
    ax1.text(95, 45, 'EW ATTACK\nFALSE TARGETS', fontsize=10, color='purple', 
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=8, label='Tracked (Good)'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='y', markersize=8, label='Degraded'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='r', markersize=8, label='Track Lost'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='purple', markersize=10, label='Growler EW'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='none', 
                  markeredgecolor='red', markersize=8, label='False Target'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='darkred', markersize=8, label='Real Missile')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=7, ncol=2)
    
    # ========== TOP RIGHT: Track Quality Evolution ==========
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.set_title('Track Quality Evolution with Drop Events', fontweight='bold')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Track Quality')
    ax2.set_xlim(0, 80)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)
    
    # Time array
    t = np.linspace(0, 80, 161)
    
    # Quality thresholds
    ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.3, label='Good')
    ax2.axhline(y=0.6, color='orange', linestyle='--', alpha=0.3, label='Fair')
    ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.3, linewidth=2, label='Drop')
    
    # EW attack markers
    ax2.axvline(x=28, color='purple', linestyle=':', linewidth=2, alpha=0.5)
    ax2.text(28, 0.9, 'EW\nATTACK', fontsize=9, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='purple', alpha=0.2))
    
    ax2.axvline(x=30, color='red', linestyle=':', linewidth=2, alpha=0.5)
    ax2.text(30, 0.7, 'REAL\nMISSILE', fontsize=9, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
    
    # Track quality curves - degradation after EW attack
    # Commercial aircraft - rapid drop
    for i, name in enumerate(['Commercial-747', 'Commercial-A320', 'Commercial-B737']):
        quality = np.ones_like(t) * 0.85
        quality[t >= 28] = 0.85 * np.exp(-0.2 * (t[t >= 28] - 28))
        quality[quality < 0.2] = 0.18
        ax2.plot(t, quality, linestyle='--', alpha=0.5, color='gray', linewidth=1)
        
        # Mark drop point
        drop_idx = np.where(quality < 0.2)[0]
        if len(drop_idx) > 0:
            drop_time = t[drop_idx[0]]
            ax2.scatter(drop_time, 0.18, s=50, c='red', marker='X', zorder=5)
    
    # Fighters - degraded but maintained
    for name in ['Fighter-F22']:
        quality = np.ones_like(t) * 0.90
        quality[t >= 28] = 0.90 - 0.35 * (1 - np.exp(-0.08 * (t[t >= 28] - 28)))
        quality[quality < 0.55] = 0.55  # F22 maintains minimum quality
        ax2.plot(t, quality, label=name, linestyle='-', color='blue', alpha=0.8)
    
    # False targets (phantoms) - consume resources
    for i in range(3):  # Show a few representative false targets
        quality = np.zeros_like(t)
        quality[t >= 28] = 0.92  # High quality tracking of false targets
        ax2.plot(t, quality, linestyle='--', linewidth=1.5, color='red', 
                alpha=0.3 + i*0.1, label=f'Phantom-{i+1}' if i < 1 else '')
    
    # Real missile - lower quality due to resource starvation
    quality_real = np.zeros_like(t)
    quality_real[t >= 30] = 0.65  # Lower quality - only 10% resources
    ax2.plot(t, quality_real, label='Real Missile', linestyle='-', 
            linewidth=2.5, color='darkred', alpha=0.9)
    
    ax2.legend(loc='upper right', fontsize=7, ncol=2)
    
    # ========== BOTTOM RIGHT: Resource Allocation ==========
    ax3 = fig.add_subplot(gs[1, 1:])
    ax3.set_title('Resource Allocation: False vs Real Targets', fontweight='bold')
    
    # Stacked bar chart showing resource waste
    categories = ['Pre-Attack', 'During EW Attack', 'Deception Success']
    false_allocation = [0, 65, 65]  # Percentage to false targets (consistent)
    real_allocation = [0, 10, 10]   # Percentage to real missile
    fighter_allocation = [30, 8, 8]  # Percentage to fighters
    commercial_allocation = [30, 2, 2]  # Percentage to commercial
    other_allocation = [40, 15, 15]  # Other/search (adjusted to total 100%)
    
    width = 0.6
    x_pos = np.arange(len(categories))
    
    # Create stacked bars
    p1 = ax3.bar(x_pos, false_allocation, width, label='False Targets', 
                color='red', alpha=0.7)
    p2 = ax3.bar(x_pos, real_allocation, width, bottom=false_allocation,
                label='Real Missile', color='darkred')
    p3 = ax3.bar(x_pos, fighter_allocation, width, 
                bottom=np.array(false_allocation)+np.array(real_allocation),
                label='Fighters', color='blue', alpha=0.7)
    p4 = ax3.bar(x_pos, commercial_allocation, width,
                bottom=np.array(false_allocation)+np.array(real_allocation)+np.array(fighter_allocation),
                label='Commercial', color='gray', alpha=0.5)
    p5 = ax3.bar(x_pos, other_allocation, width,
                bottom=np.array(false_allocation)+np.array(real_allocation)+
                       np.array(fighter_allocation)+np.array(commercial_allocation),
                label='Other/Search', color='lightgray', alpha=0.5)
    
    # Add percentage labels on significant segments
    for i, (false, real) in enumerate(zip(false_allocation, real_allocation)):
        if false > 0:
            ax3.text(i, false/2, f'{false}%', ha='center', va='center', 
                    color='white', fontweight='bold', fontsize=10)
        if real > 0:
            ax3.text(i, false + real/2, f'{real}%', ha='center', va='center',
                    color='white', fontweight='bold', fontsize=9)
    
    # Add critical annotation
    ax3.annotate('65% WASTED\non phantoms!', xy=(2, 37.5), xytext=(2.3, 60),
                arrowprops=dict(arrowstyle='->', color='red', linewidth=2),
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax3.set_ylabel('Resource Allocation (%)')
    ax3.set_ylim(0, 100)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(categories)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Add statistics text box
    stats_text = (
        "Deception Metrics:\n"
        "• False Targets: 4\n"
        "• Real Threats: 1\n" 
        "• Resources Wasted: 65%\n"
        "• Tracks Lost: 6\n"
        "• Mission Impact: SEVERE"
    )
    ax3.text(0.98, 0.45, stats_text, transform=ax3.transAxes,
            fontsize=8, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red'))
    
    # Adjust layout
    plt.subplots_adjust(top=0.92, bottom=0.08)
    
    return fig

def main():
    """Generate and save the Growler false target deception visualization"""
    
    # Create visualization
    fig = create_growler_deception_visualization()
    
    # Save figure
    output_dir = Path('results/growler_false_target_deception')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_dir / 'growler_deception_tracking.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_dir / 'growler_deception_tracking.png'}")
    
    # Also save as PDF for high quality
    fig.savefig(output_dir / 'growler_deception_tracking.pdf', bbox_inches='tight')
    print(f"Saved PDF version to {output_dir / 'growler_deception_tracking.pdf'}")
    
    plt.show()

if __name__ == "__main__":
    main()