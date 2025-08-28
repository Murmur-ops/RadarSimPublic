#!/usr/bin/env python3
"""
ATC Airport Rush Hour Visualization
Shows how civilian ATC radar prioritizes large passenger aircraft
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

def create_atc_visualization(config_file='configs/scenarios/atc_airport_rush_hour.yaml'):
    """
    Create visualization showing ATC priority management during rush hour
    """
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create figure with custom layout - wider to prevent cutoff
    fig = plt.figure(figsize=(19, 10))
    fig.suptitle('Airport ATC Radar - Extreme Rush Hour Surge\n5 Heavy Aircraft Causing Severe Resource Saturation', 
                 fontsize=14, fontweight='bold')
    
    # Create grid for subplots
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[2, 1, 1])
    
    # ========== LEFT: Airport Radar Picture ==========
    ax1 = fig.add_subplot(gs[:, 0])  # Span both rows
    ax1.set_title('Airport Approach Radar - 100km Range', fontweight='bold', pad=10)
    ax1.set_xlabel('East-West (km)')
    ax1.set_ylabel('North-South (km)')
    ax1.set_xlim(-115, 115)
    ax1.set_ylim(-115, 115)
    ax1.grid(True, alpha=0.3)
    
    # Draw airport at center
    airport = Circle((0, 0), 3, color='gray', zorder=5)
    ax1.add_patch(airport)
    ax1.text(0, 0, 'AIRPORT', ha='center', va='center', color='white', fontweight='bold', fontsize=8)
    
    # Draw runways
    runway1 = Rectangle((-8, -1), 16, 2, angle=0, color='black', alpha=0.8)
    runway2 = Rectangle((-1, -8), 2, 16, angle=0, color='black', alpha=0.8)
    ax1.add_patch(runway1)
    ax1.add_patch(runway2)
    
    # Draw range rings
    for r in [25, 50, 75, 100]:
        circle = Circle((0, 0), r, fill=False, edgecolor='green', alpha=0.2, linestyle='--')
        ax1.add_patch(circle)
        ax1.text(r, 5, f'{r}km', fontsize=7, alpha=0.4, color='green')
    
    # Draw approach corridors
    for angle in [0, 90, 180, 270]:
        rad = np.radians(angle)
        ax1.plot([0, 100*np.cos(rad)], [0, 100*np.sin(rad)], 
                'g--', alpha=0.2, linewidth=1)
    
    # Plot initial aircraft
    targets = config['targets']
    for target in targets:
        # Convert range/azimuth to x/y
        r = target['initial_position']['range'] / 1000  # Convert to km
        az = np.radians(target['initial_position']['azimuth'])
        x = r * np.sin(az)  # Note: swap sin/cos for compass bearing
        y = r * np.cos(az)
        
        # Determine marker and color based on type and outcome
        if 'general_aviation' in target['type']:
            marker = 'v'  # Small triangle
            size = 40
            default_color = 'lightblue'
        elif 'business_jet' in target['type']:
            marker = 's'  # Square
            size = 60
            default_color = 'cyan'
        elif 'regional_jet' in target['type']:
            marker = 'D'  # Diamond
            size = 80
            default_color = 'blue'
        elif 'medium_airliner' in target['type']:
            marker = 'o'  # Circle
            size = 100
            default_color = 'darkblue'
        elif 'large_airliner' in target['type']:
            marker = 'H'  # Hexagon
            size = 120
            default_color = 'navy'
        elif 'helicopter' in target['type']:
            marker = 'X'  # X for rotor
            size = 50
            default_color = 'orange'
        else:
            marker = 'o'
            size = 60
            default_color = 'gray'
        
        # Color by expected outcome
        if target['expected_outcome'] == 'LOST':
            color = 'red'
            alpha = 0.3
            edgecolor = 'darkred'
            edgewidth = 2
        elif target['expected_outcome'] == 'DEGRADED':
            color = 'yellow'
            alpha = 0.5
            edgecolor = 'orange'
            edgewidth = 1.5
        else:  # MAINTAINED
            color = default_color
            alpha = 0.8
            edgecolor = 'black'
            edgewidth = 0.5
            
        ax1.scatter(x, y, s=size, c=color, marker=marker, alpha=alpha, 
                   edgecolor=edgecolor, linewidth=edgewidth)
        
        # Add label - only for some to reduce clutter
        if target['priority'] >= 40 or 'LOST' in target['expected_outcome']:
            name_short = target['name'].split('-')[0][:6]  # Truncate long names
            ax1.text(x, y-5, name_short, fontsize=6, ha='center', alpha=0.6)
        
        # Add X mark for dropped tracks
        if target['expected_outcome'] == 'LOST':
            ax1.plot([x-3, x+3], [y-3, y+3], 'r-', linewidth=2, alpha=0.7)
            ax1.plot([x-3, x+3], [y+3, y-3], 'r-', linewidth=2, alpha=0.7)
            ax1.text(x+5, y, 'DROPPED', fontsize=5, color='darkred', 
                    fontweight='bold', style='italic')
    
    # Draw rush hour arrivals (large aircraft)
    rush_aircraft = config['scenario']['events'][0]['parameters']['aircraft']
    
    for aircraft in rush_aircraft:
        x = aircraft['position'][0]
        y = aircraft['position'][1]
        
        # Large aircraft symbols
        if 'A380' in aircraft['name']:
            marker = '*'  # Star for super-heavy
            size = 200
            color = 'gold'
        elif '747' in aircraft['name']:
            marker = 'h'  # Hexagon for heavy
            size = 150
            color = 'goldenrod'
        else:
            marker = 'H'
            size = 120
            color = 'orange'
        
        ax1.scatter(x, y, s=size, c=color, marker=marker, 
                   edgecolor='black', linewidth=2, zorder=3)
        
        # Add label with passenger count if available
        name = aircraft['name'].split('-')[0]
        if 'passengers' in aircraft:
            label = f"{name}\n({aircraft['passengers']} pax)"
        elif 'cargo_tons' in aircraft:
            label = f"{name}\n({aircraft['cargo_tons']}t cargo)"
        else:
            label = name
            
        ax1.text(x, y-8, label, fontsize=7, ha='center', 
                fontweight='bold', color='darkred')
        
        # Draw approach vector
        vx, vy = aircraft['velocity'][0]/10, aircraft['velocity'][1]/10
        ax1.arrow(x, y, vx, vy, head_width=3, head_length=4,
                 fc=color, ec='darkred', alpha=0.6, linewidth=1.5)
    
    # Add rush hour marker and drop count
    ax1.text(85, 85, 'RUSH HOUR\nARRIVALS', fontsize=10, color='darkred', 
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Add dropped tracks summary
    dropped_text = "TRACKS DROPPED:\n• 2 Cessnas\n• 2 Business Jets\n• 1 Regional Jet\n• 1 Helicopter"
    ax1.text(-100, -90, dropped_text, fontsize=8, color='darkred',
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', 
                                       edgecolor='red', linewidth=2, alpha=0.9))
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                  markersize=12, label='A380 Super Heavy'),
        plt.Line2D([0], [0], marker='H', color='w', markerfacecolor='navy', 
                  markersize=10, label='Large Airliner'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', 
                  markersize=8, label='Medium Airliner'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='blue', 
                  markersize=8, label='Regional Jet'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', 
                  markersize=7, label='Business Jet'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='red', 
                  markersize=7, label='General Aviation')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=6, ncol=1, 
              framealpha=0.8)
    
    # ========== TOP RIGHT: Track Quality Evolution ==========
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.set_title('Track Quality Evolution - Priority Impact', fontweight='bold')
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
    ax2.axhline(y=0.25, color='red', linestyle='--', alpha=0.3, linewidth=2, label='Drop')
    
    # Rush hour marker
    ax2.axvline(x=30, color='darkred', linestyle=':', linewidth=2, alpha=0.5)
    ax2.text(30, 0.5, 'RUSH\nHOUR', fontsize=8, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Track quality curves
    # General aviation - immediate drop with 5 large aircraft
    for name in ['Cessna-1', 'Cessna-2']:
        quality = np.ones_like(t) * 0.75
        quality[t >= 30] = 0.75 * np.exp(-0.25 * (t[t >= 30] - 30))  # Faster drop
        quality[quality < 0.25] = 0.23
        ax2.plot(t, quality, 'r--', alpha=0.5, linewidth=1)
        
        # Mark drop point
        drop_idx = np.where(quality < 0.25)[0]
        if len(drop_idx) > 0:
            drop_time = t[drop_idx[0]]
            ax2.scatter(drop_time, 0.23, s=50, c='red', marker='X', zorder=5)
    
    # Business jets - now also dropping
    for name in ['Gulfstream', 'Citation']:
        quality = np.ones_like(t) * 0.85
        quality[t >= 30] = 0.85 * np.exp(-0.12 * (t[t >= 30] - 30))  # Will drop
        quality[quality < 0.25] = 0.23
        ax2.plot(t, quality, 'r-', alpha=0.6, linewidth=1.5)
        
        # Mark drop point
        drop_idx = np.where(quality < 0.25)[0]
        if len(drop_idx) > 0:
            drop_time = t[drop_idx[0]]
            ax2.scatter(drop_time, 0.23, s=50, c='red', marker='X', zorder=5)
    
    # Large aircraft - excellent tracking (now 5 aircraft)
    for i, name in enumerate(['A380-1', 'A380-2', 'A380-3', '747-1', '747-2']):
        quality = np.zeros_like(t)
        quality[t >= 30] = 0.95  # High quality immediately
        colors = ['gold', 'gold', 'gold', 'goldenrod', 'goldenrod']
        ax2.plot(t, quality, linewidth=1.5, color=colors[i], 
                label=name if i < 3 else None, alpha=0.8)
    
    # Medium airliners - now degraded significantly
    quality_med = np.ones_like(t) * 0.88
    quality_med[t >= 30] = 0.88 - 0.3 * (1 - np.exp(-0.08 * (t[t >= 30] - 30)))  # Degraded
    quality_med[quality_med < 0.5] = 0.5  # Stabilize at poor quality
    ax2.plot(t, quality_med, 'y-', alpha=0.7, linewidth=1.5, label='B737/A320 (degraded)')
    
    ax2.legend(loc='lower left', fontsize=6, ncol=2)
    
    # ========== BOTTOM RIGHT: Resource Allocation ==========
    ax3 = fig.add_subplot(gs[1, 1:])
    ax3.set_title('Resource Allocation by Aircraft Size', fontweight='bold')
    
    # Categories and allocations - now more extreme with 5 large aircraft
    categories = ['Normal Traffic', 'Extreme Rush']
    super_heavy = [0, 75]  # 3 A380s get 75%
    heavy = [35, 15]  # 2 747s get 15%
    medium = [30, 7]  # 737/A320 severely reduced
    light = [20, 2]  # Business jets almost nothing
    general = [15, 1]  # Cessnas abandoned
    
    width = 0.5
    x_pos = np.arange(len(categories))
    
    # Create stacked bars
    p1 = ax3.bar(x_pos, super_heavy, width, label='Super Heavy (A380)', 
                color='gold', edgecolor='black')
    p2 = ax3.bar(x_pos, heavy, width, bottom=super_heavy,
                label='Heavy (777/747)', color='goldenrod')
    p3 = ax3.bar(x_pos, medium, width, 
                bottom=np.array(super_heavy)+np.array(heavy),
                label='Medium (737/A320)', color='darkblue', alpha=0.7)
    p4 = ax3.bar(x_pos, light, width,
                bottom=np.array(super_heavy)+np.array(heavy)+np.array(medium),
                label='Light/Business', color='cyan', alpha=0.5)
    p5 = ax3.bar(x_pos, general, width,
                bottom=np.array(super_heavy)+np.array(heavy)+np.array(medium)+np.array(light),
                label='General Aviation', color='lightblue', alpha=0.3)
    
    # Add percentage labels - updated for 5 aircraft
    ax3.text(1, 37.5, '75%', ha='center', va='center', 
            color='black', fontweight='bold', fontsize=11)
    ax3.text(1, 99, '1%', ha='center', va='center',
            color='red', fontweight='bold', fontsize=9)
    
    # Add annotation - moved left to avoid cutoff
    ax3.annotate('Small aircraft\nlose tracking!', xy=(1, 97), xytext=(0.7, 85),
                arrowprops=dict(arrowstyle='->', color='red', linewidth=2),
                fontsize=9, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.set_ylabel('Resource Allocation (%)')
    ax3.set_ylim(0, 100)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(categories)
    ax3.legend(loc='upper left', fontsize=7)
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Add safety metrics text - updated for 5 aircraft scenario
    safety_text = (
        "Safety Impact:\n"
        "• Large AC: 100%\n"
        "• Medium AC: 50%\n" 
        "• Total pax: 2,870\n"
        "• Lost: 6 tracks\n"
        "• Safe: DEGRADED"
    )
    ax3.text(0.98, 0.40, safety_text, transform=ax3.transAxes,
            fontsize=7, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, pad=0.3))  # Yellow for degraded safety
    
    # Adjust layout - more space on right for bottom panel
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, wspace=0.25, hspace=0.3)
    
    return fig

def main():
    """Generate and save the ATC airport visualization"""
    
    # Create visualization
    fig = create_atc_visualization()
    
    # Save figure
    output_dir = Path('results/atc_airport_rush_hour')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_dir / 'atc_rush_hour_tracking.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_dir / 'atc_rush_hour_tracking.png'}")
    
    # Also save as PDF for high quality
    fig.savefig(output_dir / 'atc_rush_hour_tracking.pdf', bbox_inches='tight')
    print(f"Saved PDF version to {output_dir / 'atc_rush_hour_tracking.pdf'}")
    
    plt.show()

if __name__ == "__main__":
    main()