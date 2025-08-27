#!/usr/bin/env python3
"""
Missile Salvo Impact Visualization
Generates the three-panel figure showing track dropping due to resource saturation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def create_missile_salvo_visualization(config_file='configs/scenarios/missile_salvo_impact.yaml'):
    """
    Create visualization matching the missile salvo tracking image
    """
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Missile Salvo Impact on Radar Tracking\n3-Missile Pincer Attack from Multiple Vectors', 
                 fontsize=14, fontweight='bold')
    
    # Create grid for subplots
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[2, 1, 1])
    
    # ========== TOP LEFT: Radar Tracking Picture ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Radar Tracking Picture - Salvo Attack Scenario', fontweight='bold')
    ax1.set_xlabel('Range (km)')
    ax1.set_ylabel('Cross-Range (km)')
    ax1.set_xlim(-10, 110)
    ax1.set_ylim(-50, 50)
    ax1.grid(True, alpha=0.3)
    
    # Draw radar at origin
    radar = Circle((0, 0), 2, color='blue', zorder=5)
    ax1.add_patch(radar)
    ax1.text(0, 0, 'RADAR', ha='center', va='center', color='white', fontweight='bold', fontsize=8)
    
    # Draw range rings
    for r in [20, 40, 60, 80, 100]:
        circle = Circle((0, 0), r, fill=False, edgecolor='gray', alpha=0.3, linestyle='--')
        ax1.add_patch(circle)
        ax1.text(r, 2, f'{r}km', fontsize=8, alpha=0.5)
    
    # Draw fighter screen zone (shaded area)
    fighter_zone = FancyBboxPatch((35, -15), 25, 30, 
                                  boxstyle="round,pad=0.02",
                                  facecolor='orange', alpha=0.2,
                                  edgecolor='orange', linewidth=2)
    ax1.add_patch(fighter_zone)
    ax1.text(47, 0, 'Fighter\nScreen\nZone', ha='center', va='center', 
             fontsize=10, color='darkorange', fontweight='bold')
    
    # Plot initial targets
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
            alpha = 0.6
        elif target['expected_outcome'] == 'DEGRADED':
            marker = 'v'
            color = 'yellow'
            alpha = 0.8
        else:  # MAINTAINED
            marker = 'o'
            color = 'green'
            alpha = 1.0
            
        # Determine symbol based on type
        if 'commercial' in target['type'].lower():
            marker_symbol = 'P'  # Plane
            size = 100
        elif 'fighter' in target['type'].lower():
            marker_symbol = 'v'  # Triangle
            size = 80
        else:
            marker_symbol = 'o'
            size = 60
            
        ax1.scatter(x, y, s=size, c=color, marker=marker_symbol, alpha=alpha, 
                   edgecolor='black', linewidth=0.5)
        
        # Add label
        name_short = target['name'].split('-')[1]
        ax1.text(x, y-3, name_short, fontsize=7, ha='center')
        
        # Add status annotation
        if target['expected_outcome'] == 'LOST':
            ax1.text(x+5, y, '(LOST)', fontsize=6, color='red', style='italic')
    
    # Draw missile vectors (Vampire 1, 2, 3)
    missile_positions = [
        (55, 35, 'Vampire-1'),   # North
        (50, 0, 'Vampire-2'),     # Center
        (60, -40, 'Vampire-3')    # South
    ]
    
    for x, y, name in missile_positions:
        # Draw missile icon
        ax1.scatter(x, y, s=150, c='red', marker='^', edgecolor='black', linewidth=2)
        ax1.text(x, y-5, name, fontsize=8, color='red', fontweight='bold')
        
        # Draw attack vector
        ax1.arrow(x, y, -x*0.4, -y*0.4, head_width=2, head_length=3,
                 fc='red', ec='darkred', alpha=0.6, linewidth=2)
    
    # Add SALVO DETECTED marker
    ax1.text(70, 40, 'SALVO DETECTED', fontsize=12, color='red', 
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=8, label='Tracked (Good)'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='y', markersize=8, label='Tracked (Degraded)'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='orange', markersize=8, label='Fighter'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='r', markersize=8, label='Track Lost'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='r', markersize=8, label='Missile')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=8)
    
    # ========== TOP RIGHT: Track Statistics ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Track Statistics\n(Salvo Attack)', fontweight='bold')
    
    categories = ['Initial', 'Active', 'Degraded', 'Lost', 'Missiles']
    values = [10, 4, 0, 6, 3]
    colors_bar = ['blue', 'green', 'yellow', 'red', 'darkred']
    
    bars = ax2.bar(categories, values, color=colors_bar, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax2.set_ylabel('Number of Tracks')
    ax2.set_ylim(0, 12)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # ========== TOP RIGHT: Resource Allocation Pie ==========
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title('Resource Allocation\n(Salvo Crisis Mode)', fontweight='bold')
    
    sizes = [80, 10, 7, 3]
    labels = ['3 Missiles\n(80%)', 'Fighters\n(10%)', 'Other Military\n(7%)', 'Commercial\n(3%)']
    colors_pie = ['red', 'orange', 'yellow', 'lightgray']
    explode = (0.1, 0.05, 0.05, 0.05)  # Explode missile slice
    
    wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors_pie,
                                        explode=explode, autopct='%1.0f%%',
                                        shadow=True, startangle=45)
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    # ========== BOTTOM: Track Quality Evolution ==========
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_title('Track Quality Evolution - 3-Missile Salvo Impact', fontweight='bold')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Track Quality')
    ax4.set_xlim(0, 80)
    ax4.set_ylim(0, 1.0)
    ax4.grid(True, alpha=0.3)
    
    # Time array
    t = np.linspace(0, 80, 161)
    
    # Quality thresholds
    ax4.axhline(y=0.8, color='green', linestyle='--', alpha=0.3, label='Good')
    ax4.axhline(y=0.6, color='orange', linestyle='--', alpha=0.3, label='Fair')
    ax4.axhline(y=0.2, color='red', linestyle='--', alpha=0.3, linewidth=2, label='Drop')
    
    # SALVO LAUNCH marker
    ax4.axvline(x=30, color='red', linestyle=':', linewidth=3, alpha=0.5)
    ax4.text(30, 0.5, 'SALVO\nLAUNCH', fontsize=10, fontweight='bold',
            ha='center', va='center', rotation=0,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Track quality evolution curves
    # Commercial aircraft - rapid degradation after salvo
    for i, name in enumerate(['Commercial-747', 'Commercial-A320', 'Commercial-B737']):
        quality = np.ones_like(t) * 0.85
        quality[t >= 30] = 0.85 * np.exp(-0.15 * (t[t >= 30] - 30))
        quality[quality < 0.2] = 0.18  # Just below drop threshold
        ax4.plot(t, quality, label=name, linestyle='--', alpha=0.6, color='gray')
    
    # Fighters - moderate degradation but maintained
    fighter_colors = ['blue', 'cyan', 'navy', 'darkblue']
    for i, name in enumerate(['Fighter-F22', 'Fighter-F35', 'Fighter-F18', 'Fighter-F16']):
        quality = np.ones_like(t) * 0.90
        quality[t >= 30] = 0.90 - 0.35 * (1 - np.exp(-0.05 * (t[t >= 30] - 30)))
        if 'F22' in name:
            quality[t >= 30] = 0.75  # F22 maintains higher quality
        ax4.plot(t, quality, label=name, linestyle='-', color=fighter_colors[i], alpha=0.8)
    
    # Other military - lost
    for name in ['Helicopter-MH60', 'Drone-MQ9', 'Transport-C130']:
        quality = np.ones_like(t) * 0.75
        quality[t >= 30] = 0.75 * np.exp(-0.12 * (t[t >= 30] - 30))
        quality[quality < 0.2] = 0.18
        ax4.plot(t, quality, linestyle=':', alpha=0.5, color='brown')
    
    # Missiles - high quality tracking after launch
    for i, name in enumerate(['Vampire-1', 'Vampire-2', 'Vampire-3']):
        quality = np.zeros_like(t)
        quality[t >= 30] = 0.95  # High quality tracking
        ax4.plot(t, quality, label=name, linestyle='-', linewidth=2, color='red', alpha=0.8-i*0.1)
    
    # Legend
    ax4.legend(loc='upper right', ncol=3, fontsize=8)
    
    plt.tight_layout()
    return fig

def main():
    """Generate and save the missile salvo visualization"""
    
    # Create visualization
    fig = create_missile_salvo_visualization()
    
    # Save figure
    output_dir = Path('results/missile_salvo_impact')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_dir / 'missile_salvo_tracking.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_dir / 'missile_salvo_tracking.png'}")
    
    # Also save as PDF for high quality
    fig.savefig(output_dir / 'missile_salvo_tracking.pdf', bbox_inches='tight')
    print(f"Saved PDF version to {output_dir / 'missile_salvo_tracking.pdf'}")
    
    plt.show()

if __name__ == "__main__":
    main()