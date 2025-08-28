#!/usr/bin/env python3
"""
Enhanced Missile Salvo Impact Visualization with Trajectory Tracking
Shows detection points, track gates, and drop locations along object trajectories
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def calculate_trajectory_points(initial_pos, velocity, duration=80, dt=1.0, drop_time=None):
    """
    Calculate trajectory points for a target
    
    Args:
        initial_pos: Dict with range, azimuth (degrees), elevation
        velocity: [vx, vy, vz] in m/s
        duration: Total simulation time
        dt: Time step
        drop_time: Time when track was lost (None if maintained)
    
    Returns:
        times, positions (x, y in km)
    """
    # Convert initial position to Cartesian
    r0 = initial_pos['range'] / 1000  # km
    az0 = np.radians(initial_pos.get('azimuth', 0))
    x0 = r0 * np.cos(az0)
    y0 = r0 * np.sin(az0)
    
    # Time array
    if drop_time:
        times = np.arange(0, min(drop_time, duration) + dt, dt)
    else:
        times = np.arange(0, duration + dt, dt)
    
    # Calculate positions
    vx_km_s = velocity[0] / 1000  # Convert m/s to km/s
    vy_km_s = velocity[1] / 1000 if len(velocity) > 1 else 0
    
    x_positions = x0 + vx_km_s * times
    y_positions = y0 + vy_km_s * times
    
    return times, x_positions, y_positions

def draw_track_gate(ax, x, y, quality, scale=1.0):
    """
    Draw track gate (uncertainty ellipse) at given position
    
    Args:
        ax: Matplotlib axis
        x, y: Position in km
        quality: Track quality (0-1)
        scale: Scale factor for gate size
    """
    # Gate size inversely proportional to quality
    base_size = 2.0  # km
    uncertainty = base_size * (2.0 - quality) * scale
    
    # Color based on quality
    if quality > 0.8:
        color = 'green'
        alpha = 0.3
    elif quality > 0.6:
        color = 'yellow'
        alpha = 0.4
    elif quality > 0.2:
        color = 'orange'
        alpha = 0.5
    else:
        color = 'red'
        alpha = 0.6
    
    gate = Ellipse((x, y), uncertainty * 2, uncertainty * 1.5, 
                   angle=np.random.uniform(-30, 30),  # Random orientation
                   fill=False, edgecolor=color, linewidth=1.5, 
                   linestyle='--', alpha=alpha)
    ax.add_patch(gate)
    
    return gate

def create_enhanced_tracking_visualization(config_file='configs/scenarios/missile_salvo_impact.yaml'):
    """
    Create enhanced visualization with trajectory tracking details
    """
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create figure with optimized layout
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Missile Salvo Impact - Complete Tracking Lifecycle\nShowing Detection, Track Gates, and Drop Points', 
                 fontsize=14, fontweight='bold')
    
    # Create grid: give more space to scenario view
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1.2, 0.8],
                  hspace=0.25, wspace=0.2)
    
    # ========== LEFT HALF: Radar Tracking with Trajectories (spans full height) ==========
    ax1 = fig.add_subplot(gs[:, 0])  # Span both rows, left column
    ax1.set_title('Radar Tracking Picture - Target Trajectories with Track Gates', fontweight='bold', pad=10)
    ax1.set_xlabel('Range (km)')
    ax1.set_ylabel('Cross-Range (km)')
    ax1.set_xlim(-15, 115)  # Add padding
    ax1.set_ylim(-55, 55)   # Add padding
    ax1.grid(True, alpha=0.3)
    # Remove aspect constraint to use full space
    
    # Draw radar at origin
    radar = Circle((0, 0), 2, color='blue', zorder=10)
    ax1.add_patch(radar)
    ax1.text(0, 0, 'RADAR', ha='center', va='center', color='white', 
             fontweight='bold', fontsize=8, zorder=11)
    
    # Draw range rings
    for r in [20, 40, 60, 80, 100]:
        circle = Circle((0, 0), r, fill=False, edgecolor='gray', 
                       alpha=0.2, linestyle='--', linewidth=0.5)
        ax1.add_patch(circle)
        ax1.text(r, 2, f'{r}km', fontsize=7, alpha=0.4)
    
    # Draw fighter screen zone
    fighter_zone = FancyBboxPatch((35, -15), 25, 30, 
                                  boxstyle="round,pad=0.02",
                                  facecolor='orange', alpha=0.15,
                                  edgecolor='orange', linewidth=1.5)
    ax1.add_patch(fighter_zone)
    ax1.text(47, 0, 'Fighter\nScreen\nZone', ha='center', va='center', 
             fontsize=9, color='darkorange', fontweight='bold')
    
    # Process and plot each target with trajectory
    targets = config['targets']
    
    for target in targets:
        name = target['name']
        target_type = target['type']
        initial_pos = target['initial_position']
        velocity = target['velocity']
        outcome = target['expected_outcome']
        
        # Determine drop time based on outcome
        if outcome == 'LOST':
            if 'commercial' in target_type.lower():
                drop_time = np.random.uniform(40, 45)  # Commercial drops 40-45s
            else:
                drop_time = np.random.uniform(45, 50)  # Others drop 45-50s
        else:
            drop_time = None
        
        # Calculate trajectory
        times, x_traj, y_traj = calculate_trajectory_points(
            initial_pos, velocity, duration=80, drop_time=drop_time)
        
        # Plot trajectory with color coding
        if outcome == 'LOST':
            # Green until t=30, then degrading to red
            for i in range(len(times) - 1):
                t = times[i]
                if t < 30:
                    color = 'green'
                    style = '-'
                    width = 1.5
                elif t < 35:
                    color = 'yellow'
                    style = '-'
                    width = 1.2
                elif t < drop_time - 5:
                    color = 'orange'
                    style = '--'
                    width = 1.0
                else:
                    color = 'red'
                    style = ':'
                    width = 0.8
                
                ax1.plot(x_traj[i:i+2], y_traj[i:i+2], color=color, 
                        linestyle=style, linewidth=width, alpha=0.6)
        
        elif outcome == 'DEGRADED':
            # Green until t=30, then yellow
            mask_pre = times < 30
            mask_post = times >= 30
            ax1.plot(x_traj[mask_pre], y_traj[mask_pre], 'g-', linewidth=1.5, alpha=0.7)
            ax1.plot(x_traj[mask_post], y_traj[mask_post], 'y-', linewidth=1.2, alpha=0.6)
        
        else:  # MAINTAINED
            # Green throughout
            ax1.plot(x_traj, y_traj, 'g-', linewidth=1.5, alpha=0.8)
        
        # Mark detection point (at t=0)
        ax1.scatter(x_traj[0], y_traj[0], s=100, marker='D', 
                   facecolor='cyan', edgecolor='black', linewidth=1.5,
                   zorder=5, label='Detection' if target == targets[0] else '')
        ax1.text(x_traj[0] + 2, y_traj[0], 'Det', fontsize=6, style='italic')
        
        # Draw track gates at key times
        gate_times = [10, 20, 30, 35, 40]
        if drop_time:
            gate_times = [t for t in gate_times if t <= drop_time]
        
        for t in gate_times:
            if t < len(times):
                idx = int(t)
                # Quality degrades after salvo
                if t <= 30:
                    quality = 0.85
                elif outcome == 'LOST':
                    quality = 0.85 * np.exp(-0.1 * (t - 30))
                elif outcome == 'DEGRADED':
                    quality = 0.65 - 0.01 * (t - 30)
                else:
                    quality = 0.80
                
                # Draw gate
                draw_track_gate(ax1, x_traj[idx], y_traj[idx], quality)
        
        # Mark drop point if track was lost
        if outcome == 'LOST' and drop_time:
            drop_idx = min(int(drop_time), len(x_traj) - 1)
            # Large X at drop point
            ax1.scatter(x_traj[drop_idx], y_traj[drop_idx], s=200, marker='X', 
                       color='red', edgecolor='darkred', linewidth=2, zorder=7)
            # Drop annotation - adjust position to avoid cropping
            text_x = min(x_traj[drop_idx] + 5, 105)  # Keep within bounds
            text_y = y_traj[drop_idx] + 3
            if abs(text_y) > 45:  # If near edge, adjust
                text_y = np.sign(text_y) * 45
            ax1.annotate(f'LOST\n@ t={int(drop_time)}s', 
                        xy=(x_traj[drop_idx], y_traj[drop_idx]),
                        xytext=(text_x, text_y),
                        fontsize=7, color='red', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='red', lw=1))
            # Large uncertainty gate at drop
            draw_track_gate(ax1, x_traj[drop_idx], y_traj[drop_idx], 0.15, scale=2.0)
        
        # Add target label at current/final position
        final_idx = min(40, len(x_traj) - 1) if not drop_time else min(int(drop_time), len(x_traj) - 1)
        name_short = name.split('-')[1] if '-' in name else name
        ax1.text(x_traj[final_idx], y_traj[final_idx] - 2, name_short, 
                fontsize=7, ha='center', fontweight='bold')
        
        # Add time tick marks along trajectory
        for t in [10, 20, 30, 40, 50, 60]:
            if t < len(times) and (not drop_time or t < drop_time):
                idx = int(t)
                ax1.plot(x_traj[idx], y_traj[idx], 'k.', markersize=3)
                if t % 20 == 0:
                    ax1.text(x_traj[idx], y_traj[idx] + 1, f't={t}', 
                            fontsize=5, alpha=0.5)
    
    # Add missiles (starting at t=30)
    missile_data = [
        {'name': 'Vampire-1', 'start': [55, 35], 'velocity': [-350/1000, -100/1000]},
        {'name': 'Vampire-2', 'start': [50, 0], 'velocity': [-380/1000, 0]},
        {'name': 'Vampire-3', 'start': [60, -40], 'velocity': [-340/1000, 80/1000]}
    ]
    
    for missile in missile_data:
        # Calculate missile trajectory from t=30 to t=80
        t_missile = np.arange(0, 50, 1)  # 50 seconds of flight
        x_missile = missile['start'][0] + missile['velocity'][0] * t_missile * 1000
        y_missile = missile['start'][1] + missile['velocity'][1] * t_missile * 1000
        
        # Plot missile trajectory
        ax1.plot(x_missile, y_missile, 'r-', linewidth=2, alpha=0.8)
        
        # Missile icon at launch position
        ax1.scatter(missile['start'][0], missile['start'][1], s=150, marker='^', 
                   color='red', edgecolor='black', linewidth=2, zorder=8)
        ax1.text(missile['start'][0], missile['start'][1] - 3, missile['name'], 
                fontsize=8, color='red', fontweight='bold', ha='center')
        
        # Detection diamond at launch
        ax1.scatter(missile['start'][0], missile['start'][1], s=120, marker='D', 
                   facecolor='none', edgecolor='red', linewidth=2)
        
        # Track gates for missiles (small, high quality)
        for i in range(0, min(30, len(x_missile)), 10):
            draw_track_gate(ax1, x_missile[i], y_missile[i], 0.95, scale=0.5)
        
        # Velocity vector arrow
        ax1.arrow(missile['start'][0], missile['start'][1], 
                 missile['velocity'][0] * 20000, missile['velocity'][1] * 20000,
                 head_width=1.5, head_length=2, fc='darkred', ec='darkred', alpha=0.6)
    
    # Add SALVO DETECTED marker - position to avoid cropping
    ax1.text(75, 42, 'SALVO DETECTED\n@ t=30s', fontsize=11, color='red', 
            fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='cyan', 
                  markersize=8, label='Initial Detection'),
        plt.Line2D([0], [0], color='green', linewidth=2, label='Good Track'),
        plt.Line2D([0], [0], color='yellow', linewidth=2, label='Degraded Track'),
        plt.Line2D([0], [0], color='red', linewidth=2, linestyle=':', label='Coasting/Lost'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', 
                  markersize=10, label='Track Drop Point'),
        mpatches.Ellipse((0, 0), 0.2, 0.1, fc='none', ec='green', 
                        linestyle='--', label='Track Gate (Good)'),
        mpatches.Ellipse((0, 0), 0.2, 0.1, fc='none', ec='red', 
                        linestyle='--', label='Track Gate (Poor)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                  markersize=8, label='Missile')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=7, ncol=2)
    
    # ========== TOP RIGHT: Track Quality Timeline ==========
    ax2 = fig.add_subplot(gs[0, 1])  # Top row, right column
    ax2.set_title('Track Quality Evolution with Drop Events', fontweight='bold', pad=10)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Track Quality')
    ax2.set_xlim(0, 80)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)
    
    # Time array
    t = np.linspace(0, 80, 161)
    
    # Quality thresholds
    ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax2.text(78, 0.82, 'Good', fontsize=8, color='green')
    ax2.axhline(y=0.6, color='orange', linestyle='--', alpha=0.3, linewidth=1)
    ax2.text(78, 0.62, 'Fair', fontsize=8, color='orange')
    ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax2.text(78, 0.22, 'DROP', fontsize=8, color='red', fontweight='bold')
    
    # SALVO LAUNCH marker
    ax2.axvline(x=30, color='red', linestyle=':', linewidth=3, alpha=0.5)
    ax2.text(30, 0.5, 'SALVO\nLAUNCH', fontsize=9, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Track quality curves
    # Commercial aircraft - drops at ~42s
    for i, name in enumerate(['Commercial-747', 'Commercial-A320', 'Commercial-B737']):
        quality = np.ones_like(t) * 0.85
        drop_time = 40 + i * 2  # Stagger drops
        quality[t >= 30] = 0.85 * np.exp(-0.15 * (t[t >= 30] - 30))
        quality[t >= drop_time] = 0.0  # Drop to zero
        ax2.plot(t, quality, label=name, linestyle='--', alpha=0.6)
        # Mark drop point
        ax2.scatter(drop_time, 0.2, s=100, marker='X', color='red', 
                   edgecolor='darkred', zorder=5)
        ax2.text(drop_time, 0.1, f'DROP\n{drop_time}s', fontsize=6, 
                ha='center', color='red')
    
    # Fighters - degraded but maintained
    for name in ['Fighter-F22', 'Fighter-F35']:
        quality = np.ones_like(t) * 0.90
        quality[t >= 30] = 0.65 - 0.005 * (t[t >= 30] - 30)
        ax2.plot(t, quality, label=name, linewidth=1.5, alpha=0.7)
    
    # Missiles - high quality after launch
    for name in ['Vampire-1', 'Vampire-2', 'Vampire-3']:
        quality = np.zeros_like(t)
        quality[t >= 30] = 0.95
        ax2.plot(t, quality, 'r-', linewidth=2, alpha=0.6)
        if name == 'Vampire-1':
            ax2.plot([], [], 'r-', linewidth=2, label='Missiles')  # Single legend entry
    
    ax2.legend(loc='best', fontsize=7, ncol=2)
    
    # ========== BOTTOM RIGHT: Track Statistics ==========
    ax3 = fig.add_subplot(gs[1, 1])  # Bottom row, right column
    ax3.set_title('Track Statistics Over Time', fontweight='bold', pad=10)
    
    # Time points for statistics
    time_points = [0, 30, 35, 40, 45, 80]
    stats = {
        'Tracked': [10, 10, 8, 5, 4, 4],
        'Degraded': [0, 0, 2, 2, 1, 0],
        'Lost': [0, 0, 0, 3, 5, 6],
        'Missiles': [0, 3, 3, 3, 3, 3]
    }
    
    x = np.arange(len(time_points))
    width = 0.2
    
    colors = {'Tracked': 'green', 'Degraded': 'yellow', 
             'Lost': 'red', 'Missiles': 'darkred'}
    
    for i, (label, values) in enumerate(stats.items()):
        ax3.bar(x + i * width, values, width, label=label, 
               color=colors[label], edgecolor='black', linewidth=1)
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Number of Tracks')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels([f't={t}' for t in time_points])
    ax3.legend(fontsize=8)
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Adjust subplot spacing for better fit - increase bottom margin for x-axis label
    fig.subplots_adjust(top=0.91, left=0.05, right=0.98, bottom=0.08)
    return fig

def main():
    """Generate and save the enhanced missile salvo visualization"""
    
    # Create visualization
    fig = create_enhanced_tracking_visualization()
    
    # Save figure
    output_dir = Path('results/missile_salvo_impact')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_dir / 'missile_salvo_trajectory_tracking.png', 
               dpi=150, bbox_inches='tight')
    print(f"Saved enhanced visualization to {output_dir / 'missile_salvo_trajectory_tracking.png'}")
    
    # Also save as PDF for high quality
    fig.savefig(output_dir / 'missile_salvo_trajectory_tracking.pdf', 
               bbox_inches='tight')
    print(f"Saved PDF version to {output_dir / 'missile_salvo_trajectory_tracking.pdf'}")
    
    plt.show()

if __name__ == "__main__":
    main()