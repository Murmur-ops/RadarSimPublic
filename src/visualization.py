"""
Visualization utilities for radar simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Wedge
from typing import Optional, Tuple, List, Any
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


class Visualizer:
    """Radar visualization tools"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        self.colormap = 'viridis'
        
    def plot_range_profile(self, ranges: np.ndarray, power_db: np.ndarray,
                          detections: Optional[np.ndarray] = None,
                          threshold: Optional[np.ndarray] = None,
                          title: str = "Range Profile") -> plt.Figure:
        """
        Plot range profile with optional detections
        
        Args:
            ranges: Range bins in meters
            power_db: Power in dB
            detections: Detection mask
            threshold: Detection threshold
            title: Plot title
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot range profile
        ax.plot(ranges / 1000, power_db, 'b-', linewidth=1, label='Signal')
        
        # Plot threshold if provided
        if threshold is not None:
            ax.plot(ranges / 1000, 10 * np.log10(threshold + 1e-20), 
                   'r--', linewidth=1, label='CFAR Threshold')
        
        # Mark detections
        if detections is not None:
            detection_indices = np.where(detections)[0]
            if len(detection_indices) > 0:
                ax.plot(ranges[detection_indices] / 1000, 
                       power_db[detection_indices],
                       'ro', markersize=8, label='Detections')
        
        ax.set_xlabel('Range (km)')
        ax.set_ylabel('Power (dB)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig
    
    def plot_range_doppler(self, rd_map: np.ndarray,
                         range_bins: np.ndarray,
                         velocity_bins: np.ndarray,
                         dynamic_range: float = 60,
                         title: str = "Range-Doppler Map") -> plt.Figure:
        """
        Plot range-Doppler map
        
        Args:
            rd_map: Range-Doppler data in dB
            range_bins: Range values in meters
            velocity_bins: Velocity values in m/s
            dynamic_range: Dynamic range to display in dB
            title: Plot title
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Normalize and clip data
        vmax = np.max(rd_map)
        vmin = vmax - dynamic_range
        rd_display = np.clip(rd_map, vmin, vmax)
        
        # Create mesh
        V, R = np.meshgrid(velocity_bins, range_bins / 1000)
        
        # Plot
        im = ax.pcolormesh(V, R, rd_display, cmap=self.colormap,
                          shading='auto', vmin=vmin, vmax=vmax)
        
        ax.set_xlabel('Velocity (m/s)')
        ax.set_ylabel('Range (km)')
        ax.set_title(title)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)')
        
        return fig
    
    def plot_ppi(self, ranges: np.ndarray, azimuths: np.ndarray,
                data: np.ndarray, max_range: Optional[float] = None,
                title: str = "Plan Position Indicator",
                dynamic_range: float = 40) -> plt.Figure:
        """
        Plot Plan Position Indicator (PPI) display
        
        Args:
            ranges: Range bins in meters
            azimuths: Azimuth angles in radians
            data: 2D data array (range x azimuth)
            max_range: Maximum range to display
            title: Plot title
            dynamic_range: Dynamic range in dB for display
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='polar')
        
        if max_range is None:
            max_range = np.max(ranges)
        
        # Create mesh
        theta, r = np.meshgrid(azimuths, ranges / 1000)
        
        # Convert to dB with dynamic range limiting
        data_db = 20 * np.log10(np.abs(data) + 1e-10)
        vmax = np.max(data_db)
        vmin = vmax - dynamic_range
        
        # Plot with adjusted color scaling
        im = ax.pcolormesh(theta, r, data_db,
                          cmap='hot', shading='auto',
                          vmin=vmin, vmax=vmax)
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, max_range / 1000)
        ax.set_title(title)
        
        # Add range rings with labels
        range_rings = np.arange(0, max_range / 1000, max_range / 5000)
        for ring in range_rings[1:]:
            ax.plot(azimuths, np.ones_like(azimuths) * ring, 'w--', 
                   linewidth=0.5, alpha=0.3)
            # Add range labels
            ax.text(0, ring, f'{ring:.0f}km', 
                   ha='center', va='center', 
                   fontsize=8, color='white', alpha=0.7)
        
        # Add cardinal direction labels
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        ax.set_xticklabels(['N', 'E', 'S', 'W'])
        
        plt.colorbar(im, ax=ax, label='Power (dB)')
        
        return fig
    
    def plot_3d_scenario(self, targets: List[Any],
                        radar_pos: np.ndarray = None,
                        max_range: float = 10000,
                        title: str = "3D Radar Scenario") -> plt.Figure:
        """
        Plot 3D visualization of radar scenario
        
        Args:
            targets: List of Target objects
            radar_pos: Radar position [x, y, z]
            max_range: Maximum range to display
            title: Plot title
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        if radar_pos is None:
            radar_pos = np.array([0, 0, 0])
        
        # Plot radar
        ax.scatter(radar_pos[0], radar_pos[1], radar_pos[2],
                  c='red', s=200, marker='^', label='Radar')
        
        # Plot targets
        for i, target in enumerate(targets):
            pos = target.motion.position
            ax.scatter(pos[0], pos[1], pos[2],
                      c='blue', s=100, marker='o',
                      label=f'Target {i+1}' if i < 3 else '')
            
            # Plot velocity vector
            vel = target.motion.velocity
            vel_scale = 10  # Scale factor for visualization
            ax.quiver(pos[0], pos[1], pos[2],
                     vel[0] * vel_scale, vel[1] * vel_scale, vel[2] * vel_scale,
                     color='green', alpha=0.6)
        
        # Plot detection sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = max_range * np.outer(np.cos(u), np.sin(v)) + radar_pos[0]
        y = max_range * np.outer(np.sin(u), np.sin(v)) + radar_pos[1]
        z = max_range * np.outer(np.ones(np.size(u)), np.cos(v)) + radar_pos[2]
        
        ax.plot_surface(x, y, z, alpha=0.1, color='gray')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        ax.legend()
        
        # Set equal aspect ratio
        max_dim = max_range
        ax.set_xlim([-max_dim, max_dim])
        ax.set_ylim([-max_dim, max_dim])
        ax.set_zlim([0, max_dim])
        
        return fig
    
    def plot_detection_performance(self, snr_values: np.ndarray,
                                  pd_values: np.ndarray,
                                  pfa: float = 1e-6,
                                  title: str = "Detection Performance") -> plt.Figure:
        """
        Plot probability of detection vs SNR
        
        Args:
            snr_values: SNR values in dB
            pd_values: Probability of detection values
            pfa: Probability of false alarm
            title: Plot title
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(snr_values, pd_values, 'b-', linewidth=2,
               label=f'Pfa = {pfa:.0e}')
        
        # Add reference lines
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5,
                  label='Pd = 0.5')
        ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.5,
                  label='Pd = 0.9')
        
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Probability of Detection')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([np.min(snr_values), np.max(snr_values)])
        ax.set_ylim([0, 1])
        
        return fig
    
    def plot_ambiguity_function(self, ambiguity: np.ndarray,
                               delay_axis: np.ndarray,
                               doppler_axis: np.ndarray,
                               title: str = "Ambiguity Function") -> plt.Figure:
        """
        Plot ambiguity function
        
        Args:
            ambiguity: 2D ambiguity function
            delay_axis: Delay values
            doppler_axis: Doppler values
            title: Plot title
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(14, 6))
        
        # 3D surface plot
        ax1 = fig.add_subplot(121, projection='3d')
        D, T = np.meshgrid(doppler_axis, delay_axis)
        
        surf = ax1.plot_surface(T, D, 10 * np.log10(ambiguity + 1e-10),
                               cmap=self.colormap, alpha=0.9)
        
        ax1.set_xlabel('Delay')
        ax1.set_ylabel('Doppler')
        ax1.set_zlabel('Magnitude (dB)')
        ax1.set_title(f'{title} - 3D View')
        
        # Contour plot
        ax2 = fig.add_subplot(122)
        
        levels = np.arange(-40, 1, 5)
        contour = ax2.contour(T, D, 10 * np.log10(ambiguity + 1e-10),
                             levels=levels, cmap=self.colormap)
        ax2.clabel(contour, inline=True, fontsize=8)
        
        ax2.set_xlabel('Delay')
        ax2.set_ylabel('Doppler')
        ax2.set_title(f'{title} - Contour View')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def animate_tracking(self, time_steps: List[float],
                       target_positions: List[np.ndarray],
                       tracks: Optional[List[np.ndarray]] = None,
                       max_range: float = 10000) -> FuncAnimation:
        """
        Create animation of target tracking
        
        Args:
            time_steps: Time values
            target_positions: List of target positions over time
            tracks: Optional track estimates
            max_range: Maximum range for display
            
        Returns:
            Animation object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Setup axes
        ax1.set_xlim([-max_range, max_range])
        ax1.set_ylim([-max_range, max_range])
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('X-Y View')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlim([-max_range, max_range])
        ax2.set_ylim([0, max_range])
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title('X-Z View')
        ax2.grid(True, alpha=0.3)
        
        # Initialize plots
        truth_xy, = ax1.plot([], [], 'bo', markersize=8, label='Truth')
        track_xy, = ax1.plot([], [], 'r^', markersize=6, label='Track')
        trail_xy, = ax1.plot([], [], 'b-', alpha=0.3)
        
        truth_xz, = ax2.plot([], [], 'bo', markersize=8)
        track_xz, = ax2.plot([], [], 'r^', markersize=6)
        trail_xz, = ax2.plot([], [], 'b-', alpha=0.3)
        
        ax1.legend()
        
        def update(frame):
            # Update truth
            if frame < len(target_positions):
                pos = target_positions[frame]
                truth_xy.set_data([pos[0]], [pos[1]])
                truth_xz.set_data([pos[0]], [pos[2]])
                
                # Update trail
                trail_x = [p[0] for p in target_positions[:frame+1]]
                trail_y = [p[1] for p in target_positions[:frame+1]]
                trail_z = [p[2] for p in target_positions[:frame+1]]
                
                trail_xy.set_data(trail_x, trail_y)
                trail_xz.set_data(trail_x, trail_z)
                
                # Update track if available
                if tracks and frame < len(tracks):
                    track = tracks[frame]
                    track_xy.set_data([track[0]], [track[1]])
                    track_xz.set_data([track[0]], [track[2]])
            
            return truth_xy, track_xy, trail_xy, truth_xz, track_xz, trail_xz
        
        anim = FuncAnimation(fig, update, frames=len(time_steps),
                           interval=100, blit=True, repeat=True)
        
        return anim
    
    def plot_coverage_diagram(self, max_range: float,
                            min_elevation: float = -5,
                            max_elevation: float = 60,
                            title: str = "Radar Coverage") -> plt.Figure:
        """
        Plot radar coverage diagram
        
        Args:
            max_range: Maximum detection range in meters
            min_elevation: Minimum elevation angle in degrees
            max_elevation: Maximum elevation angle in degrees
            title: Plot title
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Convert to radians
        min_el_rad = np.radians(min_elevation)
        max_el_rad = np.radians(max_elevation)
        
        # Generate coverage boundary
        elevations = np.linspace(min_el_rad, max_el_rad, 100)
        ranges = np.ones_like(elevations) * max_range
        
        # Convert to Cartesian
        x = ranges * np.cos(elevations)
        y = ranges * np.sin(elevations)
        
        # Plot coverage area
        vertices = [(0, 0)]
        vertices.extend(zip(x / 1000, y / 1000))
        vertices.append((0, 0))
        
        from matplotlib.patches import Polygon
        coverage = Polygon(vertices, alpha=0.3, facecolor='blue',
                          edgecolor='blue', linewidth=2)
        ax.add_patch(coverage)
        
        # Add range rings
        for r in np.arange(0, max_range, max_range / 5):
            if r > 0:
                circle = Circle((0, 0), r / 1000, fill=False,
                              linestyle='--', alpha=0.3)
                ax.add_patch(circle)
                ax.text(r / 1000, 0, f'{r/1000:.0f} km',
                       ha='center', va='bottom', fontsize=8)
        
        # Add elevation lines
        for el_deg in np.arange(0, max_elevation + 1, 15):
            el_rad = np.radians(el_deg)
            x_el = max_range * np.cos(el_rad) / 1000
            y_el = max_range * np.sin(el_rad) / 1000
            ax.plot([0, x_el], [0, y_el], 'k--', alpha=0.2, linewidth=0.5)
            ax.text(x_el * 1.05, y_el * 1.05, f'{el_deg}°',
                   ha='center', fontsize=8)
        
        ax.set_xlim([-max_range / 1000 * 0.1, max_range / 1000 * 1.1])
        ax.set_ylim([-max_range / 1000 * 0.1, max_range / 1000 * 1.1])
        ax.set_xlabel('Ground Range (km)')
        ax.set_ylabel('Altitude (km)')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        return fig