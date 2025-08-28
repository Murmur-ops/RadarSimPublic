#!/usr/bin/env python3
"""
Basic Radar Jamming Demonstration
Shows how noise jamming affects radar detection and tracking
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')


class Jammer:
    """Simple noise jammer model"""
    
    def __init__(self, range_m, velocity_ms=0, power_w=100, jamming_type='barrage', bandwidth_hz=1e6):
        """
        Initialize jammer
        
        Args:
            range_m: Jammer range from radar (meters)
            velocity_ms: Jammer velocity (m/s)
            power_w: Jamming power (watts)
            jamming_type: 'barrage' or 'spot'
            bandwidth_hz: Jamming bandwidth (Hz)
        """
        self.range = range_m
        self.velocity = velocity_ms
        self.power = power_w
        self.type = jamming_type
        self.bandwidth = bandwidth_hz
        self.antenna_gain = 10  # dB - omnidirectional
        
    def calculate_jsr(self, radar_range, target_rcs, radar_power=1000, radar_gain=30):
        """
        Calculate Jamming-to-Signal Ratio
        
        J/S = (P_j * G_j * R_t^2) / (P_r * G_r * σ * R_j^2)
        
        Args:
            radar_range: Range from radar to target (m)
            target_rcs: Target RCS (m^2)
            radar_power: Radar transmit power (W)
            radar_gain: Radar antenna gain (dB)
            
        Returns:
            J/S ratio in dB
        """
        # Convert gains from dB
        g_j = 10**(self.antenna_gain / 10)
        g_r = 10**(radar_gain / 10)
        
        # One-way jamming vs two-way radar
        j_power = self.power * g_j / (4 * np.pi * self.range**2)
        s_power = radar_power * g_r * target_rcs / ((4 * np.pi)**3 * radar_range**4)
        
        jsr_linear = j_power / s_power
        return 10 * np.log10(jsr_linear)
    
    def burn_through_range(self, target_rcs, radar_power=1000, radar_gain=30):
        """
        Calculate burn-through range where SNR overcomes jamming
        
        Args:
            target_rcs: Target RCS (m^2)
            radar_power: Radar power (W)
            radar_gain: Radar gain (dB)
            
        Returns:
            Burn-through range in meters
        """
        g_j = 10**(self.antenna_gain / 10)
        g_r = 10**(radar_gain / 10)
        
        # Burn-through occurs when J/S = 0 dB (unity)
        # R_bt^4 = (P_r * G_r * σ * R_j^2) / (P_j * G_j * (4π)^2)
        numerator = radar_power * g_r * target_rcs * self.range**2
        denominator = self.power * g_j * (4 * np.pi)**2
        
        r_bt = (numerator / denominator)**(0.25)
        return r_bt


class JammedRadarSimulator:
    """Radar simulator with jamming capability"""
    
    def __init__(self, max_range=20e3, range_resolution=50, velocity_resolution=1.0):
        self.max_range = max_range
        self.range_resolution = range_resolution
        self.velocity_resolution = velocity_resolution
        self.n_range_bins = int(max_range / range_resolution)
        self.n_doppler_bins = 128
        self.noise_floor = -40  # dBm
        self.detection_threshold = 10  # dB above noise
        self.radar_power = 1000  # W
        self.radar_gain = 30  # dB
        
    def generate_range_doppler_map(self, targets, jammer=None):
        """Generate range-Doppler map with optional jamming"""
        
        # Create thermal noise floor
        rd_map = (np.random.randn(self.n_range_bins, self.n_doppler_bins) + 
                  1j * np.random.randn(self.n_range_bins, self.n_doppler_bins))
        rd_map *= 10**(self.noise_floor/20)
        
        # Add targets
        for target in targets:
            range_bin = int(target['range'] / self.range_resolution)
            
            # Handle velocity as scalar or vector
            if isinstance(target['velocity'], list):
                # Use radial component (first element for approaching/receding)
                radial_velocity = target['velocity'][0] if len(target['velocity']) > 0 else 0
            else:
                radial_velocity = target['velocity']
            
            doppler_bin = int(radial_velocity / self.velocity_resolution) + self.n_doppler_bins // 2
            
            if 0 <= range_bin < self.n_range_bins and 0 <= doppler_bin < self.n_doppler_bins:
                # Calculate target SNR
                snr = self.calculate_snr(target['range'], target['rcs'])
                signal_power = 10**((self.noise_floor + snr) / 20)
                
                # Add target signal with some spread
                for dr in range(-1, 2):
                    for dv in range(-1, 2):
                        r_idx = range_bin + dr
                        v_idx = doppler_bin + dv
                        if 0 <= r_idx < self.n_range_bins and 0 <= v_idx < self.n_doppler_bins:
                            weight = np.exp(-(dr**2 + dv**2) / 2)
                            rd_map[r_idx, v_idx] += signal_power * weight * np.exp(1j * np.random.uniform(0, 2*np.pi))
        
        # Add jamming if present
        if jammer is not None:
            rd_map = self.add_jamming(rd_map, jammer)
        
        return np.abs(rd_map)**2
    
    def add_jamming(self, rd_map, jammer):
        """Add jamming signal to range-Doppler map"""
        
        jammer_range_bin = int(jammer.range / self.range_resolution)
        
        # Handle jammer velocity as scalar or vector
        if isinstance(jammer.velocity, list):
            jammer_vel = jammer.velocity[0] if len(jammer.velocity) > 0 else 0
        else:
            jammer_vel = jammer.velocity
        
        jammer_doppler_bin = int(jammer_vel / self.velocity_resolution) + self.n_doppler_bins // 2
        
        if 0 <= jammer_range_bin < self.n_range_bins:
            # Calculate jamming power at radar
            # One-way propagation for jammer
            range_loss_db = 20 * np.log10(jammer.range / 1000)  # ref at 1km
            jammer_gain_db = jammer.antenna_gain
            received_jam_power_dbm = 10 * np.log10(jammer.power * 1000) - range_loss_db + jammer_gain_db - 30
            jam_power_linear = 10**(received_jam_power_dbm / 20)
            
            if jammer.type == 'barrage':
                # Spread jamming power across all Doppler bins
                jam_signal = jam_power_linear * (np.random.randn(self.n_doppler_bins) + 
                                                  1j * np.random.randn(self.n_doppler_bins))
                jam_signal /= np.sqrt(self.n_doppler_bins)  # Normalize power distribution
                
                # Add to multiple range bins for range extent
                for dr in range(-2, 3):
                    r_idx = jammer_range_bin + dr
                    if 0 <= r_idx < self.n_range_bins:
                        weight = np.exp(-dr**2 / 4)
                        rd_map[r_idx, :] += weight * jam_signal
                        
            elif jammer.type == 'spot':
                # Concentrate jamming in specific Doppler bins
                if 0 <= jammer_doppler_bin < self.n_doppler_bins:
                    spot_width = 5  # bins
                    for dr in range(-2, 3):
                        r_idx = jammer_range_bin + dr
                        if 0 <= r_idx < self.n_range_bins:
                            for dv in range(-spot_width, spot_width + 1):
                                v_idx = jammer_doppler_bin + dv
                                if 0 <= v_idx < self.n_doppler_bins:
                                    range_weight = np.exp(-dr**2 / 4)
                                    doppler_weight = np.exp(-dv**2 / (2 * (spot_width/3)**2))
                                    jam_signal = jam_power_linear * range_weight * doppler_weight
                                    jam_signal *= (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
                                    rd_map[r_idx, v_idx] += jam_signal
        
        return rd_map
    
    def calculate_snr(self, range_m, rcs):
        """Calculate SNR using radar equation"""
        reference_snr = 40  # dB at 1km with RCS=1m²
        range_loss = 40 * np.log10(range_m / 1000)  # R^4 law
        rcs_gain = 10 * np.log10(rcs)
        return reference_snr - range_loss + rcs_gain
    
    def detect_targets(self, rd_map, jammer=None):
        """CFAR detection with jamming awareness"""
        detections = []
        power_db = 10 * np.log10(rd_map + 1e-10)
        
        # Estimate noise floor (may be elevated by jamming)
        noise_estimate = np.median(power_db)
        
        # Adaptive threshold
        if jammer is not None:
            # Increase threshold in presence of jamming
            threshold = noise_estimate + self.detection_threshold + 3  # Extra 3dB margin
        else:
            threshold = noise_estimate + self.detection_threshold
        
        # CFAR detection
        for r in range(10, self.n_range_bins - 10):
            for d in range(10, self.n_doppler_bins - 10):
                # Local peak detection
                local_region = power_db[r-2:r+3, d-2:d+3]
                if power_db[r, d] >= threshold and power_db[r, d] == np.max(local_region):
                    range_m = r * self.range_resolution
                    velocity = (d - self.n_doppler_bins // 2) * self.velocity_resolution
                    snr = power_db[r, d] - noise_estimate
                    
                    # Check if this might be jamming
                    is_jamming = False
                    if jammer is not None:
                        jammer_range_bin = int(jammer.range / self.range_resolution)
                        if abs(r - jammer_range_bin) < 5:  # Near jammer location
                            if jammer.type == 'barrage':
                                is_jamming = True  # Likely false alarm from jamming
                    
                    if not is_jamming or snr > 20:  # High SNR might be real target
                        detections.append({
                            'range': range_m,
                            'velocity': velocity,
                            'snr': snr,
                            'range_std': self.range_resolution / np.sqrt(10**(snr/10)),
                            'velocity_std': self.velocity_resolution / np.sqrt(10**(snr/10))
                        })
        
        return detections


def run_jamming_demonstration():
    """Main demonstration of jamming effects"""
    
    print("=" * 70)
    print("BASIC RADAR JAMMING DEMONSTRATION")
    print("=" * 70)
    
    # Initialize radar
    radar = JammedRadarSimulator()
    
    # Create targets
    targets = [
        {'range': 5000, 'velocity': -50, 'rcs': 100, 'name': 'Fighter'},
        {'range': 8000, 'velocity': 20, 'rcs': 50, 'name': 'Transport'},
        {'range': 12000, 'velocity': -30, 'rcs': 20, 'name': 'Drone'},
    ]
    
    # Create jammer
    jammer_barrage = Jammer(range_m=10000, velocity_ms=0, power_w=50, jamming_type='barrage')
    jammer_spot = Jammer(range_m=10000, velocity_ms=0, power_w=50, jamming_type='spot')
    
    print("\nTargets:")
    for t in targets:
        print(f"  {t['name']}: R={t['range']/1000:.1f}km, V={t['velocity']}m/s, RCS={t['rcs']}m²")
    
    print(f"\nJammer: R={jammer_barrage.range/1000:.1f}km, P={jammer_barrage.power}W")
    
    # Calculate burn-through ranges
    print("\nBurn-through Ranges:")
    for t in targets:
        r_bt = jammer_barrage.burn_through_range(t['rcs'])
        print(f"  {t['name']}: {r_bt/1000:.1f}km")
    
    # Generate range-Doppler maps
    print("\nGenerating range-Doppler maps...")
    rd_clean = radar.generate_range_doppler_map(targets, jammer=None)
    rd_barrage = radar.generate_range_doppler_map(targets, jammer=jammer_barrage)
    rd_spot = radar.generate_range_doppler_map(targets, jammer=jammer_spot)
    
    # Detect targets
    det_clean = radar.detect_targets(rd_clean)
    det_barrage = radar.detect_targets(rd_barrage, jammer=jammer_barrage)
    det_spot = radar.detect_targets(rd_spot, jammer=jammer_spot)
    
    print(f"\nDetection Results:")
    print(f"  No Jamming: {len(det_clean)} detections")
    print(f"  Barrage Jamming: {len(det_barrage)} detections")
    print(f"  Spot Jamming: {len(det_spot)} detections")
    
    # Create visualization
    create_jamming_visualization(radar, targets, jammer_barrage, 
                                rd_clean, rd_barrage, rd_spot,
                                det_clean, det_barrage, det_spot)
    
    print("\n" + "=" * 70)
    print("Demonstration complete! See 'basic_jamming_demo.png'")
    print("=" * 70)


def create_jamming_visualization(radar, targets, jammer, 
                                rd_clean, rd_barrage, rd_spot,
                                det_clean, det_barrage, det_spot):
    """Create comprehensive jamming visualization"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Convert to dB for display
    rd_clean_db = 10 * np.log10(rd_clean + 1e-10)
    rd_barrage_db = 10 * np.log10(rd_barrage + 1e-10)
    rd_spot_db = 10 * np.log10(rd_spot + 1e-10)
    
    vmin, vmax = -50, 0
    
    # 1. Clean Range-Doppler
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(rd_clean_db.T, aspect='auto', origin='lower',
                     cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title('No Jamming', fontweight='bold')
    ax1.set_xlabel('Range Bin')
    ax1.set_ylabel('Doppler Bin')
    plt.colorbar(im1, ax=ax1, label='Power (dB)')
    
    # 2. Barrage Jamming
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(rd_barrage_db.T, aspect='auto', origin='lower',
                     cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_title('Barrage Jamming', fontweight='bold')
    ax2.set_xlabel('Range Bin')
    ax2.set_ylabel('Doppler Bin')
    plt.colorbar(im2, ax=ax2, label='Power (dB)')
    
    # Mark jammer location
    jammer_bin = int(jammer.range / radar.range_resolution)
    ax2.axvline(x=jammer_bin, color='red', linestyle='--', alpha=0.5, label='Jammer')
    
    # 3. Spot Jamming
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(rd_spot_db.T, aspect='auto', origin='lower',
                     cmap='viridis', vmin=vmin, vmax=vmax)
    ax3.set_title('Spot Jamming', fontweight='bold')
    ax3.set_xlabel('Range Bin')
    ax3.set_ylabel('Doppler Bin')
    plt.colorbar(im3, ax=ax3, label='Power (dB)')
    ax3.axvline(x=jammer_bin, color='red', linestyle='--', alpha=0.5)
    
    # 4. Detection Comparison
    ax4 = fig.add_subplot(gs[1, :])
    
    # Plot true targets
    for t in targets:
        ax4.plot(t['range']/1000, 0, 'g^', markersize=10, label=f"True: {t['name']}")
    
    # Plot detections
    if det_clean:
        ranges_clean = [d['range']/1000 for d in det_clean]
        ax4.plot(ranges_clean, [1]*len(ranges_clean), 'bo', markersize=8, label='Detected (Clean)')
    
    if det_barrage:
        ranges_barrage = [d['range']/1000 for d in det_barrage]
        ax4.plot(ranges_barrage, [2]*len(ranges_barrage), 'ro', markersize=8, label='Detected (Barrage)')
    
    if det_spot:
        ranges_spot = [d['range']/1000 for d in det_spot]
        ax4.plot(ranges_spot, [3]*len(ranges_spot), 'mo', markersize=8, label='Detected (Spot)')
    
    # Mark jammer
    ax4.axvline(x=jammer.range/1000, color='red', linestyle='--', alpha=0.3, label='Jammer')
    
    # Mark burn-through ranges
    for t in targets:
        r_bt = jammer.burn_through_range(t['rcs']) / 1000
        ax4.axvspan(0, r_bt, alpha=0.1, color='red')
    
    ax4.set_xlabel('Range (km)')
    ax4.set_ylabel('Scenario')
    ax4.set_yticks([0, 1, 2, 3])
    ax4.set_yticklabels(['Truth', 'Clean', 'Barrage', 'Spot'])
    ax4.set_title('Detection Results Under Jamming', fontweight='bold')
    ax4.legend(loc='upper right', ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 20])
    
    # 5. J/S Ratio Analysis
    ax5 = fig.add_subplot(gs[2, 0])
    
    ranges_km = np.linspace(1, 20, 100)
    for t in targets:
        jsr_values = []
        for r in ranges_km * 1000:
            t_temp = {'range': r, 'rcs': t['rcs']}
            jsr = jammer.calculate_jsr(r, t['rcs'])
            jsr_values.append(jsr)
        
        ax5.plot(ranges_km, jsr_values, label=t['name'])
        
        # Mark burn-through
        r_bt = jammer.burn_through_range(t['rcs']) / 1000
        ax5.axvline(x=r_bt, color='gray', linestyle=':', alpha=0.5)
    
    ax5.axhline(y=0, color='red', linestyle='--', label='J/S = 0 dB')
    ax5.set_xlabel('Target Range (km)')
    ax5.set_ylabel('J/S Ratio (dB)')
    ax5.set_title('Jamming-to-Signal Ratio', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, 20])
    
    # 6. SNR Degradation
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Calculate SNR with and without jamming
    ranges_km = np.linspace(1, 20, 100)
    for t in targets:
        snr_clean = []
        snr_jammed = []
        
        for r in ranges_km * 1000:
            # Clean SNR
            snr = radar.calculate_snr(r, t['rcs'])
            snr_clean.append(snr)
            
            # Effective SNR with jamming
            jsr = jammer.calculate_jsr(r, t['rcs'])
            snr_eff = snr - max(0, jsr)  # Simplified model
            snr_jammed.append(snr_eff)
        
        ax6.plot(ranges_km, snr_clean, '--', alpha=0.5)
        ax6.plot(ranges_km, snr_jammed, '-', label=t['name'])
    
    ax6.axhline(y=10, color='red', linestyle=':', label='Detection Threshold')
    ax6.set_xlabel('Range (km)')
    ax6.set_ylabel('SNR (dB)')
    ax6.set_title('SNR Degradation from Jamming', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([0, 20])
    ax6.set_ylim([-10, 50])
    
    # 7. Jamming Effectiveness Summary
    ax7 = fig.add_subplot(gs[2, 2])
    
    # Bar chart of detection rates
    scenarios = ['No Jamming', 'Barrage', 'Spot']
    detected = [len(det_clean), len(det_barrage), len(det_spot)]
    total = len(targets)
    detection_rate = [d/total * 100 for d in detected]
    
    colors = ['green', 'orange', 'red']
    bars = ax7.bar(scenarios, detection_rate, color=colors, alpha=0.7)
    
    # Add value labels
    for bar, rate in zip(bars, detection_rate):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.0f}%', ha='center', va='bottom')
    
    ax7.set_ylabel('Detection Rate (%)')
    ax7.set_title('Jamming Effectiveness', fontweight='bold')
    ax7.set_ylim([0, 120])
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Main title
    fig.suptitle('Radar Jamming Effects Demonstration', fontsize=16, fontweight='bold')
    
    # Add info box
    info_text = (f"Jammer: {jammer.power}W at {jammer.range/1000:.0f}km\n"
                f"Radar: {radar.radar_power}W, {radar.radar_gain}dB gain\n"
                f"Burn-through in red shaded region")
    fig.text(0.02, 0.02, info_text, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('basic_jamming_demo.png', dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    run_jamming_demonstration()
    plt.show()