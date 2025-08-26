"""
Environmental effects modeling for radar simulation
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class WeatherType(Enum):
    """Weather condition types"""
    CLEAR = "clear"
    RAIN = "rain"
    FOG = "fog"
    SNOW = "snow"
    HAIL = "hail"


@dataclass
class AtmosphericConditions:
    """Atmospheric parameters"""
    temperature: float = 15.0  # Celsius
    pressure: float = 1013.25  # hPa
    humidity: float = 50.0  # Percent
    weather: WeatherType = WeatherType.CLEAR
    rain_rate: float = 0.0  # mm/hr


class Environment:
    """Environmental effects on radar propagation"""
    
    def __init__(self, conditions: Optional[AtmosphericConditions] = None):
        """
        Initialize environment model
        
        Args:
            conditions: Atmospheric conditions
        """
        self.conditions = conditions if conditions else AtmosphericConditions()
        self.c = 3e8  # Speed of light
        
    def atmospheric_attenuation(self, frequency: float, range_m: float) -> float:
        """
        Calculate atmospheric attenuation
        
        Args:
            frequency: Radar frequency in Hz
            range_m: Propagation range in meters
            
        Returns:
            Attenuation in dB
        """
        freq_ghz = frequency / 1e9
        range_km = range_m / 1000
        
        # Oxygen absorption (simplified ITU-R P.676 model)
        if freq_ghz < 57:
            oxygen_atten = 0.0019 * freq_ghz**2 * range_km
        else:
            oxygen_atten = 16.0 * range_km  # Peak around 60 GHz
        
        # Water vapor absorption
        water_vapor_density = self._calculate_water_vapor_density()
        water_atten = 0.00015 * freq_ghz**2 * water_vapor_density * range_km
        
        # Weather-specific attenuation
        weather_atten = self._weather_attenuation(freq_ghz, range_km)
        
        total_atten = oxygen_atten + water_atten + weather_atten
        
        return total_atten
    
    def _calculate_water_vapor_density(self) -> float:
        """Calculate water vapor density from humidity"""
        # Simplified calculation
        t = self.conditions.temperature
        rh = self.conditions.humidity
        
        # Saturation vapor pressure (Magnus formula)
        es = 6.11 * np.exp(17.27 * t / (t + 237.3))
        
        # Actual vapor pressure
        e = es * rh / 100
        
        # Water vapor density (g/m^3)
        return 217 * e / (t + 273.15)
    
    def _weather_attenuation(self, freq_ghz: float, range_km: float) -> float:
        """Calculate weather-specific attenuation"""
        if self.conditions.weather == WeatherType.CLEAR:
            return 0.0
        
        elif self.conditions.weather == WeatherType.RAIN:
            # Rain attenuation (ITU-R P.838)
            rain_rate = self.conditions.rain_rate
            if rain_rate > 0:
                # Simplified model
                k = 0.0001 * freq_ghz**2.5
                alpha = 1.0 if freq_ghz < 10 else 0.9
                return k * rain_rate**alpha * range_km
            return 0.0
        
        elif self.conditions.weather == WeatherType.FOG:
            # Fog attenuation
            visibility_km = 0.1  # Dense fog
            return 0.4 * freq_ghz**0.5 * range_km / visibility_km
        
        elif self.conditions.weather == WeatherType.SNOW:
            # Snow attenuation
            snow_rate = self.conditions.rain_rate * 0.1  # Equivalent water
            return 0.00005 * freq_ghz**2 * snow_rate * range_km
        
        return 0.0
    
    def refraction_index(self, altitude: float = 0) -> float:
        """
        Calculate refractive index of atmosphere
        
        Args:
            altitude: Altitude in meters
            
        Returns:
            Refractive index
        """
        # Standard atmosphere model
        t = self.conditions.temperature - 0.0065 * altitude
        p = self.conditions.pressure * (1 - 0.0065 * altitude / (t + 273.15))**5.255
        e = self._calculate_water_vapor_density() * (t + 273.15) / 217
        
        # Refractivity (N-units)
        N = 77.6 * p / (t + 273.15) + 3.73e5 * e / (t + 273.15)**2
        
        # Refractive index
        n = 1 + N * 1e-6
        
        return n
    
    def ducting_assessment(self, altitude_range: Tuple[float, float], 
                          num_points: int = 100) -> bool:
        """
        Assess if atmospheric ducting conditions exist
        
        Args:
            altitude_range: (min, max) altitude in meters
            num_points: Number of points to sample
            
        Returns:
            True if ducting conditions detected
        """
        altitudes = np.linspace(altitude_range[0], altitude_range[1], num_points)
        refractivity_gradient = np.zeros(num_points - 1)
        
        for i in range(num_points - 1):
            n1 = self.refraction_index(altitudes[i])
            n2 = self.refraction_index(altitudes[i + 1])
            dh = altitudes[i + 1] - altitudes[i]
            
            # Modified refractivity gradient
            refractivity_gradient[i] = (n2 - n1) / dh * 1e6 + 157
        
        # Ducting occurs when gradient < 0
        return np.any(refractivity_gradient < 0)
    
    def multipath_effect(self, direct_range: float, target_altitude: float,
                        radar_altitude: float = 0) -> Tuple[float, float]:
        """
        Calculate multipath interference effect
        
        Args:
            direct_range: Direct path range in meters
            target_altitude: Target altitude in meters
            radar_altitude: Radar altitude in meters
            
        Returns:
            Multipath factor (amplitude), phase difference (radians)
        """
        # Simplified two-ray model
        height_diff = abs(target_altitude - radar_altitude)
        
        # Direct path
        r_direct = direct_range
        
        # Reflected path (assuming flat earth)
        r_reflected = np.sqrt(direct_range**2 + (2 * height_diff)**2)
        
        # Path difference
        path_diff = r_reflected - r_direct
        
        # Phase difference (assuming reflection coefficient of -1)
        wavelength = self.c / 3e9  # Assuming 3 GHz for example
        phase_diff = 2 * np.pi * path_diff / wavelength + np.pi
        
        # Multipath factor (vector sum of direct and reflected)
        multipath_factor = abs(1 + np.exp(1j * phase_diff))
        
        return multipath_factor, phase_diff
    
    def clutter_rcs(self, range_m: float, azimuth_beamwidth: float,
                   elevation_beamwidth: float, grazing_angle: float,
                   clutter_type: str = "land") -> float:
        """
        Calculate clutter RCS
        
        Args:
            range_m: Range to clutter in meters
            azimuth_beamwidth: Azimuth beamwidth in radians
            elevation_beamwidth: Elevation beamwidth in radians
            grazing_angle: Grazing angle in radians
            clutter_type: Type of clutter ('land', 'sea', 'urban')
            
        Returns:
            Clutter RCS in m^2
        """
        # Clutter reflectivity (sigma0) in dB
        if clutter_type == "land":
            sigma0_db = -20 + 10 * np.log10(np.sin(grazing_angle) + 0.001)
        elif clutter_type == "sea":
            # Sea state dependent
            wind_speed = 10  # m/s (moderate)
            sigma0_db = -30 + 5 * np.log10(wind_speed) + 20 * np.log10(np.sin(grazing_angle) + 0.001)
        elif clutter_type == "urban":
            sigma0_db = -10 + 15 * np.log10(np.sin(grazing_angle) + 0.001)
        else:
            sigma0_db = -25
        
        sigma0 = 10**(sigma0_db / 10)
        
        # Clutter area
        range_resolution = 30  # meters (example)
        azimuth_extent = range_m * azimuth_beamwidth
        clutter_area = range_resolution * azimuth_extent
        
        # Total clutter RCS
        clutter_rcs = sigma0 * clutter_area
        
        return clutter_rcs
    
    def propagation_factor(self, range_m: float, frequency: float,
                          target_altitude: float = 1000,
                          radar_altitude: float = 0) -> float:
        """
        Calculate overall propagation factor including all effects
        
        Args:
            range_m: Range in meters
            frequency: Frequency in Hz
            target_altitude: Target altitude in meters
            radar_altitude: Radar altitude in meters
            
        Returns:
            Propagation factor (linear scale)
        """
        # Atmospheric attenuation
        atten_db = self.atmospheric_attenuation(frequency, range_m)
        atten_linear = 10**(-atten_db / 20)
        
        # Multipath effect
        multipath, _ = self.multipath_effect(range_m, target_altitude, radar_altitude)
        
        # Combined propagation factor
        prop_factor = atten_linear * multipath
        
        return prop_factor


class ClutterGenerator:
    """Generate clutter returns for simulation"""
    
    def __init__(self, environment: Environment):
        self.environment = environment
        
    def generate_ground_clutter(self, ranges: np.ndarray, 
                               azimuths: np.ndarray,
                               radar_altitude: float = 10,
                               frequency: float = 10e9) -> np.ndarray:
        """
        Generate ground clutter map
        
        Args:
            ranges: Range bins in meters
            azimuths: Azimuth angles in radians
            radar_altitude: Radar altitude in meters
            frequency: Radar frequency in Hz
            
        Returns:
            2D clutter map (range x azimuth)
        """
        clutter_map = np.zeros((len(ranges), len(azimuths)))
        
        for i, r in enumerate(ranges):
            if r > 0:
                # Calculate grazing angle
                grazing_angle = np.arcsin(radar_altitude / r) if r > radar_altitude else np.pi/2
                
                for j, az in enumerate(azimuths):
                    # Random clutter patches
                    if np.random.random() < 0.3:  # 30% probability of clutter
                        # Calculate clutter RCS
                        az_beamwidth = 0.02  # radians (example)
                        el_beamwidth = 0.02
                        
                        clutter_rcs = self.environment.clutter_rcs(
                            r, az_beamwidth, el_beamwidth, 
                            grazing_angle, "land"
                        )
                        
                        # Add random variation
                        clutter_rcs *= np.random.lognormal(0, 0.5)
                        
                        # Convert to power (simplified)
                        clutter_power = clutter_rcs / (r**4)
                        clutter_map[i, j] = clutter_power
        
        return clutter_map
    
    def generate_weather_clutter(self, ranges: np.ndarray,
                                velocities: np.ndarray,
                                weather_type: WeatherType) -> np.ndarray:
        """
        Generate weather clutter in range-Doppler space
        
        Args:
            ranges: Range bins in meters
            velocities: Velocity bins in m/s
            weather_type: Type of weather
            
        Returns:
            2D weather clutter (range x velocity)
        """
        clutter = np.zeros((len(ranges), len(velocities)))
        
        if weather_type == WeatherType.RAIN:
            # Rain clutter with wind drift
            wind_speed = np.random.normal(5, 2)  # m/s
            wind_spread = 3  # m/s
            
            for i, r in enumerate(ranges):
                if r < 50000:  # Weather within 50 km
                    # Find velocity bins near wind speed
                    for j, v in enumerate(velocities):
                        if abs(v - wind_speed) < wind_spread:
                            # Gaussian distribution around wind speed
                            clutter[i, j] = np.exp(-(v - wind_speed)**2 / (2 * wind_spread**2))
                            # Decrease with range
                            clutter[i, j] *= np.exp(-r / 20000)
                            # Add randomness
                            clutter[i, j] *= np.random.lognormal(0, 0.3)
        
        return clutter
    
    def generate_sea_clutter(self, ranges: np.ndarray,
                           velocities: np.ndarray,
                           sea_state: int = 3) -> np.ndarray:
        """
        Generate sea clutter
        
        Args:
            ranges: Range bins in meters
            velocities: Velocity bins in m/s  
            sea_state: Sea state (0-9)
            
        Returns:
            2D sea clutter (range x velocity)
        """
        clutter = np.zeros((len(ranges), len(velocities)))
        
        # Sea state parameters
        wave_height = 0.3 * (1.5**sea_state)  # Simplified
        wave_velocity = np.sqrt(9.81 * wave_height)
        
        for i, r in enumerate(ranges):
            if r < 100000:  # Sea clutter within horizon
                for j, v in enumerate(velocities):
                    # Bragg scattering creates Doppler spread
                    if abs(v) < 2 * wave_velocity:
                        clutter[i, j] = np.exp(-abs(v) / wave_velocity)
                        # Decrease with range
                        clutter[i, j] *= (1000 / (r + 1000))**2
                        # Add speckle
                        clutter[i, j] *= np.random.exponential(1.0)
        
        return clutter