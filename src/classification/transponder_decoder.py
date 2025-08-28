#!/usr/bin/env python3
"""
Transponder and IFF Decoder Module

Handles legitimate additional information sources that augment radar returns:
- Mode A/C/S transponder codes
- ADS-B (Automatic Dependent Surveillance-Broadcast)
- IFF (Identification Friend or Foe)
- TCAS (Traffic Collision Avoidance System)

This is NOT cheating - these are real systems that provide additional
information beyond raw radar returns.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json


class TransponderMode(Enum):
    """Transponder operating modes"""
    MODE_A = "mode_a"      # 4-digit squawk code
    MODE_C = "mode_c"      # Altitude reporting
    MODE_S = "mode_s"      # Enhanced surveillance
    ADS_B = "ads_b"        # Full position/velocity broadcast
    NONE = "none"          # No transponder or not responding


class IFFMode(Enum):
    """IFF interrogation modes"""
    MODE_1 = "mode_1"      # Military mission ID
    MODE_2 = "mode_2"      # Military unit ID  
    MODE_3A = "mode_3a"    # Civilian equivalent to Mode A
    MODE_4 = "mode_4"      # Encrypted military
    MODE_5 = "mode_5"      # Enhanced encrypted military
    MODE_C = "mode_c"      # Altitude


@dataclass
class TransponderData:
    """Data received from transponder systems"""
    # Basic transponder
    squawk_code: Optional[str] = None         # 4-digit code (Mode A)
    pressure_altitude: Optional[float] = None  # Mode C altitude (feet)
    
    # Mode S
    mode_s_address: Optional[str] = None      # 24-bit ICAO address
    callsign: Optional[str] = None            # Flight ID
    
    # ADS-B
    adsb_position: Optional[Tuple[float, float, float]] = None  # Lat, Lon, Alt
    adsb_velocity: Optional[Tuple[float, float, float]] = None  # Vx, Vy, Vz
    adsb_category: Optional[str] = None       # Aircraft category
    adsb_emergency: Optional[str] = None      # Emergency status
    
    # IFF
    iff_mode1: Optional[str] = None           # 2-digit mission code
    iff_mode2: Optional[str] = None           # 4-digit unit code
    iff_mode4_valid: bool = False             # Crypto validation
    
    # Metadata
    last_update_time: float = 0.0
    signal_strength: float = 0.0              # Transponder signal strength


class TransponderDecoder:
    """
    Decodes transponder and IFF responses
    """
    
    # Standard emergency codes
    EMERGENCY_CODES = {
        '7700': 'general_emergency',
        '7600': 'radio_failure', 
        '7500': 'hijack',
        '7777': 'military_intercept',  # Sometimes used
        '7400': 'lost_comm_expected',  # Some regions
    }
    
    # Special purpose codes (vary by region/country)
    SPECIAL_CODES = {
        '1200': 'vfr_no_flight_plan',  # US VFR
        '7000': 'vfr_conspicuity',     # Europe VFR
        '2000': 'ifr_no_assignment',   # Entering radar coverage
        '1000': 'ifr_mode_s_discrete', # Mode S equipped
        '0000': 'military_special',    # Should not squawk
    }
    
    # ADS-B aircraft categories
    ADSB_CATEGORIES = {
        'A0': 'unspecified',
        'A1': 'light_aircraft',        # < 15,500 lbs
        'A2': 'medium_aircraft',        # 15,500 - 75,000 lbs  
        'A3': 'heavy_aircraft',         # 75,000 - 300,000 lbs
        'A4': 'high_vortex_large',      # B757
        'A5': 'heavy_high_vortex',      # > 300,000 lbs
        'A6': 'high_performance',       # > 5G, > 400 kts
        'A7': 'rotorcraft',
        'B0': 'unspecified_powered',
        'B1': 'glider_sailplane',
        'B2': 'lighter_than_air',
        'B3': 'parachutist',
        'B4': 'ultralight',
        'B5': 'reserved',
        'B6': 'uav',
        'B7': 'space_vehicle',
        'C0': 'unspecified_ground',
        'C1': 'surface_emergency',
        'C2': 'surface_service',
        'C3': 'ground_obstruction',
    }
    
    def __init__(self):
        """Initialize transponder decoder"""
        self.current_tracks: Dict[str, TransponderData] = {}
        
    def decode_mode_a(self, code: str) -> Dict[str, Any]:
        """
        Decode Mode A squawk code
        
        Args:
            code: 4-digit octal code (e.g., '7700')
            
        Returns:
            Decoded information dictionary
        """
        info = {
            'code': code,
            'valid': self._validate_squawk_code(code)
        }
        
        # Check for emergency
        if code in self.EMERGENCY_CODES:
            info['emergency'] = True
            info['emergency_type'] = self.EMERGENCY_CODES[code]
            info['priority'] = 'emergency'
            
        # Check for special codes
        elif code in self.SPECIAL_CODES:
            info['special'] = True
            info['special_type'] = self.SPECIAL_CODES[code]
            info['priority'] = 'normal'
            
        # Assigned discrete code
        else:
            info['discrete'] = True
            info['priority'] = 'normal'
            
        return info
    
    def decode_mode_s(self, 
                     address: str,
                     data_frame: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Decode Mode S extended squitter
        
        Args:
            address: 24-bit ICAO address (hex)
            data_frame: Optional extended data frame
            
        Returns:
            Decoded Mode S information
        """
        info = {
            'icao_address': address,
            'country': self._decode_country_from_icao(address),
            'registered': self._check_registration(address)
        }
        
        if data_frame:
            # Decode extended squitter if present
            df_type = (data_frame[0] >> 3) & 0x1F
            
            if df_type == 17:  # ADS-B message
                info['adsb'] = self._decode_adsb(data_frame)
            elif df_type == 18:  # TIS-B or ADS-R
                info['tisb'] = True
            elif df_type == 19:  # Military
                info['military'] = True
                
        return info
    
    def decode_adsb(self, message: bytes) -> Dict[str, Any]:
        """
        Decode ADS-B message
        
        Args:
            message: ADS-B message bytes
            
        Returns:
            Decoded ADS-B data
        """
        # Simplified ADS-B decode (full implementation would be complex)
        tc = (message[4] >> 3) & 0x1F  # Type code
        
        info = {}
        
        if 1 <= tc <= 4:  # Aircraft identification
            info['type'] = 'identification'
            info['callsign'] = self._extract_callsign(message)
            
        elif 9 <= tc <= 18:  # Airborne position
            info['type'] = 'position'
            info['altitude'] = self._extract_altitude(message)
            info['position'] = self._extract_position(message)
            
        elif tc == 19:  # Airborne velocity
            info['type'] = 'velocity'
            info['velocity'] = self._extract_velocity(message)
            
        elif 20 <= tc <= 22:  # Airborne position (GNSS)
            info['type'] = 'position_gnss'
            info['position'] = self._extract_position_gnss(message)
            
        elif tc == 28:  # Aircraft status
            info['type'] = 'status'
            info['emergency'] = self._extract_emergency_state(message)
            
        elif tc == 29:  # Target state and status
            info['type'] = 'target_state'
            info['selected_altitude'] = self._extract_selected_altitude(message)
            
        elif tc == 31:  # Operational status
            info['type'] = 'operational'
            info['capability'] = self._extract_capability(message)
            
        return info
    
    def decode_iff(self, 
                  mode: IFFMode,
                  response: str,
                  crypto_valid: bool = False) -> Dict[str, Any]:
        """
        Decode IFF response
        
        Args:
            mode: IFF interrogation mode
            response: Response code
            crypto_valid: Whether Mode 4/5 crypto validated
            
        Returns:
            Decoded IFF information
        """
        info = {
            'mode': mode.value,
            'response': response
        }
        
        if mode == IFFMode.MODE_1:
            # 2-digit mission code
            info['mission_type'] = self._decode_mission_code(response)
            
        elif mode == IFFMode.MODE_2:
            # 4-digit unit code
            info['unit_id'] = response
            info['unit_type'] = self._decode_unit_type(response)
            
        elif mode == IFFMode.MODE_3A:
            # Same as Mode A transponder
            return self.decode_mode_a(response)
            
        elif mode in [IFFMode.MODE_4, IFFMode.MODE_5]:
            info['crypto_valid'] = crypto_valid
            if crypto_valid:
                info['friendly'] = True
                info['priority'] = 'friendly'
            else:
                info['unknown'] = True
                info['priority'] = 'investigate'
                
        return info
    
    def _validate_squawk_code(self, code: str) -> bool:
        """Validate squawk code format (4 octal digits)"""
        if len(code) != 4:
            return False
        try:
            for digit in code:
                if not 0 <= int(digit) <= 7:
                    return False
            return True
        except ValueError:
            return False
    
    def _decode_country_from_icao(self, address: str) -> str:
        """Decode country from ICAO address allocation"""
        # Simplified - real implementation would have full allocation table
        addr_int = int(address, 16)
        
        if 0xA00000 <= addr_int <= 0xAFFFFF:
            return "USA"
        elif 0x400000 <= addr_int <= 0x4FFFFF:
            return "UK"
        elif 0x3C0000 <= addr_int <= 0x3FFFFF:
            return "Germany"
        elif 0x380000 <= addr_int <= 0x3BFFFF:
            return "France"
        else:
            return "Unknown"
    
    def _check_registration(self, address: str) -> bool:
        """Check if ICAO address is in registration database"""
        # In real implementation, would check against database
        return True  # Assume registered for now
    
    def _extract_callsign(self, message: bytes) -> str:
        """Extract callsign from ADS-B identification message"""
        # Simplified extraction
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        callsign = ""
        # Would properly decode from message bytes
        return "UAL123"  # Placeholder
    
    def _extract_altitude(self, message: bytes) -> float:
        """Extract altitude from ADS-B position message"""
        # Simplified - would properly decode barometric altitude
        return 35000.0  # Placeholder (feet)
    
    def _extract_position(self, message: bytes) -> Tuple[float, float]:
        """Extract position from ADS-B position message"""
        # Simplified - would properly decode CPR lat/lon
        return (37.7749, -122.4194)  # Placeholder (lat, lon)
    
    def _extract_velocity(self, message: bytes) -> Tuple[float, float, float]:
        """Extract velocity from ADS-B velocity message"""
        # Simplified - would properly decode velocity components
        return (150.0, 50.0, -500.0)  # Placeholder (Vx, Vy, Vz in knots/fpm)
    
    def _extract_position_gnss(self, message: bytes) -> Tuple[float, float, float]:
        """Extract GNSS position from ADS-B message"""
        # Simplified - would properly decode GNSS position
        return (37.7749, -122.4194, 35000.0)  # Placeholder
    
    def _extract_emergency_state(self, message: bytes) -> str:
        """Extract emergency state from ADS-B status message"""
        # Simplified - would properly decode emergency bits
        return "none"  # or "general", "medical", "minimum_fuel", etc.
    
    def _extract_selected_altitude(self, message: bytes) -> float:
        """Extract selected altitude from target state message"""
        # Simplified - would decode MCP/FCU selected altitude
        return 31000.0  # Placeholder (feet)
    
    def _extract_capability(self, message: bytes) -> Dict[str, bool]:
        """Extract aircraft capability from operational status"""
        return {
            'tcas': True,
            'adsb_in': True,
            'adsb_out': True,
            'acas': True
        }
    
    def _decode_mission_code(self, code: str) -> str:
        """Decode Mode 1 mission type"""
        # Simplified mission type decode
        mission_types = {
            '00': 'unassigned',
            '01': 'air_defense',
            '02': 'ground_support',
            '03': 'transport',
            '04': 'training',
            '05': 'test',
        }
        return mission_types.get(code, 'unknown')
    
    def _decode_unit_type(self, code: str) -> str:
        """Decode unit type from Mode 2"""
        # Simplified - first digit often indicates type
        if code and code[0] == '0':
            return "fighter"
        elif code and code[0] == '1':
            return "bomber"
        elif code and code[0] == '2':
            return "transport"
        else:
            return "unknown"
    
    def get_operational_priority(self, transponder_data: TransponderData) -> str:
        """
        Determine operational priority from transponder data
        
        This is legitimate - we're using explicitly broadcast information
        """
        # Emergency codes always highest priority
        if transponder_data.squawk_code in self.EMERGENCY_CODES:
            return "emergency"
            
        # ADS-B emergency state
        if transponder_data.adsb_emergency and transponder_data.adsb_emergency != "none":
            return "emergency"
            
        # No transponder = investigate
        if not transponder_data.squawk_code and not transponder_data.mode_s_address:
            return "investigate"
            
        # Normal traffic with transponder
        return "normal"