"""
Arbitrary waveform generator for radar simulation
Supports various radar waveforms including LFM, NLFM, phase codes, and custom waveforms
"""

import numpy as np
from typing import Optional, Tuple, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
import scipy.signal as signal
from scipy.optimize import minimize


class WaveformType(Enum):
    """Enumeration of waveform types"""
    LFM = "lfm"  # Linear Frequency Modulation
    NLFM = "nlfm"  # Non-Linear Frequency Modulation
    BARKER = "barker"  # Barker code
    FRANK = "frank"  # Frank code
    P1 = "p1"  # P1 polyphase code
    P2 = "p2"  # P2 polyphase code
    P3 = "p3"  # P3 polyphase code
    P4 = "p4"  # P4 polyphase code
    COSTAS = "costas"  # Costas sequence
    STEPPED_FREQ = "stepped_freq"  # Stepped frequency
    OFDM = "ofdm"  # Orthogonal Frequency Division Multiplexing
    CUSTOM = "custom"  # User-defined waveform


@dataclass
class WaveformParameters:
    """Parameters for waveform generation"""
    pulse_width: float  # Pulse width in seconds
    bandwidth: float  # Bandwidth in Hz
    sample_rate: float  # Sample rate in Hz
    center_frequency: float = 0  # Center frequency in Hz (for IF/RF generation)
    
    @property
    def num_samples(self) -> int:
        """Calculate number of samples"""
        return int(self.pulse_width * self.sample_rate)
    
    @property
    def time_vector(self) -> np.ndarray:
        """Generate time vector"""
        return np.linspace(0, self.pulse_width, self.num_samples)


class WaveformGenerator:
    """Arbitrary waveform generator for radar signals"""
    
    # Barker codes dictionary
    BARKER_CODES = {
        2: [1, -1],
        3: [1, 1, -1],
        4: [1, 1, -1, 1],
        5: [1, 1, 1, -1, 1],
        7: [1, 1, 1, -1, -1, 1, -1],
        11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
        13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
    }
    
    def __init__(self, params: WaveformParameters):
        """
        Initialize waveform generator
        
        Args:
            params: Waveform parameters
        """
        self.params = params
        self.waveform_cache = {}
        
    def generate(self, waveform_type: WaveformType, **kwargs) -> np.ndarray:
        """
        Generate waveform of specified type
        
        Args:
            waveform_type: Type of waveform to generate
            **kwargs: Additional parameters specific to waveform type
            
        Returns:
            Complex waveform samples
        """
        if waveform_type == WaveformType.LFM:
            return self.lfm_chirp(**kwargs)
        elif waveform_type == WaveformType.NLFM:
            return self.nlfm_chirp(**kwargs)
        elif waveform_type == WaveformType.BARKER:
            return self.barker_code(**kwargs)
        elif waveform_type == WaveformType.FRANK:
            return self.frank_code(**kwargs)
        elif waveform_type in [WaveformType.P1, WaveformType.P2, WaveformType.P3, WaveformType.P4]:
            return self.polyphase_code(waveform_type.value, **kwargs)
        elif waveform_type == WaveformType.COSTAS:
            return self.costas_sequence(**kwargs)
        elif waveform_type == WaveformType.STEPPED_FREQ:
            return self.stepped_frequency(**kwargs)
        elif waveform_type == WaveformType.OFDM:
            return self.ofdm_waveform(**kwargs)
        elif waveform_type == WaveformType.CUSTOM:
            return self.custom_waveform(**kwargs)
        else:
            raise ValueError(f"Unknown waveform type: {waveform_type}")
    
    def lfm_chirp(self, chirp_direction: str = "up", 
                  window: Optional[str] = None) -> np.ndarray:
        """
        Generate Linear Frequency Modulation (LFM) chirp
        
        Args:
            chirp_direction: "up" or "down" chirp
            window: Optional window function ('hamming', 'hann', 'taylor', etc.)
            
        Returns:
            Complex LFM waveform
        """
        t = self.params.time_vector
        k = self.params.bandwidth / self.params.pulse_width  # Chirp rate
        
        if chirp_direction == "down":
            k = -k
        
        # Generate chirp: exp(j*pi*k*t^2)
        phase = np.pi * k * t**2
        waveform = np.exp(1j * phase)
        
        # Apply window if specified
        if window:
            waveform = self._apply_window(waveform, window)
        
        return waveform
    
    def nlfm_chirp(self, profile: str = "tangent", 
                   sidelobe_level: float = -40,
                   window: Optional[str] = None) -> np.ndarray:
        """
        Generate Non-Linear Frequency Modulation (NLFM) chirp
        Optimized for low sidelobes without amplitude weighting
        
        Args:
            profile: Type of NLFM ('tangent', 'logarithmic', 'quadratic', 'custom')
            sidelobe_level: Desired sidelobe level in dB
            window: Optional additional window
            
        Returns:
            Complex NLFM waveform
        """
        t = self.params.time_vector
        t_norm = t / self.params.pulse_width  # Normalized time [0, 1]
        
        if profile == "tangent":
            # Tangent NLFM for sidelobe suppression
            alpha = self._calculate_nlfm_alpha(sidelobe_level)
            freq_func = np.tan(alpha * (t_norm - 0.5)) / np.tan(alpha / 2)
            
        elif profile == "logarithmic":
            # Logarithmic NLFM
            a = 10  # Shape parameter
            freq_func = np.log(1 + a * t_norm) / np.log(1 + a)
            
        elif profile == "quadratic":
            # Quadratic NLFM
            freq_func = t_norm**2
            
        elif profile == "custom":
            # Allow user to provide custom frequency function
            raise NotImplementedError("Custom NLFM profile requires frequency function")
        else:
            raise ValueError(f"Unknown NLFM profile: {profile}")
        
        # Integrate frequency to get phase
        phase = 2 * np.pi * self.params.bandwidth * np.cumsum(freq_func) / self.params.sample_rate
        waveform = np.exp(1j * phase)
        
        if window:
            waveform = self._apply_window(waveform, window)
        
        return waveform
    
    def barker_code(self, length: int = 13, 
                    chip_oversampling: Optional[int] = None) -> np.ndarray:
        """
        Generate Barker code phase-modulated waveform
        
        Args:
            length: Barker code length (2, 3, 4, 5, 7, 11, or 13)
            chip_oversampling: Samples per chip (None = auto-calculate)
            
        Returns:
            Complex Barker-coded waveform
        """
        if length not in self.BARKER_CODES:
            raise ValueError(f"Invalid Barker code length: {length}. "
                           f"Valid lengths: {list(self.BARKER_CODES.keys())}")
        
        code = np.array(self.BARKER_CODES[length])
        
        if chip_oversampling is None:
            chip_oversampling = max(1, self.params.num_samples // length)
        
        # Upsample the code
        waveform = np.repeat(code, chip_oversampling)
        
        # Trim or pad to match desired length
        if len(waveform) > self.params.num_samples:
            waveform = waveform[:self.params.num_samples]
        elif len(waveform) < self.params.num_samples:
            waveform = np.pad(waveform, (0, self.params.num_samples - len(waveform)))
        
        return waveform.astype(complex)
    
    def frank_code(self, M: int = 16) -> np.ndarray:
        """
        Generate Frank polyphase code
        
        Args:
            M: Code length (should be a perfect square)
            
        Returns:
            Complex Frank-coded waveform
        """
        N = int(np.sqrt(M))
        if N * N != M:
            raise ValueError(f"M={M} must be a perfect square for Frank codes")
        
        # Generate Frank code phase matrix
        phases = np.zeros(M)
        for i in range(N):
            for j in range(N):
                phases[i * N + j] = 2 * np.pi * i * j / N
        
        code = np.exp(1j * phases)
        
        # Upsample to match pulse width
        samples_per_chip = max(1, self.params.num_samples // M)
        waveform = np.repeat(code, samples_per_chip)
        
        if len(waveform) > self.params.num_samples:
            waveform = waveform[:self.params.num_samples]
        elif len(waveform) < self.params.num_samples:
            waveform = np.pad(waveform, (0, self.params.num_samples - len(waveform)))
        
        return waveform
    
    def polyphase_code(self, code_type: str, N: int = 16) -> np.ndarray:
        """
        Generate P1, P2, P3, or P4 polyphase codes
        
        Args:
            code_type: Type of code ('p1', 'p2', 'p3', 'p4')
            N: Code length
            
        Returns:
            Complex polyphase-coded waveform
        """
        phases = np.zeros(N)
        
        if code_type == "p1":
            # P1 code
            for n in range(N):
                phases[n] = -np.pi * (n * (n - 1)) / N
                
        elif code_type == "p2":
            # P2 code
            for n in range(N):
                phases[n] = -np.pi * (n - 1)**2 / N
                
        elif code_type == "p3":
            # P3 code (Px code)
            for n in range(N):
                phases[n] = np.pi * (n - 1) * (n - N) / N
                
        elif code_type == "p4":
            # P4 code
            for n in range(N):
                phases[n] = np.pi * ((n - 1)**2 - (n - 1) * N) / N
        else:
            raise ValueError(f"Unknown polyphase code type: {code_type}")
        
        code = np.exp(1j * phases)
        
        # Upsample to match pulse width
        samples_per_chip = max(1, self.params.num_samples // N)
        waveform = np.repeat(code, samples_per_chip)
        
        if len(waveform) > self.params.num_samples:
            waveform = waveform[:self.params.num_samples]
        elif len(waveform) < self.params.num_samples:
            waveform = np.pad(waveform, (0, self.params.num_samples - len(waveform)))
        
        return waveform
    
    def costas_sequence(self, N: int = 7, sequence: Optional[list] = None) -> np.ndarray:
        """
        Generate Costas frequency-hopping sequence
        
        Args:
            N: Sequence length
            sequence: Optional custom frequency sequence (if None, generates Welch-Costas)
            
        Returns:
            Complex Costas-coded waveform
        """
        if sequence is None:
            # Generate Welch-Costas sequence
            sequence = self._generate_welch_costas(N)
        
        hop_duration = self.params.pulse_width / N
        samples_per_hop = self.params.num_samples // N
        
        waveform = np.zeros(self.params.num_samples, dtype=complex)
        
        for i, freq_idx in enumerate(sequence):
            start_idx = i * samples_per_hop
            end_idx = min((i + 1) * samples_per_hop, self.params.num_samples)
            
            # Frequency for this hop
            freq = (freq_idx - N/2) * self.params.bandwidth / N
            t = np.arange(end_idx - start_idx) / self.params.sample_rate
            
            waveform[start_idx:end_idx] = np.exp(1j * 2 * np.pi * freq * t)
        
        return waveform
    
    def stepped_frequency(self, num_steps: int = 32, 
                         step_duration: Optional[float] = None) -> np.ndarray:
        """
        Generate stepped frequency waveform
        
        Args:
            num_steps: Number of frequency steps
            step_duration: Duration of each step (None = equal division)
            
        Returns:
            Complex stepped-frequency waveform
        """
        if step_duration is None:
            step_duration = self.params.pulse_width / num_steps
        
        samples_per_step = int(step_duration * self.params.sample_rate)
        freq_step = self.params.bandwidth / num_steps
        
        waveform = np.zeros(self.params.num_samples, dtype=complex)
        
        for i in range(num_steps):
            start_idx = i * samples_per_step
            end_idx = min((i + 1) * samples_per_step, self.params.num_samples)
            
            if start_idx >= self.params.num_samples:
                break
            
            # Frequency for this step
            freq = -self.params.bandwidth/2 + i * freq_step
            t = np.arange(end_idx - start_idx) / self.params.sample_rate
            
            waveform[start_idx:end_idx] = np.exp(1j * 2 * np.pi * freq * t)
        
        return waveform
    
    def ofdm_waveform(self, num_subcarriers: int = 64,
                     cyclic_prefix_ratio: float = 0.25,
                     modulation: str = "qpsk") -> np.ndarray:
        """
        Generate OFDM radar waveform
        
        Args:
            num_subcarriers: Number of OFDM subcarriers
            cyclic_prefix_ratio: Cyclic prefix length as ratio of symbol length
            modulation: Subcarrier modulation ('bpsk', 'qpsk', 'random')
            
        Returns:
            Complex OFDM waveform
        """
        # Calculate OFDM symbol parameters
        symbol_duration = self.params.pulse_width / (1 + cyclic_prefix_ratio)
        cp_duration = symbol_duration * cyclic_prefix_ratio
        
        samples_per_symbol = int(symbol_duration * self.params.sample_rate)
        cp_samples = int(cp_duration * self.params.sample_rate)
        
        # Generate subcarrier data
        if modulation == "bpsk":
            data = 2 * np.random.randint(0, 2, num_subcarriers) - 1
        elif modulation == "qpsk":
            data = (2 * np.random.randint(0, 2, num_subcarriers) - 1 + 
                   1j * (2 * np.random.randint(0, 2, num_subcarriers) - 1)) / np.sqrt(2)
        elif modulation == "random":
            data = np.random.randn(num_subcarriers) + 1j * np.random.randn(num_subcarriers)
            data /= np.abs(data)  # Normalize to unit magnitude
        else:
            raise ValueError(f"Unknown modulation: {modulation}")
        
        # IFFT to generate time-domain signal
        time_signal = np.fft.ifft(data, samples_per_symbol)
        
        # Add cyclic prefix
        cp = time_signal[-cp_samples:]
        ofdm_symbol = np.concatenate([cp, time_signal])
        
        # Trim to match desired length
        if len(ofdm_symbol) > self.params.num_samples:
            waveform = ofdm_symbol[:self.params.num_samples]
        else:
            waveform = np.pad(ofdm_symbol, (0, self.params.num_samples - len(ofdm_symbol)))
        
        return waveform
    
    def custom_waveform(self, 
                       freq_function: Optional[Callable] = None,
                       phase_function: Optional[Callable] = None,
                       amplitude_function: Optional[Callable] = None) -> np.ndarray:
        """
        Generate custom waveform from user-defined functions
        
        Args:
            freq_function: Function that returns instantaneous frequency vs time
            phase_function: Function that returns phase vs time
            amplitude_function: Function that returns amplitude vs time
            
        Returns:
            Complex custom waveform
        """
        t = self.params.time_vector
        
        if phase_function is not None:
            # Direct phase modulation
            phase = phase_function(t)
        elif freq_function is not None:
            # Frequency modulation - integrate to get phase
            freq = freq_function(t)
            phase = 2 * np.pi * np.cumsum(freq) / self.params.sample_rate
        else:
            # Default to CW
            phase = np.zeros_like(t)
        
        # Generate complex waveform
        waveform = np.exp(1j * phase)
        
        # Apply amplitude modulation if specified
        if amplitude_function is not None:
            amplitude = amplitude_function(t)
            waveform *= amplitude
        
        return waveform
    
    def ambiguity_function(self, waveform: np.ndarray,
                          max_delay: int = 100,
                          max_doppler: int = 100,
                          normalize: bool = True) -> np.ndarray:
        """
        Calculate ambiguity function of waveform
        
        Args:
            waveform: Complex waveform
            max_delay: Maximum delay in samples
            max_doppler: Maximum Doppler in bins
            normalize: Normalize to peak value
            
        Returns:
            2D ambiguity function (delay x Doppler)
        """
        N = len(waveform)
        ambiguity = np.zeros((2 * max_delay + 1, 2 * max_doppler + 1), dtype=complex)
        
        for tau_idx, tau in enumerate(range(-max_delay, max_delay + 1)):
            for fd_idx, fd in enumerate(range(-max_doppler, max_doppler + 1)):
                # Doppler shift
                doppler = np.exp(1j * 2 * np.pi * fd * np.arange(N) / N)
                shifted_signal = waveform * doppler
                
                # Time delay and correlation
                if tau >= 0:
                    if tau < N:
                        ambiguity[tau_idx, fd_idx] = np.sum(
                            waveform[tau:] * np.conj(shifted_signal[:N - tau])
                        )
                else:
                    if -tau < N:
                        ambiguity[tau_idx, fd_idx] = np.sum(
                            waveform[:N + tau] * np.conj(shifted_signal[-tau:])
                        )
        
        ambiguity = np.abs(ambiguity)**2
        
        if normalize:
            ambiguity /= np.max(ambiguity)
        
        return ambiguity
    
    def autocorrelation(self, waveform: np.ndarray) -> np.ndarray:
        """
        Calculate autocorrelation function
        
        Args:
            waveform: Complex waveform
            
        Returns:
            Autocorrelation function
        """
        return np.correlate(waveform, waveform, mode='full')
    
    def peak_sidelobe_ratio(self, waveform: np.ndarray) -> float:
        """
        Calculate peak sidelobe ratio (PSR) after matched filtering
        
        Args:
            waveform: Complex waveform
            
        Returns:
            PSR in dB
        """
        autocorr = np.abs(self.autocorrelation(waveform))
        peak_idx = len(autocorr) // 2
        peak = autocorr[peak_idx]
        
        # Find sidelobes (exclude mainlobe)
        mainlobe_width = 3  # samples
        sidelobes = np.concatenate([
            autocorr[:peak_idx - mainlobe_width],
            autocorr[peak_idx + mainlobe_width + 1:]
        ])
        
        if len(sidelobes) > 0:
            max_sidelobe = np.max(sidelobes)
            psr = 20 * np.log10(peak / max_sidelobe)
        else:
            psr = np.inf
        
        return psr
    
    def integrated_sidelobe_ratio(self, waveform: np.ndarray) -> float:
        """
        Calculate integrated sidelobe ratio (ISR)
        
        Args:
            waveform: Complex waveform
            
        Returns:
            ISR in dB
        """
        autocorr = np.abs(self.autocorrelation(waveform))**2
        peak_idx = len(autocorr) // 2
        
        # Mainlobe energy
        mainlobe_width = 3
        mainlobe_energy = np.sum(autocorr[peak_idx - mainlobe_width:peak_idx + mainlobe_width + 1])
        
        # Total energy
        total_energy = np.sum(autocorr)
        
        # Sidelobe energy
        sidelobe_energy = total_energy - mainlobe_energy
        
        if sidelobe_energy > 0:
            isr = 10 * np.log10(mainlobe_energy / sidelobe_energy)
        else:
            isr = np.inf
        
        return isr
    
    def _apply_window(self, waveform: np.ndarray, window_type: str) -> np.ndarray:
        """
        Apply window function to waveform
        
        Args:
            waveform: Input waveform
            window_type: Type of window
            
        Returns:
            Windowed waveform
        """
        N = len(waveform)
        
        if window_type == "hamming":
            window = np.hamming(N)
        elif window_type == "hann":
            window = np.hanning(N)
        elif window_type == "blackman":
            window = np.blackman(N)
        elif window_type == "kaiser":
            window = np.kaiser(N, 5)
        elif window_type == "taylor":
            window = signal.windows.taylor(N, nbar=4, sll=-35)
        else:
            raise ValueError(f"Unknown window type: {window_type}")
        
        return waveform * window
    
    def _calculate_nlfm_alpha(self, sidelobe_level_db: float) -> float:
        """
        Calculate NLFM shape parameter for desired sidelobe level
        
        Args:
            sidelobe_level_db: Desired sidelobe level in dB
            
        Returns:
            Shape parameter alpha
        """
        # Empirical relationship for tangent NLFM
        return 0.1 * abs(sidelobe_level_db)
    
    def _generate_welch_costas(self, N: int) -> list:
        """
        Generate Welch-Costas sequence
        
        Args:
            N: Sequence length (should be prime)
            
        Returns:
            Frequency hopping sequence
        """
        # Find primitive root modulo N
        if N <= 2:
            return list(range(N))
        
        # Simple Welch construction for prime N
        g = 2  # Try 2 as primitive root
        sequence = []
        for i in range(N - 1):
            sequence.append((g**i) % N)
        
        return sequence