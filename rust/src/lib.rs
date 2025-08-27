use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{Array1, Array2, Axis};
use num_complex::Complex64;
use rustfft::{FftPlanner, num_complex::Complex};

/// Perform matched filtering using FFT-based correlation
#[pyfunction]
fn matched_filter_rust<'py>(
    py: Python<'py>,
    received: PyReadonlyArray1<'py, Complex64>,
    reference: PyReadonlyArray1<'py, Complex64>,
) -> PyResult<&'py PyArray1<Complex64>> {
    let received = received.as_array();
    let reference = reference.as_array();
    
    let n = received.len().max(reference.len());
    let padded_len = n.next_power_of_two();
    
    // Prepare FFT
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(padded_len);
    let ifft = planner.plan_fft_inverse(padded_len);
    
    // Pad signals
    let mut rec_padded = vec![Complex::new(0.0, 0.0); padded_len];
    let mut ref_padded = vec![Complex::new(0.0, 0.0); padded_len];
    
    for (i, &val) in received.iter().enumerate() {
        rec_padded[i] = Complex::new(val.re, val.im);
    }
    
    for (i, &val) in reference.iter().enumerate() {
        ref_padded[i] = Complex::new(val.re, val.im);
    }
    
    // FFT
    fft.process(&mut rec_padded);
    fft.process(&mut ref_padded);
    
    // Multiply with conjugate
    for i in 0..padded_len {
        let conj = ref_padded[i].conj();
        let prod = rec_padded[i] * conj;
        rec_padded[i] = prod;
    }
    
    // IFFT
    ifft.process(&mut rec_padded);
    
    // Normalize and convert back
    let norm = 1.0 / padded_len as f64;
    let mut result = Array1::<Complex64>::zeros(n);
    for i in 0..n {
        result[i] = Complex64::new(
            rec_padded[i].re * norm,
            rec_padded[i].im * norm
        );
    }
    
    Ok(PyArray1::from_array(py, &result))
}

/// Fast range-Doppler processing
#[pyfunction]
fn range_doppler_map_rust<'py>(
    py: Python<'py>,
    data_cube: PyReadonlyArray2<'py, Complex64>,
    window_range: bool,
    window_doppler: bool,
) -> PyResult<&'py PyArray2<f64>> {
    let data = data_cube.as_array();
    let (n_range, n_doppler) = data.dim();
    
    let mut windowed_data = data.to_owned();
    
    // Apply windowing if requested
    if window_range {
        let window = hamming_window(n_range);
        for mut col in windowed_data.axis_iter_mut(Axis(1)) {
            for (i, val) in col.iter_mut().enumerate() {
                *val = *val * window[i];
            }
        }
    }
    
    if window_doppler {
        let window = hamming_window(n_doppler);
        for mut row in windowed_data.axis_iter_mut(Axis(0)) {
            for (i, val) in row.iter_mut().enumerate() {
                *val = *val * window[i];
            }
        }
    }
    
    // 2D FFT for range-Doppler processing
    let mut planner = FftPlanner::<f64>::new();
    let fft_range = planner.plan_fft_forward(n_range);
    let fft_doppler = planner.plan_fft_forward(n_doppler);
    
    // FFT along range dimension
    for mut row in windowed_data.axis_iter_mut(Axis(0)) {
        let mut fft_data: Vec<Complex<f64>> = row.iter()
            .map(|&c| Complex::new(c.re, c.im))
            .collect();
        fft_range.process(&mut fft_data);
        for (i, val) in row.iter_mut().enumerate() {
            *val = Complex64::new(fft_data[i].re, fft_data[i].im);
        }
    }
    
    // FFT along Doppler dimension
    for mut col in windowed_data.axis_iter_mut(Axis(1)) {
        let mut fft_data: Vec<Complex<f64>> = col.iter()
            .map(|&c| Complex::new(c.re, c.im))
            .collect();
        fft_doppler.process(&mut fft_data);
        for (i, val) in col.iter_mut().enumerate() {
            *val = Complex64::new(fft_data[i].re, fft_data[i].im);
        }
    }
    
    // Convert to magnitude
    let mut magnitude = Array2::<f64>::zeros((n_range, n_doppler));
    for ((i, j), val) in windowed_data.indexed_iter() {
        magnitude[[i, j]] = val.norm();
    }
    
    Ok(PyArray2::from_array(py, &magnitude))
}

/// Generate Hamming window
fn hamming_window(n: usize) -> Vec<f64> {
    let mut window = vec![0.0; n];
    for i in 0..n {
        window[i] = 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
    }
    window
}

/// Fast CFAR detection
#[pyfunction]
fn cfar_detect_rust<'py>(
    _py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    guard_cells: usize,
    training_cells: usize,
    threshold_factor: f64,
) -> PyResult<Vec<usize>> {
    let signal = signal.as_array();
    let n = signal.len();
    let _window_size = 2 * (guard_cells + training_cells) + 1;
    
    let mut detections = Vec::new();
    
    for i in (guard_cells + training_cells)..n.saturating_sub(guard_cells + training_cells) {
        let mut sum = 0.0;
        let mut count = 0;
        
        // Left training cells
        for j in (i - guard_cells - training_cells)..i - guard_cells {
            sum += signal[j];
            count += 1;
        }
        
        // Right training cells
        for j in (i + guard_cells + 1)..=(i + guard_cells + training_cells) {
            if j < n {
                sum += signal[j];
                count += 1;
            }
        }
        
        if count > 0 {
            let noise_level = sum / count as f64;
            let threshold = threshold_factor * noise_level;
            
            if signal[i] > threshold {
                detections.push(i);
            }
        }
    }
    
    Ok(detections)
}

/// Fast pulse compression
#[pyfunction]
fn pulse_compress_rust<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, Complex64>,
    chirp_rate: f64,
    pulse_width: f64,
    sample_rate: f64,
) -> PyResult<&'py PyArray1<Complex64>> {
    let signal = signal.as_array();
    let n = signal.len();
    
    // Generate matched filter
    let samples = (pulse_width * sample_rate) as usize;
    let mut matched_filter = Array1::<Complex64>::zeros(samples);
    
    for i in 0..samples {
        let t = i as f64 / sample_rate;
        let phase = std::f64::consts::PI * chirp_rate * t * t;
        matched_filter[i] = Complex64::new(phase.cos(), -phase.sin());
    }
    
    // Perform matched filtering using FFT
    let result_len = n + samples - 1;
    let padded_len = result_len.next_power_of_two();
    
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(padded_len);
    let ifft = planner.plan_fft_inverse(padded_len);
    
    let mut sig_fft = vec![Complex::new(0.0, 0.0); padded_len];
    let mut filter_fft = vec![Complex::new(0.0, 0.0); padded_len];
    
    for (i, &val) in signal.iter().enumerate() {
        sig_fft[i] = Complex::new(val.re, val.im);
    }
    
    for (i, &val) in matched_filter.iter().enumerate() {
        filter_fft[i] = Complex::new(val.re, val.im);
    }
    
    fft.process(&mut sig_fft);
    fft.process(&mut filter_fft);
    
    for i in 0..padded_len {
        sig_fft[i] = sig_fft[i] * filter_fft[i].conj();
    }
    
    ifft.process(&mut sig_fft);
    
    let norm = 1.0 / padded_len as f64;
    let mut result = Array1::<Complex64>::zeros(n);
    for i in 0..n {
        result[i] = Complex64::new(
            sig_fft[i].re * norm,
            sig_fft[i].im * norm
        );
    }
    
    Ok(PyArray1::from_array(py, &result))
}

/// Fast matrix multiplication for tracking
#[pyfunction]
fn kalman_predict_rust<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, f64>,
    covariance: PyReadonlyArray2<'py, f64>,
    transition: PyReadonlyArray2<'py, f64>,
    process_noise: PyReadonlyArray2<'py, f64>,
) -> PyResult<(&'py PyArray1<f64>, &'py PyArray2<f64>)> {
    let x = state.as_array();
    let p = covariance.as_array();
    let f = transition.as_array();
    let q = process_noise.as_array();
    
    // Predict state: x = F @ x
    let x_pred = f.dot(&x);
    
    // Predict covariance: P = F @ P @ F.T + Q
    let p_temp = f.dot(&p);
    let p_pred = p_temp.dot(&f.t()) + q;
    
    Ok((
        PyArray1::from_array(py, &x_pred),
        PyArray2::from_array(py, &p_pred)
    ))
}

/// Python module definition
#[pymodule]
fn radar_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matched_filter_rust, m)?)?;
    m.add_function(wrap_pyfunction!(range_doppler_map_rust, m)?)?;
    m.add_function(wrap_pyfunction!(cfar_detect_rust, m)?)?;
    m.add_function(wrap_pyfunction!(pulse_compress_rust, m)?)?;
    m.add_function(wrap_pyfunction!(kalman_predict_rust, m)?)?;
    Ok(())
}