use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;

/// # Panics
///
/// panics if input slice and output slice do not have equal length.
///
/// panics if input slice has length 0
pub fn fft(time_domain: &[f32], freq_domain: &mut [f32]) {
    assert_eq!(time_domain.len(), freq_domain.len());

    let freqs = fft_(time_domain);
    for (i, f) in freqs.iter().enumerate() {
        freq_domain[i] = f.norm();
    }
}

/// the first fourth of freq_domain is the useful part
fn fft_(time_domain: &[f32]) -> Vec<Complex<f32>> {
    assert_ne!(time_domain.len(), 0);

    let mut input: Vec<Complex<f32>> = time_domain.iter().map(|f| Complex::new(*f, 0.0)).collect();
    let mut output: Vec<Complex<f32>> = vec![Complex::zero(); time_domain.len()];

    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(time_domain.len());
    fft.process(&mut input, &mut output);

    output
}
