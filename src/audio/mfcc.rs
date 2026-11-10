// This file is part of text-to-audiobook
//
// text-to-audiobook is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// text-to-audiobook is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! MFCC voice fingerprint with librosa's defaults
//! (`librosa.feature.mfcc(y, sr, n_mfcc=20)`): 2048-point STFT, hop 512,
//! centred with zero padding, 128 Slaney mel bands, power in dB with an
//! 80 dB floor, orthonormal DCT-II.

use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

const N_FFT: usize = 2048;
const HOP: usize = 512;
const N_MELS: usize = 128;
const N_MFCC: usize = 20;

fn hz_to_mel(f: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_sp * m
    }
}

/// Slaney-normalised mel filterbank, `n_mels x (n_fft/2 + 1)`.
fn mel_filterbank(sr: f64) -> Vec<Vec<f64>> {
    let n_bins = N_FFT / 2 + 1;
    let fft_freqs: Vec<f64> = (0..n_bins)
        .map(|i| i as f64 * (sr / 2.0) / (n_bins - 1) as f64)
        .collect();
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(sr / 2.0));
    let mel_f: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    let fdiff: Vec<f64> = mel_f.windows(2).map(|w| w[1] - w[0]).collect();
    (0..N_MELS)
        .map(|i| {
            let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
            fft_freqs
                .iter()
                .map(|f| {
                    let lower = (f - mel_f[i]) / fdiff[i];
                    let upper = (mel_f[i + 2] - f) / fdiff[i + 1];
                    lower.min(upper).max(0.0) * enorm
                })
                .collect()
        })
        .collect()
}

/// Mean MFCC vector (20 coefficients) of `samples`, or None for audio
/// shorter than half a second. Samples above 1.0 in magnitude are treated
/// as 16-bit PCM and scaled by 1/32768.
pub fn fingerprint(samples: &[f32], sample_rate: u32) -> Option<Vec<f64>> {
    if samples.len() < (sample_rate / 2) as usize || sample_rate == 0 {
        return None;
    }
    let peak = samples.iter().fold(0f32, |m, s| m.max(s.abs()));
    let scale = if peak > 1.0 { 1.0 / 32768.0 } else { 1.0 };
    let y: Vec<f64> = samples.iter().map(|s| *s as f64 * scale).collect();

    // Centre: pad n_fft/2 zeros on both sides.
    let pad = N_FFT / 2;
    let mut padded = vec![0f64; y.len() + 2 * pad];
    padded[pad..pad + y.len()].copy_from_slice(&y);
    let n_frames = 1 + (padded.len() - N_FFT) / HOP;

    // Periodic Hann window (scipy get_window("hann", n, fftbins=True)).
    let window: Vec<f64> = (0..N_FFT)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos())
        .collect();
    let mel = mel_filterbank(sample_rate as f64);
    let fft = FftPlanner::<f64>::new().plan_fft_forward(N_FFT);
    let n_bins = N_FFT / 2 + 1;

    // mel_spec[t][m]: frame-major
    let mut mel_spec = vec![vec![0f64; N_MELS]; n_frames];
    let mut buf = vec![Complex::new(0.0, 0.0); N_FFT];
    for (t, frame) in mel_spec.iter_mut().enumerate() {
        let start = t * HOP;
        for (n, b) in buf.iter_mut().enumerate() {
            *b = Complex::new(padded[start + n] * window[n], 0.0);
        }
        fft.process(&mut buf);
        let power: Vec<f64> = buf[..n_bins].iter().map(|c| c.norm_sqr()).collect();
        for (slot, filt) in frame.iter_mut().zip(&mel) {
            *slot = filt.iter().zip(&power).map(|(w, p)| w * p).sum();
        }
    }

    // power_to_db(ref=1.0, amin=1e-10, top_db=80)
    let mut max_db = f64::NEG_INFINITY;
    for row in mel_spec.iter_mut() {
        for v in row.iter_mut() {
            *v = 10.0 * v.max(1e-10).log10();
            max_db = max_db.max(*v);
        }
    }
    let floor = max_db - 80.0;
    for row in mel_spec.iter_mut() {
        for v in row.iter_mut() {
            *v = v.max(floor);
        }
    }

    // Orthonormal DCT-II over the mel axis, first 20 coefficients, mean over time.
    let n = N_MELS as f64;
    let mut fp = vec![0f64; N_MFCC];
    for (k, out) in fp.iter_mut().enumerate() {
        let norm = if k == 0 {
            (1.0 / n).sqrt()
        } else {
            (2.0 / n).sqrt()
        };
        let basis: Vec<f64> = (0..N_MELS)
            .map(|m| (std::f64::consts::PI * k as f64 * (2 * m + 1) as f64 / (2.0 * n)).cos())
            .collect();
        let mut sum = 0.0;
        for frame in &mel_spec {
            let c: f64 = frame.iter().zip(&basis).map(|(v, b)| v * b).sum();
            sum += c * norm;
        }
        *out = sum / n_frames as f64;
    }
    Some(fp)
}

/// Cosine similarity of two vectors.
pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denom = na * nb;
    dot / if denom == 0.0 { 1e-10 } else { denom }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_scale_roundtrip() {
        for f in [0.0, 500.0, 1000.0, 4000.0, 12000.0] {
            assert!((mel_to_hz(hz_to_mel(f)) - f).abs() < 1e-6);
        }
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-12);
    }

    #[test]
    fn fingerprint_shape_and_similarity() {
        let sr = 16000;
        let tone = |f: f32| -> Vec<f32> {
            (0..sr)
                .map(|i| 0.3 * (2.0 * std::f32::consts::PI * f * i as f32 / sr as f32).sin())
                .collect()
        };
        let a = fingerprint(&tone(220.0), sr as u32).unwrap();
        let b = fingerprint(&tone(220.0), sr as u32).unwrap();
        assert_eq!(a.len(), 20);
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-9);
        assert!(fingerprint(&[0.0; 100], 16000).is_none());
        // Int16-scaled input gives the same fingerprint.
        let scaled: Vec<f32> = tone(220.0).iter().map(|s| s * 32768.0).collect();
        let c = fingerprint(&scaled, sr as u32).unwrap();
        assert!((cosine(&a, &c) - 1.0).abs() < 1e-9);
    }
}
