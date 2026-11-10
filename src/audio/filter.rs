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

//! Butterworth filters (bilinear transform, second-order sections) and the
//! per-chunk "denoise" post-processing.

use std::path::Path;

use anyhow::{Result, bail};

use super::{read_wav, tags, write_wav_pcm16};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterKind {
    LowPass,
    HighPass,
}

/// One biquad: b0, b1, b2, a1, a2 (a0 = 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sos {
    pub b: [f64; 3],
    pub a: [f64; 2],
}

/// Digital Butterworth filter of `order` with cutoff `cutoff_hz` as cascaded
/// second-order sections (same transfer function as
/// `scipy.signal.butter(order, cutoff, kind, fs=fs, output="sos")`).
pub fn butter_sos(order: usize, cutoff_hz: f64, fs: f64, kind: FilterKind) -> Result<Vec<Sos>> {
    if order == 0 {
        bail!("filter order must be >= 1");
    }
    if !(cutoff_hz > 0.0 && cutoff_hz < fs / 2.0) {
        bail!(
            "cutoff {cutoff_hz} Hz must be between 0 and fs/2 = {} Hz",
            fs / 2.0
        );
    }
    // Pre-warped analog cutoff for the bilinear transform with T = 1.
    let k = (std::f64::consts::PI * cutoff_hz / fs).tan();
    let k2 = k * k;
    let mut sections = Vec::new();
    for i in 0..order / 2 {
        // Pole pair at angle theta: s^2 + 2 sin(theta) s + 1 (normalised)
        let theta = std::f64::consts::PI * (2 * i + 1) as f64 / (2 * order) as f64;
        let q_inv = 2.0 * theta.sin();
        let norm = 1.0 / (1.0 + k * q_inv + k2);
        let a1 = 2.0 * (k2 - 1.0) * norm;
        let a2 = (1.0 - k * q_inv + k2) * norm;
        let b = match kind {
            FilterKind::LowPass => {
                let b0 = k2 * norm;
                [b0, 2.0 * b0, b0]
            }
            FilterKind::HighPass => [norm, -2.0 * norm, norm],
        };
        sections.push(Sos { b, a: [a1, a2] });
    }
    if order % 2 == 1 {
        let norm = 1.0 / (1.0 + k);
        let a1 = (k - 1.0) * norm;
        let b = match kind {
            FilterKind::LowPass => [k * norm, k * norm, 0.0],
            FilterKind::HighPass => [norm, -norm, 0.0],
        };
        sections.push(Sos { b, a: [a1, 0.0] });
    }
    Ok(sections)
}

/// Filter `x` in place with cascaded biquads (direct form II transposed,
/// zero initial state), like `scipy.signal.sosfilt`.
pub fn sosfilt(sections: &[Sos], x: &mut [f64]) {
    for s in sections {
        let (mut z1, mut z2) = (0.0f64, 0.0f64);
        for v in x.iter_mut() {
            let input = *v;
            let y = s.b[0] * input + z1;
            z1 = s.b[1] * input - s.a[0] * y + z2;
            z2 = s.b[2] * input - s.a[1] * y;
            *v = y;
        }
    }
}

/// Remove sub-bass rumble below 60 Hz and gently roll off highs above
/// 9 kHz (harsh/metallic TTS artifacts). Rewrites the file as 16-bit PCM
/// and keeps its ID3 tag.
pub fn denoise(path: &Path) -> Result<()> {
    log::info!("Denoising: {}", path.display());
    let result = (|| -> Result<()> {
        let tag = tags::read_tag(path);
        let audio = read_wav(path)?;
        let sr = audio.sample_rate as f64;
        let hp = butter_sos(4, 60.0, sr, FilterKind::HighPass)?;
        let cutoff = (9000i64).min(audio.sample_rate as i64 / 2 - 500);
        let lp = if cutoff > 0 {
            Some(butter_sos(2, cutoff as f64, sr, FilterKind::LowPass)?)
        } else {
            None
        };
        let ch = audio.channels.max(1) as usize;
        let frames = audio.samples.len() / ch;
        let mut out = vec![0f32; frames * ch];
        for c in 0..ch {
            let mut x: Vec<f64> = (0..frames)
                .map(|i| audio.samples[i * ch + c] as f64)
                .collect();
            sosfilt(&hp, &mut x);
            if let Some(lp) = &lp {
                sosfilt(lp, &mut x);
            }
            for (i, v) in x.into_iter().enumerate() {
                out[i * ch + c] = v as f32;
            }
        }
        write_wav_pcm16(path, &out, audio.channels, audio.sample_rate)?;
        if let Some(tag) = tag {
            tags::write_tag(path, &tag);
        }
        log::debug!("Post-processing complete: {}", path.display());
        Ok(())
    })();
    if let Err(e) = &result {
        log::error!("Audio post-processing failed for {}: {e:#}", path.display());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// |H(e^jw)| of the cascade at frequency f.
    fn gain(sections: &[Sos], f: f64, fs: f64) -> f64 {
        let w = 2.0 * std::f64::consts::PI * f / fs;
        let mut g = 1.0;
        for s in sections {
            let (c1, s1) = (w.cos(), -w.sin());
            let (c2, s2) = ((2.0 * w).cos(), -(2.0 * w).sin());
            let num_re = s.b[0] + s.b[1] * c1 + s.b[2] * c2;
            let num_im = s.b[1] * s1 + s.b[2] * s2;
            let den_re = 1.0 + s.a[0] * c1 + s.a[1] * c2;
            let den_im = s.a[0] * s1 + s.a[1] * s2;
            g *= (num_re.hypot(num_im)) / den_re.hypot(den_im);
        }
        g
    }

    #[test]
    fn butterworth_is_minus_3db_at_cutoff() {
        for (order, kind) in [
            (4, FilterKind::HighPass),
            (2, FilterKind::LowPass),
            (3, FilterKind::LowPass),
        ] {
            let s = butter_sos(order, 1000.0, 24000.0, kind).unwrap();
            assert!((gain(&s, 1000.0, 24000.0) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9);
        }
        let hp = butter_sos(4, 60.0, 24000.0, FilterKind::HighPass).unwrap();
        assert!(gain(&hp, 15.0, 24000.0) < 0.01);
        assert!((gain(&hp, 1000.0, 24000.0) - 1.0).abs() < 1e-3);
        let lp = butter_sos(2, 9000.0, 24000.0, FilterKind::LowPass).unwrap();
        assert!(gain(&lp, 11900.0, 24000.0) < 0.05);
        assert!(butter_sos(2, 13000.0, 24000.0, FilterKind::LowPass).is_err());
    }

    #[test]
    fn matches_scipy() {
        // scipy.signal.butter(2, 9000, "lp", fs=24000, output="sos")
        let s = butter_sos(2, 9000.0, 24000.0, FilterKind::LowPass).unwrap();
        let expected_b = [0.5690355937288493, 1.1380711874576985, 0.5690355937288493];
        let expected_a = [0.9428090415820634, 0.33333333333333337];
        for (x, y) in s[0].b.iter().zip(expected_b) {
            assert!((x - y).abs() < 1e-12, "{x} vs {y}");
        }
        for (x, y) in s[0].a.iter().zip(expected_a) {
            assert!((x - y).abs() < 1e-12, "{x} vs {y}");
        }
        // sosfilt(butter(4, 60, "hp")) then sosfilt(butter(2, 9000, "lp")) at 24 kHz
        let mut x: Vec<f64> = (0..3000)
            .map(|n| {
                let n = n as f64;
                (0.1 * n).sin() + 0.5 * (0.37 * n).cos() + ((n as i64 % 7) as f64 - 3.0) / 10.0
            })
            .collect();
        sosfilt(
            &butter_sos(4, 60.0, 24000.0, FilterKind::HighPass).unwrap(),
            &mut x,
        );
        sosfilt(&s, &mut x);
        let expected = [
            (0, 0.11149518257418567),
            (1, 0.3173297532422338),
            (2, 0.4220525722470843),
            (100, -0.5181920662215282),
            (1000, 0.4875304026766581),
            (2999, -1.3567134103862792),
        ];
        for (i, y) in expected {
            assert!((x[i] - y).abs() < 1e-9, "sample {i}: {} vs {y}", x[i]);
        }
    }

    #[test]
    fn denoise_removes_dc_and_keeps_length() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.wav");
        let sr = 24000;
        let s: Vec<f32> = (0..sr)
            .map(|i| 0.3 + 0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();
        super::super::write_mono_wav(&p, &s, sr as u32).unwrap();
        denoise(&p).unwrap();
        let a = read_wav(&p).unwrap();
        assert_eq!(a.samples.len(), s.len());
        let mean: f32 = a.samples[12000..].iter().sum::<f32>() / 12000.0;
        assert!(mean.abs() < 0.01, "DC left: {mean}");
    }
}
