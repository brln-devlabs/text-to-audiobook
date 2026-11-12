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

//! Loudness normalization of the finished audiobook.
//!
//! The integrated loudness follows ITU-R BS.1770-4 / EBU R128: K-weighting,
//! 400 ms blocks with 75 % overlap, an absolute gate at -70 LUFS and a
//! relative gate 10 LU below the ungated mean. Pauses are gated out, so
//! they do not pull the measurement down. One gain brings the whole file to
//! the target; a look-ahead limiter keeps the peaks below
//! [`PEAK_CEILING_DBFS`] instead of clipping them.

use std::collections::VecDeque;

use super::filter::Sos;

/// Highest sample peak after normalization (headroom for the MP3 encoder).
pub const PEAK_CEILING_DBFS: f64 = -1.5;

/// Never raise the level by more than this, so near-silent output (mock
/// runs, a failed synthesis) does not become loud noise.
pub const MAX_GAIN_DB: f64 = 20.0;

const ABSOLUTE_GATE_LUFS: f64 = -70.0;
const RELATIVE_GATE_LU: f64 = -10.0;

/// Look-ahead of the limiter: the gain is lowered over this time before a
/// peak instead of jumping.
const LIMITER_LOOKAHEAD_MS: f64 = 5.0;
/// Time constant for the gain to recover after a peak.
const LIMITER_RELEASE_MS: f64 = 150.0;

pub fn db_to_gain(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

fn mean_square_to_lufs(z: f64) -> f64 {
    -0.691 + 10.0 * z.log10()
}

fn lufs_to_mean_square(l: f64) -> f64 {
    10f64.powf((l + 0.691) / 10.0)
}

/// BS.1770 K-weighting (high shelf + high pass) for any sample rate, with
/// the analog prototype of libebur128; at 48 kHz it equals the
/// coefficients printed in the standard.
pub fn k_weighting(fs: f64) -> [Sos; 2] {
    // Stage 1: high shelf, +4 dB above about 1.7 kHz (acoustic head effect).
    let f0 = 1681.974450955533;
    let g = 3.999843853973347;
    let q = 0.7071752369554196;
    let k = (std::f64::consts::PI * f0 / fs).tan();
    let vh = 10f64.powf(g / 20.0);
    let vb = vh.powf(0.4996667741545416);
    let a0 = 1.0 + k / q + k * k;
    let shelf = Sos {
        b: [
            (vh + vb * k / q + k * k) / a0,
            2.0 * (k * k - vh) / a0,
            (vh - vb * k / q + k * k) / a0,
        ],
        a: [2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
    };
    // Stage 2: RLB high pass at about 38 Hz.
    let f0 = 38.13547087602444;
    let q = 0.5003270373238773;
    let k = (std::f64::consts::PI * f0 / fs).tan();
    let a0 = 1.0 + k / q + k * k;
    let high_pass = Sos {
        b: [1.0, -2.0, 1.0],
        a: [2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
    };
    [shelf, high_pass]
}

/// Streaming integrated-loudness meter for one mono signal.
pub struct LoudnessMeter {
    filter: [Sos; 2],
    /// Direct form II transposed state per section.
    state: [[f64; 2]; 2],
    /// 100 ms in samples; a gating block is four steps.
    step: usize,
    energy: f64,
    filled: usize,
    /// Energies of the last (up to three) complete steps.
    steps: VecDeque<f64>,
    /// Mean square of every 400 ms gating block.
    blocks: Vec<f64>,
}

impl LoudnessMeter {
    pub fn new(sample_rate: u32) -> Self {
        let fs = sample_rate.max(1) as f64;
        Self {
            filter: k_weighting(fs),
            state: [[0.0; 2]; 2],
            step: ((fs / 10.0).round() as usize).max(1),
            energy: 0.0,
            filled: 0,
            steps: VecDeque::with_capacity(4),
            blocks: Vec::new(),
        }
    }

    /// Feed the next samples of the signal.
    pub fn process(&mut self, samples: &[f32]) {
        for &x in samples {
            let mut v = x as f64;
            for (s, z) in self.filter.iter().zip(self.state.iter_mut()) {
                let y = s.b[0] * v + z[0];
                z[0] = s.b[1] * v - s.a[0] * y + z[1];
                z[1] = s.b[2] * v - s.a[1] * y;
                v = y;
            }
            self.energy += v * v;
            self.filled += 1;
            if self.filled == self.step {
                self.steps.push_back(self.energy);
                self.energy = 0.0;
                self.filled = 0;
                if self.steps.len() == 4 {
                    let sum: f64 = self.steps.iter().sum();
                    self.blocks.push(sum / (4 * self.step) as f64);
                    self.steps.pop_front();
                }
            }
        }
    }

    /// Gated loudness in LUFS, `None` if nothing is louder than -70 LUFS.
    pub fn integrated_lufs(&self) -> Option<f64> {
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let absolute = lufs_to_mean_square(ABSOLUTE_GATE_LUFS);
        let loud: Vec<f64> = self
            .blocks
            .iter()
            .copied()
            .filter(|&z| z > absolute)
            .collect();
        if loud.is_empty() {
            return None;
        }
        let relative = lufs_to_mean_square(mean_square_to_lufs(mean(&loud)) + RELATIVE_GATE_LU);
        let gated: Vec<f64> = loud.into_iter().filter(|&z| z > relative).collect();
        if gated.is_empty() {
            return None;
        }
        Some(mean_square_to_lufs(mean(&gated)))
    }
}

/// Gain in dB that brings `measured` to `target` (at most [`MAX_GAIN_DB`]).
pub fn normalization_gain_db(measured: f64, target: f64) -> f64 {
    (target - measured).min(MAX_GAIN_DB)
}

/// Streaming look-ahead peak limiter.
///
/// For every sample it computes the gain that keeps it below the ceiling,
/// takes the minimum over the look-ahead window, lets it recover slowly
/// (release) and smooths it with a moving average of the window length.
/// The output is delayed by one window minus one sample, which guarantees
/// that each sample is fully attenuated when it is written. The delay is
/// hidden: [`process`](Self::process) plus [`flush`](Self::flush) return
/// exactly as many samples as went in, aligned with the input.
pub struct Limiter {
    ceiling: f64,
    window: usize,
    release: f64,
    /// Input samples waiting for their gain.
    delay: VecDeque<f32>,
    /// (index, required gain), increasing gains: sliding minimum.
    minima: VecDeque<(usize, f64)>,
    /// Released gains of the last `window` samples and their sum.
    smooth: VecDeque<f64>,
    smooth_sum: f64,
    envelope: f64,
    /// Samples fed in (input plus flush padding).
    fed: usize,
    /// Lowest gain applied so far.
    min_gain: f64,
}

impl Limiter {
    pub fn new(ceiling_dbfs: f64, sample_rate: u32) -> Self {
        let fs = sample_rate.max(1) as f64;
        let window = ((LIMITER_LOOKAHEAD_MS / 1000.0 * fs).round() as usize).max(1);
        Self {
            ceiling: db_to_gain(ceiling_dbfs),
            window,
            release: (-1.0 / (LIMITER_RELEASE_MS / 1000.0 * fs)).exp(),
            delay: VecDeque::with_capacity(window),
            minima: VecDeque::new(),
            smooth: std::iter::repeat_n(1.0, window).collect(),
            smooth_sum: window as f64,
            envelope: 1.0,
            fed: 0,
            min_gain: 1.0,
        }
    }

    fn step(&mut self, x: f32, out: &mut Vec<f32>) {
        let level = x.abs() as f64;
        let required = if level > self.ceiling {
            self.ceiling / level
        } else {
            1.0
        };
        let n = self.fed;
        self.fed += 1;
        while self.minima.back().is_some_and(|&(_, g)| g >= required) {
            self.minima.pop_back();
        }
        self.minima.push_back((n, required));
        while self
            .minima
            .front()
            .is_some_and(|&(i, _)| i + self.window <= n)
        {
            self.minima.pop_front();
        }
        let target = self.minima.front().map_or(1.0, |&(_, g)| g);
        // Instant attack, exponential release; never above the target.
        self.envelope = if target < self.envelope {
            target
        } else {
            target + (self.envelope - target) * self.release
        };
        self.smooth_sum += self.envelope - self.smooth.pop_front().unwrap_or(1.0);
        self.smooth.push_back(self.envelope);
        let gain = (self.smooth_sum / self.window as f64).min(1.0);

        self.delay.push_back(x);
        if self.delay.len() == self.window {
            let y = self.delay.pop_front().unwrap_or(0.0);
            if y != 0.0 {
                self.min_gain = self.min_gain.min(gain);
            }
            let y = y as f64 * gain;
            out.push(y.clamp(-self.ceiling, self.ceiling) as f32);
        }
    }

    /// Limit `input` and append the samples that are ready to `out`.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        out.reserve(input.len());
        for &x in input {
            self.step(x, out);
        }
    }

    /// Append the samples still held back by the look-ahead.
    pub fn flush(&mut self, out: &mut Vec<f32>) {
        for _ in 1..self.window {
            self.step(0.0, out);
        }
    }

    /// Strongest gain reduction so far in dB (0 or negative).
    pub fn max_reduction_db(&self) -> f64 {
        20.0 * self.min_gain.log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f64, amp: f64, secs: f64, sr: u32) -> Vec<f32> {
        let n = (secs * sr as f64) as usize;
        (0..n)
            .map(|i| {
                (amp * (2.0 * std::f64::consts::PI * freq * i as f64 / sr as f64).sin()) as f32
            })
            .collect()
    }

    fn loudness(samples: &[f32], sr: u32) -> Option<f64> {
        let mut m = LoudnessMeter::new(sr);
        m.process(samples);
        m.integrated_lufs()
    }

    #[test]
    fn k_weighting_matches_bs1770_at_48k() {
        let [shelf, hp] = k_weighting(48000.0);
        let expected = [
            (shelf.b[0], 1.53512485958697),
            (shelf.b[1], -2.69169618940638),
            (shelf.b[2], 1.19839281085285),
            (shelf.a[0], -1.69065929318241),
            (shelf.a[1], 0.73248077421585),
            (hp.a[0], -1.99004745483398),
            (hp.a[1], 0.99007225036621),
        ];
        for (got, want) in expected {
            assert!((got - want).abs() < 1e-8, "{got} vs {want}");
        }
    }

    #[test]
    fn full_scale_1k_sine_is_minus_3_lufs() {
        // BS.1770: a 0 dBFS 997 Hz sine reads -3.01 LUFS.
        for sr in [48000, 24000] {
            let l = loudness(&sine(997.0, 1.0, 5.0, sr), sr).unwrap();
            assert!((l + 3.01).abs() < 0.1, "{sr} Hz: {l}");
        }
        let l = loudness(&sine(997.0, 0.1, 5.0, 24000), 24000).unwrap();
        assert!((l + 23.01).abs() < 0.1, "{l}");
    }

    #[test]
    fn pauses_are_gated_out() {
        let sr = 24000;
        let tone = sine(997.0, 0.1, 3.0, sr);
        let mut with_pauses = tone.clone();
        with_pauses.extend(vec![0.0; 3 * sr as usize]);
        with_pauses.extend(&tone);
        let a = loudness(&tone, sr).unwrap();
        let b = loudness(&with_pauses, sr).unwrap();
        // Only the blocks across the edges count a little; averaging the
        // pause in would lower the level by 1.8 dB.
        assert!((a - b).abs() < 0.4, "{a} vs {b}");
        assert_eq!(loudness(&vec![0.0; sr as usize * 3], sr), None);
        assert_eq!(loudness(&[0.5; 100], sr), None);
    }

    #[test]
    fn blocks_of_any_size_give_the_same_result() {
        let sr = 24000;
        let mut x = sine(997.0, 0.1, 2.0, sr);
        x.extend(sine(300.0, 0.3, 2.0, sr));
        let mut m = LoudnessMeter::new(sr);
        for block in x.chunks(777) {
            m.process(block);
        }
        assert_eq!(m.integrated_lufs(), loudness(&x, sr));
    }

    #[test]
    fn gain_is_capped() {
        assert!((normalization_gain_db(-24.0, -16.0) - 8.0).abs() < 1e-12);
        assert!((normalization_gain_db(-12.0, -16.0) + 4.0).abs() < 1e-12);
        assert_eq!(normalization_gain_db(-60.0, -16.0), MAX_GAIN_DB);
    }

    fn limit(input: &[f32], sr: u32) -> (Vec<f32>, Limiter) {
        let mut lim = Limiter::new(PEAK_CEILING_DBFS, sr);
        let mut out = Vec::new();
        // Uneven blocks, like the mixer's flushes.
        for block in input.chunks(1000) {
            lim.process(block, &mut out);
        }
        lim.flush(&mut out);
        (out, lim)
    }

    #[test]
    fn limiter_keeps_quiet_audio_unchanged_and_aligned() {
        let sr = 24000;
        let x = sine(440.0, 0.5, 1.0, sr);
        let (y, lim) = limit(&x, sr);
        assert_eq!(y, x);
        assert_eq!(lim.max_reduction_db(), 0.0);
    }

    #[test]
    fn limiter_holds_the_ceiling() {
        let sr = 24000;
        let ceiling = db_to_gain(PEAK_CEILING_DBFS) as f32;
        // Quiet tone, then a loud burst and single spikes.
        let mut x = sine(220.0, 0.3, 2.0, sr);
        for (i, s) in x.iter_mut().enumerate() {
            if (30000..32000).contains(&i) {
                *s *= 4.0;
            }
            if i > 36000 && i % 5000 == 77 {
                *s = 1.9;
            }
        }
        let (y, lim) = limit(&x, sr);
        assert_eq!(y.len(), x.len());
        let peak = y.iter().fold(0f32, |m, s| m.max(s.abs()));
        assert!(peak <= ceiling + 1e-6, "peak {peak}");
        assert!(lim.max_reduction_db() < -6.0);
        // Before the look-ahead reaches the burst the signal is untouched.
        assert_eq!(y[..29800], x[..29800]);
        // The burst is turned down, not clipped: it keeps its shape.
        let scale = y[31000] / x[31000];
        for i in 30900..31100 {
            if x[i].abs() > 0.1 {
                assert!((y[i] / x[i] - scale).abs() < 0.05, "sample {i}");
            }
        }
    }
}
