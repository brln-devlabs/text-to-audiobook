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

//! Audio I/O and processing: WAV files, filters, assembly, quality audit.

pub mod assembler;
pub mod cleanup;
pub mod filter;
pub mod mfcc;
pub mod tags;

use std::io::{BufReader, Cursor, Read, Seek};
use std::path::Path;

use anyhow::{Context, Result, bail};

/// Decoded audio: interleaved samples in [-1, 1].
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub channels: u16,
    pub sample_rate: u32,
}

impl Audio {
    pub fn mono(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            channels: 1,
            sample_rate,
        }
    }

    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels.max(1) as usize
    }

    /// Mix down to one channel (mean of all channels).
    pub fn to_mono(&self) -> Vec<f32> {
        let ch = self.channels.max(1) as usize;
        if ch == 1 {
            return self.samples.clone();
        }
        self.samples
            .chunks_exact(ch)
            .map(|f| f.iter().sum::<f32>() / ch as f32)
            .collect()
    }

    pub fn duration_secs(&self) -> f64 {
        self.frames() as f64 / self.sample_rate.max(1) as f64
    }
}

fn decode<R: Read>(reader: hound::WavReader<R>) -> Result<Audio> {
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.into_samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample.max(1) - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    Ok(Audio {
        samples,
        channels: spec.channels,
        sample_rate: spec.sample_rate,
    })
}

/// Size of a leading ID3v2 tag (WAVs tagged by the Python version start with one).
fn id3v2_prefix_len(head: &[u8]) -> Option<usize> {
    if head.len() < 10 || &head[..3] != b"ID3" {
        return None;
    }
    let size = head[6..10]
        .iter()
        .fold(0usize, |acc, b| (acc << 7) | (*b as usize & 0x7f));
    let footer = if head[5] & 0x10 != 0 { 10 } else { 0 };
    Some(10 + size + footer)
}

/// Read a WAV file (PCM 8/16/24/32-bit or 32-bit float).
pub fn read_wav(path: &Path) -> Result<Audio> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut head = [0u8; 10];
    let n = reader.read(&mut head)?;
    reader.rewind()?;
    if let Some(skip) = id3v2_prefix_len(&head[..n]) {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        if skip >= bytes.len() {
            bail!("{}: only an ID3 tag, no audio", path.display());
        }
        let wav = hound::WavReader::new(Cursor::new(bytes[skip..].to_vec()))
            .with_context(|| format!("reading {}", path.display()))?;
        return decode(wav);
    }
    let wav =
        hound::WavReader::new(reader).with_context(|| format!("reading {}", path.display()))?;
    decode(wav)
}

/// Convert a sample in [-1, 1] to 16-bit PCM (clipped).
pub fn to_i16(x: f32) -> i16 {
    (x * 32767.0).round().clamp(-32768.0, 32767.0) as i16
}

/// Write interleaved samples as 16-bit PCM WAV.
pub fn write_wav_pcm16(
    path: &Path,
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let spec = hound::WavSpec {
        channels: channels.max(1),
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)
        .with_context(|| format!("creating {}", path.display()))?;
    {
        let mut w16 = w.get_i16_writer(samples.len() as u32);
        for s in samples {
            w16.write_sample(to_i16(*s));
        }
        w16.flush()?;
    }
    w.finalize()?;
    Ok(())
}

/// Write mono audio as 16-bit PCM WAV.
pub fn write_mono_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    write_wav_pcm16(path, samples, 1, sample_rate)
}

/// Band-limited resampling with a Hann-windowed sinc kernel.
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() || from == 0 || to == 0 {
        return x.to_vec();
    }
    let ratio = to as f64 / from as f64;
    let out_len = ((x.len() as f64) * ratio).round() as usize;
    let fc = ratio.min(1.0);
    let half_width = 16.0 / fc;
    let last = x.len() as i64 - 1;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let t = i as f64 / ratio;
        let lo = ((t - half_width).floor() as i64).max(0);
        let hi = ((t + half_width).ceil() as i64).min(last);
        let mut acc = 0.0f64;
        let mut norm = 0.0f64;
        for k in lo..=hi {
            let d = t - k as f64;
            let u = d / half_width;
            if u.abs() >= 1.0 {
                continue;
            }
            let arg = std::f64::consts::PI * fc * d;
            let sinc = if arg.abs() < 1e-12 {
                1.0
            } else {
                arg.sin() / arg
            };
            let w = 0.5 * (1.0 + (std::f64::consts::PI * u).cos());
            let h = sinc * w;
            acc += x[k as usize] as f64 * h;
            norm += h;
        }
        out.push(if norm.abs() > 1e-12 {
            (acc / norm) as f32
        } else {
            0.0
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_roundtrip_pcm16() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.wav");
        let s: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        write_mono_wav(&p, &s, 24000).unwrap();
        let a = read_wav(&p).unwrap();
        assert_eq!(a.sample_rate, 24000);
        assert_eq!(a.channels, 1);
        assert_eq!(a.samples.len(), s.len());
        assert!(a.samples.iter().zip(&s).all(|(x, y)| (x - y).abs() < 1e-3));
    }

    #[test]
    fn reads_wav_with_leading_id3_tag() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.wav");
        write_mono_wav(&p, &[0.25; 100], 16000).unwrap();
        let wav = std::fs::read(&p).unwrap();
        // ID3v2.3 header with a 20-byte (syncsafe) body
        let mut tagged = b"ID3\x03\x00\x00\x00\x00\x00\x14".to_vec();
        tagged.extend([0u8; 20]);
        tagged.extend(wav);
        std::fs::write(&p, tagged).unwrap();
        let a = read_wav(&p).unwrap();
        assert_eq!(a.samples.len(), 100);
    }

    #[test]
    fn resample_keeps_length_and_tone() {
        let from = 48000;
        let to = 24000;
        let tone: Vec<f32> = (0..48000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / from as f32).sin())
            .collect();
        let out = resample(&tone, from, to);
        assert_eq!(out.len(), 24000);
        let expected: Vec<f32> = (0..24000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / to as f32).sin())
            .collect();
        let err: f32 = out[100..23900]
            .iter()
            .zip(&expected[100..23900])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(err < 0.02, "max error {err}");
    }

    #[test]
    fn mono_mixdown() {
        let a = Audio {
            samples: vec![1.0, 0.0, 0.5, 0.5],
            channels: 2,
            sample_rate: 8000,
        };
        assert_eq!(a.to_mono(), vec![0.5, 0.5]);
        assert_eq!(a.frames(), 2);
    }
}
