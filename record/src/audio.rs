use super::common::FFT_BINS;
use super::fft::fft;
use audrey::read::BufFileReader;
use serde::{Deserialize, Serialize};
use serde_big_array::big_array;
use std::path::Path;

big_array! { BigArray; }

/// Input
#[derive(Clone, Serialize, Deserialize)]
pub struct Fft(#[serde(with = "BigArray")] pub [f32; FFT_BINS]);

pub fn load_ffts(
    audio_sample_path: impl AsRef<Path>,
    out: &mut Vec<Fft>,
) -> Result<(), Box<dyn std::error::Error>> {
    // do we resample before fft or after? Do we resample at all?
    let audio_reader = BufFileReader::open(&audio_sample_path)?;
    let raw = load_raw_samples(audio_reader)?;
    for fft in raw.windows(FFT_BINS).step_by(FFT_BINS / 2).map(|raw| {
        let mut ret = Fft([0.0; FFT_BINS]);
        fft(raw, &mut ret.0);
        ret
    }) {
        out.push(fft)
    }
    Ok(())
}

fn load_raw_samples(mut reader: BufFileReader) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // multi-channel streams are interleaved, let's take the average in the multi-channel case
    let channels = reader.description().channel_count() as usize;
    assert_ne!(channels, 0);
    let multi_channel: Vec<f32> = reader.samples().collect::<Result<_, _>>()?;
    let single_channel: Vec<f32> = multi_channel.chunks(channels).map(ave).collect();

    // for now, we don't resample based on sample_rate
    Ok(single_channel)
}

fn ave(ns: &[f32]) -> f32 {
    let sum: f32 = Iterator::sum(ns.iter());
    sum / (ns.len() as f32)
}
