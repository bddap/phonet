use super::common::FFT_BINS;
use super::fft::fft;
use audrey::read::BufFileReader;
use serde::{Deserialize, Serialize};
use serde_big_array::big_array;
use std::fs::{read_dir, DirEntry};
use std::path::{Path, PathBuf};

big_array! { BigArray; }

/// Input
#[derive(Clone, Serialize, Deserialize)]
pub struct Fft(#[serde(with = "BigArray")] pub [f32; FFT_BINS]);

pub struct TrainingData {
	pub classes: Vec<PathBuf>,
	pub training_pairs: Vec<TrainingOutput>,
}

#[derive(Serialize, Deserialize)]
pub struct TrainingOutput(Fft, usize);

impl TrainingData {
	pub fn load(training_audio_path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
		let classes: Result<Vec<DirEntry>, std::io::Error> =
			read_dir(training_audio_path)?.collect();
		let mut classes: Vec<PathBuf> = classes?.iter().map(|de| de.path()).collect();
		classes.sort_by(|pba, pbb| {
			let fna = pba.as_path().file_name().unwrap();
			let sa = fna.to_str().unwrap().to_lowercase();
			let fnb = pbb.as_path().file_name().unwrap();
			let sb = fnb.to_str().unwrap().to_lowercase();
			sa.cmp(&sb)
		});
		let mut training_pairs: Vec<TrainingOutput> = Vec::new();
		for (class, path) in classes.iter().enumerate() {
			for freqs in load_all_in_dir(path)? {
				training_pairs.push(TrainingOutput(freqs, class));
			}
		}
		Ok(TrainingData {
			classes,
			training_pairs,
		})
	}
}

fn load_all_in_dir(
	audio_sample_dir: impl AsRef<Path>,
) -> Result<Vec<Fft>, Box<dyn std::error::Error>> {
	let mut ret = Vec::new();
	for audio_file in read_dir(audio_sample_dir)? {
		let audio_file = audio_file?.path();
		load_ffts(audio_file, &mut ret)?;
	}
	Ok(ret)
}

fn load_ffts(
	audio_sample_path: impl AsRef<Path>,
	out: &mut Vec<Fft>,
) -> Result<(), Box<dyn std::error::Error>> {
	// do we resample before fft or after? Do we resample at all?
	let audio_reader = BufFileReader::open(&audio_sample_path)?;
	let raw = load_raw_samples(audio_reader)?;
	for fft in raw.windows(FFT_BINS).step_by(1).map(|raw| {
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
