mod audio;
mod common;
mod fft;

use serde::{Deserialize, Serialize};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let training_data = audio::TrainingData::load("../ipa")?;
    let training_data = training_data.input_output_pairs();
    let training_data: Vec<TrainingOutput> = training_data.iter().map(|a| a.into()).collect();
    serde_json::to_writer_pretty(std::io::stdout(), &training_data)?;
    Ok(())
}

#[derive(Serialize, Deserialize)]
struct TrainingOutput {
    freqs: audio::Fft,
    class: usize,
}

impl From<&(&audio::Fft, audio::Classification)> for TrainingOutput {
    fn from(other: &(&audio::Fft, audio::Classification)) -> Self {
        TrainingOutput {
            freqs: other.0.clone(),
            class: other.1 as usize,
        }
    }
}
