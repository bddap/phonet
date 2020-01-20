mod audio;
mod common;
mod fft;

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let training_data = audio::TrainingData::load("../training_audio")?;
	let training_data = training_data.training_pairs;
	serde_json::to_writer_pretty(std::io::stdout(), &training_data)?;
	Ok(())
}
