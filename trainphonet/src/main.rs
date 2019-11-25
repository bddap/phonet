mod audio;
mod fft;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let training_data = audio::load_training_data("../ipa")?;
	println!("{}", training_data.close_front_unrounded_vowel.len());
	println!("{}", training_data.open_back_rounded_vowel.len());
	Ok(())
}
