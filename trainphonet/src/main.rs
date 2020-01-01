// mod audio;
// mod fft;
// mod ml_model;
mod common;
mod graph_init;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args();
    let _ = args.next();
    let passes: usize = args.next().unwrap_or("100".into()).parse().unwrap();
    let hidden_width: usize = args.next().unwrap_or("10".into()).parse().unwrap();
    let hidden_height: usize = args.next().unwrap_or("256".into()).parse().unwrap();
    dbg!(passes, hidden_width, hidden_height);

    // let training_data = audio::TrainingData::load("../ipa")?;
    // dbg!(training_data.close_front_unrounded_vowel.len());
    // dbg!(training_data.open_back_rounded_vowel.len());

    // ml_model::Model::create_initial((hidden_width, hidden_height)).train(&training_data, passes);

    Ok(())
}
