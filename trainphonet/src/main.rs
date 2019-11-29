mod audio;
mod common;
mod fft;
mod ml_model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args();
    let _ = args.next();
    let passes: usize = args.next().unwrap_or("100".into()).parse().unwrap();
    dbg!(passes);
    let hidden_width = args.next().unwrap_or("10".into()).parse().unwrap();
    dbg!(hidden_width);
    let hidden_height = args.next().unwrap_or("256".into()).parse().unwrap();
    dbg!(hidden_height);

    let training_data = audio::TrainingData::load("../ipa")?;
    println!("{}", training_data.close_front_unrounded_vowel.len());
    println!("{}", training_data.open_back_rounded_vowel.len());

    ml_model::Model::create_initial((hidden_width, hidden_height)).train(&training_data, passes);

    Ok(())
}
