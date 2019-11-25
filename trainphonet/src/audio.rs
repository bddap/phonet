use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::{Path, PathBuf};

const FFT_BINS: usize = 256;

pub struct Fft([f32; FFT_BINS]);

pub struct TrainingData {
    pub close_front_unrounded_vowel: Vec<Fft>, // "i"
    pub open_back_rounded_vowel: Vec<Fft>,     // "ɒ"
}

/// ipa_path is a path to the directory containing sounds.json and the audio directory
pub fn load_training_data(
    ipa_path: impl AsRef<Path>,
) -> Result<TrainingData, Box<dyn std::error::Error>> {
    let phones: Vec<Phone> = load_phones(ipa_path.as_ref())?;
    let close_front_unrounded_vowel =
        load_ffts(get_phone_by_name(&phones, "Close_front_unrounded_vowel")?)?;
    let open_back_rounded_vowel =
        load_ffts(get_phone_by_name(&phones, "Open_back_rounded_vowel")?)?;
    Ok(TrainingData {
        close_front_unrounded_vowel,
        open_back_rounded_vowel,
    })
}

fn get_phone_by_name<'a>(
    phones: &'a [Phone],
    name: &str,
) -> Result<&'a Phone, Box<dyn std::error::Error>> {
    phones
        .iter()
        .filter(|phone| &phone.name == name)
        .next()
        .ok_or("Couldn't find phone".into())
}

/// metadata baout a specific sound
#[derive(Serialize, Deserialize)]
struct Phone {
    ipa_symbol: String,
    name: String,
    wav: PathBuf,
    mp3: PathBuf,
    ogg: PathBuf,
}

fn load_ffts(phone: &Phone) -> Result<Vec<Fft>, Box<dyn std::error::Error>> {
    Ok(vec![])
}

fn load_phones(ipa_path: &Path) -> Result<Vec<Phone>, Box<dyn std::error::Error>> {
    let file = File::open(ipa_path.join("sounds.json"))?;
    let mut phones: Vec<Phone> = serde_json::from_reader(file)?;
    for mut phone in &mut phones {
        phone.wav = ipa_path.join(&phone.wav);
        phone.mp3 = ipa_path.join(&phone.mp3);
        phone.ogg = ipa_path.join(&phone.ogg);
    }
    Ok(phones)
}
