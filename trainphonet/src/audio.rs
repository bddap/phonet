use super::fft::fft;
use audrey::read::BufFileReader;
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

/// metadata about a specific sound
#[derive(Serialize, Deserialize)]
struct Phone {
    ipa_symbol: String,
    name: String,
    wav: PathBuf,
    mp3: PathBuf,
    ogg: PathBuf,
}

fn load_ffts(phone: &Phone) -> Result<Vec<Fft>, Box<dyn std::error::Error>> {
    // do we resample before fft or after? Do we resample at all?
    let audio_reader = BufFileReader::open(&phone.ogg)?;
    let raw = load_raw_samples(audio_reader)?;
    let ret: Vec<Fft> = raw
        .windows(FFT_BINS)
        .step_by(FFT_BINS / 2)
        .map(|raw| {
            let mut ret = Fft([0.0; FFT_BINS]);
            fft(raw, &mut ret.0);
            ret
        })
        .collect();
    Ok(ret)
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
