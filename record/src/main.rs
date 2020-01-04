mod audio_input;
mod common;
mod fft;

use std::ops::Deref;
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // for every sound input, cut off 256 samples and take an fft
    // print the fft as a single line of json

    let latest_raw_audio: Arc<Mutex<Vec<f32>>> = Default::default();
    audio_input::start_sound_thread(latest_raw_audio.clone());

    let mut raw_audio_buf: Box<[f32; 256]> = Box::new([0.0; 256]);
    let mut freqs_buf = vec![0.0_f32; 256];

    loop {
        std::thread::sleep(Duration::from_millis(50));
        {
            let guard = latest_raw_audio.lock().unwrap().clone();
            if guard.len() >= 256 {
                raw_audio_buf.copy_from_slice(&guard[..256]);
            }
        };
        fft::fft(raw_audio_buf.deref(), &mut freqs_buf);
        // serde_json::to_writer(std::io::stdout(), &freqs_buf).unwrap();
        // println!("");
    }

    Ok(())
}
