mod audio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    audio::load_training_data("../ipa").map(|_| ())
}
