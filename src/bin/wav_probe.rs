//! CPU-only WAV probe: verify a real-world WAV parses through our
//! `audio::wav::probe_duration` and that load_wav_f32 gets samples in
//! [-1, 1]. No GPU, no encoder — used to sanity-check unusual WAVs
//! (8-bit, weird rates) before launching a heavy GPU job.

use anyhow::Result;
use std::path::PathBuf;

fn main() -> Result<()> {
    let path = std::env::args().nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: wav_probe <path/to.wav>"))?;
    let (dur, rate, ch) = inference_flame::audio::wav::probe_duration(&path)?;
    println!("probe: {:.3}s, {} Hz, {} ch", dur, rate, ch);
    let (samples, _, _) = inference_flame::audio::wav::load_wav_f32(&path)?;
    let n = samples.len();
    let absmax = samples.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let mean = samples.iter().sum::<f32>() / n.max(1) as f32;
    println!("samples: count={n}  |max|={absmax:.4}  mean={mean:.4}");
    let derived_num_frames = (dur * 25.0).round() as usize;
    println!("would set num_frames = {} for talking-head (dur * 25)", derived_num_frames);
    Ok(())
}
