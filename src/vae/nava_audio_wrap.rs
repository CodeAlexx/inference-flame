//! NAVA audio-VAE wrapper.
//!
//! NAVA's audio path produces a 128-channel latent stream `[B, 128, L]`. The
//! underlying decoder is the LTX-2.3 audio VAE (already ported in
//! [`crate::vae::ltx2_audio_vae`] + [`crate::vae::ltx2_vocoder`]). This module
//! is the thin glue NAVA needs around them — it mirrors
//! `nava_src/vae/local_audio_vae.py::LtxAudioVAE.wrapped_decode`
//! (lines 76-94) and `LocalAudioVAEAdapter.decode` (lines 180-209):
//!
//!   1. `[B, 128, L] -> [B, L, 128]`        (transpose, py:79)
//!   2. unpatchify `128 -> [8 latent_ch, 16 mel_bins]` (py:80-86 via
//!      `AudioPatchifier.unpatchify`, `components/patchifiers.py:303-327`).
//!   3. decode `[B, 8, L, 16] -> mel`        (py:87, `LTX2AudioVaeDecoder::decode`)
//!   4. vocoder+BWE `mel -> waveform @ 48 kHz` (py:88, `LTX2VocoderWithBWE::forward`)
//!   5. resample 48 kHz -> 16 kHz            (py:89-93, NAVA target = 16000)
//!
//! Reuses the existing decoder, vocoder, and `crate::audio::resample` — it does
//! NOT reimplement the VAE or vocoder.

use anyhow::{anyhow, Result};
use flame_core::{DType, Shape, Tensor};

use crate::audio::resample::resample_to;
use crate::vae::{LTX2AudioVaeDecoder, LTX2VocoderWithBWE};

/// Number of latent channels the LTX audio decoder consumes (`z_channels`).
/// Mirrors `local_audio_vae.py:82` (`channels=self.decoder.z_channels`) and the
/// Rust decoder doc (`ltx2_audio_vae.rs:539`, input `[B, 8, T, F]`).
pub const LATENT_CHANNELS: usize = 8;

/// Number of mel bins per latent channel. Computed in Python as
/// `latents.shape[2] // z_channels = 128 // 8 = 16` (`local_audio_vae.py:84`).
pub const MEL_BINS: usize = 16;

/// Packed audio-latent channel count = `LATENT_CHANNELS * MEL_BINS`.
pub const PACKED_CHANNELS: usize = LATENT_CHANNELS * MEL_BINS; // 128

/// NAVA's target waveform sample rate (`local_audio_vae.py:22`).
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Unpatchify a packed audio-latent `[B, L, 128]` into the VAE-decoder layout
/// `[B, 8, L, 16]`.
///
/// **Axis order (skeptic-bait, line-cited):** the Python patchifier flattens
/// with `einops.rearrange(x, "b c t f -> b t (c f)")` (`patchifiers.py:296-299`)
/// and unpatchifies with `"b t (c f) -> b c t f", c=channels, f=mel_bins`
/// (`patchifiers.py:320-325`). In `(c f)` the **channel axis is the OUTER /
/// slower-varying** index and the **mel axis is the INNER / faster-varying**
/// index. So the packed 128 splits as `[c=8 outer, f=16 inner]`, NOT
/// `[mel outer, channel inner]`. Reshaping `[B, L, 128] -> [B, L, 8, 16]`
/// reproduces exactly that contiguous `(c, f)` decomposition; we then move the
/// channel axis ahead of time with `permute([0, 2, 1, 3])`.
///
/// Takes `[B, L, 128]` (already transposed from the incoming `[B, 128, L]`).
fn unpatchify_audio_latents(latents_blc: &Tensor) -> Result<Tensor> {
    let dims = latents_blc.dims();
    if dims.len() != 3 {
        return Err(anyhow!(
            "nava_audio_wrap: expected [B, L, 128], got {:?}",
            dims
        ));
    }
    let (b, l, packed) = (dims[0], dims[1], dims[2]);
    if packed != PACKED_CHANNELS {
        return Err(anyhow!(
            "nava_audio_wrap: packed channel dim = {}, expected {} (= {} latent_ch * {} mel_bins)",
            packed,
            PACKED_CHANNELS,
            LATENT_CHANNELS,
            MEL_BINS
        ));
    }

    // "b t (c f) -> b t c f": split the packed dim, channel OUTER, mel INNER.
    // `reshape` materializes if the input is a non-contiguous view, so no
    // explicit `.contiguous()` is needed before it.
    let split = latents_blc.reshape(&[b, l, LATENT_CHANNELS, MEL_BINS])?;
    // "b t c f -> b c t f": bring the channel axis to position 1.
    // `.contiguous()` IS required here: the permute leaves a strided view, and
    // the downstream decoder conv path (CausalConv2d) strides its input
    // naively — a non-contiguous tensor would read scrambled elements
    // (the cat-not-contig fingerprint). This is the one legitimate
    // materialization in the wrapper.
    let unpatched = split.permute(&[0, 2, 1, 3])?.contiguous()?;
    Ok(unpatched)
}

/// Decode NAVA audio latents to a 16 kHz waveform.
///
/// `audio_latents`: `[B, 128, L]` (BF16). Returns the waveform tensor at
/// `TARGET_SAMPLE_RATE` (16 kHz), shape `[B, n_ch, T_16k]`.
///
/// Mirrors `local_audio_vae.py::wrapped_decode` (lines 76-94). Reuses
/// `LTX2AudioVaeDecoder::decode` + `LTX2VocoderWithBWE::forward` +
/// `crate::audio::resample::resample_to`.
pub fn nava_decode_audio(
    audio_latents: &Tensor,
    decoder: &LTX2AudioVaeDecoder,
    vocoder: &LTX2VocoderWithBWE,
) -> Result<Tensor> {
    let in_dims = audio_latents.dims();
    if in_dims.len() != 3 || in_dims[1] != PACKED_CHANNELS {
        return Err(anyhow!(
            "nava_decode_audio: expected [B, {}, L], got {:?}",
            PACKED_CHANNELS,
            in_dims
        ));
    }

    // The decoder + vocoder run in BF16 (inference path). Match the storage
    // dtype going in.
    let latents = audio_latents.to_dtype(DType::BF16)?;

    // 1. [B, 128, L] -> [B, L, 128]  (py:79 `latents.transpose(1, 2)`).
    let latents_blc = latents.transpose_dims(1, 2)?;

    // 2. unpatchify 128 -> [8 latent_ch, 16 mel_bins] -> [B, 8, L, 16] (py:80-86).
    let latents_bctf = unpatchify_audio_latents(&latents_blc)?;

    // 3. decode [B, 8, L, 16] -> mel spectrogram (py:87).
    let mel = decoder
        .decode(&latents_bctf)
        .map_err(|e| anyhow!("nava_decode_audio: LTX2 audio decode: {e}"))?;

    // 4. vocoder + BWE -> waveform at the vocoder's output SR (48 kHz) (py:88).
    let waveform = vocoder
        .forward(&mel)
        .map_err(|e| anyhow!("nava_decode_audio: vocoder+BWE: {e}"))?;
    let vocoder_sr = vocoder.output_sample_rate();

    // 5. resample vocoder_sr -> 16 kHz (py:89-93). `resample_to` works on CPU
    //    mono f32 slices, so move to host, resample each channel, rebuild.
    if vocoder_sr == TARGET_SAMPLE_RATE {
        return Ok(waveform);
    }

    let wf_dims = waveform.dims().to_vec();
    if wf_dims.len() != 3 {
        return Err(anyhow!(
            "nava_decode_audio: expected vocoder waveform [B, C, T], got {:?}",
            wf_dims
        ));
    }
    let (b, c, t) = (wf_dims[0], wf_dims[1], wf_dims[2]);
    let host = waveform.to_dtype(DType::F32)?.to_vec_f32()?;

    // Resample each (batch, channel) row independently, then reassemble in the
    // same [B, C, T'] row-major layout.
    let mut out_rows: Vec<Vec<f32>> = Vec::with_capacity(b * c);
    let mut out_len: Option<usize> = None;
    for row in 0..(b * c) {
        let start = row * t;
        let chan = &host[start..start + t];
        let resampled = resample_to(chan, vocoder_sr, TARGET_SAMPLE_RATE)
            .map_err(|e| anyhow!("nava_decode_audio: resample {vocoder_sr}->{TARGET_SAMPLE_RATE}: {e}"))?;
        match out_len {
            None => out_len = Some(resampled.len()),
            Some(prev) if prev != resampled.len() => {
                return Err(anyhow!(
                    "nava_decode_audio: resampled rows differ in length ({} vs {})",
                    prev,
                    resampled.len()
                ));
            }
            _ => {}
        }
        out_rows.push(resampled);
    }
    let t_out = out_len.unwrap_or(0);
    let mut flat = Vec::with_capacity(b * c * t_out);
    for row in out_rows {
        flat.extend_from_slice(&row);
    }

    let out = Tensor::from_vec(
        flat,
        Shape::from_dims(&[b, c, t_out]),
        audio_latents.device().clone(),
    )?
    .to_dtype(DType::BF16)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shape + axis-order contract for the 128 -> [8, 16] unpatchify.
    ///
    /// Builds a synthetic `[B, L, 128]` where each element is set to its
    /// `(c, f)` source index packed as `c * 100 + f`, exactly as the Python
    /// `(c f)` flatten lays it out (channel outer, mel inner). After
    /// `unpatchify_audio_latents` we expect `[B, 8, L, 16]` with
    /// `out[b, c, l, f] == c * 100 + f`, confirming the channel-major split.
    #[test]
    fn unpatchify_shape_and_axis_order() -> Result<()> {
        let device = match flame_core::CudaDevice::new(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skip: no CUDA device available for unpatchify test");
                return Ok(());
            }
        };

        let b = 1usize;
        let l = 3usize;
        // Packed layout: index 128 = c * 16 + f  (c outer, f inner).
        // Store a recognizable value c*100 + f at each packed slot.
        let mut data = Vec::with_capacity(b * l * PACKED_CHANNELS);
        for _bi in 0..b {
            for _li in 0..l {
                for c in 0..LATENT_CHANNELS {
                    for f in 0..MEL_BINS {
                        data.push((c * 100 + f) as f32);
                    }
                }
            }
        }
        let packed = Tensor::from_vec(
            data,
            Shape::from_dims(&[b, l, PACKED_CHANNELS]),
            device.clone(),
        )?;

        let unpatched = unpatchify_audio_latents(&packed)?;
        assert_eq!(
            unpatched.dims(),
            &[b, LATENT_CHANNELS, l, MEL_BINS],
            "unpatchify must yield [B, 8, L, 16]"
        );

        let host = unpatched.to_dtype(DType::F32)?.to_vec_f32()?;
        // Row-major [B, C, L, F] index.
        let idx = |bi: usize, c: usize, li: usize, f: usize| {
            ((bi * LATENT_CHANNELS + c) * l + li) * MEL_BINS + f
        };
        // Spot-check a few positions: value must equal c*100 + f for ALL l,
        // proving channel is the outer axis of the 128 split and mel the inner.
        for &(c, f) in &[(0usize, 0usize), (0, 15), (1, 0), (7, 15), (3, 9)] {
            for li in 0..l {
                let got = host[idx(0, c, li, f)];
                let want = (c * 100 + f) as f32;
                assert_eq!(
                    got, want,
                    "axis order wrong at (c={c}, l={li}, f={f}): got {got}, want {want}"
                );
            }
        }
        Ok(())
    }
}
