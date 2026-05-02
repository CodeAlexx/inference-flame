//! Stable Audio Open Oobleck VAE — pure-Rust decoder for MagiHuman audio path.
//!
//! Decode-only port of the Oobleck VAE used by `stabilityai/stable-audio-open-1.0`,
//! configured to match the MagiHuman audio output stage. Maps a
//! `[B, latent_dim=64, T_lat]` F32 latent into `[B, 2, T_lat * 2048]` F32 stereo
//! waveform (no final tanh; values typically in `[-1, 1]`).
//!
//! Architecture (from `inference/model/sa_audio/sa_audio_module.py` plus the
//! shipped `model_config.json` pretransform.config):
//!     channels=128, latent_dim=64, c_mults=[1,2,4,8,16], strides=[2,4,4,8,8]
//!     use_snake=True, final_tanh=False
//!     downsampling_ratio=2048 (=2*4*4*8*8), io_channels=2, sample_rate=44100
//!
//! Decoder layer chain (after `[1] + c_mults` prepend → c_mults=[1,1,2,4,8,16]):
//!     1. WNConv1d(latent_dim=64 → 2048, k=7, pad=3)            (init projection)
//!     2. DecoderBlock(2048 → 1024, stride=8)                    (× 8 upsample)
//!     3. DecoderBlock(1024 →  512, stride=8)
//!     4. DecoderBlock( 512 →  256, stride=4)
//!     5. DecoderBlock( 256 →  128, stride=4)
//!     6. DecoderBlock( 128 →  128, stride=2)
//!     7. SnakeBeta(128)
//!     8. WNConv1d(128 → 2, k=7, pad=3, bias=False)              (final projection)
//!
//! Each `DecoderBlock` is:
//!     SnakeBeta(in_ch)
//!     → ConvTranspose1d(in_ch, out_ch, k=2*stride, stride=stride, pad=ceil(stride/2))
//!     → ResidualUnit(out_ch, dilation=1)
//!     → ResidualUnit(out_ch, dilation=3)
//!     → ResidualUnit(out_ch, dilation=9)
//!
//! Each `ResidualUnit(C, D)` is:
//!     SnakeBeta(C) → Conv1d(C, C, k=7, dilation=D, pad=3*D)
//!     → SnakeBeta(C) → Conv1d(C, C, k=1)
//!     → residual add input
//!
//! Weight format: PyTorch `weight_norm` parametrization (`weight_g` + `weight_v`)
//! is reconstructed at convert time (`scripts/convert_stable_audio_vae_to_safetensors.py`)
//! into a single `weight` tensor per conv. So the Rust loader only sees plain
//! `.weight` / `.bias` keys — no runtime weight_norm reconstruction needed.

use flame_core::conv1d::{conv1d, conv_transpose1d_dilated};
use flame_core::serialization::load_file;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

type Weights = HashMap<String, Tensor>;

fn get(weights: &Weights, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::InvalidOperation(format!("OobleckVAE: missing weight: {key}")))
}

// ===========================================================================
// SnakeBeta activation
// ===========================================================================
//
// Python:
//     alpha = self.alpha[None, :, None]  # → [1, C, 1]
//     beta  = self.beta[None, :, None]
//     if alpha_logscale: alpha = exp(alpha); beta = exp(beta)
//     return x + (1 / (beta + 1e-9)) * sin(x * alpha)^2
//
// `alpha_logscale=True` per default — both alpha and beta come from the
// checkpoint already in log space (they're trained as `log(alpha)`).

struct SnakeBeta {
    alpha: Tensor, // [1, C, 1] F32 (log-domain raw values)
    beta: Tensor,  // [1, C, 1] F32 (log-domain raw values)
}

impl SnakeBeta {
    fn load(weights: &Weights, prefix: &str, channels: usize) -> Result<Self> {
        let alpha = get(weights, &format!("{prefix}.alpha"))?
            .to_dtype(DType::F32)?
            .reshape(&[1, channels, 1])?;
        let beta = get(weights, &format!("{prefix}.beta"))?
            .to_dtype(DType::F32)?
            .reshape(&[1, channels, 1])?;
        Ok(Self { alpha, beta })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Operate in F32 for numerical stability (sin/exp).
        let x_f32 = x.to_dtype(DType::F32)?;
        // exp(log_alpha), exp(log_beta) — both broadcast as [1, C, 1].
        let alpha = self.alpha.exp()?;
        let beta = self.beta.exp()?;
        // sin(x * alpha) ^ 2
        let xa = x_f32.mul(&alpha)?;
        let s = xa.sin()?;
        let s2 = s.mul(&s)?;
        // (beta + 1e-9), used as denominator
        let beta_eps = beta.add_scalar(1e-9)?;
        // out = x + s^2 / (beta + 1e-9)
        let scaled = s2.div(&beta_eps)?;
        let out = x_f32.add(&scaled)?;
        if x.dtype() == DType::F32 {
            Ok(out)
        } else {
            out.to_dtype(x.dtype())
        }
    }
}

// ===========================================================================
// Plain Conv1d / ConvTranspose1d wrappers that keep weights resident
// ===========================================================================

struct Conv1dLayer {
    weight: Tensor,        // [C_out, C_in, K] F32
    bias: Option<Tensor>,  // [C_out] F32 or None
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Conv1dLayer {
    fn load(
        weights: &Weights,
        prefix: &str,
        c_in: usize,
        c_out: usize,
        k: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        with_bias: bool,
    ) -> Result<Self> {
        let weight = get(weights, &format!("{prefix}.weight"))?.to_dtype(DType::F32)?;
        let w_dims = weight.shape().dims();
        if w_dims != [c_out, c_in, k] {
            return Err(Error::InvalidOperation(format!(
                "OobleckVAE Conv1d {prefix}: expected shape [{c_out}, {c_in}, {k}] got {w_dims:?}"
            )));
        }
        let bias = if with_bias {
            Some(get(weights, &format!("{prefix}.bias"))?.to_dtype(DType::F32)?)
        } else {
            None
        };
        Ok(Self { weight, bias, stride, padding, dilation })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // flame's conv1d has a k=1/stride=1/pad=0/dilation=1 fast path that
        // converts to matmul via permuted (non-contiguous) views. flame's
        // matmul silently produces wrong results when fed non-contiguous
        // operands (see `feedback_rope_fused_autograd.md` / soul.md). We
        // hand-roll the k=1 case here with explicit .contiguous() calls.
        let w_dims = self.weight.shape().dims();
        let k = w_dims[2];
        if k == 1 && self.stride == 1 && self.padding == 0 && self.dilation == 1 {
            let in_dims = x.shape().dims();
            let (b, c_in, l) = (in_dims[0], in_dims[1], in_dims[2]);
            let c_out = w_dims[0];
            // weight: [C_out, C_in, 1] → [C_out, C_in]
            let w_2d = self.weight.reshape(&[c_out, c_in])?;
            // x: [B, C_in, L] → [B, L, C_in] (contiguous!)
            let x_t = x.permute(&[0, 2, 1])?.contiguous()?;
            // [B, L, C_in] @ [C_in, C_out] = [B, L, C_out]
            let w_t = w_2d.permute(&[1, 0])?.contiguous()?;
            let out = x_t.matmul(&w_t)?;
            // [B, L, C_out] → [B, C_out, L]
            let mut result = out.permute(&[0, 2, 1])?.contiguous()?;
            if let Some(ref b_t) = self.bias {
                let b_3d = b_t.reshape(&[1, c_out, 1])?;
                result = result.add(&b_3d)?;
            }
            let _ = (b, l);
            return Ok(result);
        }
        conv1d(
            x,
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
            1,
        )
    }
}

struct ConvT1dLayer {
    weight: Tensor,        // [C_in, C_out, K] F32 (PyTorch ConvTranspose1d layout)
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl ConvT1dLayer {
    fn load(
        weights: &Weights,
        prefix: &str,
        c_in: usize,
        c_out: usize,
        k: usize,
        stride: usize,
        padding: usize,
        with_bias: bool,
    ) -> Result<Self> {
        let weight = get(weights, &format!("{prefix}.weight"))?.to_dtype(DType::F32)?;
        let w_dims = weight.shape().dims();
        if w_dims != [c_in, c_out, k] {
            return Err(Error::InvalidOperation(format!(
                "OobleckVAE ConvT1d {prefix}: expected shape [{c_in}, {c_out}, {k}] got {w_dims:?}"
            )));
        }
        let bias = if with_bias {
            Some(get(weights, &format!("{prefix}.bias"))?.to_dtype(DType::F32)?)
        } else {
            None
        };
        Ok(Self { weight, bias, stride, padding })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        conv_transpose1d_dilated(
            x,
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            0, // output_padding
            1, // dilation
            1, // groups
        )
    }
}

// ===========================================================================
// ResidualUnit
// ===========================================================================
//
// Python sequential:
//     [SnakeBeta(C), Conv1d(C, C, k=7, dilation=D, pad=3*D),
//      SnakeBeta(C), Conv1d(C, C, k=1)]
//   → (forward) layers(x) + x

struct ResidualUnit {
    snake1: SnakeBeta,
    conv1: Conv1dLayer,
    snake2: SnakeBeta,
    conv2: Conv1dLayer,
}

impl ResidualUnit {
    fn load(
        weights: &Weights,
        prefix: &str,
        channels: usize,
        dilation: usize,
    ) -> Result<Self> {
        // layers.0 = SnakeBeta, layers.1 = Conv1d, layers.2 = SnakeBeta, layers.3 = Conv1d
        let snake1 = SnakeBeta::load(weights, &format!("{prefix}.layers.0"), channels)?;
        let conv1 = Conv1dLayer::load(
            weights,
            &format!("{prefix}.layers.1"),
            channels,
            channels,
            7,
            1,
            dilation * 3, // pad = (dilation * (k-1)) / 2 = (dilation * 6) / 2 = dilation * 3
            dilation,
            true,
        )?;
        let snake2 = SnakeBeta::load(weights, &format!("{prefix}.layers.2"), channels)?;
        let conv2 = Conv1dLayer::load(
            weights,
            &format!("{prefix}.layers.3"),
            channels,
            channels,
            1,
            1,
            0,
            1,
            true,
        )?;
        Ok(Self { snake1, conv1, snake2, conv2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.snake1.forward(x)?;
        let h = self.conv1.forward(&h)?;
        let h = self.snake2.forward(&h)?;
        let h = self.conv2.forward(&h)?;
        h.add(x)
    }
}

// ===========================================================================
// DecoderBlock
// ===========================================================================
//
// Python sequential:
//     [SnakeBeta(in_ch),
//      WNConvTranspose1d(in_ch, out_ch, k=2*stride, stride=stride, pad=ceil(stride/2)),
//      ResidualUnit(out_ch, 1),
//      ResidualUnit(out_ch, 3),
//      ResidualUnit(out_ch, 9)]

struct DecoderBlock {
    snake: SnakeBeta,
    upsample: ConvT1dLayer,
    res1: ResidualUnit,
    res2: ResidualUnit,
    res3: ResidualUnit,
}

impl DecoderBlock {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        stride: usize,
    ) -> Result<Self> {
        // layers.0 = SnakeBeta(in_ch)
        let snake = SnakeBeta::load(weights, &format!("{prefix}.layers.0"), in_ch)?;
        // layers.1 = ConvTranspose1d(in_ch, out_ch, k=2*stride, stride, pad=ceil(stride/2))
        let kernel = 2 * stride;
        let padding = stride.div_ceil(2); // ceil(stride/2)
        let upsample = ConvT1dLayer::load(
            weights,
            &format!("{prefix}.layers.1"),
            in_ch,
            out_ch,
            kernel,
            stride,
            padding,
            true,
        )?;
        // layers.2 / 3 / 4 = ResidualUnit(out_ch, dilation=1/3/9)
        let res1 = ResidualUnit::load(weights, &format!("{prefix}.layers.2"), out_ch, 1)?;
        let res2 = ResidualUnit::load(weights, &format!("{prefix}.layers.3"), out_ch, 3)?;
        let res3 = ResidualUnit::load(weights, &format!("{prefix}.layers.4"), out_ch, 9)?;
        Ok(Self { snake, upsample, res1, res2, res3 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.snake.forward(x)?;
        let h = self.upsample.forward(&h)?;
        let h = self.res1.forward(&h)?;
        let h = self.res2.forward(&h)?;
        self.res3.forward(&h)
    }
}

// ===========================================================================
// OobleckDecoder
// ===========================================================================

pub struct OobleckDecoder {
    init_conv: Conv1dLayer,
    blocks: Vec<DecoderBlock>,
    final_snake: SnakeBeta,
    final_conv: Conv1dLayer,
}

impl OobleckDecoder {
    /// Build a decoder matching the shipped `stable-audio-open-1.0` config:
    /// channels=128, latent_dim=64, c_mults=[1,2,4,8,16], strides=[2,4,4,8,8],
    /// use_snake=True, final_tanh=False.
    pub fn load_default(
        path: &str,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let weights = load_file(path, device)?;
        println!(
            "[OobleckVAE] Loaded {} weight tensors from {}",
            weights.len(),
            path
        );
        Self::from_weights_default(&weights)
    }

    pub fn from_weights_default(weights: &Weights) -> Result<Self> {
        // Hardcoded config for stable-audio-open-1.0 (no need for runtime config
        // since the MagiHuman pipeline only uses this exact VAE).
        let channels: usize = 128;
        let latent_dim: usize = 64;
        let c_mults: [usize; 6] = [1, 1, 2, 4, 8, 16]; // [1] + [1,2,4,8,16]
        let strides: [usize; 5] = [2, 4, 4, 8, 8];
        let depth = c_mults.len(); // 6

        // 1. layers.0: WNConv1d(latent_dim, c_mults[-1]*channels=2048, k=7, pad=3)
        let top = c_mults[depth - 1] * channels; // 2048
        let init_conv = Conv1dLayer::load(
            weights,
            "decoder.layers.0",
            latent_dim,
            top,
            7,
            1,
            3,
            1,
            true,
        )?;

        // 2-6. layers.1..5: 5 DecoderBlocks, iterating i = depth-1 .. 1 (5..1 inclusive)
        let mut blocks = Vec::with_capacity(depth - 1);
        let mut layer_idx = 1usize;
        for i in (1..=depth - 1).rev() {
            let in_ch = c_mults[i] * channels;
            let out_ch = c_mults[i - 1] * channels;
            let stride = strides[i - 1];
            let prefix = format!("decoder.layers.{layer_idx}");
            blocks.push(DecoderBlock::load(weights, &prefix, in_ch, out_ch, stride)?);
            layer_idx += 1;
        }

        // 7. layers.6: SnakeBeta(c_mults[0]*channels=128)
        let final_ch = c_mults[0] * channels;
        let final_snake = SnakeBeta::load(weights, &format!("decoder.layers.{layer_idx}"), final_ch)?;
        layer_idx += 1;

        // 8. layers.7: WNConv1d(128, out_channels=2, k=7, pad=3, bias=False)
        let final_conv = Conv1dLayer::load(
            weights,
            &format!("decoder.layers.{layer_idx}"),
            final_ch,
            2,
            7,
            1,
            3,
            1,
            false,
        )?;

        // (final tanh is `Identity` per `final_tanh=False` in shipped config)

        Ok(Self {
            init_conv,
            blocks,
            final_snake,
            final_conv,
        })
    }

    /// Decode a `[B, 64, T_lat]` F32 latent to `[B, 2, T_lat * 2048]` F32 stereo
    /// waveform.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let z_f32 = if z.dtype() != DType::F32 {
            z.to_dtype(DType::F32)?
        } else {
            z.clone()
        };
        let mut h = self.init_conv.forward(&z_f32)?;
        for blk in &self.blocks {
            h = blk.forward(&h)?;
        }
        h = self.final_snake.forward(&h)?;
        self.final_conv.forward(&h)
    }

    /// Bisect for ResidualUnit's inner Sequential (db1.res1.layers).
    pub fn db1_resu_inner(&self, sub_idx: usize, x: &Tensor) -> Result<Tensor> {
        let r = &self.blocks[0].res1;
        match sub_idx {
            0 => r.snake1.forward(x),
            1 => r.conv1.forward(x),
            2 => r.snake2.forward(x),
            3 => r.conv2.forward(x),
            _ => Err(Error::InvalidOperation(format!("resu inner idx {sub_idx} oor"))),
        }
    }

    /// Run one of the inner DecoderBlocks' sublayers individually for
    /// targeted bisect against Python `oobleck_db1_bisect.safetensors`.
    pub fn db1_sub(&self, sub_idx: usize, x: &Tensor) -> Result<Tensor> {
        let blk = &self.blocks[0];
        match sub_idx {
            0 => blk.snake.forward(x),
            1 => blk.upsample.forward(x),
            2 => blk.res1.forward(x),
            3 => blk.res2.forward(x),
            4 => blk.res3.forward(x),
            _ => Err(Error::InvalidOperation(format!("db1 sub idx {sub_idx} out of range"))),
        }
    }

    /// Bisect helper: returns intermediates with names matching
    /// `oobleck_bisect.safetensors` Python dump.
    pub fn decode_with_dumps(&self, z: &Tensor) -> Result<Vec<(String, Tensor)>> {
        let z_f32 = if z.dtype() != DType::F32 { z.to_dtype(DType::F32)? } else { z.clone() };
        let mut dumps = Vec::new();
        let mut h = self.init_conv.forward(&z_f32)?;
        dumps.push(("after_layer_0_Conv1d".to_string(), h.clone()));
        for (i, blk) in self.blocks.iter().enumerate() {
            h = blk.forward(&h)?;
            dumps.push((format!("after_layer_{}_DecoderBlock", i + 1), h.clone()));
        }
        h = self.final_snake.forward(&h)?;
        dumps.push(("after_layer_6_SnakeBeta".to_string(), h.clone()));
        h = self.final_conv.forward(&h)?;
        dumps.push(("after_layer_7_Conv1d".to_string(), h));
        Ok(dumps)
    }
}

// Suppress dead-code warnings for fields not yet used externally.
#[allow(dead_code)]
fn _used_imports() {
    let _ = Shape::from_dims(&[1]);
}
