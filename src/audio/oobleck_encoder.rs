//! Stable Audio Open 1.0 Oobleck VAE encoder — pure flame-core.
//!
//! Mirrors `inference/model/sa_audio/sa_audio_module.py:OobleckEncoder` with
//! the variational bottleneck applied at inference time (mean only — no KL
//! sampling needed for talking-head conditioning).
//!
//! Architecture (verified against the actual safetensors):
//! ```text
//! encoder.layers.0:  WNConv1d(2 → 128,    k=7, pad=3)        # initial
//! encoder.layers.1:  EncoderBlock(128 → 128,  stride=2)       # block 0
//! encoder.layers.2:  EncoderBlock(128 → 256,  stride=4)       # block 1
//! encoder.layers.3:  EncoderBlock(256 → 512,  stride=4)       # block 2
//! encoder.layers.4:  EncoderBlock(512 → 1024, stride=8)       # block 3
//! encoder.layers.5:  EncoderBlock(1024 → 2048, stride=8)      # block 4
//! encoder.layers.6:  SnakeBeta(2048)
//! encoder.layers.7:  WNConv1d(2048 → 128,  k=3, pad=1)        # latent head
//! ```
//!
//! Each EncoderBlock = 3× ResidualUnit (dilations 1, 3, 9) + SnakeBeta + WNConv1d
//! downsample (kernel 2*stride, stride, pad=ceil(stride/2)).
//!
//! Each ResidualUnit = SnakeBeta → WNConv1d(k=7, dilation=d, pad=d*3) →
//!                     SnakeBeta → WNConv1d(k=1) → add residual.
//!
//! Total downsample: 2 * 4 * 4 * 8 * 8 = 2048. Encoder output T_lat =
//! T_audio / 2048. After bottleneck split: 128 → 64 channels (mean).
//!
//! For 1 sec @ 51,200 Hz: 51,200 / 2048 = 25 latent tokens, exactly
//! MagiHuman's `num_frames = 25` for 1-sec @ 25 fps video.
//!
//! NOTE: Stable Audio Open 1.0 ships the VAE with weight-norm ALREADY FUSED
//! into single `weight` tensors — no `weight_g`/`weight_v` split. This is
//! different from the ACE-Step VAE (in `vae/acestep_vae.rs`) where we fuse
//! at load.

use anyhow::{anyhow, Context, Result};
use flame_core::conv1d::conv1d;
use flame_core::{serialization, DType, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

const SNAKE_EPS: f32 = 1e-9;

// SA Open 1.0 architecture constants (per checkpoint inspection)
const IN_CHANNELS: usize = 2; // stereo
const BASE_CHANNELS: usize = 128;
const C_MULTS: [usize; 6] = [1, 1, 2, 4, 8, 16];
const STRIDES: [usize; 5] = [2, 4, 4, 8, 8];
const NUM_BLOCKS: usize = 5;
const FINAL_LATENT_DIM: usize = 128; // pre-bottleneck (split → 64 mean + 64 scale)
const DILATIONS: [usize; 3] = [1, 3, 9];

type Weights = HashMap<String, Tensor>;

fn get_bf16(weights: &Weights, key: &str) -> Result<Tensor> {
    let t = weights
        .get(key)
        .ok_or_else(|| anyhow!("OobleckVaeEncoder: missing weight '{key}'"))?;
    t.to_dtype(DType::BF16)
        .map_err(|e| anyhow!("OobleckVaeEncoder: '{key}' to_dtype: {e:?}"))
}

// ---------------------------------------------------------------------------
// SnakeBeta activation
// ---------------------------------------------------------------------------
//
// y = x + (1 / (β + ε)) * sin²(α * x)
//
// Stored as `alpha`, `beta` 1D tensors `(C,)`; logscale=True so we exp() at
// load time and store `(1, C, 1)` for broadcasting against `[B, C, L]`.

struct Snake {
    /// exp(alpha) reshaped to [1, C, 1]
    alpha_exp: Tensor,
    /// 1 / (exp(beta) + eps), [1, C, 1]
    inv_beta: Tensor,
}

impl Snake {
    fn load(weights: &Weights, prefix: &str) -> Result<Self> {
        let alpha = get_bf16(weights, &format!("{prefix}.alpha"))?;
        let beta = get_bf16(weights, &format!("{prefix}.beta"))?;
        let alpha_dims = alpha.shape().dims().to_vec();
        let c = alpha_dims.last().copied().unwrap_or(0);
        if c == 0 {
            anyhow::bail!("Snake: empty alpha shape {:?}", alpha_dims);
        }

        let alpha_f32 = alpha.to_dtype(DType::F32)?;
        let beta_f32 = beta.to_dtype(DType::F32)?;
        let alpha_exp = alpha_f32
            .exp()?
            .reshape(&[1, c, 1])?
            .to_dtype(DType::BF16)?;
        let beta_exp = beta_f32.exp()?;
        let inv_beta = beta_exp
            .add_scalar(SNAKE_EPS)?
            .reciprocal()?
            .reshape(&[1, c, 1])?
            .to_dtype(DType::BF16)?;
        Ok(Self { alpha_exp, inv_beta })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let ax = x.mul(&self.alpha_exp)?;
        let s = ax.sin()?;
        let s2 = s.mul(&s)?;
        let scaled = s2.mul(&self.inv_beta)?;
        x.add(&scaled).map_err(|e| anyhow!("Snake add: {e:?}"))
    }
}

// ---------------------------------------------------------------------------
// Conv1d wrapper (already-fused weights from SA Open ckpt)
// ---------------------------------------------------------------------------

struct Wnconv {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Wnconv {
    fn load(
        weights: &Weights,
        prefix: &str,
        stride: usize,
        padding: usize,
        dilation: usize,
        with_bias: bool,
    ) -> Result<Self> {
        let weight = get_bf16(weights, &format!("{prefix}.weight"))?;
        let bias = if with_bias {
            Some(get_bf16(weights, &format!("{prefix}.bias"))?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            dilation,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        conv1d(
            x,
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
            1,
        )
        .map_err(|e| anyhow!("conv1d: {e:?}"))
    }
}

// ---------------------------------------------------------------------------
// ResidualUnit: snake → conv7(dilated) → snake → conv1 → +residual
// ---------------------------------------------------------------------------

struct ResidualUnit {
    snake1: Snake,
    conv1: Wnconv, // k=7, dilation=d, pad=d*3
    snake2: Snake,
    conv2: Wnconv, // k=1
}

impl ResidualUnit {
    fn load(weights: &Weights, prefix: &str, channels: usize, dilation: usize) -> Result<Self> {
        let _ = channels; // shape inferred from weights
        let snake1 = Snake::load(weights, &format!("{prefix}.layers.0"))
            .with_context(|| format!("ResidualUnit snake1 at {prefix}"))?;
        let pad7 = dilation * 3;
        let conv1 = Wnconv::load(weights, &format!("{prefix}.layers.1"), 1, pad7, dilation, true)?;
        let snake2 = Snake::load(weights, &format!("{prefix}.layers.2"))?;
        let conv2 = Wnconv::load(weights, &format!("{prefix}.layers.3"), 1, 0, 1, true)?;
        Ok(Self {
            snake1,
            conv1,
            snake2,
            conv2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.snake1.forward(x)?;
        let h = self.conv1.forward(&h)?;
        let h = self.snake2.forward(&h)?;
        let h = self.conv2.forward(&h)?;
        x.add(&h).map_err(|e| anyhow!("ResidualUnit add: {e:?}"))
    }
}

// ---------------------------------------------------------------------------
// EncoderBlock: 3 ResidualUnits + Snake + downsample conv
// ---------------------------------------------------------------------------

struct EncoderBlock {
    res_units: [ResidualUnit; 3],
    snake: Snake,
    downsample: Wnconv, // k=2*stride, stride=stride, pad=ceil(stride/2)
}

impl EncoderBlock {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_channels: usize,
        stride: usize,
    ) -> Result<Self> {
        let r0 = ResidualUnit::load(
            weights,
            &format!("{prefix}.layers.0"),
            in_channels,
            DILATIONS[0],
        )?;
        let r1 = ResidualUnit::load(
            weights,
            &format!("{prefix}.layers.1"),
            in_channels,
            DILATIONS[1],
        )?;
        let r2 = ResidualUnit::load(
            weights,
            &format!("{prefix}.layers.2"),
            in_channels,
            DILATIONS[2],
        )?;
        let snake = Snake::load(weights, &format!("{prefix}.layers.3"))?;
        let pad_ds = (stride + 1) / 2; // matches Python `math.ceil(stride/2)`
        let downsample = Wnconv::load(
            weights,
            &format!("{prefix}.layers.4"),
            stride,
            pad_ds,
            1,
            true,
        )?;
        Ok(Self {
            res_units: [r0, r1, r2],
            snake,
            downsample,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.res_units[0].forward(x)?;
        h = self.res_units[1].forward(&h)?;
        h = self.res_units[2].forward(&h)?;
        h = self.snake.forward(&h)?;
        self.downsample.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// OobleckVaeEncoder
// ---------------------------------------------------------------------------

pub struct OobleckVaeEncoder {
    initial: Wnconv, // k=7, pad=3
    blocks: [EncoderBlock; NUM_BLOCKS],
    final_snake: Snake,
    final_conv: Wnconv, // k=3, pad=1, latent_dim=128
}

impl OobleckVaeEncoder {
    /// Load the SA Open 1.0 VAE encoder (encoder.* keys) from a safetensors.
    /// Decoder weights are present in the same file but ignored here.
    pub fn load(
        path: &Path,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        // Filter to encoder.* only — saves loading the decoder unnecessarily.
        let weights = serialization::load_file_filtered(
            path,
            device,
            |k: &str| k.starts_with("encoder."),
        )
        .map_err(|e| anyhow!("OobleckVaeEncoder weight load: {e:?}"))?;
        if weights.is_empty() {
            anyhow::bail!(
                "OobleckVaeEncoder: no encoder.* keys in {} — wrong file?",
                path.display()
            );
        }
        eprintln!(
            "[audio] loaded OobleckVaeEncoder ({} weight tensors) from {}",
            weights.len(),
            path.display()
        );

        // encoder.layers.0: WNConv1d(2 → 128, k=7, pad=3)
        let initial = Wnconv::load(&weights, "encoder.layers.0", 1, 3, 1, true)?;

        // encoder.layers.1..5: 5 EncoderBlocks
        let mut blocks: Vec<EncoderBlock> = Vec::with_capacity(NUM_BLOCKS);
        for i in 0..NUM_BLOCKS {
            let in_ch = C_MULTS[i] * BASE_CHANNELS;
            let stride = STRIDES[i];
            let block = EncoderBlock::load(
                &weights,
                &format!("encoder.layers.{}", i + 1),
                in_ch,
                stride,
            )?;
            blocks.push(block);
        }

        // encoder.layers.6: final SnakeBeta on 2048 channels
        let final_snake = Snake::load(&weights, "encoder.layers.6")?;
        // encoder.layers.7: WNConv1d(2048 → 128, k=3, pad=1)
        let final_conv = Wnconv::load(&weights, "encoder.layers.7", 1, 1, 1, true)?;

        let blocks: [EncoderBlock; NUM_BLOCKS] = blocks
            .try_into()
            .map_err(|_| anyhow!("OobleckVaeEncoder: block count mismatch"))?;
        Ok(Self {
            initial,
            blocks,
            final_snake,
            final_conv,
        })
    }

    /// Encode stereo audio `[1, 2, T_audio]` BF16 → pre-bottleneck latent
    /// `[1, 128, T_lat]` BF16 where T_lat = T_audio / 2048.
    pub fn encode(&self, audio: &Tensor) -> Result<Tensor> {
        let dims = audio.shape().dims();
        if dims != &[1, IN_CHANNELS, dims[2]] {
            anyhow::bail!(
                "OobleckVaeEncoder.encode: expected [1, {IN_CHANNELS}, T], got {:?}",
                dims
            );
        }
        let mut h = self.initial.forward(audio)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        h = self.final_snake.forward(&h)?;
        self.final_conv.forward(&h)
    }

    /// Apply the variational bottleneck's *mean-only* path: split last channel
    /// dim in half (mean | scale), return mean. At inference for talking-head
    /// conditioning we don't sample — we use the mean to keep the audio
    /// signal deterministic and well-defined.
    pub fn bottleneck_mean(&self, pre_bottleneck: &Tensor) -> Result<Tensor> {
        let dims = pre_bottleneck.shape().dims();
        if dims.len() != 3 || dims[1] != FINAL_LATENT_DIM {
            anyhow::bail!(
                "bottleneck_mean: expected [B, {FINAL_LATENT_DIM}, T], got {:?}",
                dims
            );
        }
        let half = FINAL_LATENT_DIM / 2; // 64
        let mean = pre_bottleneck
            .narrow(1, 0, half)?
            .contiguous()
            .map_err(|e| anyhow!("bottleneck_mean narrow: {e:?}"))?;
        Ok(mean)
    }
}

#[allow(dead_code)]
fn _shape(t: &Tensor) -> Vec<usize> {
    t.shape().dims().to_vec()
}

#[allow(dead_code)]
fn _check_shape(t: &Tensor, expected: &[usize], where_: &str) -> Result<()> {
    let got = t.shape().dims();
    if got != expected {
        anyhow::bail!("{}: expected {:?}, got {:?}", where_, expected, got);
    }
    Ok(())
}

// Re-export the device alias used in the public API to keep the module
// signature self-contained.
pub type Device = Arc<cudarc::driver::CudaDevice>;

// Need to bring Shape into scope for use sites that call directly.
#[allow(unused_imports)]
use Shape as _Shape;
