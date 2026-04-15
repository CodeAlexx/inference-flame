//! ACE-Step Oobleck VAE decoder — pure flame-core, production checkpoint parity.
//!
//! Decodes latent `[B, 64, T_lat]` to stereo 48 kHz waveform `[B, 2, T_audio]`.
//!
//! Architecture (from diffusers `AutoencoderOobleck` decoder):
//!
//!   conv1: WNConv1d(64 → 2048, k=7, pad=3)
//!   5 decoder blocks (reversed upsampling ratios [10, 6, 4, 4, 2]):
//!     Block i: Snake1d → WN ConvTranspose1d → 3 residual units
//!       ConvTranspose1d: kernel=2*stride, padding=ceil(stride/2)
//!       Block 0: 2048 → 1024, stride=10
//!       Block 1: 1024 →  512, stride=6
//!       Block 2:  512 →  256, stride=4
//!       Block 3:  256 →  128, stride=4
//!       Block 4:  128 →  128, stride=2
//!   snake1: Snake1d(128)
//!   conv2: WNConv1d(128 → 2, k=7, pad=3, no bias)
//!
//! Each residual unit: snake1 → conv1(k=7, dilation=d) → snake2 → conv2(k=1)
//!   dilations per block: [1, 3, 9]
//!
//! Snake1d: x + (1/(exp(beta)+eps)) * sin²(exp(alpha) * x)
//!   alpha, beta are [1, C, 1] learnable parameters (logscale)
//!
//! Weight normalization stored as weight_g [C_out, 1, 1] and weight_v [C_out, C_in, K].
//! Fused at load: weight = weight_g * weight_v / ||weight_v||_{dim=[1,2]}
//!
//! ## Performance optimizations
//!
//! - ConvTranspose1d weights are pre-flipped+transposed at load time (avoids
//!   O(K) flip + permute per forward pass).
//! - Conv1d k=1 uses matmul fast path (avoids cuDNN overhead for trivial ops).
//! - flip_last_axis uses GPU kernel instead of element-by-element narrow+cat.

use flame_core::conv1d::{conv1d, conv_transpose1d_prepare_weight, conv_transpose1d_with_prepared_weight};
use flame_core::{serialization, CudaDevice, DType, Error, Result, Tensor};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

type Weights = HashMap<String, Tensor>;

const SNAKE_EPS: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Weight loading helpers
// ---------------------------------------------------------------------------

fn get_bf16(weights: &Weights, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .ok_or_else(|| Error::InvalidOperation(format!("ACEStepVAE: missing weight: {key}")))?
        .to_dtype(DType::BF16)
}

fn has_key(weights: &Weights, key: &str) -> bool {
    weights.contains_key(key)
}

/// Fuse weight_g and weight_v into a single weight tensor.
/// weight = weight_g * weight_v / norm(weight_v, dim=[1,2], keepdim=True)
/// weight_g: [C_out, 1, 1], weight_v: [C_out, C_in, K]
fn fuse_weight_norm(weight_g: &Tensor, weight_v: &Tensor) -> Result<Tensor> {
    let dims = weight_v.shape().dims().to_vec();
    let (c_out, c_in, k) = (dims[0], dims[1], dims[2]);

    // Flatten to [C_out, C_in*K] for norm computation
    let v_f32 = weight_v.to_dtype(DType::F32)?;
    let v_flat = v_f32.reshape(&[c_out, c_in * k])?;
    let v_sq = v_flat.mul(&v_flat)?;
    // sum_dim(1) removes dim 1: [C_out, C_in*K] -> [C_out]
    let sum_sq = v_sq.sum_dim(1)?;
    let norm = sum_sq.sqrt()?;
    // Reshape norm to [C_out, 1, 1] for broadcasting
    let norm_bcast = norm.reshape(&[c_out, 1, 1])?;

    // weight = weight_g * weight_v / norm
    let g_f32 = weight_g.to_dtype(DType::F32)?;
    let v_normalized = v_f32.div(&norm_bcast)?;
    let result = v_normalized.mul(&g_f32)?;
    result.to_dtype(DType::BF16)
}

/// Load a weight-normalized Conv1d weight (and optional bias).
/// Returns (fused_weight, Option<bias>).
fn load_wn_conv1d(weights: &Weights, prefix: &str) -> Result<(Tensor, Option<Tensor>)> {
    let weight_g = get_bf16(weights, &format!("{prefix}.weight_g"))?;
    let weight_v = get_bf16(weights, &format!("{prefix}.weight_v"))?;
    let weight = fuse_weight_norm(&weight_g, &weight_v)?;
    let bias_key = format!("{prefix}.bias");
    let bias = if has_key(weights, &bias_key) {
        Some(get_bf16(weights, &bias_key)?)
    } else {
        None
    };
    Ok((weight, bias))
}

// ---------------------------------------------------------------------------
// Snake1d activation
// ---------------------------------------------------------------------------

struct Snake1d {
    /// exp(alpha), precomputed, [1, C, 1]
    alpha_exp: Tensor,
    /// 1 / (exp(beta) + eps), precomputed, [1, C, 1]
    inv_beta: Tensor,
}

impl Snake1d {
    fn load(weights: &Weights, prefix: &str) -> Result<Self> {
        let alpha = get_bf16(weights, &format!("{prefix}.alpha"))?;
        let beta = get_bf16(weights, &format!("{prefix}.beta"))?;

        // Precompute in F32 for numerical stability
        let alpha_f32 = alpha.to_dtype(DType::F32)?;
        let beta_f32 = beta.to_dtype(DType::F32)?;
        let alpha_exp = alpha_f32.exp()?.to_dtype(DType::BF16)?;
        let beta_exp = beta_f32.exp()?;
        let inv_beta = beta_exp.add_scalar(SNAKE_EPS as f32)?.reciprocal()?.to_dtype(DType::BF16)?;

        Ok(Self { alpha_exp, inv_beta })
    }

    /// x + (1/(exp(beta)+eps)) * sin²(exp(alpha)*x)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let ax = x.mul(&self.alpha_exp)?;
        let sin_ax = ax.sin()?;
        let sin_sq = sin_ax.mul(&sin_ax)?;
        let scaled = sin_sq.mul(&self.inv_beta)?;
        x.add(&scaled)
    }
}

// ---------------------------------------------------------------------------
// OobleckResidualUnit
// ---------------------------------------------------------------------------

struct OobleckResidualUnit {
    snake1: Snake1d,
    conv1_w: Tensor,
    conv1_b: Tensor,
    snake2: Snake1d,
    conv2_w: Tensor,
    conv2_b: Tensor,
    dilation: usize,
}

impl OobleckResidualUnit {
    fn load(weights: &Weights, prefix: &str, dilation: usize) -> Result<Self> {
        let snake1 = Snake1d::load(weights, &format!("{prefix}.snake1"))?;
        let (conv1_w, conv1_b) = load_wn_conv1d(weights, &format!("{prefix}.conv1"))?;
        let snake2 = Snake1d::load(weights, &format!("{prefix}.snake2"))?;
        let (conv2_w, conv2_b) = load_wn_conv1d(weights, &format!("{prefix}.conv2"))?;

        Ok(Self {
            snake1,
            conv1_w,
            conv1_b: conv1_b.ok_or_else(|| {
                Error::InvalidOperation(format!("{prefix}.conv1 missing bias"))
            })?,
            snake2,
            conv2_w,
            conv2_b: conv2_b.ok_or_else(|| {
                Error::InvalidOperation(format!("{prefix}.conv2 missing bias"))
            })?,
            dilation,
        })
    }

    fn forward(&self, hidden_state: &Tensor) -> Result<Tensor> {
        // snake1 → conv1(k=7, dilation=d, pad=(7-1)*d/2) → snake2 → conv2(k=1)
        let h = self.snake1.forward(hidden_state)?;
        let pad1 = ((7 - 1) * self.dilation) / 2;
        let h = conv1d(&h, &self.conv1_w, Some(&self.conv1_b), 1, pad1, self.dilation, 1)?;
        let h = self.snake2.forward(&h)?;
        // conv2 is k=1: conv1d fast path will use matmul
        let h = conv1d(&h, &self.conv2_w, Some(&self.conv2_b), 1, 0, 1, 1)?;

        // Residual connection — handle possible length mismatch from dilation padding
        let in_len = hidden_state.shape().dims()[2];
        let out_len = h.shape().dims()[2];
        if in_len != out_len {
            let padding = (in_len - out_len) / 2;
            let trimmed = hidden_state.narrow(2, padding, out_len)?;
            trimmed.add(&h)
        } else {
            hidden_state.add(&h)
        }
    }
}

// ---------------------------------------------------------------------------
// OobleckDecoderBlock
// ---------------------------------------------------------------------------

struct OobleckDecoderBlock {
    snake1: Snake1d,
    /// Pre-prepared ConvTranspose1d weight: already flipped+transposed to
    /// regular Conv1d layout [C_out, C_in, K]. Avoids per-call flip+permute.
    conv_t1_w_prepared: Tensor,
    conv_t1_b: Tensor,
    conv_t1_stride: usize,
    conv_t1_padding: usize,
    res_units: Vec<OobleckResidualUnit>,
    block_idx: usize,
}

impl OobleckDecoderBlock {
    fn load(weights: &Weights, prefix: &str, stride: usize, block_idx: usize) -> Result<Self> {
        let snake1 = Snake1d::load(weights, &format!("{prefix}.snake1"))?;
        let (conv_t1_w, conv_t1_b) = load_wn_conv1d(weights, &format!("{prefix}.conv_t1"))?;
        let padding = (stride + 1) / 2; // ceil(stride/2)

        // Pre-compute the flipped+transposed weight at load time.
        // This avoids O(K) narrow+cat flip + permute on every forward pass.
        let conv_t1_w_prepared = conv_transpose1d_prepare_weight(&conv_t1_w, 1)?;

        let dilations = [1, 3, 9];
        let mut res_units = Vec::with_capacity(3);
        for (i, &d) in dilations.iter().enumerate() {
            res_units.push(OobleckResidualUnit::load(
                weights,
                &format!("{prefix}.res_unit{}", i + 1),
                d,
            )?);
        }

        Ok(Self {
            snake1,
            conv_t1_w_prepared,
            conv_t1_b: conv_t1_b.ok_or_else(|| {
                Error::InvalidOperation(format!("{prefix}.conv_t1 missing bias"))
            })?,
            conv_t1_stride: stride,
            conv_t1_padding: padding,
            res_units,
            block_idx,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t_block = Instant::now();

        let mut h = self.snake1.forward(x)?;

        let t_upsample = Instant::now();
        h = conv_transpose1d_with_prepared_weight(
            &h,
            &self.conv_t1_w_prepared,
            Some(&self.conv_t1_b),
            self.conv_t1_stride,
            self.conv_t1_padding,
            0, // output_padding
            1, // dilation
            1, // groups
        )?;
        // Sync to get accurate timing
        h.device().synchronize()?;
        let upsample_ms = t_upsample.elapsed().as_secs_f64() * 1000.0;

        let t_res = Instant::now();
        for unit in &self.res_units {
            h = unit.forward(&h)?;
        }
        h.device().synchronize()?;
        let res_ms = t_res.elapsed().as_secs_f64() * 1000.0;

        let block_ms = t_block.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "  block[{}]: {:.1}ms total (upsample: {:.1}ms, residuals: {:.1}ms) shape={:?}",
            self.block_idx, block_ms, upsample_ms, res_ms, h.shape().dims()
        );

        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// Public decoder
// ---------------------------------------------------------------------------

pub struct OobleckVaeDecoder {
    conv1_w: Tensor,
    conv1_b: Tensor,
    blocks: Vec<OobleckDecoderBlock>,
    snake1: Snake1d,
    conv2_w: Tensor,
}

impl OobleckVaeDecoder {
    /// Load decoder weights from a pre-parsed weight map.
    /// Keys should start with "decoder." (e.g. "decoder.conv1.weight_g").
    fn load(weights: &Weights) -> Result<Self> {
        let (conv1_w, conv1_b) = load_wn_conv1d(weights, "decoder.conv1")?;
        let conv1_b = conv1_b.ok_or_else(|| {
            Error::InvalidOperation("decoder.conv1 missing bias".into())
        })?;

        // Upsampling ratios (reversed downsampling): [10, 6, 4, 4, 2]
        let strides = [10, 6, 4, 4, 2];
        let mut blocks = Vec::with_capacity(5);
        for (i, &stride) in strides.iter().enumerate() {
            blocks.push(OobleckDecoderBlock::load(
                weights,
                &format!("decoder.block.{i}"),
                stride,
                i,
            )?);
        }

        let snake1 = Snake1d::load(weights, "decoder.snake1")?;
        let (conv2_w, _conv2_b) = load_wn_conv1d(weights, "decoder.conv2")?;
        // conv2 has no bias per the checkpoint

        Ok(Self {
            conv1_w,
            conv1_b,
            blocks,
            snake1,
            conv2_w,
        })
    }

    /// Load from safetensors file. Only loads decoder keys.
    pub fn from_safetensors(path: &str, device: &Arc<CudaDevice>) -> Result<Self> {
        eprintln!("Loading ACE-Step Oobleck VAE decoder from: {path}");
        let raw = serialization::load_file_filtered(
            std::path::Path::new(path),
            device,
            |k| k.starts_with("decoder."),
        )?;

        let dec_count = raw.len();
        eprintln!("  {dec_count} decoder keys loaded");

        Self::load(&raw)
    }

    /// Decode latent `[B, 64, T_lat]` to waveform `[B, 2, T_audio]`.
    ///
    /// T_audio = T_lat * product(upsampling_ratios) = T_lat * 1920
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let t_total = Instant::now();

        // conv1: 64 → 2048, k=7, pad=3
        let t0 = Instant::now();
        let mut h = conv1d(latent, &self.conv1_w, Some(&self.conv1_b), 1, 3, 1, 1)?;
        h.device().synchronize()?;
        eprintln!("  conv1: {:.1}ms shape={:?}", t0.elapsed().as_secs_f64() * 1000.0, h.shape().dims());

        // 5 decoder blocks
        for block in &self.blocks {
            h = block.forward(&h)?;
        }

        // snake1 → conv2
        let t0 = Instant::now();
        h = self.snake1.forward(&h)?;
        h.device().synchronize()?;
        eprintln!("  snake1: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

        let t0 = Instant::now();
        h = conv1d(&h, &self.conv2_w, None, 1, 3, 1, 1)?;
        h.device().synchronize()?;
        eprintln!("  conv2: {:.1}ms shape={:?}", t0.elapsed().as_secs_f64() * 1000.0, h.shape().dims());

        eprintln!("  VAE decode total: {:.1}ms", t_total.elapsed().as_secs_f64() * 1000.0);

        Ok(h)
    }
}
