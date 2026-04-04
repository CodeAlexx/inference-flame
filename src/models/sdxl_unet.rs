//! SDXL UNet — pure Rust, LDM-format weight keys.
//!
//! Architecture: Standard LDM UNet for SDXL (Stable Diffusion XL).
//! - model_channels=320, channel_mult=(1,2,4), num_res_blocks=2
//! - context_dim=2048 (CLIP-L 768 + CLIP-G 1280)
//! - adm_in_channels=2816 (pooled CLIP-G 1280 + time_ids)
//! - transformer_depth_input=[0,0,2,2,10,10], middle=10
//! - transformer_depth_output=[0,0,0,2,2,2,10,10,10]
//! - use_linear_in_transformer=true (Linear proj_in/proj_out, not Conv2d 1x1)
//! - 64 dim per head (num_heads = channels / 64)
//!
//! LDM key format:
//!   time_embed.0.weight/bias, time_embed.2.weight/bias
//!   label_emb.0.0.weight/bias, label_emb.0.2.weight/bias
//!   input_blocks.0.0.weight/bias                    (conv_in: 4->320)
//!   input_blocks.{n}.0.in_layers.0/2.*              (ResBlock norm1/conv1)
//!   input_blocks.{n}.0.out_layers.0/3.*             (ResBlock norm2/conv2)
//!   input_blocks.{n}.0.emb_layers.1.*               (time embed proj)
//!   input_blocks.{n}.0.skip_connection.*             (channel mismatch 1x1)
//!   input_blocks.{n}.1.norm/proj_in/proj_out.*       (SpatialTransformer)
//!   input_blocks.{n}.1.transformer_blocks.{j}.*      (BasicTransformerBlock)
//!   middle_block.{0,1,2}.*                           (ResBlock+SpatialTransformer+ResBlock)
//!   output_blocks.{n}.*                              (mirror of input + skip concat)
//!   out.0.weight/bias, out.2.weight/bias             (final GroupNorm + Conv)
//!
//! Weight loading: HashMap approach with block-level offloading via load_file_filtered.
//! NCHW layout throughout. GroupNorm converts NCHW->NHWC internally.
//! BF16 in/out, F32 compute in kernels.

use flame_core::cuda_kernels::CudaKernels;
use flame_core::cuda_ops::GpuOps;
use flame_core::group_norm::group_norm;
use flame_core::layer_norm::layer_norm;
use flame_core::sdpa::forward as sdpa_forward;
use flame_core::serialization::load_file_filtered;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Layout helpers — GroupNorm and Conv kernel both want NHWC; UNet data flow is NCHW
// ---------------------------------------------------------------------------

/// NCHW -> NHWC (GPU-optimized, avoids CPU roundtrip)
fn to_nhwc(x: &Tensor) -> Result<Tensor> {
    GpuOps::permute_nchw_to_nhwc(x)
}

/// NHWC -> NCHW (GPU-optimized, avoids CPU roundtrip)
fn to_nchw(x: &Tensor) -> Result<Tensor> {
    GpuOps::permute_nhwc_to_nchw(x)
}

/// GroupNorm on NCHW tensor (converts to NHWC internally, converts back)
fn group_norm_nchw(
    x: &Tensor,
    num_groups: usize,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let nhwc = to_nhwc(x)?;
    let out_nhwc = group_norm(&nhwc, num_groups, weight, bias, eps)?;
    to_nchw(&out_nhwc)
}

/// Transpose a 2D tensor [M, N] -> [N, M]
fn transpose_2d(t: &Tensor) -> Result<Tensor> {
    t.permute(&[1, 0])
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// SDXL UNet configuration.
pub struct SDXLConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub model_channels: usize,
    pub channel_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub context_dim: usize,
    pub head_dim: usize,
    pub use_linear_in_transformer: bool,
    /// Flat list: one entry per res-block in input half
    pub transformer_depth_input: Vec<usize>,
    pub transformer_depth_middle: usize,
    /// Flat list: one entry per res-block in output half
    pub transformer_depth_output: Vec<usize>,
    pub adm_in_channels: usize,
}

impl Default for SDXLConfig {
    fn default() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            model_channels: 320,
            channel_mult: vec![1, 2, 4],
            num_res_blocks: 2,
            context_dim: 2048,
            head_dim: 64,
            use_linear_in_transformer: true,
            transformer_depth_input: vec![0, 0, 2, 2, 10, 10],
            transformer_depth_middle: 10,
            transformer_depth_output: vec![10, 10, 10, 2, 2, 2, 0, 0, 0],
            adm_in_channels: 2816,
        }
    }
}

// ---------------------------------------------------------------------------
// Block descriptors — describe what each input/output block contains
// ---------------------------------------------------------------------------

/// Describes one flat input_block or output_block for weight loading.
#[derive(Debug, Clone)]
enum BlockType {
    /// Initial conv_in (input_blocks.0.0)
    ConvIn,
    /// ResBlock (+ optional SpatialTransformer)
    ResBlockAttn {
        in_ch: usize,
        out_ch: usize,
        transformer_depth: usize,
    },
    /// Downsample (stride-2 conv)
    Downsample { channels: usize },
    /// Output ResBlock (+ optional SpatialTransformer + optional Upsample)
    /// in_ch includes skip concat channel count
    OutputResBlockAttn {
        in_ch: usize,
        out_ch: usize,
        transformer_depth: usize,
        has_upsample: bool,
    },
}

// ---------------------------------------------------------------------------
// SDXLUNet — weight-backed model with block-level offloading
// ---------------------------------------------------------------------------

/// SDXL UNet with HashMap-based weight storage and block-level offloading.
pub struct SDXLUNet {
    config: SDXLConfig,
    /// Small weights that stay on GPU permanently (time_embed, label_emb, conv_in, out)
    resident: HashMap<String, Tensor>,
    /// Path to safetensors for on-demand block loading via mmap
    model_path: String,
    device: Arc<cudarc::driver::CudaDevice>,
    /// Temporarily loaded block weights (loaded on demand, dropped after use)
    block_cache: HashMap<String, Tensor>,
    /// CUDA kernels for upsample etc.
    kernels: CudaKernels,
    /// Block descriptors for input_blocks
    input_block_descs: Vec<BlockType>,
    /// Block descriptors for output_blocks
    output_block_descs: Vec<BlockType>,
    /// Channel counts stored by each input block (for skip connections)
    input_block_channels: Vec<usize>,
    /// When true, all weights are in `resident` — skip load/unload
    all_on_gpu: bool,
}

impl SDXLUNet {
    /// Create a new SDXLUNet.
    ///
    /// `resident` should contain the small always-on-GPU weights:
    ///   time_embed.*, label_emb.*, input_blocks.0.0.*, out.*
    ///
    /// `model_path` is the safetensors file for block-level loading.
    pub fn new(
        model_path: String,
        resident: HashMap<String, Tensor>,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let config = SDXLConfig::default();
        let kernels = CudaKernels::new(device.clone())?;

        let mut unet = Self {
            config,
            resident,
            model_path,
            device,
            block_cache: HashMap::new(),
            kernels,
            input_block_descs: Vec::new(),
            output_block_descs: Vec::new(),
            input_block_channels: Vec::new(),
            all_on_gpu: false,
        };
        unet.build_block_descriptors();
        Ok(unet)
    }

    /// Build block descriptors matching the exact LDM flat indexing.
    ///
    /// SDXL input_blocks layout (9 blocks total):
    ///   0: conv_in (4->320)
    ///   1: ResBlock(320->320)                          td=0
    ///   2: ResBlock(320->320)                          td=0
    ///   3: Downsample(320)
    ///   4: ResBlock(320->640) + SpatialTransformer     td=2
    ///   5: ResBlock(640->640) + SpatialTransformer     td=2
    ///   6: Downsample(640)
    ///   7: ResBlock(640->1280) + SpatialTransformer    td=10
    ///   8: ResBlock(1280->1280) + SpatialTransformer   td=10
    ///
    /// SDXL output_blocks layout (9 blocks total):
    ///   0: ResBlock(1280+1280=2560->1280) + ST         td=10
    ///   1: ResBlock(1280+1280=2560->1280) + ST         td=10
    ///   2: ResBlock(1280+640=1920->1280) + ST + Up     td=10
    ///   3: ResBlock(1280+640=1920->640) + ST           td=2
    ///   4: ResBlock(640+640=1280->640) + ST            td=2
    ///   5: ResBlock(640+320=960->640) + ST + Upsample  td=2
    ///   6: ResBlock(640+320=960->320)                  td=0
    ///   7: ResBlock(320+320=640->320)                  td=0
    ///   8: ResBlock(320+320=640->320)                  td=0
    fn build_block_descriptors(&mut self) {
        let mc = self.config.model_channels;
        let mut td_input = self.config.transformer_depth_input.clone();
        td_input.reverse(); // we'll pop from the end (equivalent to pop(0) in Python)

        // --- Input blocks ---
        // Block 0: conv_in
        self.input_block_descs.push(BlockType::ConvIn);
        self.input_block_channels.push(mc);

        let mut ch = mc;
        for (level, &mult) in self.config.channel_mult.iter().enumerate() {
            let out_ch = mc * mult;
            for _ in 0..self.config.num_res_blocks {
                let num_transformers = td_input.pop().unwrap_or(0);
                self.input_block_descs.push(BlockType::ResBlockAttn {
                    in_ch: ch,
                    out_ch,
                    transformer_depth: num_transformers,
                });
                ch = out_ch;
                self.input_block_channels.push(ch);
            }
            // Downsample (except last level)
            if level < self.config.channel_mult.len() - 1 {
                self.input_block_descs.push(BlockType::Downsample { channels: ch });
                self.input_block_channels.push(ch);
            }
        }

        // --- Output blocks ---
        let mut td_output = self.config.transformer_depth_output.clone();
        td_output.reverse(); // pop from end = pop(0) in Python
        let mut input_channels = self.input_block_channels.clone();

        let num_levels = self.config.channel_mult.len();
        ch = mc * self.config.channel_mult[num_levels - 1];

        for level in (0..num_levels).rev() {
            let out_ch = mc * self.config.channel_mult[level];
            for i in 0..self.config.num_res_blocks + 1 {
                let skip_ch = input_channels.pop().unwrap_or(0);
                let in_ch = ch + skip_ch;
                let num_transformers = td_output.pop().unwrap_or(0);
                let has_upsample = level > 0 && i == self.config.num_res_blocks;

                self.output_block_descs.push(BlockType::OutputResBlockAttn {
                    in_ch,
                    out_ch,
                    transformer_depth: num_transformers,
                    has_upsample,
                });
                ch = out_ch;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Block loading / weight access
    // -----------------------------------------------------------------------

    /// Load a block's weights from disk (mmap) into GPU.
    /// Pre-computes HWIO permutations for conv weights in this block.
    fn load_block(&mut self, prefix: &str) -> Result<()> {
        if self.all_on_gpu { return Ok(()); }
        self.block_cache.clear();
        let prefix_dot = format!("{prefix}.");
        let ckpt_prefix = "model.diffusion_model.";
        let ckpt_prefix_dot = format!("{ckpt_prefix}{prefix_dot}");
        let raw = load_file_filtered(&self.model_path, &self.device, |key| {
            key.starts_with(&prefix_dot) || key.starts_with(&ckpt_prefix_dot)
        })?;
        // Strip "model.diffusion_model." prefix if present, convert to BF16
        let mut stripped = HashMap::with_capacity(raw.len());
        for (key, val) in raw {
            let k = key.strip_prefix(ckpt_prefix).unwrap_or(&key).to_string();
            let v = if val.dtype() != DType::BF16 { val.to_dtype(DType::BF16)? } else { val };
            stripped.insert(k, v);
        }

        // Pre-compute HWIO weight permutations for conv weights in this block
        let conv_keys: Vec<String> = stripped
            .keys()
            .filter(|k| k.ends_with(".weight") && stripped[k.as_str()].shape().dims().len() == 4)
            .cloned()
            .collect();
        for key in &conv_keys {
            let w = &stripped[key.as_str()];
            let w_f32 = if w.dtype() != DType::F32 { w.to_dtype(DType::F32)? } else { w.clone_result()? };
            let w_hwio = GpuOps::weight_ocickhkw_to_khwkicoc(&w_f32)?.to_dtype(DType::BF16)?;
            let hwio_key = key.replace(".weight", ".weight_hwio");
            stripped.insert(hwio_key, w_hwio);
        }

        println!(
            "    [offload] Loaded {} tensors for {prefix} ({} conv HWIO pre-computed)",
            stripped.len(), conv_keys.len()
        );
        self.block_cache = stripped;
        Ok(())
    }

    /// Drop current block weights to free VRAM.
    fn unload_block(&mut self) {
        if self.all_on_gpu { return; }
        self.block_cache.clear();
    }

    /// Get a weight tensor by key — checks block_cache first, then resident.
    fn w(&self, key: &str) -> Result<&Tensor> {
        self.block_cache
            .get(key)
            .or_else(|| self.resident.get(key))
            .ok_or_else(|| Error::InvalidInput(format!("Missing weight key: {key}")))
    }

    /// Check if a weight key exists in block_cache or resident.
    fn has_key(&self, key: &str) -> bool {
        self.block_cache.contains_key(key) || self.resident.contains_key(key)
    }

    // -----------------------------------------------------------------------
    // Linear helpers
    // -----------------------------------------------------------------------

    /// x @ weight.T  (weight shape: [out, in], no bias)
    fn linear_no_bias(&self, x: &Tensor, weight_key: &str) -> Result<Tensor> {
        let weight = self.w(weight_key)?;
        let x_dims = x.shape().dims().to_vec();
        let in_features = *x_dims.last().unwrap();
        let batch: usize = x_dims[..x_dims.len() - 1].iter().product();
        let out_features = weight.shape().dims()[0];

        let x_2d = x.reshape(&[batch, in_features])?;
        let wt = transpose_2d(weight)?;
        let out_2d = x_2d.matmul(&wt)?;

        let mut out_shape = x_dims[..x_dims.len() - 1].to_vec();
        out_shape.push(out_features);
        out_2d.reshape(&out_shape)
    }

    /// x @ weight.T + bias
    fn linear_with_bias(&self, x: &Tensor, weight_key: &str, bias_key: &str) -> Result<Tensor> {
        let out = self.linear_no_bias(x, weight_key)?;
        let bias = self.w(bias_key)?;
        let out_dims = out.shape().dims().to_vec();
        let batch: usize = out_dims[..out_dims.len() - 1].iter().product();
        let out_feat = *out_dims.last().unwrap();
        let bias_1d = bias.reshape(&[1, out_feat])?;
        let out_2d = out.reshape(&[batch, out_feat])?;
        let result_2d = out_2d.add(&bias_1d)?;
        result_2d.reshape(&out_dims)
    }

    // -----------------------------------------------------------------------
    // Timestep embedding
    // -----------------------------------------------------------------------

    /// Sinusoidal timestep embedding: t -> [B, model_channels]
    fn timestep_embedding(&self, t: &Tensor) -> Result<Tensor> {
        let dim = self.config.model_channels; // 320
        let half = dim / 2;
        let max_period: f32 = 10000.0;

        let t_data = t.to_vec()?;
        let batch = t_data.len();

        // Build [cos, sin] embedding
        let mut emb_data = vec![0.0f32; batch * dim];
        for b in 0..batch {
            let t_val = t_data[b];
            for i in 0..half {
                let freq = (-f32::ln(max_period) * (i as f32) / (half as f32)).exp();
                let angle = t_val * freq;
                emb_data[b * dim + i] = angle.cos();
                emb_data[b * dim + half + i] = angle.sin();
            }
        }

        Tensor::from_vec_dtype(
            emb_data,
            Shape::from_dims(&[batch, dim]),
            self.device.clone(),
            DType::BF16,
        )
    }

    /// Full time embedding: sinusoidal -> time_embed MLP -> [B, emb_ch]
    fn time_embed(&self, t: &Tensor) -> Result<Tensor> {
        let t_emb = self.timestep_embedding(t)?;
        // time_embed.0: Linear(320, 1280)
        let h = self.linear_with_bias(&t_emb, "time_embed.0.weight", "time_embed.0.bias")?;
        let h = h.silu()?;
        // time_embed.2: Linear(1280, 1280)
        self.linear_with_bias(&h, "time_embed.2.weight", "time_embed.2.bias")
    }

    /// Label/vector embedding (SDXL ADM): y -> [B, emb_ch]
    fn label_embed(&self, y: &Tensor) -> Result<Tensor> {
        // label_emb.0.0: Linear(2816, 1280)
        let h = self.linear_with_bias(y, "label_emb.0.0.weight", "label_emb.0.0.bias")?;
        let h = h.silu()?;
        // label_emb.0.2: Linear(1280, 1280)
        self.linear_with_bias(&h, "label_emb.0.2.weight", "label_emb.0.2.bias")
    }

    // -----------------------------------------------------------------------
    // Conv2d helper — uses pre-computed HWIO weights from HashMap
    // -----------------------------------------------------------------------

    /// Execute a Conv2d operation directly using pre-computed HWIO weight.
    /// Weight OIHW->HWIO permutation is done once at load time, not per-call.
    fn conv2d_forward(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let hwio_key = format!("{prefix}.weight_hwio");

        // Use pre-computed HWIO weight (computed at load time).
        // Fallback to on-the-fly permutation only if missing (shouldn't happen).
        let w_oihw;
        let w_hwio_ref: &Tensor = if let Ok(cached) = self.w(&hwio_key) {
            cached
        } else {
            let w = self.w(&format!("{prefix}.weight"))?;
            let w_f32_tmp = if w.dtype() != DType::F32 { w.to_dtype(DType::F32)? } else { w.clone_result()? };
            w_oihw = GpuOps::weight_ocickhkw_to_khwkicoc(&w_f32_tmp)?.to_dtype(DType::BF16)?;
            &w_oihw
        };

        // Determine stride/padding from the HWIO weight shape
        let hwio_dims = w_hwio_ref.shape().dims();
        let kh = hwio_dims[0]; // HWIO: first dim is kernel height

        let (stride, padding) = if kh == 1 {
            (1, 0)
        } else if prefix.contains(".op") {
            (2, 1) // downsample
        } else {
            (1, 1) // standard 3x3
        };

        let bias = self.w(&format!("{prefix}.bias")).ok();

        // NCHW -> NHWC (GPU kernel), conv, NHWC -> NCHW (GPU kernel)
        let x_nhwc = to_nhwc(x)?;
        let out_nhwc = flame_core::cuda_ops_bf16::conv2d_bf16(
            &x_nhwc,
            w_hwio_ref,
            bias,
            (stride as i32, stride as i32),
            (padding as i32, padding as i32),
            (1, 1),
            1, // groups
            flame_core::cuda_ops_bf16::ConvActivation::None,
        )?;
        to_nchw(&out_nhwc)
    }

    // -----------------------------------------------------------------------
    // ResBlock
    // -----------------------------------------------------------------------

    /// ResBlock forward with time embedding injection.
    ///
    /// Key prefix: e.g. "input_blocks.1.0" or "middle_block.0"
    ///
    /// Structure:
    ///   in_layers:  GroupNorm(32) -> SiLU -> Conv3x3
    ///   emb_layers: SiLU -> Linear(emb_ch, out_ch)
    ///   out_layers: GroupNorm(32) -> SiLU -> Dropout(0) -> Conv3x3
    ///   skip_connection: optional 1x1 Conv (when in_ch != out_ch)
    fn resblock(&self, x: &Tensor, emb: &Tensor, prefix: &str) -> Result<Tensor> {
        // in_layers: GroupNorm(32) -> SiLU -> Conv3x3
        let gn1_w = self.w(&format!("{prefix}.in_layers.0.weight"))?;
        let gn1_b = self.w(&format!("{prefix}.in_layers.0.bias"))?;
        let h = group_norm_nchw(x, 32, Some(gn1_w), Some(gn1_b), 1e-5)?;
        let h = h.silu()?;
        let h = self.conv2d_forward(&h, &format!("{prefix}.in_layers.2"))?;

        // emb_layers: SiLU -> Linear(emb_ch, out_ch)
        let emb_h = emb.silu()?;
        let emb_out = self.linear_with_bias(
            &emb_h,
            &format!("{prefix}.emb_layers.1.weight"),
            &format!("{prefix}.emb_layers.1.bias"),
        )?;
        // emb_out: [B, out_ch] -> [B, out_ch, 1, 1] for spatial broadcast
        let emb_out = emb_out.unsqueeze(2)?.unsqueeze(3)?;
        let h = h.add(&emb_out)?;

        // out_layers: GroupNorm(32) -> SiLU -> Dropout(0) -> Conv3x3
        let gn2_w = self.w(&format!("{prefix}.out_layers.0.weight"))?;
        let gn2_b = self.w(&format!("{prefix}.out_layers.0.bias"))?;
        let h = group_norm_nchw(&h, 32, Some(gn2_w), Some(gn2_b), 1e-5)?;
        let h = h.silu()?;
        // out_layers.3 (index 3 because of Dropout at index 2)
        let h = self.conv2d_forward(&h, &format!("{prefix}.out_layers.3"))?;

        // Skip connection (1x1 conv if channels mismatch)
        let residual = if self.has_key(&format!("{prefix}.skip_connection.weight")) {
            self.conv2d_forward(x, &format!("{prefix}.skip_connection"))?
        } else {
            x.clone_result()?
        };

        residual.add(&h)
    }

    // -----------------------------------------------------------------------
    // SpatialTransformer (CrossAttention blocks)
    // -----------------------------------------------------------------------

    /// GEGLU: Linear(dim, ff_dim*2) -> split -> x * gelu(gate)
    fn geglu(&self, x: &Tensor, weight_key: &str, bias_key: &str) -> Result<Tensor> {
        let projected = self.linear_with_bias(x, weight_key, bias_key)?;
        let dims = projected.shape().dims().to_vec();
        let last_dim = *dims.last().unwrap();
        let half_dim = last_dim / 2;

        // Split on last dimension: narrow along last axis
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let flat = projected.reshape(&[batch, last_dim])?;
        let x_part = flat.narrow(1, 0, half_dim)?;
        let gate_part = flat.narrow(1, half_dim, half_dim)?;
        let gate = gate_part.gelu()?;
        let result = x_part.mul(&gate)?;

        let mut out_shape = dims[..dims.len() - 1].to_vec();
        out_shape.push(half_dim);
        result.reshape(&out_shape)
    }

    /// Cross/self-attention.
    ///
    /// Q from x, K/V from context (or x for self-attention).
    /// prefix: e.g. "input_blocks.3.1.transformer_blocks.0.attn1"
    fn cross_attention(
        &self,
        x: &Tensor,       // [B, N, C]
        context: &Tensor,  // [B, M, ctx_dim] (or same as x for self-attn)
        prefix: &str,
    ) -> Result<Tensor> {
        let x_dims = x.shape().dims().to_vec();
        let (b, n, c) = (x_dims[0], x_dims[1], x_dims[2]);
        let num_heads = c / self.config.head_dim;
        let d = self.config.head_dim;

        // Q from x
        let q = self.linear_no_bias(x, &format!("{prefix}.to_q.weight"))?;
        // K/V from context
        let k = self.linear_no_bias(context, &format!("{prefix}.to_k.weight"))?;
        let v = self.linear_no_bias(context, &format!("{prefix}.to_v.weight"))?;

        let ctx_n = context.shape().dims()[1];

        // Reshape to [B, H, N, D] for SDPA
        let q = q.reshape(&[b, n, num_heads, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, ctx_n, num_heads, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, ctx_n, num_heads, d])?.permute(&[0, 2, 1, 3])?;

        // Scaled dot-product attention: [B, H, N, D]
        let out = sdpa_forward(&q, &k, &v, None)?;

        // Back to [B, N, H*D]
        let out = out.permute(&[0, 2, 1, 3])?.reshape(&[b, n, c])?;

        // Output projection: to_out.0
        self.linear_with_bias(
            &out,
            &format!("{prefix}.to_out.0.weight"),
            &format!("{prefix}.to_out.0.bias"),
        )
    }

    /// BasicTransformerBlock: norm1 -> self-attn -> norm2 -> cross-attn -> norm3 -> FF
    ///
    /// prefix: e.g. "input_blocks.3.1.transformer_blocks.0"
    fn basic_transformer_block(
        &self,
        x: &Tensor,       // [B, N, C]
        context: &Tensor,  // [B, M, ctx_dim]
        prefix: &str,
    ) -> Result<Tensor> {
        let c = *x.shape().dims().last().unwrap();

        // Self-attention: norm1 -> attn1 (Q/K/V all from x)
        let x_norm1 = layer_norm(
            x,
            &[c],
            Some(self.w(&format!("{prefix}.norm1.weight"))?),
            Some(self.w(&format!("{prefix}.norm1.bias"))?),
            1e-5,
        )?;
        let attn1_out = self.cross_attention(&x_norm1, &x_norm1, &format!("{prefix}.attn1"))?;
        let x = x.add(&attn1_out)?;

        // Cross-attention: norm2 -> attn2 (Q from image, K/V from text context)
        let x_norm2 = layer_norm(
            &x,
            &[c],
            Some(self.w(&format!("{prefix}.norm2.weight"))?),
            Some(self.w(&format!("{prefix}.norm2.bias"))?),
            1e-5,
        )?;
        let attn2_out = self.cross_attention(&x_norm2, context, &format!("{prefix}.attn2"))?;
        let x = x.add(&attn2_out)?;

        // Feed-forward: norm3 -> GEGLU -> Linear
        let x_norm3 = layer_norm(
            &x,
            &[c],
            Some(self.w(&format!("{prefix}.norm3.weight"))?),
            Some(self.w(&format!("{prefix}.norm3.bias"))?),
            1e-5,
        )?;
        // ff.net.0.proj (GEGLU: projects to 2*ff_dim, splits, applies gelu gate)
        let ff_out = self.geglu(
            &x_norm3,
            &format!("{prefix}.ff.net.0.proj.weight"),
            &format!("{prefix}.ff.net.0.proj.bias"),
        )?;
        // ff.net.2 (Linear down-projection)
        let ff_out = self.linear_with_bias(
            &ff_out,
            &format!("{prefix}.ff.net.2.weight"),
            &format!("{prefix}.ff.net.2.bias"),
        )?;

        x.add(&ff_out)
    }

    /// SpatialTransformer: GroupNorm -> proj_in -> N x BasicTransformerBlock -> proj_out
    ///
    /// prefix: e.g. "input_blocks.3.1"
    /// SDXL uses linear proj_in/proj_out (use_linear_in_transformer=true)
    fn spatial_transformer(
        &self,
        x: &Tensor,       // [B, C, H, W] NCHW
        context: &Tensor,  // [B, M, ctx_dim]
        prefix: &str,
        depth: usize,
    ) -> Result<Tensor> {
        let x_dims = x.shape().dims().to_vec();
        let (b, c, h, w) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
        let residual = x.clone_result()?;

        // GroupNorm (eps=1e-6 matches PyTorch reference)
        let gn_w = self.w(&format!("{prefix}.norm.weight"))?;
        let gn_b = self.w(&format!("{prefix}.norm.bias"))?;
        let x_normed = group_norm_nchw(x, 32, Some(gn_w), Some(gn_b), 1e-6)?;

        // SDXL: linear proj_in (use_linear_in_transformer=true)
        // Reshape NCHW -> [B, H*W, C] first, then Linear
        let x_flat = if self.config.use_linear_in_transformer {
            let flat = x_normed.permute(&[0, 2, 3, 1])?.reshape(&[b, h * w, c])?;
            self.linear_with_bias(
                &flat,
                &format!("{prefix}.proj_in.weight"),
                &format!("{prefix}.proj_in.bias"),
            )?
        } else {
            // SD1.5: Conv2d 1x1 proj_in, then reshape
            let proj = self.conv2d_forward(&x_normed, &format!("{prefix}.proj_in"))?;
            proj.permute(&[0, 2, 3, 1])?.reshape(&[b, h * w, c])?
        };

        // N transformer blocks
        let mut hidden = x_flat;
        for j in 0..depth {
            hidden = self.basic_transformer_block(
                &hidden,
                context,
                &format!("{prefix}.transformer_blocks.{j}"),
            )?;
        }

        // proj_out and reshape back to NCHW
        let out = if self.config.use_linear_in_transformer {
            let proj = self.linear_with_bias(
                &hidden,
                &format!("{prefix}.proj_out.weight"),
                &format!("{prefix}.proj_out.bias"),
            )?;
            proj.reshape(&[b, h, w, c])?.permute(&[0, 3, 1, 2])?
        } else {
            let spatial = hidden.reshape(&[b, h, w, c])?.permute(&[0, 3, 1, 2])?;
            self.conv2d_forward(&spatial, &format!("{prefix}.proj_out"))?
        };

        residual.add(&out)
    }

    // -----------------------------------------------------------------------
    // Downsample / Upsample
    // -----------------------------------------------------------------------

    /// Downsample: Conv2d(stride=2, kernel=3, padding=1)
    /// Key: {prefix}.0.op.weight/bias
    fn downsample(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        self.conv2d_forward(x, &format!("{prefix}.op"))
    }

    /// Upsample: nearest 2x interpolation -> Conv2d(3, padding=1)
    /// Key: {prefix}.conv.weight/bias
    fn upsample(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let dims = x.shape().dims();
        let h_out = dims[2] * 2;
        let w_out = dims[3] * 2;
        let x_up = self.kernels.upsample2d_nearest(x, (h_out, w_out))?;
        self.conv2d_forward(&x_up, &format!("{prefix}.conv"))
    }

    // -----------------------------------------------------------------------
    // Full forward pass
    // -----------------------------------------------------------------------

    /// Forward pass through the SDXL UNet.
    ///
    /// - `x`: noisy latents [B, 4, H, W] in BF16
    /// - `timesteps`: discrete timesteps [B] in F32
    /// - `context`: text encoder hidden states [B, seq_len, 2048] in BF16
    /// - `y`: vector conditioning (pooled + time_ids) [B, 2816] in BF16
    ///
    /// Returns: predicted noise [B, 4, H, W] in BF16
    pub fn forward(
        &mut self,
        x: &Tensor,
        timesteps: &Tensor,
        context: &Tensor,
        y: &Tensor,
    ) -> Result<Tensor> {
        // 1. Time embedding + label embedding -> [B, 1280]
        let emb = self.time_embed(timesteps)?;
        let label_emb = self.label_embed(y)?;
        let emb = emb.add(&label_emb)?;

        // 2. Input blocks — store hidden states for skip connections
        let mut hs: Vec<Tensor> = Vec::new();
        let mut h = x.clone_result()?;

        let num_input_blocks = self.input_block_descs.len();
        for block_idx in 0..num_input_blocks {
            let desc = self.input_block_descs[block_idx].clone();
            let prefix = format!("input_blocks.{block_idx}");

            match desc {
                BlockType::ConvIn => {
                    // conv_in is in resident weights: input_blocks.0.0
                    h = self.conv2d_forward(&h, &format!("{prefix}.0"))?;
                }
                BlockType::ResBlockAttn {
                    transformer_depth, ..
                } => {
                    self.load_block(&prefix)?;
                    h = self.resblock(&h, &emb, &format!("{prefix}.0"))?;
                    if transformer_depth > 0 {
                        h = self.spatial_transformer(
                            &h,
                            context,
                            &format!("{prefix}.1"),
                            transformer_depth,
                        )?;
                    }
                    self.unload_block();
                }
                BlockType::Downsample { .. } => {
                    self.load_block(&prefix)?;
                    // Downsample is at sub-index 0 within this block
                    h = self.downsample(&h, &format!("{prefix}.0"))?;
                    self.unload_block();
                }
                _ => unreachable!("Input block has invalid descriptor type"),
            }

            hs.push(h.clone_result()?);
        }

        // 3. Middle block: ResBlock(0) + SpatialTransformer(1) + ResBlock(2)
        self.load_block("middle_block")?;
        h = self.resblock(&h, &emb, "middle_block.0")?;
        if self.config.transformer_depth_middle > 0 {
            h = self.spatial_transformer(
                &h,
                context,
                "middle_block.1",
                self.config.transformer_depth_middle,
            )?;
        }
        h = self.resblock(&h, &emb, "middle_block.2")?;
        self.unload_block();

        // 4. Output blocks — pop skip connections and concat
        let num_output_blocks = self.output_block_descs.len();
        for block_idx in 0..num_output_blocks {
            let desc = self.output_block_descs[block_idx].clone();
            let prefix = format!("output_blocks.{block_idx}");

            // Pop skip connection from encoder and concat along channel dim (dim=1 in NCHW)
            let skip = hs.pop().ok_or_else(|| {
                Error::InvalidInput("Ran out of skip connections".into())
            })?;
            h = Tensor::cat(&[&h, &skip], 1)?;

            match desc {
                BlockType::OutputResBlockAttn {
                    transformer_depth,
                    has_upsample,
                    ..
                } => {
                    self.load_block(&prefix)?;

                    // ResBlock (handles channel mismatch via skip_connection 1x1 conv)
                    h = self.resblock(&h, &emb, &format!("{prefix}.0"))?;

                    // SpatialTransformer (if present, at sub-index 1)
                    if transformer_depth > 0 {
                        h = self.spatial_transformer(
                            &h,
                            context,
                            &format!("{prefix}.1"),
                            transformer_depth,
                        )?;
                    }

                    // Upsample (if present, at the last sub-index)
                    if has_upsample {
                        // Sub-index: 1 if no transformer, 2 if transformer present
                        let up_idx = if transformer_depth > 0 { 2 } else { 1 };
                        h = self.upsample(&h, &format!("{prefix}.{up_idx}"))?;
                    }

                    self.unload_block();
                }
                _ => unreachable!("Output block has invalid descriptor type"),
            }
        }

        // 5. Final output: out.0 (GroupNorm) -> SiLU -> out.2 (Conv2d)
        let out_gn_w = self.w("out.0.weight")?;
        let out_gn_b = self.w("out.0.bias")?;
        h = group_norm_nchw(&h, 32, Some(out_gn_w), Some(out_gn_b), 1e-5)?;
        h = h.silu()?;
        h = self.conv2d_forward(&h, "out.2")?;

        Ok(h)
    }

    // -----------------------------------------------------------------------
    // Factory methods
    // -----------------------------------------------------------------------

    /// Load all resident (always-on-GPU) weights from a safetensors file.
    ///
    /// Resident weights are small enough to keep permanently:
    /// time_embed.*, label_emb.*, input_blocks.0.0.* (conv_in), out.*
    pub fn load_resident(
        path: &str,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<HashMap<String, Tensor>> {
        // SDXL combined checkpoints prefix UNet keys with "model.diffusion_model."
        // Strip this prefix so internal code can reference "time_embed.*" etc.
        let prefix = "model.diffusion_model.";
        let raw = load_file_filtered(path, device, |key| {
            let k = key.strip_prefix(prefix).unwrap_or(key);
            k.starts_with("time_embed.")
                || k.starts_with("label_emb.")
                || k.starts_with("input_blocks.0.0.")
                || k.starts_with("out.")
        })?;
        let mut stripped = HashMap::with_capacity(raw.len());
        for (key, val) in raw {
            let k = key.strip_prefix(prefix).unwrap_or(&key).to_string();
            let v = if val.dtype() != DType::BF16 { val.to_dtype(DType::BF16)? } else { val };
            stripped.insert(k, v);
        }
        // Pre-compute HWIO permutations for resident conv weights (conv_in, out.2)
        let conv_keys: Vec<String> = stripped
            .keys()
            .filter(|k| k.ends_with(".weight") && stripped[k.as_str()].shape().dims().len() == 4)
            .cloned()
            .collect();
        for key in &conv_keys {
            let w = &stripped[key.as_str()];
            let w_f32 = if w.dtype() != DType::F32 { w.to_dtype(DType::F32)? } else { w.clone_result()? };
            let w_hwio = GpuOps::weight_ocickhkw_to_khwkicoc(&w_f32)?.to_dtype(DType::BF16)?;
            let hwio_key = key.replace(".weight", ".weight_hwio");
            stripped.insert(hwio_key, w_hwio);
        }
        Ok(stripped)
    }

    /// Convenience constructor: loads resident weights only (block offloading mode).
    pub fn from_safetensors(
        path: &str,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let resident = Self::load_resident(path, device)?;
        println!(
            "[SDXLUNet] Loaded {} resident weight tensors",
            resident.len()
        );
        Self::new(path.to_string(), resident, device.clone())
    }

    /// Load ALL UNet weights onto GPU (no block offloading). ~5GB VRAM.
    /// Expects a pre-extracted BF16 safetensors file with stripped keys.
    ///
    /// Pre-computes HWIO weight permutations for all conv layers so
    /// `conv2d_forward` never has to permute weights at inference time.
    pub fn from_safetensors_all_gpu(
        path: &str,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let mut all_weights = flame_core::serialization::load_file(
            std::path::Path::new(path), device,
        )?;
        println!("[SDXLUNet] Loaded {} weight tensors (all-on-GPU)", all_weights.len());

        // Pre-compute HWIO weight permutations for every conv weight.
        // OIHW [OC,IC,KH,KW] -> HWIO [KH,KW,IC,OC] — done once, used every forward pass.
        let conv_keys: Vec<String> = all_weights
            .keys()
            .filter(|k| k.ends_with(".weight") && {
                let dims = all_weights[k.as_str()].shape().dims();
                dims.len() == 4 // conv weights are 4D [O,I,H,W]
            })
            .cloned()
            .collect();

        let mut hwio_count = 0;
        for key in &conv_keys {
            let w = &all_weights[key.as_str()];
            let w_f32 = if w.dtype() != DType::F32 { w.to_dtype(DType::F32)? } else { w.clone_result()? };
            let w_hwio = GpuOps::weight_ocickhkw_to_khwkicoc(&w_f32)?.to_dtype(DType::BF16)?;
            let hwio_key = key.replace(".weight", ".weight_hwio");
            all_weights.insert(hwio_key, w_hwio);
            hwio_count += 1;
        }
        println!("[SDXLUNet] Pre-computed {} HWIO weight permutations", hwio_count);

        let config = SDXLConfig::default();
        let kernels = CudaKernels::new(device.clone())?;
        let mut unet = Self {
            config,
            resident: all_weights,
            model_path: path.to_string(),
            device: device.clone(),
            block_cache: HashMap::new(),
            kernels,
            input_block_descs: Vec::new(),
            output_block_descs: Vec::new(),
            input_block_channels: Vec::new(),
            all_on_gpu: true,
        };
        unet.build_block_descriptors();
        Ok(unet)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = SDXLConfig::default();
        assert_eq!(config.model_channels, 320);
        assert_eq!(config.channel_mult, vec![1, 2, 4]);
        assert_eq!(config.context_dim, 2048);
        assert_eq!(config.adm_in_channels, 2816);
        assert_eq!(config.transformer_depth_input, vec![0, 0, 2, 2, 10, 10]);
        assert_eq!(config.transformer_depth_middle, 10);
        assert_eq!(
            config.transformer_depth_output,
            vec![10, 10, 10, 2, 2, 2, 0, 0, 0]
        );
    }

    #[test]
    fn test_block_descriptor_counts() {
        // Verify the flat indexing produces the right number of blocks.
        // Input blocks: 1 (conv_in) + 3 levels * 2 res_blocks + 2 downsamples = 9
        //   Level 0: 2 res + 1 down = 3 blocks
        //   Level 1: 2 res + 1 down = 3 blocks
        //   Level 2: 2 res           = 2 blocks
        //   Total: 1 + 3 + 3 + 2 = 9
        let expected_input = 9;

        // Output blocks: 3 levels * (num_res_blocks + 1) = 3 * 3 = 9
        let expected_output = 9;

        // Build descriptors manually (same logic as build_block_descriptors)
        let config = SDXLConfig::default();
        let mc = config.model_channels;

        let mut td_input: Vec<usize> = config.transformer_depth_input.iter().copied().rev().collect();
        let mut descs_input = Vec::new();
        let mut input_channels = Vec::new();

        descs_input.push(BlockType::ConvIn);
        input_channels.push(mc);

        let mut ch = mc;
        for (level, &mult) in config.channel_mult.iter().enumerate() {
            let out_ch = mc * mult;
            for _ in 0..config.num_res_blocks {
                let num_t = td_input.pop().unwrap_or(0);
                descs_input.push(BlockType::ResBlockAttn {
                    in_ch: ch,
                    out_ch,
                    transformer_depth: num_t,
                });
                ch = out_ch;
                input_channels.push(ch);
            }
            if level < config.channel_mult.len() - 1 {
                descs_input.push(BlockType::Downsample { channels: ch });
                input_channels.push(ch);
            }
        }
        assert_eq!(descs_input.len(), expected_input);

        // Output blocks
        let mut td_output: Vec<usize> = config.transformer_depth_output.iter().copied().rev().collect();
        let mut descs_output = Vec::new();
        ch = mc * config.channel_mult[config.channel_mult.len() - 1];
        let num_levels = config.channel_mult.len();

        for level in (0..num_levels).rev() {
            let out_ch = mc * config.channel_mult[level];
            for i in 0..config.num_res_blocks + 1 {
                let skip_ch = input_channels.pop().unwrap_or(0);
                let in_ch = ch + skip_ch;
                let num_t = td_output.pop().unwrap_or(0);
                let has_up = level > 0 && i == config.num_res_blocks;
                descs_output.push(BlockType::OutputResBlockAttn {
                    in_ch,
                    out_ch,
                    transformer_depth: num_t,
                    has_upsample: has_up,
                });
                ch = out_ch;
            }
        }
        assert_eq!(descs_output.len(), expected_output);
    }

    #[test]
    fn test_input_channel_tracking() {
        // Verify channel tracking matches SDXL architecture
        let config = SDXLConfig::default();
        let mc = config.model_channels; // 320

        // Expected input block channels (stored for skip connections):
        //   Block 0 (conv_in):       320
        //   Block 1 (res, 320->320):  320
        //   Block 2 (res, 320->320):  320
        //   Block 3 (downsample):     320
        //   Block 4 (res, 320->640):  640
        //   Block 5 (res, 640->640):  640
        //   Block 6 (downsample):     640
        //   Block 7 (res, 640->1280): 1280
        //   Block 8 (res, 1280->1280):1280
        let expected = vec![320, 320, 320, 320, 640, 640, 640, 1280, 1280];

        let mut channels = Vec::new();
        channels.push(mc); // conv_in

        let mut ch = mc;
        for (level, &mult) in config.channel_mult.iter().enumerate() {
            let out_ch = mc * mult;
            for _ in 0..config.num_res_blocks {
                ch = out_ch;
                channels.push(ch);
            }
            if level < config.channel_mult.len() - 1 {
                channels.push(ch);
            }
        }

        assert_eq!(channels, expected);
    }

    #[test]
    fn test_output_block_in_channels() {
        // Verify output block input channels (after skip concat) match expected values
        let config = SDXLConfig::default();
        let mc = config.model_channels;
        let input_ch = vec![320usize, 320, 320, 320, 640, 640, 640, 1280, 1280];
        let mut ic = input_ch.clone();

        let mut expected_in_ch = Vec::new();
        let num_levels = config.channel_mult.len();
        let mut ch = mc * config.channel_mult[num_levels - 1]; // 1280

        for level in (0..num_levels).rev() {
            let out_ch = mc * config.channel_mult[level];
            for _ in 0..config.num_res_blocks + 1 {
                let skip = ic.pop().unwrap();
                expected_in_ch.push(ch + skip);
                ch = out_ch;
            }
        }

        // output_blocks.0: 1280 + 1280 = 2560
        // output_blocks.1: 1280 + 1280 = 2560
        // output_blocks.2: 1280 + 640  = 1920
        // output_blocks.3: 1280 + 640  = 1920  (ch was set to 1280 from out_ch of level 2)
        // Wait, after level 2 completes, ch = out_ch of level 2 = 1280.
        // Then level 1 starts: out_ch = 640.
        // output_blocks.3: ch=1280, skip=640 -> 1920, then ch becomes 640
        // output_blocks.4: ch=640, skip=640 -> 1280
        // output_blocks.5: ch=640, skip=320 -> 960
        // Then level 0: out_ch = 320
        // output_blocks.6: ch=640, skip=320 -> 960, then ch becomes 320
        // output_blocks.7: ch=320, skip=320 -> 640
        // output_blocks.8: ch=320, skip=320 -> 640
        let expected = vec![2560, 2560, 1920, 1920, 1280, 960, 960, 640, 640];
        assert_eq!(expected_in_ch, expected);
    }
}
