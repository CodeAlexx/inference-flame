//! Stable Diffusion 1.5 UNet — pure Rust, LDM-format weight keys.
//!
//! Architecture: Standard LDM UNet for SD 1.5.
//! - model_channels=320, channel_mult=(1,2,4,4), num_res_blocks=2
//! - context_dim=768 (CLIP-L only)
//! - NO label_emb / ADM conditioning (unlike SDXL)
//! - transformer_depth_input  = [1,1,1,1,1,1,0,0]
//! - transformer_depth_middle = 1
//! - transformer_depth_output = [0,0,0,1,1,1,1,1,1,1,1,1]
//! - use_linear_in_transformer=false (Conv2d 1x1 proj_in/proj_out, NOT Linear)
//! - **num_heads=8 (constant across levels), head_dim = channels / 8**
//!   (40 at level 0, 80 at L1, 160 at L2/L3/mid). The diffusers config
//!   stores `"attention_head_dim": 8`, but for SD 1.5 that field is
//!   interpreted as **num_heads**, not head_dim — verified against the
//!   actual `UNet2DConditionModel` runtime (`attn1.heads == 8`).
//!
//! SD 1.5 diffusers weight keys (what's on disk in unet/diffusion_pytorch_model.safetensors):
//!   conv_in.{weight,bias}                             (320,4,3,3)
//!   conv_norm_out.{weight,bias}                       (320,)
//!   conv_out.{weight,bias}                            (4,320,3,3)
//!   time_embedding.linear_{1,2}.{weight,bias}
//!   down_blocks.{0..3}.resnets.{0,1}.{conv1,conv2,norm1,norm2,time_emb_proj,(conv_shortcut)}.*
//!   down_blocks.{0..2}.attentions.{0,1}.{norm,proj_in,proj_out}.*
//!   down_blocks.{0..2}.attentions.{0,1}.transformer_blocks.0.{attn1,attn2,ff,norm{1,2,3}}.*
//!   down_blocks.{0..2}.downsamplers.0.conv.*
//!   mid_block.resnets.{0,1}.*
//!   mid_block.attentions.0.*
//!   up_blocks.{0..3}.resnets.{0,1,2}.*
//!   up_blocks.{1..3}.attentions.{0,1,2}.*
//!   up_blocks.{0..2}.upsamplers.0.conv.*
//!
//! These are remapped to the LDM key format that the rest of the code (copied from
//! SDXL) expects:
//!   time_embed.0/2, input_blocks.{0..11}.*, middle_block.{0,1,2}.*,
//!   output_blocks.{0..11}.*, out.0/2.*
//!
//! Weight loading: HashMap approach with block-level offloading via load_file_filtered.
//! NCHW layout throughout. GroupNorm converts NCHW->NHWC internally.
//! BF16 in/out, F32 compute in kernels.

use flame_core::cuda_kernels::CudaKernels;
use flame_core::cuda_ops::GpuOps;
use flame_core::group_norm::group_norm;
use flame_core::layer_norm::layer_norm;
use flame_core::serialization::load_file_filtered;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Layout helpers — GroupNorm and Conv kernel both want NHWC; UNet data flow is NCHW
// ---------------------------------------------------------------------------

fn to_nhwc(x: &Tensor) -> Result<Tensor> {
    GpuOps::permute_nchw_to_nhwc(x)
}

fn to_nchw(x: &Tensor) -> Result<Tensor> {
    GpuOps::permute_nhwc_to_nchw(x)
}

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

fn transpose_2d(t: &Tensor) -> Result<Tensor> {
    t.permute(&[1, 0])
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// SD 1.5 UNet configuration.
pub struct SD15Config {
    pub in_channels: usize,
    pub out_channels: usize,
    pub model_channels: usize,
    pub channel_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub context_dim: usize,
    /// SD 1.5 has a FIXED 8 heads at every attention (head_dim varies with
    /// the channel count). Storing num_heads, deriving head_dim at call site.
    pub num_heads: usize,
    pub use_linear_in_transformer: bool,
    pub transformer_depth_input: Vec<usize>,
    pub transformer_depth_middle: usize,
    pub transformer_depth_output: Vec<usize>,
}

impl Default for SD15Config {
    fn default() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            model_channels: 320,
            channel_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            context_dim: 768,
            num_heads: 8,
            use_linear_in_transformer: false,
            transformer_depth_input: vec![1, 1, 1, 1, 1, 1, 0, 0],
            transformer_depth_middle: 1,
            transformer_depth_output: vec![0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    }
}

// ---------------------------------------------------------------------------
// Block descriptors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum BlockType {
    ConvIn,
    ResBlockAttn {
        in_ch: usize,
        out_ch: usize,
        transformer_depth: usize,
    },
    Downsample { channels: usize },
    OutputResBlockAttn {
        in_ch: usize,
        out_ch: usize,
        transformer_depth: usize,
        has_upsample: bool,
    },
}

// ---------------------------------------------------------------------------
// Diffusers → LDM key remapping
// ---------------------------------------------------------------------------

/// Return the LDM-format key for a single diffusers SD 1.5 UNet key, or None
/// if the key is not recognized (caller should retain it unchanged).
fn remap_one_sd15_key(key: &str) -> Option<String> {
    // conv_in → input_blocks.0.0
    if let Some(rest) = key.strip_prefix("conv_in.") {
        return Some(format!("input_blocks.0.0.{rest}"));
    }
    // conv_norm_out → out.0
    if let Some(rest) = key.strip_prefix("conv_norm_out.") {
        return Some(format!("out.0.{rest}"));
    }
    // conv_out → out.2
    if let Some(rest) = key.strip_prefix("conv_out.") {
        return Some(format!("out.2.{rest}"));
    }
    // time_embedding.linear_1 → time_embed.0 ; linear_2 → time_embed.2
    if let Some(rest) = key.strip_prefix("time_embedding.linear_1.") {
        return Some(format!("time_embed.0.{rest}"));
    }
    if let Some(rest) = key.strip_prefix("time_embedding.linear_2.") {
        return Some(format!("time_embed.2.{rest}"));
    }

    // Helpers for per-component suffix remapping
    fn remap_resnet_suffix(suffix: &str) -> String {
        // norm1 → in_layers.0, conv1 → in_layers.2
        // norm2 → out_layers.0, conv2 → out_layers.3
        // time_emb_proj → emb_layers.1
        // conv_shortcut → skip_connection
        if let Some(rest) = suffix.strip_prefix("norm1.") {
            return format!("in_layers.0.{rest}");
        }
        if let Some(rest) = suffix.strip_prefix("conv1.") {
            return format!("in_layers.2.{rest}");
        }
        if let Some(rest) = suffix.strip_prefix("norm2.") {
            return format!("out_layers.0.{rest}");
        }
        if let Some(rest) = suffix.strip_prefix("conv2.") {
            return format!("out_layers.3.{rest}");
        }
        if let Some(rest) = suffix.strip_prefix("time_emb_proj.") {
            return format!("emb_layers.1.{rest}");
        }
        if let Some(rest) = suffix.strip_prefix("conv_shortcut.") {
            return format!("skip_connection.{rest}");
        }
        // unrecognized — pass through
        suffix.to_string()
    }

    // down_blocks.{i}.resnets.{j}.*  →  input_blocks.{1+3*i+j}.0.{remapped}
    if let Some(rest) = key.strip_prefix("down_blocks.") {
        // parse i
        if let Some(dot1) = rest.find('.') {
            let i: usize = rest[..dot1].parse().ok()?;
            let rest2 = &rest[dot1 + 1..];
            if let Some(inner) = rest2.strip_prefix("resnets.") {
                let dot2 = inner.find('.')?;
                let j: usize = inner[..dot2].parse().ok()?;
                let suffix = &inner[dot2 + 1..];
                let block_idx = 1 + 3 * i + j; // 1,2,4,5,7,8,10,11
                let new_suffix = remap_resnet_suffix(suffix);
                return Some(format!("input_blocks.{block_idx}.0.{new_suffix}"));
            }
            if let Some(inner) = rest2.strip_prefix("attentions.") {
                let dot2 = inner.find('.')?;
                let j: usize = inner[..dot2].parse().ok()?;
                let suffix = &inner[dot2 + 1..];
                let block_idx = 1 + 3 * i + j; // 1,2,4,5,7,8
                return Some(format!("input_blocks.{block_idx}.1.{suffix}"));
            }
            if let Some(inner) = rest2.strip_prefix("downsamplers.0.conv.") {
                // down_blocks.i.downsamplers.0.conv.* → input_blocks.{3*(i+1)}.0.op.*
                let block_idx = 3 * (i + 1); // 3,6,9
                return Some(format!("input_blocks.{block_idx}.0.op.{inner}"));
            }
        }
    }

    // mid_block.resnets.{0,1}.* → middle_block.{0,2}.{remapped}
    if let Some(rest) = key.strip_prefix("mid_block.resnets.") {
        let dot = rest.find('.')?;
        let j: usize = rest[..dot].parse().ok()?;
        let suffix = &rest[dot + 1..];
        let mb_idx = if j == 0 { 0 } else { 2 };
        let new_suffix = remap_resnet_suffix(suffix);
        return Some(format!("middle_block.{mb_idx}.{new_suffix}"));
    }
    if let Some(rest) = key.strip_prefix("mid_block.attentions.0.") {
        return Some(format!("middle_block.1.{rest}"));
    }

    // up_blocks.{i}.resnets.{j}.*  →  output_blocks.{3*i+j}.0.{remapped}
    // up_blocks.{i}.attentions.{j}.* → output_blocks.{3*i+j}.1.{suffix}  (i in 1..4)
    // up_blocks.{i}.upsamplers.0.conv.* → output_blocks.{3*i+2}.{sub}.conv.{suffix}
    //   sub = 1 when the corresponding output block has NO transformer (i.e. i==0)
    //   sub = 2 when it has a transformer (i==1 or 2).
    if let Some(rest) = key.strip_prefix("up_blocks.") {
        if let Some(dot1) = rest.find('.') {
            let i: usize = rest[..dot1].parse().ok()?;
            let rest2 = &rest[dot1 + 1..];
            if let Some(inner) = rest2.strip_prefix("resnets.") {
                let dot2 = inner.find('.')?;
                let j: usize = inner[..dot2].parse().ok()?;
                let suffix = &inner[dot2 + 1..];
                let block_idx = 3 * i + j; // 0..11
                let new_suffix = remap_resnet_suffix(suffix);
                return Some(format!("output_blocks.{block_idx}.0.{new_suffix}"));
            }
            if let Some(inner) = rest2.strip_prefix("attentions.") {
                let dot2 = inner.find('.')?;
                let j: usize = inner[..dot2].parse().ok()?;
                let suffix = &inner[dot2 + 1..];
                let block_idx = 3 * i + j;
                return Some(format!("output_blocks.{block_idx}.1.{suffix}"));
            }
            if let Some(inner) = rest2.strip_prefix("upsamplers.0.conv.") {
                // Upsampler exists only on i = 0, 1, 2 in SD 1.5.
                // output_blocks.2 has no attn (i=0) → sub-index 1
                // output_blocks.5 and .8 have attn (i=1,2) → sub-index 2
                let block_idx = 3 * i + 2; // 2, 5, 8
                let sub = if i == 0 { 1 } else { 2 };
                return Some(format!(
                    "output_blocks.{block_idx}.{sub}.conv.{inner}"
                ));
            }
        }
    }

    None
}

/// Remap a whole HashMap of diffusers SD 1.5 UNet weights to LDM format.
pub fn remap_diffusers_to_ldm_unet(w: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    let is_diffusers = w
        .keys()
        .any(|k| k.starts_with("down_blocks.") || k.starts_with("up_blocks.") || k.starts_with("mid_block."));
    if !is_diffusers {
        return w;
    }
    println!("[SD15UNet] Remapping diffusers keys to LDM format");
    let mut out = HashMap::with_capacity(w.len());
    let mut unchanged = 0usize;
    for (key, val) in w {
        match remap_one_sd15_key(&key) {
            Some(new_key) => {
                out.insert(new_key, val);
            }
            None => {
                unchanged += 1;
                out.insert(key, val);
            }
        }
    }
    if unchanged > 0 {
        println!("[SD15UNet] {unchanged} keys passed through unchanged");
    }
    out
}

// ---------------------------------------------------------------------------
// SD15UNet — weight-backed model
// ---------------------------------------------------------------------------

pub struct SD15UNet {
    config: SD15Config,
    resident: HashMap<String, Tensor>,
    model_path: String,
    device: Arc<cudarc::driver::CudaDevice>,
    block_cache: HashMap<String, Tensor>,
    kernels: CudaKernels,
    input_block_descs: Vec<BlockType>,
    output_block_descs: Vec<BlockType>,
    input_block_channels: Vec<usize>,
    all_on_gpu: bool,
}

impl SD15UNet {
    pub fn new(
        model_path: String,
        resident: HashMap<String, Tensor>,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let config = SD15Config::default();
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

    /// Build block descriptors matching the exact LDM flat indexing for SD 1.5.
    ///
    /// SD 1.5 input_blocks layout (12 blocks total):
    ///   0:  conv_in (4->320)
    ///   1:  ResBlock(320->320)  + ST  td=1
    ///   2:  ResBlock(320->320)  + ST  td=1
    ///   3:  Downsample(320)
    ///   4:  ResBlock(320->640)  + ST  td=1
    ///   5:  ResBlock(640->640)  + ST  td=1
    ///   6:  Downsample(640)
    ///   7:  ResBlock(640->1280) + ST  td=1
    ///   8:  ResBlock(1280->1280)+ ST  td=1
    ///   9:  Downsample(1280)
    ///   10: ResBlock(1280->1280)       td=0
    ///   11: ResBlock(1280->1280)       td=0
    ///
    /// SD 1.5 output_blocks layout (12 blocks total):
    ///   0:  ResBlock(1280+1280=2560->1280)            td=0
    ///   1:  ResBlock(1280+1280=2560->1280)            td=0
    ///   2:  ResBlock(1280+1280=2560->1280) + Up       td=0
    ///   3:  ResBlock(1280+1280=2560->1280) + ST       td=1
    ///   4:  ResBlock(1280+1280=2560->1280) + ST       td=1
    ///   5:  ResBlock(1280+640=1920->1280)  + ST + Up  td=1
    ///   6:  ResBlock(1280+640=1920->640)   + ST       td=1
    ///   7:  ResBlock(640+640=1280->640)    + ST       td=1
    ///   8:  ResBlock(640+320=960->640)     + ST + Up  td=1
    ///   9:  ResBlock(640+320=960->320)     + ST       td=1
    ///   10: ResBlock(320+320=640->320)     + ST       td=1
    ///   11: ResBlock(320+320=640->320)     + ST       td=1
    fn build_block_descriptors(&mut self) {
        let mc = self.config.model_channels;
        let mut td_input = self.config.transformer_depth_input.clone();
        td_input.reverse();

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
            if level < self.config.channel_mult.len() - 1 {
                self.input_block_descs
                    .push(BlockType::Downsample { channels: ch });
                self.input_block_channels.push(ch);
            }
        }

        let mut td_output = self.config.transformer_depth_output.clone();
        td_output.reverse();
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

    /// Load a block's weights from disk (mmap) into GPU. Remaps diffusers keys
    /// on the fly if present.
    fn load_block(&mut self, prefix: &str) -> Result<()> {
        if self.all_on_gpu { return Ok(()); }
        self.block_cache.clear();

        // For block-offload mode we don't precompute the inverse remap, so we
        // load the *whole* file via filter that matches either format. In
        // practice this path is only used via `from_safetensors` which we
        // don't exercise in the sd15 inference bin — but we still handle it
        // defensively. Cheapest correct behavior: filter against the
        // (LDM) prefix — callers always request LDM prefixes — and do key
        // remap after load.
        let prefix_dot = format!("{prefix}.");
        // Best-effort: match anything that *could* be the raw diffusers key
        // for this LDM prefix. Since the mapping is many-to-one we let it
        // load a superset and filter/remap afterwards.
        let raw = load_file_filtered(&self.model_path, &self.device, |_| true)?;
        let remapped = remap_diffusers_to_ldm_unet(raw);

        let mut stripped = HashMap::new();
        for (key, val) in remapped {
            if !key.starts_with(&prefix_dot) {
                continue;
            }
            let v = if val.dtype() != DType::BF16 { val.to_dtype(DType::BF16)? } else { val };
            stripped.insert(key, v);
        }

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

    fn unload_block(&mut self) {
        if self.all_on_gpu { return; }
        self.block_cache.clear();
    }

    fn w(&self, key: &str) -> Result<&Tensor> {
        self.block_cache
            .get(key)
            .or_else(|| self.resident.get(key))
            .ok_or_else(|| Error::InvalidInput(format!("Missing weight key: {key}")))
    }

    fn has_key(&self, key: &str) -> bool {
        self.block_cache.contains_key(key) || self.resident.contains_key(key)
    }

    // -----------------------------------------------------------------------
    // Linear helpers
    // -----------------------------------------------------------------------

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

    /// Sinusoidal timestep embedding.
    ///
    /// SD 1.5 UNet config: `flip_sin_to_cos=True, freq_shift=0`. Diffusers
    /// computes frequencies as `exp(-ln(max_period) * arange(half) / half)`
    /// (with `downscale_freq_shift=0` so the denominator is plain `half`).
    /// With `flip_sin_to_cos=True`, the first half of the output is cos,
    /// the second half is sin. Matches the LDM/SDXL convention.
    fn timestep_embedding(&self, t: &Tensor) -> Result<Tensor> {
        let dim = self.config.model_channels;
        let half = dim / 2;
        let max_period: f32 = 10000.0;

        let t_data = t.to_vec()?;
        let batch = t_data.len();

        let mut emb_data = vec![0.0f32; batch * dim];
        for b in 0..batch {
            let t_val = t_data[b];
            for i in 0..half {
                let freq = (-f32::ln(max_period) * (i as f32) / (half as f32)).exp();
                let angle = t_val * freq;
                // flip_sin_to_cos=True ⇒ first half cos, second half sin
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

    fn time_embed(&self, t: &Tensor) -> Result<Tensor> {
        let t_emb = self.timestep_embedding(t)?;
        let h = self.linear_with_bias(&t_emb, "time_embed.0.weight", "time_embed.0.bias")?;
        let h = h.silu()?;
        self.linear_with_bias(&h, "time_embed.2.weight", "time_embed.2.bias")
    }

    // -----------------------------------------------------------------------
    // Conv2d helper
    // -----------------------------------------------------------------------

    fn conv2d_forward(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let hwio_key = format!("{prefix}.weight_hwio");

        let w_oihw;
        let w_hwio_ref: &Tensor = if let Ok(cached) = self.w(&hwio_key) {
            cached
        } else {
            let w = self.w(&format!("{prefix}.weight"))?;
            let w_f32_tmp = if w.dtype() != DType::F32 { w.to_dtype(DType::F32)? } else { w.clone_result()? };
            w_oihw = GpuOps::weight_ocickhkw_to_khwkicoc(&w_f32_tmp)?.to_dtype(DType::BF16)?;
            &w_oihw
        };

        let hwio_dims = w_hwio_ref.shape().dims();
        let kh = hwio_dims[0];

        let (stride, padding) = if kh == 1 {
            (1, 0)
        } else if prefix.contains(".op") {
            (2, 1)
        } else {
            (1, 1)
        };

        let bias = self.w(&format!("{prefix}.bias")).ok();

        let x_nhwc = to_nhwc(x)?;
        let out_nhwc = flame_core::cuda_ops_bf16::conv2d_bf16(
            &x_nhwc,
            w_hwio_ref,
            bias,
            (stride as i32, stride as i32),
            (padding as i32, padding as i32),
            (1, 1),
            1,
            flame_core::cuda_ops_bf16::ConvActivation::None,
        )?;
        to_nchw(&out_nhwc)
    }

    // -----------------------------------------------------------------------
    // ResBlock
    // -----------------------------------------------------------------------

    fn resblock(&self, x: &Tensor, emb: &Tensor, prefix: &str) -> Result<Tensor> {
        let gn1_w = self.w(&format!("{prefix}.in_layers.0.weight"))?;
        let gn1_b = self.w(&format!("{prefix}.in_layers.0.bias"))?;
        let h = group_norm_nchw(x, 32, Some(gn1_w), Some(gn1_b), 1e-5)?;
        let h = h.silu()?;
        let h = self.conv2d_forward(&h, &format!("{prefix}.in_layers.2"))?;

        let emb_h = emb.silu()?;
        let emb_out = self.linear_with_bias(
            &emb_h,
            &format!("{prefix}.emb_layers.1.weight"),
            &format!("{prefix}.emb_layers.1.bias"),
        )?;
        let emb_out = emb_out.unsqueeze(2)?.unsqueeze(3)?;
        let h = h.add(&emb_out)?;

        let gn2_w = self.w(&format!("{prefix}.out_layers.0.weight"))?;
        let gn2_b = self.w(&format!("{prefix}.out_layers.0.bias"))?;
        let h = group_norm_nchw(&h, 32, Some(gn2_w), Some(gn2_b), 1e-5)?;
        let h = h.silu()?;
        let h = self.conv2d_forward(&h, &format!("{prefix}.out_layers.3"))?;

        let residual = if self.has_key(&format!("{prefix}.skip_connection.weight")) {
            self.conv2d_forward(x, &format!("{prefix}.skip_connection"))?
        } else {
            x.clone_result()?
        };

        let out = residual.to_dtype(DType::F32)?.add(&h.to_dtype(DType::F32)?)?;
        out.to_dtype(DType::BF16)
    }

    // -----------------------------------------------------------------------
    // SpatialTransformer
    // -----------------------------------------------------------------------

    fn geglu(&self, x: &Tensor, weight_key: &str, bias_key: &str) -> Result<Tensor> {
        let projected = self.linear_with_bias(x, weight_key, bias_key)?;
        let dims = projected.shape().dims().to_vec();
        let last_dim = *dims.last().unwrap();
        let half_dim = last_dim / 2;

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

    fn cross_attention(
        &self,
        x: &Tensor,
        context: &Tensor,
        prefix: &str,
    ) -> Result<Tensor> {
        let x_dims = x.shape().dims().to_vec();
        let (b, n, c) = (x_dims[0], x_dims[1], x_dims[2]);
        // SD 1.5: fixed num_heads=8, head_dim varies per level (40/80/160).
        let num_heads = self.config.num_heads;
        let d = c / num_heads;

        let q = self.linear_no_bias(x, &format!("{prefix}.to_q.weight"))?;
        let k = self.linear_no_bias(context, &format!("{prefix}.to_k.weight"))?;
        let v = self.linear_no_bias(context, &format!("{prefix}.to_v.weight"))?;

        let ctx_n = context.shape().dims()[1];

        let q = q.reshape(&[b, n, num_heads, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, ctx_n, num_heads, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, ctx_n, num_heads, d])?.permute(&[0, 2, 1, 3])?;

        // head_dim values for SD 1.5 (num_heads=8 fixed): {40, 80, 160}. None
        // are in {64, 96, 128}, so the flash-attention kernel is skipped and
        // `forward_bf16_fallback` handles them via cuBLASLt BF16 GEMMs.
        let out = flame_core::sdpa::forward(&q, &k, &v, None)?;

        let out = out.permute(&[0, 2, 1, 3])?.reshape(&[b, n, c])?;

        self.linear_with_bias(
            &out,
            &format!("{prefix}.to_out.0.weight"),
            &format!("{prefix}.to_out.0.bias"),
        )
    }

    fn basic_transformer_block(
        &self,
        x: &Tensor,
        context: &Tensor,
        prefix: &str,
    ) -> Result<Tensor> {
        let c = *x.shape().dims().last().unwrap();

        let x_norm1 = layer_norm(
            x,
            &[c],
            Some(self.w(&format!("{prefix}.norm1.weight"))?),
            Some(self.w(&format!("{prefix}.norm1.bias"))?),
            1e-5,
        )?;
        let attn1_out = self.cross_attention(&x_norm1, &x_norm1, &format!("{prefix}.attn1"))?;
        let x = x.to_dtype(DType::F32)?.add(&attn1_out.to_dtype(DType::F32)?)?.to_dtype(DType::BF16)?;

        let x_norm2 = layer_norm(
            &x,
            &[c],
            Some(self.w(&format!("{prefix}.norm2.weight"))?),
            Some(self.w(&format!("{prefix}.norm2.bias"))?),
            1e-5,
        )?;
        let attn2_out = self.cross_attention(&x_norm2, context, &format!("{prefix}.attn2"))?;
        let x = x.to_dtype(DType::F32)?.add(&attn2_out.to_dtype(DType::F32)?)?.to_dtype(DType::BF16)?;

        let x_norm3 = layer_norm(
            &x,
            &[c],
            Some(self.w(&format!("{prefix}.norm3.weight"))?),
            Some(self.w(&format!("{prefix}.norm3.bias"))?),
            1e-5,
        )?;
        let ff_out = self.geglu(
            &x_norm3,
            &format!("{prefix}.ff.net.0.proj.weight"),
            &format!("{prefix}.ff.net.0.proj.bias"),
        )?;
        let ff_out = self.linear_with_bias(
            &ff_out,
            &format!("{prefix}.ff.net.2.weight"),
            &format!("{prefix}.ff.net.2.bias"),
        )?;

        x.to_dtype(DType::F32)?.add(&ff_out.to_dtype(DType::F32)?)?.to_dtype(DType::BF16)
    }

    fn spatial_transformer(
        &self,
        x: &Tensor,
        context: &Tensor,
        prefix: &str,
        depth: usize,
    ) -> Result<Tensor> {
        let x_dims = x.shape().dims().to_vec();
        let (b, c, h, w) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
        let residual = x.clone_result()?;

        let gn_w = self.w(&format!("{prefix}.norm.weight"))?;
        let gn_b = self.w(&format!("{prefix}.norm.bias"))?;
        let x_normed = group_norm_nchw(x, 32, Some(gn_w), Some(gn_b), 1e-6)?;

        // SD 1.5: Conv2d 1x1 proj_in (use_linear_in_transformer=false)
        let x_flat = if self.config.use_linear_in_transformer {
            let flat = x_normed.permute(&[0, 2, 3, 1])?.reshape(&[b, h * w, c])?;
            self.linear_with_bias(
                &flat,
                &format!("{prefix}.proj_in.weight"),
                &format!("{prefix}.proj_in.bias"),
            )?
        } else {
            let proj = self.conv2d_forward(&x_normed, &format!("{prefix}.proj_in"))?;
            proj.permute(&[0, 2, 3, 1])?.reshape(&[b, h * w, c])?
        };

        let mut hidden = x_flat;
        for j in 0..depth {
            hidden = self.basic_transformer_block(
                &hidden,
                context,
                &format!("{prefix}.transformer_blocks.{j}"),
            )?;
        }

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

        residual.to_dtype(DType::F32)?.add(&out.to_dtype(DType::F32)?)?.to_dtype(DType::BF16)
    }

    // -----------------------------------------------------------------------
    // Downsample / Upsample
    // -----------------------------------------------------------------------

    fn downsample(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        self.conv2d_forward(x, &format!("{prefix}.op"))
    }

    fn upsample(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let dims = x.shape().dims();
        let h_out = dims[2] * 2;
        let w_out = dims[3] * 2;
        let x_up = self.kernels.upsample2d_nearest(x, (h_out, w_out))?;
        self.conv2d_forward(&x_up, &format!("{prefix}.conv"))
    }

    // -----------------------------------------------------------------------
    // Full forward pass (NO pooled / ADM conditioning)
    // -----------------------------------------------------------------------

    /// Forward pass through the SD 1.5 UNet.
    ///
    /// - `x`: noisy latents [B, 4, H, W] in BF16
    /// - `timesteps`: discrete timesteps [B] in F32/BF16
    /// - `context`: CLIP-L hidden states [B, 77, 768] in BF16
    ///
    /// Returns: predicted noise [B, 4, H, W] in BF16
    pub fn forward(
        &mut self,
        x: &Tensor,
        timesteps: &Tensor,
        context: &Tensor,
    ) -> Result<Tensor> {
        // 1. Time embedding only (no label_emb in SD 1.5)
        let emb = self.time_embed(timesteps)?;

        // 2. Input blocks
        let mut hs: Vec<Tensor> = Vec::new();
        let mut h = x.clone_result()?;

        let num_input_blocks = self.input_block_descs.len();
        for block_idx in 0..num_input_blocks {
            let desc = self.input_block_descs[block_idx].clone();
            let prefix = format!("input_blocks.{block_idx}");

            match desc {
                BlockType::ConvIn => {
                    h = self.conv2d_forward(&h, &format!("{prefix}.0"))?;
                }
                BlockType::ResBlockAttn { transformer_depth, .. } => {
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
                    h = self.downsample(&h, &format!("{prefix}.0"))?;
                    self.unload_block();
                }
                _ => unreachable!("Input block has invalid descriptor type"),
            }
            hs.push(h.clone_result()?);
        }

        // 3. Middle block
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

        // 4. Output blocks
        let num_output_blocks = self.output_block_descs.len();
        for block_idx in 0..num_output_blocks {
            let desc = self.output_block_descs[block_idx].clone();
            let prefix = format!("output_blocks.{block_idx}");

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
                    h = self.resblock(&h, &emb, &format!("{prefix}.0"))?;
                    if transformer_depth > 0 {
                        h = self.spatial_transformer(
                            &h,
                            context,
                            &format!("{prefix}.1"),
                            transformer_depth,
                        )?;
                    }
                    if has_upsample {
                        let up_idx = if transformer_depth > 0 { 2 } else { 1 };
                        h = self.upsample(&h, &format!("{prefix}.{up_idx}"))?;
                    }
                    self.unload_block();
                }
                _ => unreachable!("Output block has invalid descriptor type"),
            }
        }

        // 5. Final output
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

    /// Load ALL UNet weights onto GPU (no block offloading) from a diffusers-
    /// format SD 1.5 UNet safetensors. ~3.4GB in BF16.
    ///
    /// Remaps diffusers keys to LDM format, then pre-computes HWIO weight
    /// permutations for every conv layer.
    pub fn from_safetensors_all_gpu(
        path: &str,
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let raw = flame_core::serialization::load_file(
            std::path::Path::new(path), device,
        )?;
        println!("[SD15UNet] Loaded {} raw weight tensors", raw.len());

        // Remap diffusers → LDM keys
        let remapped = remap_diffusers_to_ldm_unet(raw);

        // Ensure BF16 everywhere
        let mut all_weights: HashMap<String, Tensor> = HashMap::with_capacity(remapped.len());
        for (k, v) in remapped {
            let v = if v.dtype() != DType::BF16 { v.to_dtype(DType::BF16)? } else { v };
            all_weights.insert(k, v);
        }
        println!("[SD15UNet] Prepared {} LDM-format weight tensors", all_weights.len());

        // Pre-compute HWIO weight permutations for every conv weight.
        let conv_keys: Vec<String> = all_weights
            .keys()
            .filter(|k| k.ends_with(".weight") && {
                let dims = all_weights[k.as_str()].shape().dims();
                dims.len() == 4
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
        println!("[SD15UNet] Pre-computed {hwio_count} HWIO weight permutations");

        let config = SD15Config::default();
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
        let cfg = SD15Config::default();
        assert_eq!(cfg.model_channels, 320);
        assert_eq!(cfg.channel_mult, vec![1, 2, 4, 4]);
        assert_eq!(cfg.context_dim, 768);
        assert_eq!(cfg.num_heads, 8);
        assert!(!cfg.use_linear_in_transformer);
        assert_eq!(cfg.transformer_depth_input, vec![1, 1, 1, 1, 1, 1, 0, 0]);
        assert_eq!(cfg.transformer_depth_middle, 1);
        assert_eq!(
            cfg.transformer_depth_output,
            vec![0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        );
    }

    #[test]
    fn test_block_descriptor_counts() {
        // 12 input_blocks, 12 output_blocks for SD 1.5.
        let expected_input = 12;
        let expected_output = 12;

        let config = SD15Config::default();
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
    fn test_key_remap_samples() {
        assert_eq!(
            remap_one_sd15_key("conv_in.weight").as_deref(),
            Some("input_blocks.0.0.weight")
        );
        assert_eq!(
            remap_one_sd15_key("conv_norm_out.bias").as_deref(),
            Some("out.0.bias")
        );
        assert_eq!(
            remap_one_sd15_key("conv_out.weight").as_deref(),
            Some("out.2.weight")
        );
        assert_eq!(
            remap_one_sd15_key("time_embedding.linear_1.weight").as_deref(),
            Some("time_embed.0.weight")
        );
        assert_eq!(
            remap_one_sd15_key("time_embedding.linear_2.bias").as_deref(),
            Some("time_embed.2.bias")
        );

        // down resnets
        assert_eq!(
            remap_one_sd15_key("down_blocks.0.resnets.0.conv1.weight").as_deref(),
            Some("input_blocks.1.0.in_layers.2.weight")
        );
        assert_eq!(
            remap_one_sd15_key("down_blocks.0.resnets.1.conv_shortcut.weight").as_deref(),
            Some("input_blocks.2.0.skip_connection.weight")
        );
        assert_eq!(
            remap_one_sd15_key("down_blocks.1.resnets.0.time_emb_proj.bias").as_deref(),
            Some("input_blocks.4.0.emb_layers.1.bias")
        );
        assert_eq!(
            remap_one_sd15_key("down_blocks.3.resnets.1.conv2.weight").as_deref(),
            Some("input_blocks.11.0.out_layers.3.weight")
        );

        // down attentions
        assert_eq!(
            remap_one_sd15_key("down_blocks.0.attentions.0.proj_in.weight").as_deref(),
            Some("input_blocks.1.1.proj_in.weight")
        );
        assert_eq!(
            remap_one_sd15_key("down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.weight").as_deref(),
            Some("input_blocks.8.1.transformer_blocks.0.attn1.to_q.weight")
        );

        // downsamplers
        assert_eq!(
            remap_one_sd15_key("down_blocks.0.downsamplers.0.conv.weight").as_deref(),
            Some("input_blocks.3.0.op.weight")
        );
        assert_eq!(
            remap_one_sd15_key("down_blocks.2.downsamplers.0.conv.bias").as_deref(),
            Some("input_blocks.9.0.op.bias")
        );

        // mid
        assert_eq!(
            remap_one_sd15_key("mid_block.resnets.0.conv1.weight").as_deref(),
            Some("middle_block.0.in_layers.2.weight")
        );
        assert_eq!(
            remap_one_sd15_key("mid_block.resnets.1.conv2.weight").as_deref(),
            Some("middle_block.2.out_layers.3.weight")
        );
        assert_eq!(
            remap_one_sd15_key("mid_block.attentions.0.norm.weight").as_deref(),
            Some("middle_block.1.norm.weight")
        );

        // up resnets
        assert_eq!(
            remap_one_sd15_key("up_blocks.0.resnets.0.conv1.weight").as_deref(),
            Some("output_blocks.0.0.in_layers.2.weight")
        );
        assert_eq!(
            remap_one_sd15_key("up_blocks.3.resnets.2.conv_shortcut.weight").as_deref(),
            Some("output_blocks.11.0.skip_connection.weight")
        );

        // up attentions
        assert_eq!(
            remap_one_sd15_key("up_blocks.1.attentions.0.proj_in.weight").as_deref(),
            Some("output_blocks.3.1.proj_in.weight")
        );
        assert_eq!(
            remap_one_sd15_key("up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_k.weight").as_deref(),
            Some("output_blocks.11.1.transformer_blocks.0.attn2.to_k.weight")
        );

        // upsamplers
        assert_eq!(
            remap_one_sd15_key("up_blocks.0.upsamplers.0.conv.weight").as_deref(),
            Some("output_blocks.2.1.conv.weight")
        );
        assert_eq!(
            remap_one_sd15_key("up_blocks.1.upsamplers.0.conv.weight").as_deref(),
            Some("output_blocks.5.2.conv.weight")
        );
        assert_eq!(
            remap_one_sd15_key("up_blocks.2.upsamplers.0.conv.bias").as_deref(),
            Some("output_blocks.8.2.conv.bias")
        );
    }
}
