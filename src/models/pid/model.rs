//! PiD `PidNet` forward — pure Rust, key-exact to the released SD3 ckpt.
//!
//! Source of truth (read in full, line numbers cited inline):
//!   pixeldit_official.py — PixDiT_T2I.forward (l.1318-1438 base; PidNet
//!     overrides it in pid_net.py l.265-469), MMDiTBlockT2I (l.627-682),
//!     MMDiTJointAttention (l.517-624), PiTBlock (l.416-509),
//!     RotaryAttention (l.209-277), RMSNorm (l.111-122), FeedForward (l.125-135),
//!     TimestepConditioner (l.80-108), precompute_freqs_cis_2d_ntk (l.154-193),
//!     apply_rotary_emb (l.196-206), PixelTokenEmbedder (l.335-413),
//!     get_2d_sincos_pos_embed (l.27-73), FinalLayer (l.298-307).
//!   pid_net.py — PidNet.forward (l.265-469), _run_patch_blocks (l.189-237).
//!   lq_projection_2d.py — LQProjection2D (l.97-413), ResBlock (l.74-89),
//!     SigmaAwareGatePerTokenPerDim (l.28-56).
//!
//! All math runs in BF16 to match the released bf16 ckpt and the flame
//! inference path, EXCEPT the RMSNorm/RoPE/timestep-sinusoid internals which
//! flame already upcasts to F32 internally (matching PyTorch RMSNorm's
//! `.to(float32)` at pixeldit_official.py:119 and the F32 freqs build).
//!
//! Layout conventions (flame):
//!   - `Conv2d::forward` is NCHW, BF16.
//!   - `flame_core::group_norm` is NHWC, BF16.
//!   - `upsample_nearest2d` is NCHW.
//!   - `flame_core::bf16_ops::rope_fused_bf16` is INTERLEAVED complex RoPE
//!     (pairs adjacent dims 2d,2d+1) with cos/sin tables [1, N, head_dim/2] and
//!     input [B*H, N, head_dim] — this is exactly PiD's `apply_rotary_emb`
//!     (view_as_complex on reshape(...,-1,2)). The freqs_cis interleaving of
//!     (x_freq_k, y_freq_k) per complex pair is reproduced when we build the
//!     cos/sin table (see `build_ntk_rope_2d`).
//!   - `flame_core::attention::sdpa` takes q,k,v as [B, H, N, D] and an
//!     optional additive/bool mask; scale defaults to 1/sqrt(D).

use flame_core::attention::sdpa;
use flame_core::conv::Conv2d;
use flame_core::group_norm::group_norm;
use flame_core::norm::rms_norm;
use flame_core::serialization::load_file_filtered;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PidConfig {
    pub hidden_size: usize,        // 1536
    pub num_groups: usize,         // 24   (patch-stream attention heads)
    pub head_dim: usize,           // 64   (= hidden_size / num_groups)
    pub patch_depth: usize,        // 14   (MMDiT blocks)
    pub pixel_hidden_size: usize,  // 16   (per-pixel embedding dim, NOT 64)
    pub pixel_depth: usize,        // 2    (PiT blocks)
    pub pixel_attn_hidden: usize,  // 1152
    pub pixel_num_groups: usize,   // 16   (pixel attention heads)
    pub pixel_head_dim: usize,     // 72   (= pixel_attn_hidden / pixel_num_groups)
    pub patch_size: usize,         // 16
    pub in_channels: usize,        // 3
    pub txt_embed_dim: usize,      // 2304
    pub txt_max_length: usize,     // 300
    pub text_rope_theta: f32,      // 10000.0
    pub rope_ref_grid_h: usize,    // 64 (= rope_ref_h / patch_size = 1024/16)
    pub rope_ref_grid_w: usize,    // 64
    pub lq_latent_channels: usize, // 16
    pub lq_hidden_dim: usize,      // 512
    pub lq_num_res_blocks: usize,  // 4
    pub lq_interval: usize,        // 2
    pub sr_scale: usize,           // 4
    pub latent_spatial_down: usize, // 8
    pub timestep_freq_dim: usize,  // 256
    pub timestep_max_period: f32,  // 10.0  (NOT 10000 — TimestepConditioner default)
}

impl PidConfig {
    /// Released SD3 `res2k_sr4x` distilled-4step config.
    pub fn sd3_res2k_sr4x() -> Self {
        let hidden_size = 1536;
        let num_groups = 24;
        let pixel_attn_hidden = 1152;
        let pixel_num_groups = 16;
        Self {
            hidden_size,
            num_groups,
            head_dim: hidden_size / num_groups, // 64
            patch_depth: 14,
            pixel_hidden_size: 16,
            pixel_depth: 2,
            pixel_attn_hidden,
            pixel_num_groups,
            pixel_head_dim: pixel_attn_hidden / pixel_num_groups, // 72
            patch_size: 16,
            in_channels: 3,
            txt_embed_dim: 2304,
            txt_max_length: 300,
            text_rope_theta: 10000.0,
            // rope_ref_grid = rope_ref / patch_size (pixeldit_official.py:1120-1121).
            rope_ref_grid_h: 1024 / 16,
            rope_ref_grid_w: 1024 / 16,
            lq_latent_channels: 16,
            lq_hidden_dim: 512,
            lq_num_res_blocks: 4,
            lq_interval: 2,
            sr_scale: 4,
            latent_spatial_down: 8,
            timestep_freq_dim: 256,
            timestep_max_period: 10.0,
        }
    }

    /// num_lq_outputs = ceil(patch_depth / lq_interval) (pid_net.py:131).
    pub fn num_lq_outputs(&self) -> usize {
        (self.patch_depth + self.lq_interval - 1) / self.lq_interval
    }
}

// ---------------------------------------------------------------------------
// Model — all weights resident (released ckpt ≈2.72 GB bf16, fits 24 GB
// alongside one decode). Block-streaming can be added later if 2k/4k
// activation memory demands it (mirrors sd3_mmdit::BlockLoader).
// ---------------------------------------------------------------------------

pub struct PidNet {
    pub config: PidConfig,
    weights: HashMap<String, Tensor>,
    device: Arc<cudarc::driver::CudaDevice>,
}

impl PidNet {
    pub fn new(
        weights: HashMap<String, Tensor>,
        config: PidConfig,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Self {
        Self { config, weights, device }
    }

    fn w(&self, key: &str) -> Result<&Tensor> {
        self.weights
            .get(key)
            .ok_or_else(|| Error::InvalidInput(format!("PidNet missing weight: {key}")))
    }

    fn has(&self, key: &str) -> bool {
        self.weights.contains_key(key)
    }

    // -- Linear helpers (weight stored [out, in], PyTorch nn.Linear) ----------

    /// x @ W.T   (no bias). x: [.., in] -> [.., out].
    fn linear_nb(&self, x: &Tensor, w_key: &str) -> Result<Tensor> {
        let w = self.w(w_key)?;
        let xd = x.shape().dims().to_vec();
        let in_f = *xd.last().unwrap();
        let batch: usize = xd[..xd.len() - 1].iter().product();
        let out_f = w.shape().dims()[0];
        let x2 = x.reshape(&[batch, in_f])?;
        let y2 = flame_core::ops::gemm_bf16::matmul_bf16_trans(&x2, w, false, true)?;
        let mut out_shape = xd[..xd.len() - 1].to_vec();
        out_shape.push(out_f);
        y2.reshape(&out_shape)
    }

    /// x @ W.T + b.
    fn linear_b(&self, x: &Tensor, w_key: &str, b_key: &str) -> Result<Tensor> {
        let y = self.linear_nb(x, w_key)?;
        let b = self.w(b_key)?;
        let yd = y.shape().dims().to_vec();
        let out_f = *yd.last().unwrap();
        let b1 = b.reshape(&[1, out_f])?;
        let batch: usize = yd[..yd.len() - 1].iter().product();
        let y2 = y.reshape(&[batch, out_f])?;
        y2.add(&b1)?.reshape(&yd)
    }

    // -- RMSNorm (pixeldit_official.py:111-122) -------------------------------
    // weight*( x * rsqrt(mean(x^2)+eps) ), eps=1e-6, F32 upcast handled by flame.
    fn rmsnorm(&self, x: &Tensor, w_key: &str) -> Result<Tensor> {
        let w = self.w(w_key)?;
        let dim = w.shape().dims()[0];
        rms_norm(x, &[dim], Some(w), 1e-6)
    }

    // -- apply_adaln: x*(1+scale)+shift (pixeldit_official.py:76-77) ----------
    // scale/shift broadcast over the token axis when they are [B, 1, D].
    fn apply_adaln(&self, x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
        let one_plus = scale.add_scalar(1.0)?;
        x.mul(&one_plus)?.add(shift)
    }

    // -- SwiGLU FeedForward: w2(silu(w1 x) * w3 x) (pixeldit_official.py:133-134)
    // All linears bias-free. mlp prefix e.g. "patch_blocks.0.mlp_x".
    fn swiglu_ffn(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let g = self.linear_nb(x, &format!("{prefix}.w1.weight"))?;
        let u = self.linear_nb(x, &format!("{prefix}.w3.weight"))?;
        let act = g.silu()?.mul(&u)?;
        self.linear_nb(&act, &format!("{prefix}.w2.weight"))
    }

    // =====================================================================
    // Timestep conditioner (pixeldit_official.py:80-108).
    //   emb = cat([cos(t*freqs), sin(t*freqs)])  (COS first), max_period=10.
    //   freqs = exp(-ln(mp)*arange(half)/half)
    //   t_emb = Linear(256->1536) -> SiLU -> Linear(1536->1536)
    // =====================================================================
    fn timestep_embed(&self, t: &Tensor) -> Result<Tensor> {
        let dim = self.config.timestep_freq_dim;
        let half = dim / 2;
        let mp = self.config.timestep_max_period;
        let t_vals = t.to_dtype(DType::F32)?.to_vec()?;
        let batch = t_vals.len();
        let mut emb = vec![0.0f32; batch * dim];
        for b in 0..batch {
            let tv = t_vals[b];
            for i in 0..half {
                let freq = (-mp.ln() * (i as f32) / (half as f32)).exp();
                let arg = tv * freq;
                emb[b * dim + i] = arg.cos(); // cos first
                emb[b * dim + half + i] = arg.sin();
            }
        }
        let emb = Tensor::from_vec_dtype(
            emb,
            Shape::from_dims(&[batch, dim]),
            self.device.clone(),
            DType::BF16,
        )?;
        let h = self.linear_b(&emb, "t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias")?;
        let h = h.silu()?;
        self.linear_b(&h, "t_embedder.mlp.2.weight", "t_embedder.mlp.2.bias")
    }

    // =====================================================================
    // NTK-aware 2D RoPE table for the patch (image) stream and pixel stream.
    // Reproduces precompute_freqs_cis_2d_ntk (pixeldit_official.py:154-193):
    //   dim_axis = dim//2
    //   ntk = (cur/ref)^(dim_axis/(dim_axis-2)) (if dim_axis>2 else 1)
    //   theta_h = theta*h_ntk ; theta_w = theta*w_ntk
    //   x_pos = linspace(0, scale, W); y_pos = linspace(0, scale, H); meshgrid ij
    //   freqs_w = 1/(theta_w^(arange(0,dim,4)[:dim//4]/dim))   (dim//4 entries)
    //   freqs_h likewise with theta_h
    //   x_cis = polar(1, outer(x_pos, freqs_w)); y_cis likewise
    //   freqs_cis = cat([x_cis[...,None], y_cis[...,None]], -1).reshape(L, dim//2)
    // i.e. for complex-pair k in [0,dim//4): col 2k uses x-axis freq_k @ x_pos,
    // col 2k+1 uses y-axis freq_k @ y_pos.  cos = real(freqs_cis), sin = imag.
    //
    // Returns (cos, sin) each [1, L, dim/2] BF16, where L = H*W. This is the
    // exact table layout rope_fused_bf16 expects ([1, N, half]); the interleaved
    // (x,y) packing matches apply_rotary_emb's view_as_complex pairing.
    // =====================================================================
    fn build_ntk_rope_2d(
        &self,
        dim: usize, // head_dim
        h: usize,
        w: usize,
        ref_h: usize,
        ref_w: usize,
        theta: f32,
        scale: f32,
    ) -> Result<(Tensor, Tensor)> {
        let dim_axis = dim / 2;
        let quarter = dim / 4;
        let half = dim / 2;
        let p = if dim_axis > 2 {
            (dim_axis as f64) / ((dim_axis - 2) as f64)
        } else {
            // ntk factor = 1 (no scaling) when dim_axis<=2
            0.0
        };
        let h_scale = h as f64 / ref_h as f64;
        let w_scale = w as f64 / ref_w as f64;
        let (h_theta, w_theta) = if dim_axis > 2 {
            (
                theta as f64 * h_scale.powf(p),
                theta as f64 * w_scale.powf(p),
            )
        } else {
            (theta as f64, theta as f64)
        };

        let l = h * w;
        let mut cos = vec![0.0f32; l * half];
        let mut sin = vec![0.0f32; l * half];
        // freqs_axis[k] = 1 / (theta_axis ^ (4k / dim))   (arange(0,dim,4)[k]=4k)
        for pos in 0..l {
            let py = pos / w;
            let px = pos % w;
            let x_pos = if w > 1 {
                scale as f64 * (px as f64) / ((w - 1) as f64)
            } else {
                0.0
            };
            let y_pos = if h > 1 {
                scale as f64 * (py as f64) / ((h - 1) as f64)
            } else {
                0.0
            };
            for k in 0..quarter {
                let expo = (4 * k) as f64 / dim as f64;
                let freq_w = 1.0 / w_theta.powf(expo);
                let freq_h = 1.0 / h_theta.powf(expo);
                let ang_x = x_pos * freq_w; // x axis -> even column (2k)
                let ang_y = y_pos * freq_h; // y axis -> odd column (2k+1)
                let base = pos * half;
                cos[base + 2 * k] = ang_x.cos() as f32;
                sin[base + 2 * k] = ang_x.sin() as f32;
                cos[base + 2 * k + 1] = ang_y.cos() as f32;
                sin[base + 2 * k + 1] = ang_y.sin() as f32;
            }
        }
        let cos = Tensor::from_vec_dtype(
            cos,
            Shape::from_dims(&[1, l, half]),
            self.device.clone(),
            DType::BF16,
        )?;
        let sin = Tensor::from_vec_dtype(
            sin,
            Shape::from_dims(&[1, l, half]),
            self.device.clone(),
            DType::BF16,
        )?;
        Ok((cos, sin))
    }

    // =====================================================================
    // 1D text RoPE table (pixeldit_official.py:1291-1302 fetch_pos_text).
    //   freqs = 1/(theta^(arange(0,head_dim,2)/head_dim))   (head_dim//2 entries)
    //   angles = positions[:,None] * freqs[None,:]          [L, head_dim//2]
    //   freqs_cis = polar(1, angles)
    // NOTE: this is a plain 1D RoPE — the half-dim entries are NOT interleaved
    // x/y; each complex pair k uses freq_k at the same position. apply_rotary_emb
    // still pairs adjacent (2k,2k+1) reals, so cos/sin[k] applies to pair k.
    // cos = real, sin = imag. Returns [1, L, head_dim/2] BF16.
    // =====================================================================
    fn build_text_rope_1d(&self, length: usize) -> Result<(Tensor, Tensor)> {
        let head_dim = self.config.head_dim;
        let half = head_dim / 2;
        let theta = self.config.text_rope_theta as f64;
        let mut cos = vec![0.0f32; length * half];
        let mut sin = vec![0.0f32; length * half];
        for pos in 0..length {
            for k in 0..half {
                // arange(0, head_dim, 2)[k] = 2k
                let expo = (2 * k) as f64 / head_dim as f64;
                let freq = 1.0 / theta.powf(expo);
                let ang = (pos as f64) * freq;
                cos[pos * half + k] = ang.cos() as f32;
                sin[pos * half + k] = ang.sin() as f32;
            }
        }
        let cos = Tensor::from_vec_dtype(
            cos,
            Shape::from_dims(&[1, length, half]),
            self.device.clone(),
            DType::BF16,
        )?;
        let sin = Tensor::from_vec_dtype(
            sin,
            Shape::from_dims(&[1, length, half]),
            self.device.clone(),
            DType::BF16,
        )?;
        Ok((cos, sin))
    }

    // =====================================================================
    // 2D sin/cos absolute pos embed for the pixel embedder
    // (get_2d_sincos_pos_embed / get_2d_sincos_pos_embed_from_grid,
    //  pixeldit_official.py:27-73). embed_dim = pixel_hidden_size (16).
    //   grid = meshgrid(arange(W), arange(H))  (w first), shape [2,1,H,W]
    //   emb_h = sincos_1d(D/2, grid[0]); emb_w = sincos_1d(D/2, grid[1])
    //   emb = concat([emb_h, emb_w], dim=1)  -> [H*W, D]
    //   sincos_1d(d, pos): omega = 1/10000^(arange(d/2)/(d/2));
    //     out = outer(pos, omega); emb = concat([sin(out), cos(out)])  (SIN first)
    // grid is reshaped from [2,1,H,W]; grid[0] (the meshgrid(w,h) "first" output)
    // is the W-coordinate broadcast — row-major flatten gives, per (h,w):
    //   grid[0][h,w] = w (the meshgrid w-first convention), grid[1][h,w] = h.
    // emb_h uses grid[0] = w-coordinate; emb_w uses grid[1] = h-coordinate.
    // Returns [H*W, D] F32 row-major (pos index = h*W + w).
    // =====================================================================
    fn build_pixel_pos_embed(&self, h: usize, w: usize) -> Result<Tensor> {
        let d = self.config.pixel_hidden_size; // 16
        // Each half-embed has dim d/2 = 8, itself split into d/4 sin + d/4 cos.
        let dh = d / 2; // 8
        let dq = dh / 2; // 4
        let l = h * w;
        let mut out = vec![0.0f32; l * d];
        for hh in 0..h {
            for ww in 0..w {
                let pos = hh * w + ww;
                // np.meshgrid(grid_w, grid_h) with w first => grid[0]=w-coord,
                // grid[1]=h-coord (see pixeldit_official.py:33-38).
                let coord_a = ww as f64; // grid[0] -> emb_h
                let coord_b = hh as f64; // grid[1] -> emb_w
                // emb_h: sincos_1d(d/2=8, coord_a) -> 8 values (4 sin, 4 cos)
                for k in 0..dq {
                    let omega = 1.0 / 10000f64.powf(k as f64 / dq as f64);
                    let arg_a = coord_a * omega;
                    out[pos * d + k] = arg_a.sin() as f32; // sin first
                    out[pos * d + dq + k] = arg_a.cos() as f32;
                    let arg_b = coord_b * omega;
                    out[pos * d + dh + k] = arg_b.sin() as f32;
                    out[pos * d + dh + dq + k] = arg_b.cos() as f32;
                }
            }
        }
        Tensor::from_vec_dtype(
            out,
            Shape::from_dims(&[l, d]),
            self.device.clone(),
            DType::BF16,
        )
    }

    // =====================================================================
    // MMDiTJointAttention (pixeldit_official.py:517-624), no-CP, no-mask.
    // Separate qkv_x / qkv_y (bias-free), per-head RMSNorm QK, image RoPE on
    // x-stream, text RoPE on y-stream, joint [text, image] SDPA, per-stream proj.
    //   q/k/v reshape (B,N,3,H,Hc).permute(2,0,1,3,4) -> [3,B,N,H,Hc].
    // Returns (out_x [B,Nx,C], out_y [B,Ny,C]).
    // =====================================================================
    #[allow(clippy::too_many_arguments)]
    fn joint_attention(
        &self,
        x_n: &Tensor, // [B, Nx, C]
        y_n: &Tensor, // [B, Ny, C]
        prefix: &str, // "patch_blocks.{i}.attn"
        img_cos: &Tensor,
        img_sin: &Tensor, // [1, Nx, hd/2]
        txt_cos: &Tensor,
        txt_sin: &Tensor, // [1, Ny, hd/2]
    ) -> Result<(Tensor, Tensor)> {
        let h = self.config.num_groups;
        let hd = self.config.head_dim;
        let b = x_n.shape().dims()[0];
        let nx = x_n.shape().dims()[1];
        let ny = y_n.shape().dims()[1];
        let c = self.config.hidden_size;

        // QKV projections (bias-free, qkv_x.weight [3C, C]).
        let qkv_x = self.linear_nb(x_n, &format!("{prefix}.qkv_x.weight"))?;
        let qkv_y = self.linear_nb(y_n, &format!("{prefix}.qkv_y.weight"))?;

        // reshape [B,N,3,H,Hc] then to [3,B,H,N,Hc] (permute(2,0,3,1,4)).
        let qkv_x = qkv_x.reshape(&[b, nx, 3, h, hd])?.permute(&[2, 0, 3, 1, 4])?;
        let qkv_y = qkv_y.reshape(&[b, ny, 3, h, hd])?.permute(&[2, 0, 3, 1, 4])?;
        // split q/k/v: each [B,H,N,Hc]
        let qx = qkv_x.narrow(0, 0, 1)?.squeeze(Some(0))?;
        let kx = qkv_x.narrow(0, 1, 1)?.squeeze(Some(0))?;
        let vx = qkv_x.narrow(0, 2, 1)?.squeeze(Some(0))?;
        let qy = qkv_y.narrow(0, 0, 1)?.squeeze(Some(0))?;
        let ky = qkv_y.narrow(0, 1, 1)?.squeeze(Some(0))?;
        let vy = qkv_y.narrow(0, 2, 1)?.squeeze(Some(0))?;

        // per-head RMSNorm over Hc (q_norm_x/k_norm_x/q_norm_y/k_norm_y).
        let qx = self.qk_norm_4d(&qx, &format!("{prefix}.q_norm_x"))?;
        let kx = self.qk_norm_4d(&kx, &format!("{prefix}.k_norm_x"))?;
        let qy = self.qk_norm_4d(&qy, &format!("{prefix}.q_norm_y"))?;
        let ky = self.qk_norm_4d(&ky, &format!("{prefix}.k_norm_y"))?;

        // RoPE: interleaved complex via rope_fused_bf16 (input [B*H, N, Hc],
        // table [1, N, Hc/2]). q/k are [B,H,N,Hc] -> flatten B,H.
        let qx = self.apply_rope_bhnd(&qx, img_cos, img_sin)?;
        let kx = self.apply_rope_bhnd(&kx, img_cos, img_sin)?;
        // text RoPE (use_text_rope=True for released ckpt).
        let qy = self.apply_rope_bhnd(&qy, txt_cos, txt_sin)?;
        let ky = self.apply_rope_bhnd(&ky, txt_cos, txt_sin)?;

        // joint sequence: cat([text, image]) along seq dim (=2 in [B,H,N,Hc]).
        let q = Tensor::cat(&[&qy, &qx], 2)?;
        let k = Tensor::cat(&[&ky, &kx], 2)?;
        let v = Tensor::cat(&[&vy, &vx], 2)?;

        let out = sdpa(&q, &k, &v, None)?; // [B,H,Ny+Nx,Hc], scale=1/sqrt(Hc)

        // split back [text, image].
        let out_y = out.narrow(2, 0, ny)?;
        let out_x = out.narrow(2, ny, nx)?;
        // merge heads [B,H,N,Hc] -> [B,N,C].
        let out_y = out_y.permute(&[0, 2, 1, 3])?.reshape(&[b, ny, c])?;
        let out_x = out_x.permute(&[0, 2, 1, 3])?.reshape(&[b, nx, c])?;

        // per-stream output proj (with bias).
        let out_x = self.linear_b(&out_x, &format!("{prefix}.proj_x.weight"), &format!("{prefix}.proj_x.bias"))?;
        let out_y = self.linear_b(&out_y, &format!("{prefix}.proj_y.weight"), &format!("{prefix}.proj_y.bias"))?;
        Ok((out_x, out_y))
    }

    /// Per-head RMSNorm over head_dim on a [B, H, N, Hc] tensor (eps 1e-6).
    fn qk_norm_4d(&self, x: &Tensor, w_key: &str) -> Result<Tensor> {
        let d = x.shape().dims().to_vec();
        let (b, h, n, hc) = (d[0], d[1], d[2], d[3]);
        let flat = x.reshape(&[b * h * n, hc])?;
        let w = self.w(&format!("{w_key}.weight"))?;
        let dim = w.shape().dims()[0];
        let normed = rms_norm(&flat, &[dim], Some(w), 1e-6)?;
        normed.reshape(&[b, h, n, hc])
    }

    /// Apply interleaved RoPE to a [B, H, N, Hc] tensor with [1, N, Hc/2] tables.
    /// `rope_fused_bf16` reshapes x to [B*H, N, Hc] internally, which requires a
    /// contiguous buffer; q/k arrive here via narrow+squeeze+permute so we force
    /// contiguity first.
    fn apply_rope_bhnd(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x = x.contiguous()?;
        flame_core::bf16_ops::rope_fused_bf16(&x, cos, sin)
    }

    // =====================================================================
    // MMDiTBlockT2I.forward (pixeldit_official.py:663-682).
    //   img/txt AdaLN: Linear(C->6C) on c, chunk6 -> shift/scale/gate {msa,mlp}
    //   x_n = adaln(norm_x1(x), shift_msa_x, scale_msa_x); y_n likewise
    //   attn_x, attn_y = JointAttention(x_n, y_n, pos_img, pos_txt)
    //   x = x + gate_msa_x * attn_x;  y = y + gate_msa_y * attn_y
    //   x = x + gate_mlp_x * mlp_x(adaln(norm_x2(x), shift_mlp_x, scale_mlp_x))
    //   y likewise
    // c: [B, 1, C] -> chunks are [B, 1, C], broadcast over tokens.
    // =====================================================================
    #[allow(clippy::too_many_arguments)]
    fn mmdit_block(
        &self,
        x: &Tensor, // [B, Nx, C]
        y: &Tensor, // [B, Ny, C]
        c: &Tensor, // [B, 1, C]
        i: usize,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let p = format!("patch_blocks.{i}");
        let c_dim = self.config.hidden_size;

        let ci = self.linear_b(c, &format!("{p}.adaLN_modulation_img.0.weight"), &format!("{p}.adaLN_modulation_img.0.bias"))?;
        let ct = self.linear_b(c, &format!("{p}.adaLN_modulation_txt.0.weight"), &format!("{p}.adaLN_modulation_txt.0.bias"))?;
        let last = ci.shape().dims().len() - 1;
        let cx = ci.chunk(6, last)?;
        let cy = ct.chunk(6, last)?;
        // chunk order: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        let (sh_msa_x, sc_msa_x, g_msa_x, sh_mlp_x, sc_mlp_x, g_mlp_x) =
            (&cx[0], &cx[1], &cx[2], &cx[3], &cx[4], &cx[5]);
        let (sh_msa_y, sc_msa_y, g_msa_y, sh_mlp_y, sc_mlp_y, g_mlp_y) =
            (&cy[0], &cy[1], &cy[2], &cy[3], &cy[4], &cy[5]);
        let _ = c_dim;

        // 1) joint attention.
        let x_n = self.apply_adaln(&self.rmsnorm(x, &format!("{p}.norm_x1.weight"))?, sh_msa_x, sc_msa_x)?;
        let y_n = self.apply_adaln(&self.rmsnorm(y, &format!("{p}.norm_y1.weight"))?, sh_msa_y, sc_msa_y)?;
        let (attn_x, attn_y) = self.joint_attention(
            &x_n, &y_n, &format!("{p}.attn"), img_cos, img_sin, txt_cos, txt_sin,
        )?;
        let x1 = x.add(&g_msa_x.mul(&attn_x)?)?;
        let y1 = y.add(&g_msa_y.mul(&attn_y)?)?;

        // 2) per-stream SwiGLU FFN.
        let x_mlp_in = self.apply_adaln(&self.rmsnorm(&x1, &format!("{p}.norm_x2.weight"))?, sh_mlp_x, sc_mlp_x)?;
        let y_mlp_in = self.apply_adaln(&self.rmsnorm(&y1, &format!("{p}.norm_y2.weight"))?, sh_mlp_y, sc_mlp_y)?;
        let x_mlp = self.swiglu_ffn(&x_mlp_in, &format!("{p}.mlp_x"))?;
        let y_mlp = self.swiglu_ffn(&y_mlp_in, &format!("{p}.mlp_y"))?;
        let x_out = x1.add(&g_mlp_x.mul(&x_mlp)?)?;
        let y_out = y1.add(&g_mlp_y.mul(&y_mlp)?)?;
        Ok((x_out, y_out))
    }

    // =====================================================================
    // RotaryAttention used inside PiTBlock (pixeldit_official.py:209-277),
    // no-CP, mask=None. x_comp: [B, L, attn_dim]. qkv bias-free, qk_norm RMSNorm,
    // interleaved RoPE over the L (patch-grid) axis, SDPA scale=1/sqrt(head_dim),
    // proj with bias.
    // =====================================================================
    fn rotary_attention(
        &self,
        x_comp: &Tensor, // [B, L, attn_dim]
        prefix: &str,    // "pixel_blocks.{j}.attn"
        cos: &Tensor,
        sin: &Tensor, // [1, L, head_dim/2]
    ) -> Result<Tensor> {
        let b = x_comp.shape().dims()[0];
        let l = x_comp.shape().dims()[1];
        let h = self.config.pixel_num_groups; // 16
        let hd = self.config.pixel_head_dim; // 72
        let c = self.config.pixel_attn_hidden; // 1152

        let qkv = self.linear_nb(x_comp, &format!("{prefix}.qkv.weight"))?; // [B,L,3C]
        // reshape (B,L,3,H,Hc).permute(2,0,3,1,4) -> [3,B,H,L,Hc]
        let qkv = qkv.reshape(&[b, l, 3, h, hd])?.permute(&[2, 0, 3, 1, 4])?;
        let q = qkv.narrow(0, 0, 1)?.squeeze(Some(0))?; // [B,H,L,Hc]
        let k = qkv.narrow(0, 1, 1)?.squeeze(Some(0))?;
        let v = qkv.narrow(0, 2, 1)?.squeeze(Some(0))?;

        let q = self.pixel_qk_norm_4d(&q, &format!("{prefix}.q_norm"))?;
        let k = self.pixel_qk_norm_4d(&k, &format!("{prefix}.k_norm"))?;

        let q = self.apply_rope_bhnd(&q, cos, sin)?;
        let k = self.apply_rope_bhnd(&k, cos, sin)?;

        let out = sdpa(&q, &k, &v, None)?; // [B,H,L,Hc], scale=1/sqrt(Hc)
        let out = out.permute(&[0, 2, 1, 3])?.reshape(&[b, l, c])?;
        self.linear_b(&out, &format!("{prefix}.proj.weight"), &format!("{prefix}.proj.bias"))
    }

    fn pixel_qk_norm_4d(&self, x: &Tensor, w_key: &str) -> Result<Tensor> {
        let d = x.shape().dims().to_vec();
        let (b, h, n, hc) = (d[0], d[1], d[2], d[3]);
        let flat = x.reshape(&[b * h * n, hc])?;
        let w = self.w(&format!("{w_key}.weight"))?;
        let dim = w.shape().dims()[0];
        let normed = rms_norm(&flat, &[dim], Some(w), 1e-6)?;
        normed.reshape(&[b, h, n, hc])
    }

    // =====================================================================
    // PiTBlock.forward (pixeldit_official.py:470-509), no-CP, mask=None.
    //   x: [B*L, P2, pixel_dim]; s_cond: [B*L, hidden_size]
    //   cond = adaLN(s_cond).view(BL, P2, 6*pixel_dim); chunk6 along last dim
    //   x_norm = adaln(norm1(x), shift_msa, scale_msa)           per-token [BL,P2,C]
    //   x_comp = compress_to_attn(x_norm.view(BL, P2*C)).view(B, L, attn_dim)
    //   attn = RotaryAttention(x_comp, pos)                      [B, L, attn_dim]
    //   attn_exp = expand_from_attn(attn.view(B*L, attn_dim)).view(BL, P2, C)
    //   x = x + gate_msa * attn_exp
    //   x = x + gate_mlp * mlp(adaln(norm2(x), shift_mlp, scale_mlp))   (exact GELU)
    // =====================================================================
    #[allow(clippy::too_many_arguments)]
    fn pit_block(
        &self,
        x: &Tensor,       // [B*L, P2, pixel_dim]
        s_cond: &Tensor,  // [B*L, hidden_size]
        j: usize,
        b: usize,
        l: usize,
        cos: &Tensor,
        sin: &Tensor,     // pixel-grid NTK RoPE [1, L, pixel_head_dim/2]
    ) -> Result<Tensor> {
        let p = format!("pixel_blocks.{j}");
        let pixel_dim = self.config.pixel_hidden_size; // 16
        let p2 = self.config.patch_size * self.config.patch_size; // 256
        let attn_dim = self.config.pixel_attn_hidden; // 1152
        let bl = b * l;

        // adaLN -> [BL, 6*pixel_dim*P2] -> view [BL, P2, 6*pixel_dim].
        let cond = self.linear_b(s_cond, &format!("{p}.adaLN_modulation.0.weight"), &format!("{p}.adaLN_modulation.0.bias"))?;
        let cond = cond.reshape(&[bl, p2, 6 * pixel_dim])?;
        let last = 2; // last dim of [BL, P2, 6C]
        // chunk6 along last dim: each [BL, P2, pixel_dim]. These are PER-TOKEN
        // (per pixel-within-patch) shift/scale/gate — NOT per-channel broadcast.
        let shift_msa = cond.narrow(last, 0, pixel_dim)?;
        let scale_msa = cond.narrow(last, pixel_dim, pixel_dim)?;
        let gate_msa = cond.narrow(last, 2 * pixel_dim, pixel_dim)?;
        let shift_mlp = cond.narrow(last, 3 * pixel_dim, pixel_dim)?;
        let scale_mlp = cond.narrow(last, 4 * pixel_dim, pixel_dim)?;
        let gate_mlp = cond.narrow(last, 5 * pixel_dim, pixel_dim)?;

        // x_norm = norm1(x)*(1+scale_msa)+shift_msa  (per-token broadcast over C).
        let n1 = self.rmsnorm(x, &format!("{p}.norm1.weight"))?; // [BL,P2,C]
        let x_norm = n1.mul(&scale_msa.add_scalar(1.0)?)?.add(&shift_msa)?;

        // compress -> [B, L, attn_dim].
        let x_flat = x_norm.reshape(&[bl, p2 * pixel_dim])?;
        let x_comp = self.linear_b(&x_flat, &format!("{p}.compress_to_attn.weight"), &format!("{p}.compress_to_attn.bias"))?;
        let x_comp = x_comp.reshape(&[b, l, attn_dim])?;

        let attn = self.rotary_attention(&x_comp, &format!("{p}.attn"), cos, sin)?; // [B,L,attn_dim]

        // expand -> [BL, P2, C].
        let attn_2d = attn.reshape(&[bl, attn_dim])?;
        let attn_exp = self.linear_b(&attn_2d, &format!("{p}.expand_from_attn.weight"), &format!("{p}.expand_from_attn.bias"))?;
        let attn_exp = attn_exp.reshape(&[bl, p2, pixel_dim])?;

        let x1 = x.add(&gate_msa.mul(&attn_exp)?)?;

        // MLP: fc1 (16->64), exact GELU, fc2 (64->16).  MLP uses nn.GELU() default
        // = EXACT erf gelu, NOT the tanh approx (pixeldit_official.py:285).
        let n2 = self.rmsnorm(&x1, &format!("{p}.norm2.weight"))?;
        let mlp_in = n2.mul(&scale_mlp.add_scalar(1.0)?)?.add(&shift_mlp)?;
        let h1 = self.linear_b(&mlp_in, &format!("{p}.mlp.fc1.weight"), &format!("{p}.mlp.fc1.bias"))?;
        let h1 = h1.gelu_exact()?;
        let mlp_out = self.linear_b(&h1, &format!("{p}.mlp.fc2.weight"), &format!("{p}.mlp.fc2.bias"))?;

        x1.add(&gate_mlp.mul(&mlp_out)?)
    }

    // =====================================================================
    // LQProjection2D latent branch (lq_projection_2d.py:97-413), latent-only,
    // released SD3 ckpt: z_to_patch_ratio = (sr_scale*lsdf)/patch_size =
    // (4*8)/16 = 2.0 > 1  => NEAREST upsample latent (zH,zW) -> (pH,pW), then
    // latent_proj = [Conv2d(16->512,k3p1), SiLU, Conv2d(512->512,k3p1), ResBlock×4].
    // Returns the shared token map [B, L, hidden_dim=512] (merged.flatten(2).T).
    // The per-injection output_heads (Linear 512->1536) are applied lazily at
    // each gate site in `forward`.
    //
    // NCHW vs NHWC: Conv2d is NCHW; group_norm is NHWC. We keep the conv stack
    // in NCHW, converting to NHWC only around the GroupNorm calls inside the
    // ResBlock.
    // =====================================================================
    fn lq_latent_tokens(
        &self,
        lq_latent: &Tensor, // [B, 16, zH, zW]
        ph: usize,
        pw: usize,
    ) -> Result<Tensor> {
        let b = lq_latent.shape().dims()[0];
        let zc = self.config.lq_latent_channels;
        let hid = self.config.lq_hidden_dim;

        // z_to_patch_ratio = 2.0 > 1 => nearest upsample to (pH, pW) (NCHW).
        let z_aligned = lq_latent.upsample_nearest2d(ph, pw)?; // [B,16,pH,pW]

        // latent_proj.0: Conv2d(16->512, k3, s1, p1) + bias.
        let mut conv0 = Conv2d::new_with_bias(zc, hid, 3, 1, 1, self.device.clone(), true)?;
        conv0.copy_weight_from(self.w("lq_proj.latent_proj.0.weight")?)?;
        conv0.copy_bias_from(self.w("lq_proj.latent_proj.0.bias")?)?;
        let h = conv0.forward(&z_aligned)?; // [B,512,pH,pW]
        let h = h.silu()?;

        // latent_proj.2: Conv2d(512->512, k3, s1, p1) + bias.
        let mut conv1 = Conv2d::new_with_bias(hid, hid, 3, 1, 1, self.device.clone(), true)?;
        conv1.copy_weight_from(self.w("lq_proj.latent_proj.2.weight")?)?;
        conv1.copy_bias_from(self.w("lq_proj.latent_proj.2.bias")?)?;
        let mut h = conv1.forward(&h)?;

        // latent_proj.3..6: 4 pre-act ResBlocks.
        for idx in 3..3 + self.config.lq_num_res_blocks {
            h = self.lq_resblock(&h, &format!("lq_proj.latent_proj.{idx}"), hid, ph, pw)?;
        }

        // tokens: merged.flatten(2).transpose(1,2) => [B, hid, pH, pW] -> [B, L, hid].
        // NCHW [B,hid,pH,pW] -> [B,hid,L] -> [B,L,hid].
        let l = ph * pw;
        let h = h.reshape(&[b, hid, l])?;
        h.permute(&[0, 2, 1])
    }

    /// Pre-activation ResBlock (lq_projection_2d.py:74-89):
    ///   x + Conv(SiLU(GN(Conv(SiLU(GN(x)))))).
    /// Keys: <prefix>.block.{0:GN, 2:Conv, 3:GN, 5:Conv}. GroupNorm groups=4,
    /// eps=1e-5 (PyTorch nn.GroupNorm default). Input NCHW [B,C,pH,pW].
    fn lq_resblock(&self, x: &Tensor, prefix: &str, ch: usize, ph: usize, pw: usize) -> Result<Tensor> {
        let groups = 4;
        let b = x.shape().dims()[0];

        // GN0 (NHWC) -> SiLU -> Conv0 (NCHW).
        let h = self.group_norm_nchw(x, &format!("{prefix}.block.0"), groups, b, ch, ph, pw)?;
        let h = h.silu()?;
        let mut conv0 = Conv2d::new_with_bias(ch, ch, 3, 1, 1, self.device.clone(), true)?;
        conv0.copy_weight_from(self.w(&format!("{prefix}.block.2.weight"))?)?;
        conv0.copy_bias_from(self.w(&format!("{prefix}.block.2.bias"))?)?;
        let h = conv0.forward(&h)?;

        // GN1 -> SiLU -> Conv1.
        let h = self.group_norm_nchw(&h, &format!("{prefix}.block.3"), groups, b, ch, ph, pw)?;
        let h = h.silu()?;
        let mut conv1 = Conv2d::new_with_bias(ch, ch, 3, 1, 1, self.device.clone(), true)?;
        conv1.copy_weight_from(self.w(&format!("{prefix}.block.5.weight"))?)?;
        conv1.copy_bias_from(self.w(&format!("{prefix}.block.5.bias"))?)?;
        let h = conv1.forward(&h)?;

        x.add(&h)
    }

    /// GroupNorm on an NCHW tensor by converting to/from NHWC around the
    /// flame `group_norm` (which is NHWC).
    fn group_norm_nchw(&self, x: &Tensor, prefix: &str, groups: usize, b: usize, c: usize, ph: usize, pw: usize) -> Result<Tensor> {
        let w = self.w(&format!("{prefix}.weight"))?;
        let bias = self.w(&format!("{prefix}.bias"))?;
        // NCHW [B,C,H,W] -> NHWC [B,H,W,C]
        let nhwc = x.permute(&[0, 2, 3, 1])?.reshape(&[b, ph, pw, c])?;
        let gn = group_norm(&nhwc, groups, Some(w), Some(bias), 1e-5)?;
        // back to NCHW
        gn.reshape(&[b, ph, pw, c])?.permute(&[0, 3, 1, 2])
    }

    // =====================================================================
    // SigmaAwareGatePerTokenPerDim (lq_projection_2d.py:28-56):
    //   content_logit = content_proj(cat([x, lq], -1))     [B, N, D]
    //   sigma_offset  = -exp(log_alpha) * sigma             (scalar per sample)
    //   gate = sigmoid(content_logit + sigma_offset)
    //   out  = x + gate * lq
    // For released ckpts degrade_sigma is None at inference (latent-only), which
    // PyTorch would error on (assert sigma is not None) — but the released
    // pipeline passes degrade_sigma_tensor (defaulting to 0). We take sigma=0 by
    // default, so sigma_offset=0 and gate = sigmoid(content_logit).
    // x, lq: [B, L, hidden_size]. content_proj.weight [D, 2D].
    // =====================================================================
    fn sigma_gate(
        &self,
        x: &Tensor,
        lq: &Tensor,
        out_idx: usize,
        sigma: f32,
    ) -> Result<Tensor> {
        let p = format!("lq_proj.gate_modules.{out_idx}");
        let cat = Tensor::cat(&[x, lq], 2)?; // [B, L, 2D]
        let logit = self.linear_b(&cat, &format!("{p}.content_proj.weight"), &format!("{p}.content_proj.bias"))?;
        // sigma_offset = -exp(log_alpha)*sigma (scalar). log_alpha is rank-0.
        let logit = if sigma != 0.0 {
            let log_alpha = self.w(&format!("{p}.log_alpha"))?.to_dtype(DType::F32)?.to_vec()?[0];
            let offset = -log_alpha.exp() * sigma;
            logit.add_scalar(offset)?
        } else {
            logit
        };
        let gate = logit.sigmoid()?;
        x.add(&gate.mul(lq)?)
    }

    // =====================================================================
    // Patchify x [B,C,H,W] -> [B, L, C*P2] (== F.unfold(ks=ps,stride=ps).T).
    //   token[b, l, c*P2 + kh*ps + kw] = x[b, c, ph*ps+kh, pw*ps+kw]
    //   L = pH*pW, l = ph*pW + pw  (row-major patch grid).
    // Verified against torch in the mojo skeptic basics audit (cos=1.0).
    // Implemented via reshape/permute on the contiguous NCHW buffer.
    //   x[B,C,pH,ps,pW,ps] -> permute(0,2,4,1,3,5) -> [B,pH,pW,C,ps,ps]
    //   -> reshape [B, L, C*P2].
    // =====================================================================
    fn patchify(&self, x: &Tensor) -> Result<Tensor> {
        let d = x.shape().dims().to_vec();
        let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
        let ps = self.config.patch_size;
        let (ph, pw) = (h / ps, w / ps);
        let x = x.reshape(&[b, c, ph, ps, pw, ps])?;
        // -> [B, pH, pW, C, ps, ps]
        let x = x.permute(&[0, 2, 4, 1, 3, 5])?;
        x.reshape(&[b, ph * pw, c * ps * ps])
    }

    // =====================================================================
    // Pixel embedder image-mode (pixeldit_official.py:390-411):
    //   x = proj(x.permute(0,2,3,1))               [B,H,W,16]   (Linear 3->16)
    //   x = x + pos_full.view(H,W,16)              add 2D sincos pos at grid
    //   x = x.view(B,Hs,ps,Ws,ps,16).permute(0,1,3,2,4,5).view(B*L, P2, 16)
    // Returns [B*L, P2, 16].
    // =====================================================================
    fn pixel_embed(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let d = x.shape().dims().to_vec();
        let b = d[0];
        let ps = self.config.patch_size;
        let pd = self.config.pixel_hidden_size; // 16
        let (hs, ws) = (h / ps, w / ps);
        let p2 = ps * ps;
        let l = hs * ws;

        // per-pixel Linear(3->16) over NHWC.
        let x_nhwc = x.permute(&[0, 2, 3, 1])?.reshape(&[b, h, w, self.config.in_channels])?;
        let x_proj = self.linear_b(&x_nhwc, "pixel_embedder.proj.weight", "pixel_embedder.proj.bias")?; // [B,H,W,16]

        // add full-image 2D sincos pos [H*W,16] -> [H,W,16], broadcast over B.
        let pos = self.build_pixel_pos_embed(h, w)?.reshape(&[1, h, w, pd])?;
        let x_pos = x_proj.add(&pos)?; // [B,H,W,16]

        // view(B,Hs,ps,Ws,ps,16).permute(0,1,3,2,4,5).view(B*L, P2, 16).
        let x_pos = x_pos.reshape(&[b, hs, ps, ws, ps, pd])?;
        let x_pos = x_pos.permute(&[0, 1, 3, 2, 4, 5])?; // [B,Hs,Ws,ps,ps,16]
        x_pos.reshape(&[b * l, p2, pd])
    }

    // =====================================================================
    // FinalLayer + fold back to image (pixeldit_official.py:1432-1437):
    //   x = final_layer(x_pixels) = linear(norm(x))    [B*L, P2, C_out=3]
    //   x = x.view(B,L,P2,C).permute(0,3,2,1).view(B, C*P2, L)
    //   img = F.fold(x, (H,W), ks=ps, stride=ps)
    // We reproduce fold for stride==ps as an exact inverse scatter:
    //   img[b, c, ph*ps+kh, pw*ps+kw] = x_pixels[b*L+l, kh*ps+kw, c]
    //   with l = ph*pW + pw.
    // Implemented via reshape/permute mirroring the inverse of `patchify`/pixel
    // ordering: the fold-input channel index is c*P2 + (kh*ps+kw).
    // =====================================================================
    fn final_layer_fold(&self, x_pixels: &Tensor, b: usize, h: usize, w: usize) -> Result<Tensor> {
        let ps = self.config.patch_size;
        let p2 = ps * ps;
        let c_out = self.config.in_channels; // 3
        let (ph, pw) = (h / ps, w / ps);
        let l = ph * pw;

        // final_layer: RMSNorm(16) -> Linear(16->3).
        let xn = self.rmsnorm(x_pixels, "final_layer.norm.weight")?; // [B*L,P2,16]
        let xo = self.linear_b(&xn, "final_layer.linear.weight", "final_layer.linear.bias")?; // [B*L,P2,3]

        // x.view(B,L,P2,C).permute(0,3,2,1) -> [B, C, P2, L].
        let xo = xo.reshape(&[b, l, p2, c_out])?;
        let xo = xo.permute(&[0, 3, 2, 1])?; // [B, C, P2, L]
        // -> [B, C*P2, L]; the fold-input row index is (c*P2 + p2idx).
        // Reconstruct the image: for each (c, p2idx=kh*ps+kw, l=ph*pW+pw),
        // place at (c, ph*ps+kh, pw*ps+kw). Express via reshape/permute:
        //   [B, C, P2, L] = [B, C, (ps,ps), (pH,pW)]
        //   -> permute to [B, C, pH, ps, pW, ps] -> reshape [B, C, H, W].
        let xo = xo.reshape(&[b, c_out, ps, ps, ph, pw])?; // [B,C,kh,kw,pH,pW]
        let xo = xo.permute(&[0, 1, 4, 2, 5, 3])?; // [B,C,pH,kh,pW,kw]
        xo.reshape(&[b, c_out, h, w])
    }

    // =====================================================================
    // Full PidNet.forward (pid_net.py:265-469), latent-only, no-CP, no-ED,
    // released SD3 cut. Returns the velocity prediction [B, 3, H, W].
    //
    // Inputs:
    //   x:          [B, 3, H, W]   pixel noise/state (BF16)
    //   t_scaled:   [B]            timestep already multiplied by timescale=1000
    //   caption:    [B, Ltxt, 2304] Gemma caption embeds (BF16)
    //   lq_latent:  [B, 16, zH, zW] upstream VAE latent (BF16)
    //   sigma:      degrade_sigma scalar (0.0 for released latent-only ckpts)
    // =====================================================================
    pub fn forward(
        &self,
        x: &Tensor,
        t_scaled: &Tensor,
        caption: &Tensor,
        lq_latent: &Tensor,
        sigma: f32,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let d = x.shape().dims().to_vec();
        let (b, h, w) = (d[0], d[2], d[3]);
        let ps = cfg.patch_size;
        let (hs, ws) = (h / ps, w / ps);
        let l = hs * ws;

        // --- LQ features: project shared token map (heads applied lazily) -----
        let lq_tokens = self.lq_latent_tokens(lq_latent, hs, ws)?; // [B, L, 512]

        // --- image RoPE (NTK 2D) over the patch grid (Hs, Ws) ----------------
        // pixeldit_official.py:1278-1289 fetch_pos: head_dim = hidden/num_groups.
        let (img_cos, img_sin) = self.build_ntk_rope_2d(
            cfg.head_dim, hs, ws, cfg.rope_ref_grid_h, cfg.rope_ref_grid_w, 10000.0, 16.0,
        )?;

        // --- patch tokens + s_embedder ---------------------------------------
        let x_patches = self.patchify(x)?; // [B, L, 3*256=768]
        let s = self.linear_b(&x_patches, "s_embedder.proj.weight", "s_embedder.proj.bias")?; // [B,L,1536]

        // --- timestep -> condition = silu(t_emb) -----------------------------
        let t_emb = self.timestep_embed(t_scaled)?; // [B, 1536]
        let t_emb_3 = t_emb.reshape(&[b, 1, cfg.hidden_size])?; // [B,1,1536]
        let condition = t_emb_3.silu()?; // [B,1,1536]

        // --- text embed: y_embedder (Linear + RMSNorm) + y_pos_embedding ------
        let ltxt = caption.shape().dims()[1].min(cfg.txt_max_length);
        let caption = caption.narrow(1, 0, ltxt)?;
        // PatchTokenEmbedder(y) = norm(proj(y)) — y_embedder has .proj then .norm
        // (pixeldit_official.py:329-332). proj Linear(2304->1536), norm RMSNorm.
        let y_proj = self.linear_b(&caption, "y_embedder.proj.weight", "y_embedder.proj.bias")?;
        let y_norm = self.rmsnorm(&y_proj, "y_embedder.norm.weight")?;
        let ypos = self.w("y_pos_embedding")?.narrow(1, 0, ltxt)?; // [1, Ltxt, 1536]
        let mut y_emb = y_norm.add(&ypos)?; // [B, Ltxt, 1536]

        // --- text RoPE (1D), use_text_rope=True ------------------------------
        let (txt_cos, txt_sin) = self.build_text_rope_1d(ltxt)?;

        // --- 14 MMDiT blocks with LQ gate every `lq_interval` blocks ----------
        let mut s_cur = s;
        for i in 0..cfg.patch_depth {
            // is_gate_active(i): interval>1 -> i%interval==0 (lq_projection_2d.py:288-292).
            if cfg.lq_interval > 1 && i % cfg.lq_interval == 0 {
                let out_idx = i / cfg.lq_interval; // _get_output_index
                // lq_feature[out_idx] = output_heads[out_idx](shared tokens).
                let hp = format!("lq_proj.output_heads.{out_idx}");
                let lq_feat = self.linear_b(&lq_tokens, &format!("{hp}.weight"), &format!("{hp}.bias"))?;
                s_cur = self.sigma_gate(&s_cur, &lq_feat, out_idx, sigma)?;
            }
            let (nx, ny) = self.mmdit_block(
                &s_cur, &y_emb, &condition, i, &img_cos, &img_sin, &txt_cos, &txt_sin,
            )?;
            s_cur = nx;
            y_emb = ny;
        }

        // --- s = silu(t_emb + s) (pid_net.py:412) ----------------------------
        // t_emb_3 [B,1,1536] broadcasts over L.
        let s_sum = s_cur.add(&t_emb_3)?;
        let s_act = s_sum.silu()?;
        let s_cond = s_act.reshape(&[b * l, cfg.hidden_size])?;

        // --- pixel embedder + 2 PiT blocks -----------------------------------
        let mut x_pixels = self.pixel_embed(x, h, w)?; // [B*L, 256, 16]
        // pixel-block NTK RoPE over the patch grid, head_dim = pixel_head_dim (72)
        // (PiTBlock._fetch_pos, pixeldit_official.py:456-468).
        let (pix_cos, pix_sin) = self.build_ntk_rope_2d(
            cfg.pixel_head_dim, hs, ws, cfg.rope_ref_grid_h, cfg.rope_ref_grid_w, 10000.0, 16.0,
        )?;
        for j in 0..cfg.pixel_depth {
            x_pixels = self.pit_block(&x_pixels, &s_cond, j, b, l, &pix_cos, &pix_sin)?;
        }

        // --- final layer + fold -> [B,3,H,W] ---------------------------------
        self.final_layer_fold(&x_pixels, b, h, w)
    }
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

/// Load all PiD net weights from the released safetensors (e.g.
/// `model_ema_bf16.safetensors`). Released checkpoints store everything bf16;
/// we keep them BF16. There is no key prefix in the released ckpt
/// (`pid_distill_model.load_state_dict` strips a leading `net.` if present, so
/// we also strip it defensively). Returns a HashMap keyed by the net-relative
/// names (e.g. `patch_blocks.0.attn.qkv_x.weight`).
pub fn load_pid_resident(
    model_path: &str,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    let raw = load_file_filtered(model_path, device, |_| true)?;
    let mut weights = HashMap::with_capacity(raw.len());
    for (key, val) in raw {
        let k = key.strip_prefix("net.").unwrap_or(&key).to_string();
        let val = if val.dtype() != DType::BF16 {
            val.to_dtype(DType::BF16)?
        } else {
            val
        };
        weights.insert(k, val);
    }
    Ok(weights)
}

#[allow(dead_code)]
fn _assert_keys_present(net: &PidNet) -> Result<()> {
    // Sanity helper for the smoke harness — confirms the released key set is
    // complete for the configured depth before the first forward.
    net.w("s_embedder.proj.weight")?;
    net.w("t_embedder.mlp.0.weight")?;
    net.w("y_embedder.proj.weight")?;
    net.w("y_embedder.norm.weight")?;
    net.w("y_pos_embedding")?;
    net.w("pixel_embedder.proj.weight")?;
    net.w("final_layer.norm.weight")?;
    net.w("final_layer.linear.weight")?;
    net.w("lq_proj.latent_proj.0.weight")?;
    for i in 0..net.config.patch_depth {
        net.w(&format!("patch_blocks.{i}.attn.qkv_x.weight"))?;
    }
    for j in 0..net.config.pixel_depth {
        net.w(&format!("pixel_blocks.{j}.compress_to_attn.weight"))?;
    }
    for o in 0..net.config.num_lq_outputs() {
        net.w(&format!("lq_proj.output_heads.{o}.weight"))?;
        net.w(&format!("lq_proj.gate_modules.{o}.content_proj.weight"))?;
        if !net.has(&format!("lq_proj.gate_modules.{o}.log_alpha")) {
            return Err(Error::InvalidInput(format!(
                "missing lq_proj.gate_modules.{o}.log_alpha"
            )));
        }
    }
    Ok(())
}
