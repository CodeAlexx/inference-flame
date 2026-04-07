# FLUX 1 Dev — VAE Decoder + Sampling Audit (DELTA vs BFL)

Source of truth:
- `/home/alex/black-forest-labs-flux/src/flux/modules/autoencoder.py`
- `/home/alex/black-forest-labs-flux/src/flux/sampling.py`
- `/home/alex/black-forest-labs-flux/src/flux/util.py` (ae_params: flux-dev)

Under audit:
- `/home/alex/EriDiffusion/inference-flame/src/vae/ldm_decoder.rs`
- `/home/alex/EriDiffusion/inference-flame/src/sampling/klein_sampling.rs`
- `/home/alex/EriDiffusion/inference-flame/src/sampling/euler.rs`
- `/home/alex/EriDiffusion/inference-flame/src/bin/flux1_test.rs`
- (No dedicated flux1 sampling / no flux1 end-to-end pipeline exists yet)

FLUX 1 Dev ae_params (util.py line 318-328): resolution=256, in_channels=3, ch=128,
out_ch=3, ch_mult=[1,2,4,4], num_res_blocks=2, z_channels=16, scale_factor=0.3611,
shift_factor=0.1159.

---

## SUMMARY

**5 CRITICAL, 4 HIGH, 3 LOW deltas found**

Critical = will break correctness or missing whole pipeline; High = observable quality
drift or wrong numeric ops; Low = cosmetics / doc.

The VAE decoder module (`ldm_decoder.rs`) is **structurally correct** and will work for
FLUX 1 once called with `(in_channels=16, scaling_factor=0.3611, shift_factor=0.1159)`.
The main problems are:

1. **No FLUX 1 sampling module exists.** Nothing wires time-shift/mu/denoise for FLUX 1.
   `klein_sampling.rs` uses Flux 2's *empirical mu*, not FLUX 1's linear estimator
   (y1=0.5 @ x1=256, y2=1.15 @ x2=4096).
2. **No FLUX 1 end-to-end binary** — `flux1_test.rs` is component-only, produces no
   image, never calls the VAE, never runs the denoise loop.
3. **VAE decode latent-normalization order is WRONG** for BFL.
4. **No pack/unpack helpers** (rearrange `b c (h ph) (w pw) -> b (h w) (c ph pw)`).
5. **No output denorm ([-1,1] → [0,255] uint8).**

---

## CRITICAL DELTAS

### C1. VAE `decode()` latent-normalization order is reversed

- **What:** BFL decodes via `z = z / scale_factor + shift_factor` (divide first, then
  add), then runs decoder. Our Rust does `(z - shift_factor) * (1/scale_factor)`.
- **Source:** `autoencoder.py:313-315`
  ```python
  def decode(self, z: Tensor) -> Tensor:
      z = z / self.scale_factor + self.shift_factor
      return self.decoder(z)
  ```
- **Rust:** `ldm_decoder.rs:576-580`
  ```rust
  let z = z.add_scalar(-self.shift_factor)?
      .mul_scalar(1.0 / self.scaling_factor)?;
  ```
- **Math:** BFL computes `z/s + f`. Rust computes `(z - f)/s = z/s - f/s`. These
  differ by a constant offset of `f + f/s = 0.1159 + 0.1159/0.3611 ≈ 0.437`, applied
  to every latent channel. For Z-Image this worked only because Z-Image uses the same
  operation the opposite way at encode time (and the current file was written for
  Z-Image, see header comment). For **FLUX 1 this will produce a colour-shifted /
  broken decode.**
- **Severity:** CRITICAL
- **Verbatim fix:** Replace lines 578-579 of `ldm_decoder.rs` with:
  ```rust
  // BFL: z = z / scale_factor + shift_factor
  let z = z.mul_scalar(1.0 / self.scaling_factor)?
      .add_scalar(self.shift_factor)?;
  ```
  (Either add a `mode: LatentNormMode` enum or make this the only path and fix
  Z-Image's encode-side to match. The BFL order above is the correct one for FLUX 1.)

### C2. No FLUX 1 sampling module / `flux_time_shift` / `get_schedule(base_shift=0.5, max_shift=1.15)`

- **What:** BFL FLUX 1 schedule: `linspace(1,0,num_steps+1)`, then if shift=True apply
  `time_shift(mu, 1.0, t)` where `mu` comes from a linear estimator
  `y = m·seq_len + b` fitted through `(x1=256, y1=0.5), (x2=4096, y2=1.15)`.
- **Source:** `sampling.py:277-305`
  ```python
  def time_shift(mu, sigma, t):
      return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
  def get_lin_function(x1=256, y1=0.5, x2=4096, y2=1.15):
      m = (y2 - y1) / (x2 - x1); b = y1 - m*x1
      return lambda x: m*x + b
  def get_schedule(num_steps, image_seq_len, base_shift=0.5, max_shift=1.15, shift=True):
      timesteps = torch.linspace(1, 0, num_steps + 1)
      if shift:
          mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
          timesteps = time_shift(mu, 1.0, timesteps)
      return timesteps.tolist()
  ```
- **Rust:** MISSING for FLUX 1. `klein_sampling.rs:19-37` uses the Flux **2** empirical
  mu (a1=8.73809524e-05, b1=1.89833333, a2=0.00016927, b2=0.45666666) — this is the
  **wrong formula for FLUX 1**. Using it will produce a different schedule and
  visibly worse images.
- **Severity:** CRITICAL
- **Verbatim fix:** Add `src/sampling/flux1_sampling.rs`:
  ```rust
  //! FLUX 1 Dev sampling helpers — ported verbatim from BFL sampling.py.

  /// BFL flux1 time_shift: exp(mu) / (exp(mu) + (1/t - 1)^sigma)
  pub fn time_shift(mu: f64, sigma: f64, t: f64) -> f64 {
      let em = mu.exp();
      em / (em + (1.0 / t - 1.0).powf(sigma))
  }

  /// BFL flux1 linear mu estimator: line through (256, 0.5) and (4096, 1.15).
  pub fn flux1_mu(image_seq_len: usize, base_shift: f64, max_shift: f64) -> f64 {
      let (x1, y1) = (256.0f64, base_shift);
      let (x2, y2) = (4096.0f64, max_shift);
      let m = (y2 - y1) / (x2 - x1);
      let b = y1 - m * x1;
      m * image_seq_len as f64 + b
  }

  /// BFL get_schedule for FLUX 1. Returns num_steps+1 timesteps, t[0]=1, t[-1]=0.
  pub fn get_schedule(num_steps: usize, image_seq_len: usize, shift: bool) -> Vec<f32> {
      // linspace(1, 0, num_steps+1)
      let mut ts: Vec<f64> = (0..=num_steps)
          .map(|i| 1.0 - (i as f64) / (num_steps as f64))
          .collect();
      if shift {
          let mu = flux1_mu(image_seq_len, 0.5, 1.15);
          for t in ts.iter_mut() {
              if *t > 0.0 && *t < 1.0 {
                  *t = time_shift(mu, 1.0, *t);
              }
          }
      }
      ts.into_iter().map(|v| v as f32).collect()
  }
  ```
  Register in `sampling/mod.rs`.

### C3. No FLUX 1 denoise loop wired up

- **What:** BFL denoise is a **single-pass Euler** with `guidance_vec = ones*guidance`
  passed as a conditioning input; **no classifier-free guidance**, single forward per step.
  Step: `img = img + (t_prev - t_curr) * pred`.
- **Source:** `sampling.py:308-353`
- **Rust:** No FLUX 1 caller exists. `flux1_test.rs` runs ONE forward and exits; no
  loop. `klein_sampling.rs::euler_denoise` is generic in the model closure but never
  wired to `Flux1DiT::forward`, and its schedule is wrong (see C2).
- **Severity:** CRITICAL
- **Verbatim fix:** Add a `flux1_denoise` that mirrors BFL exactly:
  ```rust
  pub fn flux1_denoise(
      mut model: impl FnMut(&Tensor /*img*/, f32 /*t*/, &Tensor /*guidance*/) -> Result<Tensor>,
      mut img: Tensor,
      timesteps: &[f32],
      guidance: f32,
  ) -> Result<Tensor> {
      let g = Tensor::from_vec(
          vec![guidance; img.shape().dims()[0]],
          Shape::from_dims(&[img.shape().dims()[0]]),
          img.device().clone(),
      )?.to_dtype(img.dtype())?;
      for w in timesteps.windows(2) {
          let (t_curr, t_prev) = (w[0], w[1]);
          let pred = model(&img, t_curr, &g)?;
          img = img.add(&pred.mul_scalar(t_prev - t_curr)?)?;
      }
      Ok(img)
  }
  ```
  NOTE: Do **not** do CFG. FLUX 1 Dev is a guidance-distilled model — it takes the
  guidance scalar as a model input, not via two forwards.

### C4. No pack / unpack (rearrange) helpers for FLUX 1

- **What:** BFL packs latent `(b,16,H,W) → (b, H*W/4, 64)` via
  `rearrange(b c (h ph) (w pw) -> b (h w) (c ph pw), ph=2, pw=2)`, and unpacks at end:
  `(b, h*w, 64) → (b, 16, 2h, 2w)`.
- **Source:** `sampling.py:41` (pack inside `prepare`), `sampling.py:356-364` (`unpack`)
- **Rust:** MISSING. `flux1_test.rs:106-110` fakes it by building random `[1,4096,64]`
  directly, never proving the layout. There is **no code** that does the actual
  `(h ph)(w pw)`→`(c ph pw)` reindex, nor the inverse.
- **Severity:** CRITICAL
- **Verbatim fix:** Add to `flux1_sampling.rs` (operating on NCHW BF16 tensors):
  ```rust
  /// BFL prepare(): rearrange "b c (h ph) (w pw) -> b (h w) (c ph pw)" with ph=pw=2.
  /// Input  [B, 16, H, W] with H,W even.
  /// Output [B, (H/2)*(W/2), 64].
  pub fn pack_latent(z: &Tensor) -> Result<Tensor> {
      let d = z.shape().dims(); // [B, C, H, W]
      let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
      assert!(h % 2 == 0 && w % 2 == 0);
      let (h2, w2) = (h / 2, w / 2);
      // [B, C, h2, 2, w2, 2]
      let t = z.reshape(&[b, c, h2, 2, w2, 2])?;
      // -> [B, h2, w2, C, 2, 2]
      let t = t.permute(&[0, 2, 4, 1, 3, 5])?;
      // -> [B, h2*w2, C*4]
      t.reshape(&[b, h2 * w2, c * 4])
  }

  /// BFL unpack(): inverse of pack. Returns [B, 16, 2*ceil(H/16), 2*ceil(W/16)].
  pub fn unpack_latent(x: &Tensor, height: usize, width: usize) -> Result<Tensor> {
      let d = x.shape().dims(); // [B, N, 64]
      let b = d[0];
      let h2 = (height + 15) / 16 * 1; // = ceil(H/16); full latent h = 2*h2
      let w2 = (width + 15) / 16 * 1;
      let c = d[2] / 4;
      // [B, h2, w2, C, 2, 2]
      let t = x.reshape(&[b, h2, w2, c, 2, 2])?;
      // -> [B, C, h2, 2, w2, 2]
      let t = t.permute(&[0, 3, 1, 4, 2, 5])?;
      t.reshape(&[b, c, h2 * 2, w2 * 2])
  }
  ```
  (BFL `get_noise` shape is `(B, 16, 2*ceil(H/16), 2*ceil(W/16))`, so the latent H,W
  passed to `pack_latent` are **already** `2*ceil(H/16)`; `unpack` must divide
  accordingly. The formula above matches sampling.py:360-364 verbatim.)

### C5. No [-1,1] → uint8 PNG denormalization path

- **What:** After VAE decode the pixel tensor is in `[-1, 1]` and must be mapped to
  `[0, 255]` uint8 (`x = ((x.clamp(-1,1) + 1)/2 * 255).round().to(u8)`). PNG writing
  needs HWC order.
- **Source:** BFL scripts do it in their saver (`cli.py` uses `(127.5 * (x+1)).uint8()`).
- **Rust:** `ldm_decoder.rs::decode` returns raw `[-1,1]`; no writer in inference-flame
  consumes it for FLUX 1. `flux1_test.rs` never calls decode at all.
- **Severity:** CRITICAL (for end-to-end)
- **Verbatim fix:** In the FLUX 1 end-to-end binary:
  ```rust
  let img = vae.decode(&latent)?;                  // [B,3,H,W], bf16, [-1,1]
  let img = img.to_dtype(DType::F32)?;
  let img = img.clamp(-1.0, 1.0)?
                .add_scalar(1.0)?
                .mul_scalar(127.5)?;
  // Then permute NCHW -> NHWC, round, cast to u8, write PNG.
  ```

---

## HIGH DELTAS

### H1. AttnBlock Q·K ordering differs (head layout)

- **What:** BFL rearranges `b c h w -> b 1 (h w) c` for Q/K/V (sequence=h·w,
  dim=channels, one head of size C) and calls `F.scaled_dot_product_attention(q,k,v)`.
- **Source:** `autoencoder.py:42-49`
- **Rust:** `ldm_decoder.rs:179-238` reshapes `[B,C,H,W] → [B,H*W,C]` via permute, does
  Q/K/V via a **linear_3d** (matmul with `Conv1x1` weights squeezed to `[C,C]`), then
  `unsqueeze(1)` to `[B,1,N,C]` and `sdpa_forward`. Functionally equivalent to BFL
  (`Conv1x1` on `[B,C,H,W]` == linear on `[B,H*W,C]`), so the math is the same.
  **However** this relies on `sdpa_forward` using scale `1/sqrt(C)`. At C=512 that's
  `~0.0442`. Verify `flame_core::sdpa::forward` defaults to `1/sqrt(head_dim)` — if it
  instead infers head_dim from a different axis, outputs drift.
- **Severity:** HIGH (verification item, not a known bug)
- **Verbatim fix:** Add a runtime assert in `AttnBlock::forward` after the sdpa call:
  log a spot value for a known input and compare against a PyTorch reference once
  (e.g., `torch.nn.functional.scaled_dot_product_attention` on the same weights).
  No code change if scale is `1/sqrt(C)`.

### H2. z_channels is hard-coded via `in_channels` parameter only; conv_in out_channels hard-coded to 512

- **What:** For FLUX 1, latent z_channels=16 → `decoder.conv_in` is `Conv2d(16, 512, 3)`.
- **Source:** `autoencoder.py:208`, util.py:325 `z_channels=16`.
- **Rust:** `ldm_decoder.rs:520-529` hard-codes `ch=128`, `ch_mult=[1,2,4,4]`,
  `num_resnets=3`, `top_ch = 128*4 = 512`. This matches FLUX 1 exactly (good) but
  does NOT match FLUX 1's `num_res_blocks=2` semantically — BFL defines
  `num_res_blocks=2` in ae_params and the decoder builds `num_res_blocks + 1 = 3`
  resnets per up block (autoencoder.py:222). Rust's `num_resnets=3` is correct
  because it's the already-incremented value. Document this so nobody "fixes" it.
- **Severity:** HIGH (maintenance trap, correct today)
- **Verbatim fix:** Add a comment:
  ```rust
  // BFL's AE param num_res_blocks=2; decoder instantiates num_res_blocks+1 per up.
  // For FLUX 1 this yields 3 resnets per up block. Do NOT change to 2.
  let num_resnets: usize = 3;
  ```

### H3. Upsample mode assumed "nearest" — verify kernel

- **What:** BFL `Upsample` is `F.interpolate(x, scale=2, mode='nearest')` followed by
  a 3x3 `Conv2d`.
- **Source:** `autoencoder.py:98-106`
- **Rust:** `ldm_decoder.rs:344` calls `kernels.upsample2d_nearest(&x, (h*2,w*2))`
  then the 3x3 conv. Good — provided `upsample2d_nearest` is true nearest-neighbor and
  not bilinear. Not confirmed in this file.
- **Severity:** HIGH
- **Verbatim fix:** Search `flame_core/src/cuda_kernels.rs` for `upsample2d_nearest`
  and confirm it does integer replication, not bilinear. If ambiguous, add a unit test:
  feed `[[[[1,2],[3,4]]]]` and expect `[[[[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]]]`.

### H4. No VAE loader call-site that passes the **correct** FLUX 1 params

- **What:** `LdmVAEDecoder::from_safetensors(path, in_channels=16, scaling_factor=0.3611, shift_factor=0.1159, device)` is the only correct invocation for FLUX 1.
- **Rust:** No call site exists anywhere under `src/bin/` for FLUX 1. `flux1_test.rs`
  only loads DiT.
- **Severity:** HIGH
- **Verbatim fix:** Add to `flux1_test.rs` (or new `flux1_infer.rs`):
  ```rust
  const AE_SCALE: f32 = 0.3611;
  const AE_SHIFT: f32 = 0.1159;
  let vae = inference_flame::vae::ldm_decoder::LdmVAEDecoder::from_safetensors(
      VAE_PATH, 16, AE_SCALE, AE_SHIFT, &device,
  )?;
  ```
  Then, after running `flux1_denoise`, call `unpack_latent(&x, H, W)` and
  `vae.decode(&latent)`.

---

## LOW DELTAS

### L1. Header comment says "z = (z - shift_factor) / scaling_factor"

- **What:** File header claims the BFL rule, but the math in code is different (see C1).
- **Rust:** `ldm_decoder.rs:8`
- **Severity:** LOW (misleading doc)
- **Fix:** Update comment to the BFL form `z = z / scaling_factor + shift_factor` after
  C1 lands.

### L2. `num_res_blocks` comment is ambiguous

- **What:** Line 522 `num_resnets: usize = 3; // layers_per_block + 1`. "layers_per_block"
  is diffusers terminology. BFL calls this `num_res_blocks=2`, + 1 for the decoder.
- **Severity:** LOW
- **Fix:** Change comment to `// BFL num_res_blocks(2) + 1 extra resnet per up-block`.

### L3. `flux1_test.rs` builds `img_ids` with wrong tiling order

- **What:** BFL sets `img_ids[..., 1] += arange(h//2)[:,None]` and `[..., 2] += arange(w//2)[None,:]`, flattened `h w c -> (h w) c`. `flux1_test.rs:163-172` does a nested
  loop writing `[idx*3+1]=h, [idx*3+2]=w` at `idx = h*packed_w + w`. This happens to
  match, but there's no test. Once C4's `pack_latent` lands, make sure `img_ids`
  ordering matches the flatten order of `pack_latent` (row-major over `(h2,w2)`).
- **Severity:** LOW (correct but fragile)
- **Fix:** Assert in a test that `build_flux1_ids(4,4,0)` returns the expected 16×3 table.

---

## CHECKLIST RESULTS

| Item                                                                 | Status |
|----------------------------------------------------------------------|--------|
| ae_params (resolution, ch=128, ch_mult=[1,2,4,4], num_res_blocks=2, z_channels=16, scale=0.3611, shift=0.1159) | Structure OK — caller must pass scale/shift (H4) |
| `decode` op order `(z/scale)+shift`                                  | **WRONG** (C1) |
| Decoder structure conv_in → mid → up(3..0) → norm → SiLU → conv_out  | OK |
| ResnetBlock: norm1→SiLU→conv1→norm2→SiLU→conv2 + shortcut            | OK (ldm_decoder.rs:119-133) |
| AttnBlock: norm → q,k,v → softmax(qk/√c) @ v → proj + residual        | OK modulo H1 verification |
| GroupNorm(32, eps=1e-6)                                              | OK (ldm_decoder.rs:120,123,211,593) |
| Upsample = nearest ×2 then 3x3 conv                                  | OK pending H3 kernel check |
| z_channels=16 for FLUX 1                                             | Supported, not yet used (H4) |
| Output denormalization [-1,1]→uint8                                  | **MISSING** (C5) |
| `flux_time_shift(mu, sigma=1, t)`                                    | **MISSING** (C2) |
| linear mu: y1=0.5 @ 256, y2=1.15 @ 4096                              | **MISSING** (C2); klein uses wrong Flux-2 formula |
| `get_schedule(num_steps, seq_len, base=0.5, max=1.15, shift=True)`   | **MISSING** (C2) |
| Denoise: for each (t_curr, t_prev) Euler `img += (t_prev - t_curr)*pred` | **MISSING** (C3) |
| `prepare` pack `(b,c,h,w)→(b,hw/4,c*4)` with ph=pw=2                 | **MISSING** (C4) |
| `unpack` inverse                                                     | **MISSING** (C4) |
| `img_ids` = 3-axis (0, h_idx, w_idx); `txt_ids` = zeros              | Present in `flux1_test.rs` only (L3) |
| Single forward per step with `guidance_vec = ones*guidance` (NO CFG) | **MISSING** (C3); current test passes guidance but no loop |
| `get_noise` shape `(B, 16, 2*ceil(H/16), 2*ceil(W/16))`              | **MISSING** — `flux1_test.rs` fakes packed shape directly |

---

**Tally:** 5 CRITICAL, 4 HIGH, 3 LOW deltas found.
