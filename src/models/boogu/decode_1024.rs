//! Boogu-Image — tiled VAE decode for the 1024² production path.
//!
//! The DiT denoise at `--size 1024` produces a valid `[1,16,128,128]` F32
//! latent (resident, ~21.7 GB peak), but the **monolithic** [`LdmVAEDecoder`]
//! decode at 128×128 latent → 1024×1024 image needs ~+2.25 GB (conv im2col @
//! 1024² + mid-block attention over (1024/8·…)² positions) → OOM on a 24 GB
//! RTX 3090 Ti. The monolithic decode fits ≤768².
//!
//! Fix = **tiled decode**, mirroring the verified Mojo pipeline
//! (`serenitymojo/pipeline/boogu_pipeline.mojo::_decode_and_save` reusing
//! `models/vae/ideogram4_tiled_decode.mojo::_blend3`):
//!
//! - Split the `[1,16,128,128]` latent into a **fixed 3×3 grid of overlapping
//!   `TILE=LAT/2` (64×64) crops** at row/col offsets `{0, HALF=LAT/4=32, TILE=64}`.
//! - Decode each crop with the SAME [`LdmVAEDecoder`] (rescale folded inside
//!   `decode` → pass the RAW latent crop). Each 64×64 crop → a 512×512 image tile.
//! - **Feather-blend** the nine 512×512 tiles into a seamless 1024×1024 image:
//!   blend the three tiles of each row along W (`_blend3` dim 3), then blend the
//!   three rows along H (`_blend3` dim 2). The crossfade weights are
//!   half-pixel-centered linear ramps `(i+0.5)/n` — byte-for-byte the Mojo
//!   `_weight_tensor` / `_xfade` math.
//!
//! Per-tile decode peak ≈ the 512²-image monolithic decode, which fits easily;
//! the latent crops + nine 512×512 F32 tiles are small. So the 1024 path now
//! stays well under 24 GB.
//!
//! Pure-Rust, autograd OFF — no backward registration. Reuses `LdmVAEDecoder`
//! verbatim (no VAE reimplementation).

use anyhow::{anyhow, Result};

use flame_core::{DType, Shape, Tensor};

use crate::vae::ldm_decoder::LdmVAEDecoder;

/// Half-pixel-centered linear feather weights, shaped to broadcast against an
/// NCHW tensor along `dim` (2 = H, 3 = W). Matches the Mojo `_weight_tensor`:
/// for `n` positions, `w[i] = (i+0.5)/n` (ascending) or `1 - (i+0.5)/n`
/// (descending). Built on CPU as F32 and uploaded (the tiles are F32).
fn weight_tensor(n: usize, dim: usize, ascending: bool, device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Result<Tensor> {
    let mut h = Vec::with_capacity(n);
    for i in 0..n {
        let t = (i as f32 + 0.5) / n as f32;
        h.push(if ascending { t } else { 1.0 - t });
    }
    // [1,1,n,1] for H (dim 2), [1,1,1,n] for W (dim 3) — broadcasts over B,C and
    // the other spatial axis.
    let shape = if dim == 2 {
        Shape::from_dims(&[1, 1, n, 1])
    } else {
        Shape::from_dims(&[1, 1, 1, n])
    };
    Tensor::from_vec(h, shape, device.clone()).map_err(|e| anyhow!("weight_tensor: {e:?}"))
}

/// Linear crossfade of two equal-width strips along `dim`:
/// `left*descending + right*ascending`. Mirrors Mojo `_xfade`.
fn xfade(left: &Tensor, right: &Tensor, dim: usize) -> Result<Tensor> {
    let n = left.shape().dims()[dim];
    let device = left.device();
    let wl = weight_tensor(n, dim, false, device)?; // descending: weights the LEFT (earlier) tile
    let wr = weight_tensor(n, dim, true, device)?; // ascending: weights the RIGHT (later) tile
    let a = left.mul(&wl).map_err(|e| anyhow!("xfade mul left: {e:?}"))?;
    let b = right.mul(&wr).map_err(|e| anyhow!("xfade mul right: {e:?}"))?;
    a.add(&b).map_err(|e| anyhow!("xfade add: {e:?}"))
}

/// Blend three equal-size, half-overlapping tiles into one along `dim`.
/// Mirrors Mojo `_blend3` exactly:
///   t = tile size; s = t/2 (solo); ov = t - s (overlap)
///   out = [ t0[0:s] | xfade(t0[s:t], t1[0:ov]) | xfade(t1[ov:2ov], t2[0:ov]) | t2[ov:t] ]
/// For t=512 → s=256, ov=256 → out length 256+256+256+256 = 1024 (seamless).
fn blend3(t0: &Tensor, t1: &Tensor, t2: &Tensor, dim: usize) -> Result<Tensor> {
    let t = t0.shape().dims()[dim];
    let s = t / 2;
    let ov = t - s;
    let a = t0.narrow(dim, 0, s).map_err(|e| anyhow!("blend3 a: {e:?}"))?;
    let b = xfade(
        &t0.narrow(dim, s, ov).map_err(|e| anyhow!("blend3 b.l: {e:?}"))?,
        &t1.narrow(dim, 0, ov).map_err(|e| anyhow!("blend3 b.r: {e:?}"))?,
        dim,
    )?;
    let c = xfade(
        &t1.narrow(dim, ov, ov).map_err(|e| anyhow!("blend3 c.l: {e:?}"))?,
        &t2.narrow(dim, 0, ov).map_err(|e| anyhow!("blend3 c.r: {e:?}"))?,
        dim,
    )?;
    let d = t2.narrow(dim, ov, s).map_err(|e| anyhow!("blend3 d: {e:?}"))?;
    // `Tensor::cat` materializes any strided narrow-views internally, so no
    // explicit `.contiguous()` is needed here.
    Tensor::cat(&[&a, &b, &c, &d], dim).map_err(|e| anyhow!("blend3 cat: {e:?}"))
}

/// Decode one raw latent crop `[1,16,th,tw]` with the shared VAE and return the
/// resulting image tile as **F32** `[1,3,8·th,8·tw]`. The rescale
/// (`z/scale + shift`) is folded inside `LdmVAEDecoder::decode`, so pass the
/// RAW (un-rescaled) latent crop; the conv path is BF16-only so cast in, F32 out
/// (matches the Mojo `cast_tensor(dec.decode(...), F32)`).
fn decode_tile(vae: &LdmVAEDecoder, crop: &Tensor) -> Result<Tensor> {
    // `.contiguous()` after the narrow chain: a narrowed crop is a strided view;
    // the VAE conv (im2col) path assumes a contiguous NCHW input — this is the
    // documented narrow→conv trap, so materialize before decode (mirrors the
    // helios `decode_tiled` contiguous() and the Mojo `slice` which copies).
    let crop_bf16 = crop
        .contiguous()
        .map_err(|e| anyhow!("decode_tile contiguous: {e:?}"))?
        .to_dtype(DType::BF16)
        .map_err(|e| anyhow!("decode_tile ->bf16: {e:?}"))?;
    let tile = vae
        .decode(&crop_bf16)
        .map_err(|e| anyhow!("decode_tile VAE decode: {e}"))?;
    // Free per-tile transient pool allocations so peak stays bounded.
    flame_core::cuda_alloc_pool::clear_pool_cache();
    tile.to_dtype(DType::F32).map_err(|e| anyhow!("decode_tile ->f32: {e:?}"))
}

/// Tiled VAE decode of a `[1,16,LAT,LAT]` latent into a seamless
/// `[1,3,8·LAT,8·LAT]` F32 image, using a fixed 3×3 grid of overlapping
/// `TILE=LAT/2` latent crops + feathered 3×3 blend. Matches the Mojo
/// `_decode_and_save` geometry exactly.
///
/// `vae` is the SAME [`LdmVAEDecoder`] used for the monolithic path; here it is
/// simply invoked nine times on 64×64 crops instead of once on the full 128×128.
pub fn decode_tiled_1024(vae: &LdmVAEDecoder, latent: &Tensor) -> Result<Tensor> {
    let dims = latent.shape().dims().to_vec();
    if dims.len() != 4 {
        return Err(anyhow!(
            "decode_tiled_1024: expected [1,C,H,W] latent, got {dims:?}"
        ));
    }
    let (lat_h, lat_w) = (dims[2], dims[3]);
    if lat_h % 2 != 0 || lat_w % 2 != 0 {
        return Err(anyhow!(
            "decode_tiled_1024: latent H/W must be even, got {lat_h}x{lat_w}"
        ));
    }
    let tile_h = lat_h / 2; // 64
    let tile_w = lat_w / 2; // 64
    let half_h = tile_h / 2; // 32
    let half_w = tile_w / 2; // 32

    // Row/col crop offsets: {0, HALF, TILE} → 3 overlapping crops per axis.
    let row_offsets = [0usize, half_h, tile_h];
    let col_offsets = [0usize, half_w, tile_w];

    // Decode all nine crops, blending each row of three along W (dim 3), then
    // blend the three rows along H (dim 2).
    let mut rows: Vec<Tensor> = Vec::with_capacity(3);
    for &r in &row_offsets {
        // Latent row band [1,16,tile_h,LAT] (narrow on H first, then per-col W).
        let band = latent
            .narrow(2, r, tile_h)
            .map_err(|e| anyhow!("decode_tiled_1024 row narrow @{r}: {e:?}"))?;
        let mut tiles: Vec<Tensor> = Vec::with_capacity(3);
        for &c in &col_offsets {
            let crop = band
                .narrow(3, c, tile_w)
                .map_err(|e| anyhow!("decode_tiled_1024 col narrow @{c}: {e:?}"))?;
            let t_tile = std::time::Instant::now();
            tiles.push(decode_tile(vae, &crop)?);
            println!(
                "    tile (row@{r}, col@{c}) {tile_h}x{tile_w} latent -> {0}x{0} px in {1:.2}s",
                tile_h * 8,
                t_tile.elapsed().as_secs_f32()
            );
        }
        let row_img = blend3(&tiles[0], &tiles[1], &tiles[2], 3)?;
        rows.push(row_img);
    }
    blend3(&rows[0], &rows[1], &rows[2], 2)
}
