//! L2P safetensors → internal-key translator.
//!
//! Loads `model-1k-merge.safetensors` and walks the HF-shape key set into
//! the internal key layout that `L2pDiT::new_resident` and
//! `MicroDiffusionModel::new` consume.
//!
//! Substantive transformations:
//!
//! 1. **QKV fusion.** L2P stores three separate `attention.to_{q,k,v}.weight`
//!    matrices per attention block. The Rust DiT (mirroring NextDiT) reads
//!    a single fused `attention.qkv.weight`. Per-block:
//!      cat([Wq, Wk, Wv], dim=0)  →  attention.qkv.weight
//!    Shapes: each Wq/Wk/Wv is `[3840, 3840] = [num_heads*head_dim, dim]`.
//!    Concatenated along dim 0 → `[11520, 3840]`. The DiT pre-transposes
//!    every 2D weight `[out, in] → [in, out]` in `new_resident`, so the
//!    runtime matmul reads `[3840, 11520]` and the `chunk(3, dim=-1)`
//!    inside `joint_attention` peels Q/K/V apart along the final-axis
//!    `11520` blocks. Fused dim **= 0**, single `.contiguous()` after the
//!    cat (kernel contract).
//!
//! 2. **ModuleList / Sequential unwrap.**
//!    - `attention.to_out.0.weight` → `attention.out.weight`
//!    - `local_decoder.enc{N}.0.weight/bias` → `local_decoder.enc{N}.conv.weight/bias`
//!      (Sequential[Conv, SiLU], Conv at index 0)
//!    - `local_decoder.up{N}.1.weight/bias` → `local_decoder.up{N}.conv.weight/bias`
//!      (Sequential[Upsample, Conv], Conv at index 1)
//!    - `local_decoder.dec{N}.0.weight/bias` → `local_decoder.dec{N}.conv.weight/bias`
//!      (Sequential[Conv, SiLU], Conv at index 0)
//!    - `local_decoder.bottleneck.0.weight/bias` → `local_decoder.bottleneck.conv.weight/bias`
//!      (Sequential[Conv, SiLU], Conv at index 0)
//!    - `local_decoder.out_conv.weight/bias` → passthrough (raw Conv2d)
//!
//! 3. **Attention norm rename.**
//!    - `attention.norm_q.weight` → `attention.q_norm.weight`
//!    - `attention.norm_k.weight` → `attention.k_norm.weight`
//!
//! 4. **Embedder rename.** Python uses an `nn.ModuleDict` keyed by the
//!    literal string `f"{patch_size}-{f_patch_size}"`, here `"16-1"`:
//!    - `all_x_embedder.16-1.weight/bias` → `x_embedder.weight/bias`
//!
//! All other keys (refiners' adaLN_modulation / feed_forward / norms /
//! t_embedder.mlp / cap_embedder / x_pad_token / cap_pad_token / out_conv)
//! pass through verbatim.

use flame_core::serialization::load_file;
use flame_core::{DType, Error, Result, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Block prefixes that need QKV fusion + norm rename + to_out rename:
///   noise_refiner.{0,1}, context_refiner.{0,1}, layers.{0..29}
fn all_attention_prefixes() -> Vec<String> {
    let mut v = Vec::with_capacity(34);
    for i in 0..2 {
        v.push(format!("noise_refiner.{i}"));
    }
    for i in 0..2 {
        v.push(format!("context_refiner.{i}"));
    }
    for i in 0..30 {
        v.push(format!("layers.{i}"));
    }
    v
}

/// In-memory translation. Pulls out as a separate fn so the test can build
/// a synthetic source HashMap without touching disk.
///
/// Consumes the source map (so renamed keys can move tensors without
/// clone) and returns the internal map.
/// Cast a tensor to BF16 if it's not already. The L2P merged checkpoint
/// ships roughly half its weights as F32 (~14.5 GB of 19.6 GB). Downstream
/// fused-kernel paths (`fused_rms_norm`, `fused_linear3d_native`,
/// `swiglu_fused_bf16`, `rope_fused_bf16`, `Conv2d::forward`) all require
/// BF16 — running with mixed dtypes either errors at the kernel boundary
/// or silently produces wrong output. We force-cast at load time so the
/// rest of the pipeline can assume BF16-only.
fn ensure_bf16(t: Tensor) -> Result<Tensor> {
    if t.dtype() == DType::BF16 {
        Ok(t)
    } else {
        t.to_dtype(DType::BF16)
    }
}

pub fn translate_l2p_keys(
    mut source: HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>> {
    let mut out: HashMap<String, Tensor> = HashMap::with_capacity(source.len());

    // ---------------------------------------------------------------------
    // 1. QKV fusion + attention.to_out / norm renames per attention block.
    // ---------------------------------------------------------------------
    for prefix in all_attention_prefixes() {
        let q_key = format!("{prefix}.attention.to_q.weight");
        let k_key = format!("{prefix}.attention.to_k.weight");
        let v_key = format!("{prefix}.attention.to_v.weight");

        let q = ensure_bf16(source.remove(&q_key).ok_or_else(|| {
            Error::InvalidInput(format!("L2P loader: missing source key {q_key}"))
        })?)?;
        let k = ensure_bf16(source.remove(&k_key).ok_or_else(|| {
            Error::InvalidInput(format!("L2P loader: missing source key {k_key}"))
        })?)?;
        let v = ensure_bf16(source.remove(&v_key).ok_or_else(|| {
            Error::InvalidInput(format!("L2P loader: missing source key {v_key}"))
        })?)?;

        // Fuse Q/K/V along dim 0 (the OUT-features dim of the [out, in]
        // safetensors layout). The DiT's `new_resident` pre-transposes
        // 2D weights to [in, out]; the runtime `chunk(3, dim=-1)` in
        // `joint_attention` then peels back along the final axis.
        // Mandatory `.contiguous()` after cat per CONTEXT.md "Known traps".
        // Q/K/V are already BF16 (cast above) so the cat output is BF16.
        let qkv = Tensor::cat(&[&q, &k, &v], 0)?.contiguous()?;
        out.insert(format!("{prefix}.attention.qkv.weight"), qkv);

        // attention.to_out.0.weight  →  attention.out.weight
        let to_out_key = format!("{prefix}.attention.to_out.0.weight");
        let to_out = ensure_bf16(source.remove(&to_out_key).ok_or_else(|| {
            Error::InvalidInput(format!("L2P loader: missing source key {to_out_key}"))
        })?)?;
        out.insert(format!("{prefix}.attention.out.weight"), to_out);

        // attention.norm_q.weight  →  attention.q_norm.weight
        // attention.norm_k.weight  →  attention.k_norm.weight
        let nq_key = format!("{prefix}.attention.norm_q.weight");
        let nk_key = format!("{prefix}.attention.norm_k.weight");
        let nq = ensure_bf16(source.remove(&nq_key).ok_or_else(|| {
            Error::InvalidInput(format!("L2P loader: missing source key {nq_key}"))
        })?)?;
        let nk = ensure_bf16(source.remove(&nk_key).ok_or_else(|| {
            Error::InvalidInput(format!("L2P loader: missing source key {nk_key}"))
        })?)?;
        out.insert(format!("{prefix}.attention.q_norm.weight"), nq);
        out.insert(format!("{prefix}.attention.k_norm.weight"), nk);
    }

    // ---------------------------------------------------------------------
    // 2. Embedder rename: all_x_embedder.16-1.{weight,bias} → x_embedder.*
    // ---------------------------------------------------------------------
    for suffix in &["weight", "bias"] {
        let src_key = format!("all_x_embedder.16-1.{suffix}");
        let dst_key = format!("x_embedder.{suffix}");
        let t = ensure_bf16(source.remove(&src_key).ok_or_else(|| {
            Error::InvalidInput(format!("L2P loader: missing source key {src_key}"))
        })?)?;
        out.insert(dst_key, t);
    }

    // ---------------------------------------------------------------------
    // 3. local_decoder Sequential unwraps.
    //    Sequential[Conv, SiLU]:  enc{1..4}, dec{1..4}, bottleneck  (Conv at .0.)
    //    Sequential[Upsample, Conv]: up{1..4}                       (Conv at .1.)
    // ---------------------------------------------------------------------
    let conv_zero_groups: [&str; 9] = [
        "local_decoder.enc1",
        "local_decoder.enc2",
        "local_decoder.enc3",
        "local_decoder.enc4",
        "local_decoder.dec1",
        "local_decoder.dec2",
        "local_decoder.dec3",
        "local_decoder.dec4",
        "local_decoder.bottleneck",
    ];
    for prefix in conv_zero_groups.iter() {
        for suffix in &["weight", "bias"] {
            let src_key = format!("{prefix}.0.{suffix}");
            let dst_key = format!("{prefix}.conv.{suffix}");
            let t = ensure_bf16(source.remove(&src_key).ok_or_else(|| {
                Error::InvalidInput(format!("L2P loader: missing source key {src_key}"))
            })?)?;
            out.insert(dst_key, t);
        }
    }
    let up_groups: [&str; 4] = [
        "local_decoder.up1",
        "local_decoder.up2",
        "local_decoder.up3",
        "local_decoder.up4",
    ];
    for prefix in up_groups.iter() {
        for suffix in &["weight", "bias"] {
            let src_key = format!("{prefix}.1.{suffix}");
            let dst_key = format!("{prefix}.conv.{suffix}");
            let t = ensure_bf16(source.remove(&src_key).ok_or_else(|| {
                Error::InvalidInput(format!("L2P loader: missing source key {src_key}"))
            })?)?;
            out.insert(dst_key, t);
        }
    }
    // local_decoder.out_conv.weight/bias  passthrough (raw Conv2d, no Sequential)
    for suffix in &["weight", "bias"] {
        let key = format!("local_decoder.out_conv.{suffix}");
        let t = ensure_bf16(source.remove(&key).ok_or_else(|| {
            Error::InvalidInput(format!("L2P loader: missing source key {key}"))
        })?)?;
        out.insert(key, t);
    }

    // ---------------------------------------------------------------------
    // 4. Verbatim passthrough for remaining keys.
    //
    //    - noise_refiner.{0,1}.{attention_norm1,attention_norm2,ffn_norm1,ffn_norm2}.weight
    //    - noise_refiner.{0,1}.adaLN_modulation.0.{weight,bias}
    //    - noise_refiner.{0,1}.feed_forward.{w1,w2,w3}.weight
    //    - context_refiner.{0,1}.{...norms, feed_forward} (no adaLN)
    //    - layers.{0..29}.{...norms, adaLN_modulation.0, feed_forward}
    //    - t_embedder.mlp.{0,2}.{weight,bias}
    //    - cap_embedder.0.weight
    //    - cap_embedder.1.{weight,bias}
    //    - x_pad_token, cap_pad_token
    //    - local_decoder.pool{N} has no params → not in source, nothing to do
    //
    //    We're permissive on the source side: every remaining key passes
    //    through unmodified. If a wholly-unknown key sneaks in (e.g. a
    //    `.opt_state.` accidentally saved into the merged file) it will
    //    pass through too — the downstream model just ignores unknown
    //    keys. That's looser than a strict allow-list but matches the
    //    spirit of every other loader in inference-flame.
    // ---------------------------------------------------------------------
    for (key, tensor) in source.drain() {
        out.insert(key, ensure_bf16(tensor)?);
    }

    // ---------------------------------------------------------------------
    // 5. Sanity: confirm every internal key the model reads is present.
    //    Cheap "missing key surfaces here, not deep inside forward".
    // ---------------------------------------------------------------------
    validate_internal_keys(&out)?;

    Ok(out)
}

/// Confirm the translated map contains every key `L2pDiT::new_resident` +
/// `MicroDiffusionModel::new` will read at construction. Catches stray
/// missing keys before the model panics on first forward.
fn validate_internal_keys(map: &HashMap<String, Tensor>) -> Result<()> {
    let mut missing: Vec<String> = Vec::new();

    let check = |missing: &mut Vec<String>, key: &str| {
        if !map.contains_key(key) {
            missing.push(key.to_string());
        }
    };

    // Attention block keys (34 blocks: 2 noise + 2 context + 30 layers)
    for prefix in all_attention_prefixes() {
        check(&mut missing, &format!("{prefix}.attention.qkv.weight"));
        check(&mut missing, &format!("{prefix}.attention.out.weight"));
        check(&mut missing, &format!("{prefix}.attention.q_norm.weight"));
        check(&mut missing, &format!("{prefix}.attention.k_norm.weight"));
        check(&mut missing, &format!("{prefix}.attention_norm1.weight"));
        check(&mut missing, &format!("{prefix}.attention_norm2.weight"));
        check(&mut missing, &format!("{prefix}.ffn_norm1.weight"));
        check(&mut missing, &format!("{prefix}.ffn_norm2.weight"));
        check(&mut missing, &format!("{prefix}.feed_forward.w1.weight"));
        check(&mut missing, &format!("{prefix}.feed_forward.w2.weight"));
        check(&mut missing, &format!("{prefix}.feed_forward.w3.weight"));

        // adaLN_modulation is present on noise_refiner.* and layers.*, but
        // NOT on context_refiner.* (modulation=false there).
        if !prefix.starts_with("context_refiner.") {
            check(
                &mut missing,
                &format!("{prefix}.adaLN_modulation.0.weight"),
            );
            check(&mut missing, &format!("{prefix}.adaLN_modulation.0.bias"));
        }
    }

    // Top-level embedders + pad tokens
    check(&mut missing, "x_embedder.weight");
    check(&mut missing, "x_embedder.bias");
    check(&mut missing, "t_embedder.mlp.0.weight");
    check(&mut missing, "t_embedder.mlp.0.bias");
    check(&mut missing, "t_embedder.mlp.2.weight");
    check(&mut missing, "t_embedder.mlp.2.bias");
    check(&mut missing, "cap_embedder.0.weight");
    check(&mut missing, "cap_embedder.1.weight");
    check(&mut missing, "cap_embedder.1.bias");
    check(&mut missing, "x_pad_token");
    check(&mut missing, "cap_pad_token");

    // U-Net convs
    for n in 1..=4 {
        check(
            &mut missing,
            &format!("local_decoder.enc{n}.conv.weight"),
        );
        check(&mut missing, &format!("local_decoder.enc{n}.conv.bias"));
        check(&mut missing, &format!("local_decoder.up{n}.conv.weight"));
        check(&mut missing, &format!("local_decoder.up{n}.conv.bias"));
        check(
            &mut missing,
            &format!("local_decoder.dec{n}.conv.weight"),
        );
        check(&mut missing, &format!("local_decoder.dec{n}.conv.bias"));
    }
    check(&mut missing, "local_decoder.bottleneck.conv.weight");
    check(&mut missing, "local_decoder.bottleneck.conv.bias");
    check(&mut missing, "local_decoder.out_conv.weight");
    check(&mut missing, "local_decoder.out_conv.bias");

    if !missing.is_empty() {
        // Print first 8 to keep the error readable.
        let preview: Vec<_> = missing.iter().take(8).cloned().collect();
        return Err(Error::InvalidInput(format!(
            "L2P loader: {n} required internal keys missing after translation. First 8: {preview:?}",
            n = missing.len(),
        )));
    }

    Ok(())
}

/// Load an L2P safetensors file and translate keys into the internal
/// layout used by `L2pDiT::new_resident` / `MicroDiffusionModel::new`.
///
/// Uses `flame_core::serialization::load_file` (mmap path). On success
/// returns a HashMap with every weight the model reads. On any missing
/// source-side key or post-translation gap, returns `Error::InvalidInput`
/// naming the first offending key.
pub fn load_l2p_safetensors(
    path: &Path,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    let source = load_file(path, device)?;
    translate_l2p_keys(source)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use flame_core::{DType, Shape};
    use std::sync::Arc;

    /// Build a minimal synthetic source map covering ONE attention block
    /// (`noise_refiner.0`) + every other required key, all as 1-element
    /// tensors of the right dtype. Exercises the rename + QKV fusion code
    /// path without any disk I/O or real weight shapes.
    ///
    /// QKV weights are sized so cat-along-dim-0 produces a meaningful
    /// `[3*K, K]` shape — gives the test something concrete to assert.
    fn build_minimal_source(
        device: &Arc<cudarc::driver::CudaDevice>,
    ) -> HashMap<String, Tensor> {
        let mut s = HashMap::new();

        // Helpers: tiny tensors. Real shapes don't matter for the
        // rename-correctness assertions below.
        let scalar = |val: f32| {
            Tensor::from_vec_dtype(
                vec![val; 1],
                Shape::from_dims(&[1]),
                device.clone(),
                DType::BF16,
            )
            .unwrap()
        };
        // QKV: each [4, 4] BF16 → cat dim=0 → [12, 4]
        let mat44 = |val: f32| {
            Tensor::from_vec_dtype(
                vec![val; 16],
                Shape::from_dims(&[4, 4]),
                device.clone(),
                DType::BF16,
            )
            .unwrap()
        };

        // -- attention block (one prefix: noise_refiner.0) -----------------
        // We populate ALL 34 attention prefixes so validate_internal_keys
        // accepts. The minimal source has to be a complete set.
        for prefix in all_attention_prefixes() {
            s.insert(
                format!("{prefix}.attention.to_q.weight"),
                mat44(1.0),
            );
            s.insert(
                format!("{prefix}.attention.to_k.weight"),
                mat44(2.0),
            );
            s.insert(
                format!("{prefix}.attention.to_v.weight"),
                mat44(3.0),
            );
            s.insert(
                format!("{prefix}.attention.to_out.0.weight"),
                scalar(0.5),
            );
            s.insert(
                format!("{prefix}.attention.norm_q.weight"),
                scalar(0.5),
            );
            s.insert(
                format!("{prefix}.attention.norm_k.weight"),
                scalar(0.5),
            );
            s.insert(format!("{prefix}.attention_norm1.weight"), scalar(0.5));
            s.insert(format!("{prefix}.attention_norm2.weight"), scalar(0.5));
            s.insert(format!("{prefix}.ffn_norm1.weight"), scalar(0.5));
            s.insert(format!("{prefix}.ffn_norm2.weight"), scalar(0.5));
            s.insert(format!("{prefix}.feed_forward.w1.weight"), scalar(0.5));
            s.insert(format!("{prefix}.feed_forward.w2.weight"), scalar(0.5));
            s.insert(format!("{prefix}.feed_forward.w3.weight"), scalar(0.5));
            if !prefix.starts_with("context_refiner.") {
                s.insert(
                    format!("{prefix}.adaLN_modulation.0.weight"),
                    scalar(0.5),
                );
                s.insert(
                    format!("{prefix}.adaLN_modulation.0.bias"),
                    scalar(0.5),
                );
            }
        }

        // -- Top-level ----------------------------------------------------
        s.insert("all_x_embedder.16-1.weight".into(), scalar(0.5));
        s.insert("all_x_embedder.16-1.bias".into(), scalar(0.5));
        s.insert("t_embedder.mlp.0.weight".into(), scalar(0.5));
        s.insert("t_embedder.mlp.0.bias".into(), scalar(0.5));
        s.insert("t_embedder.mlp.2.weight".into(), scalar(0.5));
        s.insert("t_embedder.mlp.2.bias".into(), scalar(0.5));
        s.insert("cap_embedder.0.weight".into(), scalar(0.5));
        s.insert("cap_embedder.1.weight".into(), scalar(0.5));
        s.insert("cap_embedder.1.bias".into(), scalar(0.5));
        s.insert("x_pad_token".into(), scalar(0.5));
        s.insert("cap_pad_token".into(), scalar(0.5));

        // -- U-Net local_decoder ------------------------------------------
        for n in 1..=4 {
            s.insert(format!("local_decoder.enc{n}.0.weight"), scalar(0.5));
            s.insert(format!("local_decoder.enc{n}.0.bias"), scalar(0.5));
            s.insert(format!("local_decoder.up{n}.1.weight"), scalar(0.5));
            s.insert(format!("local_decoder.up{n}.1.bias"), scalar(0.5));
            s.insert(format!("local_decoder.dec{n}.0.weight"), scalar(0.5));
            s.insert(format!("local_decoder.dec{n}.0.bias"), scalar(0.5));
        }
        s.insert("local_decoder.bottleneck.0.weight".into(), scalar(0.5));
        s.insert("local_decoder.bottleneck.0.bias".into(), scalar(0.5));
        s.insert("local_decoder.out_conv.weight".into(), scalar(0.5));
        s.insert("local_decoder.out_conv.bias".into(), scalar(0.5));

        s
    }

    #[test]
    fn translate_renames_and_fuses_qkv() {
        let device = cudarc::driver::CudaDevice::new(0)
            .expect("cuda dev 0; run on a box with a GPU");
        let source = build_minimal_source(&device);
        let out = translate_l2p_keys(source).expect("translate ok");

        // QKV fusion shape: [4, 4] cat-on-dim-0 three times → [12, 4]
        let qkv = out
            .get("noise_refiner.0.attention.qkv.weight")
            .expect("fused qkv key present");
        let dims = qkv.shape().dims();
        assert_eq!(dims, &[12, 4], "qkv shape after cat(dim=0)");

        // Renames
        assert!(out.contains_key("noise_refiner.0.attention.out.weight"));
        assert!(out.contains_key("noise_refiner.0.attention.q_norm.weight"));
        assert!(out.contains_key("noise_refiner.0.attention.k_norm.weight"));
        assert!(out.contains_key("layers.29.attention.qkv.weight"));
        assert!(out.contains_key("context_refiner.1.attention.qkv.weight"));

        // Embedder rename
        assert!(out.contains_key("x_embedder.weight"));
        assert!(out.contains_key("x_embedder.bias"));
        assert!(!out.contains_key("all_x_embedder.16-1.weight"));

        // local_decoder Sequential unwraps
        assert!(out.contains_key("local_decoder.enc1.conv.weight"));
        assert!(out.contains_key("local_decoder.enc1.conv.bias"));
        assert!(out.contains_key("local_decoder.up3.conv.weight"));
        assert!(out.contains_key("local_decoder.dec2.conv.bias"));
        assert!(out.contains_key("local_decoder.bottleneck.conv.weight"));
        assert!(out.contains_key("local_decoder.out_conv.weight"));
        // The old Sequential-index keys must NOT survive the translate.
        assert!(!out.contains_key("local_decoder.enc1.0.weight"));
        assert!(!out.contains_key("local_decoder.up3.1.weight"));
    }

    #[test]
    fn translate_errors_on_missing_qkv() {
        let device = cudarc::driver::CudaDevice::new(0)
            .expect("cuda dev 0; run on a box with a GPU");
        let mut source = build_minimal_source(&device);
        // Remove one required key — expect a clear-typed error mentioning it.
        let dropped = "noise_refiner.0.attention.to_q.weight";
        source.remove(dropped);

        let err = translate_l2p_keys(source).expect_err("must error");
        let msg = format!("{err}");
        assert!(
            msg.contains(dropped),
            "error msg should name the missing key, got: {msg}"
        );
    }
}
