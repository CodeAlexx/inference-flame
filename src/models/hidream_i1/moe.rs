//! Mixture-of-Experts FeedForward (SwiGLU experts + shared expert).
//!
//! Mirrors `edv2-reference/.../hidream/src/models/moe.py`:
//!
//! ```python
//! class MOEFeedForwardSwiGLU(nn.Module):
//!     def __init__(self, dim, hidden_dim, num_routed_experts, num_activated_experts):
//!         self.shared_experts = FeedForwardSwiGLU(dim, hidden_dim // 2)
//!         self.experts        = nn.ModuleList([FeedForwardSwiGLU(dim, hidden_dim) for _ in range(num_routed_experts)])
//!         self.gate           = MoEGate(dim, num_routed_experts, num_activated_experts)
//!
//!     def forward(self, x):
//!         identity      = x
//!         topk_idx, topk_w, _ = self.gate(x)    # [B*S, top_k], [B*S, top_k]
//!         x_flat        = x.view(-1, h)
//!         y_flat        = self.moe_infer(x_flat, topk_idx.view(-1), topk_w.view(-1, 1))
//!         return y_flat.view_as(x) + self.shared_experts(identity)
//! ```
//!
//! `moe_infer` is a sorted-routing dispatch:
//!   1. `idxs = argsort(flat_topk_idx)` — group token-expert assignments by expert.
//!   2. For each expert `e`, gather its assigned token rows, run the SwiGLU
//!      expert, multiply by the per-assignment top-k weight, scatter-add back
//!      to the destination row (`token_idx = idxs // top_k`).
//!
//! ## I1 specifics + LoRA filter
//! edv2-reference's canonical YAML (`train_lora_hidream_48.yaml`) excludes
//! `ff_i.experts` and `ff_i.gate` from LoRA (the router stays frozen). The
//! forward path still has to compute everything. We tag the candidate LoRA
//! sites with `// LORA-TARGET: ...` comments so the trainer can wrap them
//! later in M4 without touching this file.
//!
//! ## Implementation choices
//! - **No flame-core changes.** The gate softmax + top-k is computed by
//!   small CPU passes on the host gate logits (`scores` shape `[B*S, E]`,
//!   E=4). For B*S<200K and E=4, that's a few ms.
//! - **Expert dispatch via `index_select0`.** Per-expert we gather a
//!   `[N_e, D]` contiguous slab, run SwiGLU, then write back with
//!   `Tensor::cat` reconstruction. This avoids needing a CUDA scatter-add
//!   primitive flame-core doesn't expose.
//! - **scatter back**: instead of in-place scatter (no GPU primitive), we
//!   build a `[B*S, top_k, D]` weighted-output tensor where dim 1 is the
//!   k-th assignment, then sum over `dim=1`. Equivalent math, vectorized
//!   on GPU.
//!
//! See `transformer_hidream_image.py` `single_transformer_block.forward`
//! line 101 for the usage pattern (`ff_output_i = gate_mlp_i * self.ff_i(...)`).

use std::collections::HashMap;

use flame_core::{CudaDevice, DType, Result, Shape, Tensor};

/// Group of weights for one MoE block, parsed from the block's `weights`
/// HashMap. All entries are owning clones (Tensors are Arc'd internally).
pub struct MoeWeights<'a> {
    /// `shared_experts.w1.weight` `[hidden_shared, dim]`
    pub shared_w1: &'a Tensor,
    /// `shared_experts.w2.weight` `[dim, hidden_shared]`
    pub shared_w2: &'a Tensor,
    /// `shared_experts.w3.weight` `[hidden_shared, dim]`
    pub shared_w3: &'a Tensor,
    /// `experts.{e}.w1.weight` `[hidden_expert, dim]` — vector of len `num_routed_experts`
    pub expert_w1: Vec<&'a Tensor>,
    /// `experts.{e}.w2.weight` `[dim, hidden_expert]`
    pub expert_w2: Vec<&'a Tensor>,
    /// `experts.{e}.w3.weight` `[hidden_expert, dim]`
    pub expert_w3: Vec<&'a Tensor>,
    /// `gate.weight` `[num_routed_experts, dim]` — softmax router.
    pub gate_w: &'a Tensor,
}

impl<'a> MoeWeights<'a> {
    /// Collect the MoE weights for one block from a flat name map.
    /// `prefix` is the FFN namespace (e.g. `"double_stream_blocks.0.block.ff_i"`).
    pub fn from_block(
        weights: &'a HashMap<String, Tensor>,
        prefix: &str,
        num_routed_experts: usize,
    ) -> Result<Self> {
        let g = |k: &str| -> Result<&Tensor> {
            let key = format!("{prefix}.{k}");
            weights.get(&key).ok_or_else(|| {
                flame_core::Error::InvalidInput(format!("Missing MoE weight: {key}"))
            })
        };

        let shared_w1 = g("shared_experts.w1.weight")?;
        let shared_w2 = g("shared_experts.w2.weight")?;
        let shared_w3 = g("shared_experts.w3.weight")?;
        let gate_w = g("gate.weight")?;

        let mut expert_w1 = Vec::with_capacity(num_routed_experts);
        let mut expert_w2 = Vec::with_capacity(num_routed_experts);
        let mut expert_w3 = Vec::with_capacity(num_routed_experts);
        for e in 0..num_routed_experts {
            expert_w1.push(g(&format!("experts.{e}.w1.weight"))?);
            expert_w2.push(g(&format!("experts.{e}.w2.weight"))?);
            expert_w3.push(g(&format!("experts.{e}.w3.weight"))?);
        }

        Ok(Self {
            shared_w1,
            shared_w2,
            shared_w3,
            expert_w1,
            expert_w2,
            expert_w3,
            gate_w,
        })
    }
}

/// SwiGLU FeedForward forward (no bias, matches `FeedForwardSwiGLU` from
/// `attention.py`): `w2(silu(w1(x)) * w3(x))`.
///
/// Weights are `[hidden, dim]` layout (after `BlockOffloader`'s 2D
/// auto-transpose: original ckpt is `[hidden, dim]` for w1/w3 and
/// `[dim, hidden]` for w2, which `fused_linear3d_native` consumes directly).
///
/// `x`: `[N, dim]` BF16; returns `[N, dim]` BF16.
///
/// `fused_linear3d_native` requires 3D inputs (BUG #1 fix), so wrap with
/// `linear_compat` which unsqueezes 2D inputs.
fn swiglu_ffn(x: &Tensor, w1: &Tensor, w2: &Tensor, w3: &Tensor) -> Result<Tensor> {
    // h1 = silu(w1(x))   shape [N, hidden]
    let h1 = super::model::linear_compat(x, w1, None)?;
    let h1 = h1.silu()?;
    // h3 = w3(x)         shape [N, hidden]
    let h3 = super::model::linear_compat(x, w3, None)?;
    let gated = h1.mul(&h3)?;
    // w2(gated)          shape [N, dim]
    super::model::linear_compat(&gated, w2, None)
}

/// Top-k expert selection.
///
/// `scores`: `[N, E]` softmax probabilities BF16. Returns:
///   - `topk_idx`:  Vec<u32> of length `N * top_k` — flat expert indices.
///   - `topk_w`:    Vec<f32> of length `N * top_k` — corresponding weights.
///
/// Done on CPU; for N up to ~64K and E=4 this is a sub-millisecond pass.
fn cpu_topk(scores_f32: &[f32], n: usize, e_count: usize, top_k: usize) -> (Vec<u32>, Vec<f32>) {
    let mut idx_out = Vec::with_capacity(n * top_k);
    let mut w_out = Vec::with_capacity(n * top_k);
    // For each row, find top_k indices in `scores[row]`. E=4, top_k=2 → tiny.
    for row in 0..n {
        let s = &scores_f32[row * e_count..(row + 1) * e_count];
        // Selection sort for top_k (E is tiny, so no need for partial-heap).
        let mut taken = [false; 32]; // upper bound; E <= 32 for HiDream
        for _ in 0..top_k {
            let mut best_i = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in s.iter().enumerate() {
                if !taken[i] && v > best_v {
                    best_v = v;
                    best_i = i;
                }
            }
            taken[best_i] = true;
            idx_out.push(best_i as u32);
            w_out.push(best_v);
        }
    }
    (idx_out, w_out)
}

/// MoE FFN forward.
///
/// `x`: `[B, S, dim]` BF16. Returns `[B, S, dim]` BF16.
///
/// Algorithm:
///   1. **Gate**: `logits = x @ gate_w.T`, then `scores = softmax(logits, -1)`,
///      then per-row top-k.
///   2. **Shared expert**: `shared = shared_experts(x)`.
///   3. **Routed experts**: for each expert e, gather the rows where any of
///      the top-k assignments equals e, run the expert, scale by the matching
///      top-k weight, and accumulate into an `[N, top_k, dim]` workspace
///      indexed by (token row, which-of-its-k-slots assigned).
///   4. **Combine**: `routed = sum(workspace, dim=1)`.
///   5. Return `shared + routed`.
///
/// # Performance notes
/// - Each expert call is one SwiGLU on its share of tokens (~25% of total
///   for E=4, k=2: each token visits 2 experts, evenly = N/2 per expert).
/// - We use `index_select0` (already on GPU via gather_rows). No CUDA
///   scatter primitive needed — the workspace approach uses standard
///   ops only.
/// - The CPU topk + bincount adds ~1ms for 4K tokens on a 3090. Worth
///   optimizing later, not now.
///
/// LORA-TARGET: `gate.weight` is a Linear weight; the YAML excludes it
/// from LoRA by default, but the trainer's exclusion filter is the
/// authoritative gate, not this file.
/// LORA-TARGET: `experts.{e}.{w1,w2,w3}.weight` are Linear weights;
/// same exclusion-by-default treatment.
/// LORA-TARGET: `shared_experts.{w1,w2,w3}.weight` are Linear weights;
/// NOT in the YAML exclusion list, so they ARE LoRA'd.
pub fn moe_ffn_forward(
    x: &Tensor,
    weights: &MoeWeights<'_>,
    num_activated_experts: usize,
    device: &std::sync::Arc<CudaDevice>,
) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(flame_core::Error::InvalidInput(format!(
            "moe_ffn_forward expects [B, S, dim], got {dims:?}"
        )));
    }
    let (b, s, d) = (dims[0], dims[1], dims[2]);
    let n = b * s; // total tokens

    // TODO(M2.2 / BUG #12): the gate's softmax + top-k below pulls a
    // `[N, E]` F32 buffer to host every block (48 syncs/step). For B>1 or
    // training this should be a GPU top-k. Also TODO(BUG #13): verify
    // `swiglu_ffn` autograd registration when LoRA targets are added to
    // shared_experts (currently only matters for inference).

    // ----- 1. Gate: logits + softmax + top-k --------------------------------
    let x_flat = x.reshape(&[n, d])?;
    // gate_w: [E, dim] → logits = x_flat @ gate_w.T (2D input → linear_compat per BUG #1)
    let logits = super::model::linear_compat(&x_flat, weights.gate_w, None)?;
    let e_count = logits.shape().dims()[1];
    // softmax along last dim, BF16 has a fast path.
    let scores = logits.softmax(-1)?;
    // Pull to CPU as f32 for the tiny top-k pass.
    let scores_f32 = scores.to_dtype(DType::F32)?.to_vec()?;
    let (topk_idx, topk_w) = cpu_topk(&scores_f32, n, e_count, num_activated_experts);

    // ----- 2. Shared expert -------------------------------------------------
    let shared = swiglu_ffn(&x_flat, weights.shared_w1, weights.shared_w2, weights.shared_w3)?;

    // ----- 3. Routed experts ------------------------------------------------
    // Per-expert routing tables. token_for_expert[e] = vec of (token_row, slot_k)
    let routed_count = weights.expert_w1.len();
    let mut tokens_per_expert: Vec<Vec<(u32, u32)>> = vec![Vec::new(); routed_count];
    for token in 0..n {
        for k in 0..num_activated_experts {
            let flat = token * num_activated_experts + k;
            let e = topk_idx[flat] as usize;
            tokens_per_expert[e].push((token as u32, k as u32));
        }
    }

    // BUG #11 fix: removed dead `workspace` alloc (was allocated and dropped
    // unused — the vectorized per-slot path below has no scatter target).

    // For each expert: gather its assigned token rows, run SwiGLU, scale by
    // per-assignment weight, then scatter back via cat-based reconstruction.
    //
    // Because flame-core has no in-place scatter, we accumulate via:
    //   workspace[token, slot, :] = expert_out_for_that_assignment * weight
    // implemented by building per-slot contributions and `Tensor::cat` ing
    // along the slot dim — but that needs careful re-ordering. Simpler:
    // run all experts, get a [sum_N_e, D] output per expert, then iterate
    // host-side over assignments and use `narrow + add` to write the
    // workspace slab. Still on-GPU per slot because each slot occupies a
    // contiguous slab of workspace.
    //
    // For first-pass simplicity we use the per-slot rebuild approach: for
    // each slot k in 0..top_k, gather *one* expert per token (the one
    // assigned at slot k), produce a [N, D] tensor, multiply by the weights
    // for that slot, then stack them along dim 1.
    let mut per_slot_outputs: Vec<Tensor> = Vec::with_capacity(num_activated_experts);
    for k in 0..num_activated_experts {
        // For each token, which expert did it pick at slot k? Build the
        // index list and grouping for this slot only.
        let mut tokens_by_expert_at_slot: Vec<Vec<u32>> = vec![Vec::new(); routed_count];
        let mut weights_at_slot: Vec<f32> = vec![0.0; n];
        for token in 0..n {
            let flat = token * num_activated_experts + k;
            let e = topk_idx[flat] as usize;
            let w = topk_w[flat];
            tokens_by_expert_at_slot[e].push(token as u32);
            weights_at_slot[token] = w;
        }

        // Result for this slot: [N, D] BF16, computed by routing per expert.
        // For each expert e: gather the rows it owns, run SwiGLU, then
        // we'll reorder back to original token order via index_select with
        // a permutation.
        //
        // To avoid an explicit scatter we use a single index_select0 over
        // the *concatenated per-expert outputs* with a precomputed
        // permutation index.
        //
        // Simpler implementation: compute one per-expert output, copy each
        // row back into a CPU-staged buffer (size n*d), then upload as one
        // tensor. For BF16 with n up to ~64K and d=2560 that's ~330MB — too
        // big to round-trip per step. Skip CPU staging.
        //
        // Vectorized GPU implementation: for each expert e with N_e tokens
        // out of N total:
        //   - rows_e:        I32 [N_e]   token indices owned by expert e
        //   - x_e:           [N_e, D]    = x_flat.index_select0(rows_e)
        //   - y_e:           [N_e, D]    = swiglu(x_e)
        //   - place y_e back into a [N, D] buffer at rows rows_e.
        //
        // The place-back step needs a scatter-by-rows. flame-core lacks
        // scatter, but we can fake it as: build a permutation that
        // re-orders concatenated y across all experts into original token
        // order, then `index_select0` that permutation.
        //
        // perm[token_row] = position in concatenated buffer.
        let mut concat_outputs: Vec<Tensor> = Vec::with_capacity(routed_count);
        let mut cumulative = 0usize;
        let mut perm: Vec<u32> = vec![0; n];
        let mut expert_has_any = false;
        for (e, rows) in tokens_by_expert_at_slot.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            expert_has_any = true;
            // Build I32 row-index tensor on device via F32 → I32 cast (per
            // flame-core convention: from_vec_dtype doesn't support I32 directly).
            let rows_f32: Vec<f32> = rows.iter().map(|&r| r as f32).collect();
            let row_ids = Tensor::from_vec(
                rows_f32,
                Shape::from_dims(&[rows.len()]),
                device.clone(),
            )?
            .to_dtype(DType::I32)?;
            // Gather → run expert → enqueue.
            let x_e = x_flat.index_select0(&row_ids)?;
            let y_e = swiglu_ffn(
                &x_e,
                weights.expert_w1[e],
                weights.expert_w2[e],
                weights.expert_w3[e],
            )?;
            // Record where each gathered token lands in the concatenated buf.
            for (i, &row) in rows.iter().enumerate() {
                perm[row as usize] = (cumulative + i) as u32;
            }
            cumulative += rows.len();
            concat_outputs.push(y_e);
        }

        let slot_y = if !expert_has_any {
            // No tokens routed to any expert at this slot (shouldn't happen
            // with softmax routing) — produce zeros.
            Tensor::zeros_dtype(Shape::from_dims(&[n, d]), DType::BF16, device.clone())?
        } else {
            // Concat all per-expert outputs along the row dim, then permute
            // back to original token order via index_select0.
            let cat_refs: Vec<&Tensor> = concat_outputs.iter().collect();
            let concat = Tensor::cat(&cat_refs, 0)?;
            let perm_f32: Vec<f32> = perm.iter().map(|&v| v as f32).collect();
            let perm_ids = Tensor::from_vec(
                perm_f32,
                Shape::from_dims(&[n]),
                device.clone(),
            )?
            .to_dtype(DType::I32)?;
            concat.index_select0(&perm_ids)?
        };

        // Scale by weights for this slot.
        let w_tensor = Tensor::from_vec_dtype(
            weights_at_slot.clone(),
            Shape::from_dims(&[n, 1]),
            device.clone(),
            DType::BF16,
        )?;
        let scaled = slot_y.mul(&w_tensor)?;
        per_slot_outputs.push(scaled);
    }

    // Sum over slots: workspace built as stack along dim 1, then sum dim 1.
    // We can equivalently just add: routed = sum(per_slot_outputs).
    let mut routed = per_slot_outputs.swap_remove(0);
    for slab in per_slot_outputs.iter() {
        routed = routed.add(slab)?;
    }
    let combined = routed.add(&shared)?;
    combined.reshape(&[b, s, d])
}
