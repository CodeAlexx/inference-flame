//! Temperature sampling for Gemma-4 decode.
//!
//! Reference: `prompt_agent.py:174-179` — `do_sample=True` when
//! `temperature > 0`, else greedy argmax. The Python uses HF
//! `generate()` with implicit top-p=1.0 / top-k=0, i.e. pure
//! temperature sampling over the full vocab (no truncation).
//!
//! ```text
//! probs = softmax(logits / temperature)
//! next_token ~ Categorical(probs)
//! ```
//!
//! ## Host vs GPU — AGENT-DEFAULT (Builder 3, 2026-05-21)
//!
//! The skeleton called for an all-GPU sampler (logits → scale → softmax
//! → cumsum → uniform → searchsorted). flame-core today exposes
//! `Tensor::softmax` (BF16 last-dim fast path) but has no `argmax`,
//! `cumsum`, or `searchsorted` primitives — see audit notes in the
//! Builder 3 report at `/tmp/builder3_orchestration_report.md`.
//!
//! Rather than land three new flame-core kernels for one consumer
//! (against TENET 5 — workarounds live in the wrong place), we follow
//! the precedent set at `inference-flame/src/models/sensenova_u1.rs:1640`:
//!
//! 1. Compute the softmax / scaling on GPU when needed.
//! 2. dtoh the `[1, vocab]` row (`262_144 × 2 B = ~520 KB`) — the same
//!    transfer sensenova_u1 already pays per decoded token.
//! 3. Do argmax / multinomial on the host (microseconds at vocab=262K).
//! 4. Wrap the chosen id back into a `[B]` I32 device tensor so the
//!    caller's API stays "Tensor in, Tensor out".
//!
//! The host round-trip is ~520 KB per token; at 4096-token generation
//! that's ~2 GB across the run, which is small next to the per-token
//! 11-weight BlockOffloader page-in. If a future flame-core sampler
//! lands, this file is the single point that swaps over.

use flame_core::{DType, Result, Shape, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Stateless sampler — RNG state lives in the struct so generate()
/// stays single-threaded and reproducible.
pub struct TemperatureSampler {
    /// Master seed; consumed at `new`. The internal `rng` is the
    /// per-call state.
    pub seed: u64,
    /// Deterministic uniform RNG. Reseeded once at `new`; advances per
    /// sample call.
    rng: StdRng,
}

impl TemperatureSampler {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Reset the RNG state to the seed. Useful for parity testing
    /// across two consecutive decode calls.
    pub fn reset(&mut self) {
        self.rng = StdRng::seed_from_u64(self.seed);
    }

    /// Sample one token id per batch row from a logits tensor.
    ///
    /// `logits`: `[B, vocab_size]` BF16 (post-softcap from
    /// `Gemma4ForCausalLM::lm_head`).
    /// `temperature`: if < 1e-5, return argmax (greedy). Else divide
    /// logits and sample from the resulting categorical distribution.
    ///
    /// Returns a `[B]` I32 device tensor holding the sampled token ids.
    /// Callers typically immediately read this via `to_vec_i32()` to
    /// build the next-step input — the dtoh is unavoidable because
    /// the autoregressive feedback path needs the id as a host integer
    /// to assemble the next `[B, 1]` input_ids tensor.
    pub fn sample(&mut self, logits: &Tensor, temperature: f32) -> Result<Tensor> {
        let dims = logits.shape().dims().to_vec();
        if dims.len() != 2 {
            return Err(flame_core::Error::InvalidInput(format!(
                "TemperatureSampler::sample: expected [B, vocab], got {dims:?}"
            )));
        }
        let b = dims[0];
        let vocab = dims[1];

        // Branch on greedy vs multinomial. Both paths end with a host-side
        // u32 per batch row that we package into an I32 device tensor.
        let mut ids: Vec<i32> = Vec::with_capacity(b);

        if temperature < 1e-5 {
            // Greedy: argmax on host. dtoh the BF16→F32 tensor once.
            let host = logits.to_vec_f32()?;
            if host.len() != b * vocab {
                return Err(flame_core::Error::InvalidInput(format!(
                    "TemperatureSampler::sample (greedy): host buffer {} != B*V {}",
                    host.len(),
                    b * vocab
                )));
            }
            for row in 0..b {
                let row_start = row * vocab;
                let row_slice = &host[row_start..row_start + vocab];
                let mut best_i: usize = 0;
                let mut best_x: f32 = row_slice[0];
                for (i, &x) in row_slice.iter().enumerate().skip(1) {
                    if x > best_x {
                        best_x = x;
                        best_i = i;
                    }
                }
                ids.push(best_i as i32);
            }
        } else {
            // Multinomial via temperature-scaled softmax.
            //
            // We do the scale + softmax on GPU (BF16 last-dim fast path
            // when present), then dtoh the probability row and run the
            // cumsum + binary-search on host. Doing it this way keeps
            // the heavy elementwise work on the GPU while sidestepping
            // the missing cumsum/searchsorted primitives.
            let scale = 1.0f32 / temperature;
            let scaled = logits.mul_scalar(scale)?;
            // Tensor::softmax dispatches to bf16_elementwise::softmax_lastdim_bf16
            // for the BF16 last-dim fast path (no scratch allocs).
            let probs = scaled.softmax(-1)?;
            let host = probs.to_vec_f32()?;
            if host.len() != b * vocab {
                return Err(flame_core::Error::InvalidInput(format!(
                    "TemperatureSampler::sample (multinomial): host buffer {} != B*V {}",
                    host.len(),
                    b * vocab
                )));
            }
            for row in 0..b {
                let u: f32 = self.rng.gen_range(0.0f32..1.0f32);
                let row_start = row * vocab;
                // Inline cumsum-and-pick: walk the row accumulating
                // probability until we hit `u`. On numerical underflow
                // (sum < u from BF16 round-off), fall back to the last
                // index. This matches PyTorch's torch.multinomial
                // behaviour at the BF16 boundary.
                let mut acc: f32 = 0.0;
                let mut picked: usize = vocab - 1;
                for i in 0..vocab {
                    acc += host[row_start + i];
                    if u <= acc {
                        picked = i;
                        break;
                    }
                }
                ids.push(picked as i32);
            }
        }

        // Pack host ids back into a device I32 tensor of shape [B].
        let dev_arc = logits.device().clone();
        let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
        Tensor::from_vec(ids_f32, Shape::from_dims(&[b]), dev_arc)?.to_dtype(DType::I32)
    }
}
