//! HeliosDMDScheduler — pure-Rust port of
//! `diffusers/schedulers/scheduling_helios_dmd.py` (Helios-Distilled config).
//!
//! 3-stage pyramid sampler with γ-renoise correction at stage transitions
//! (the latent resolution doubles between stages, so noise is re-injected
//! at the boundary). Distilled config runs `[2, 2, 2]` denoise steps × 3
//! stages = 6 total per chunk.
//!
//! Schedule construction is done in `f64` to match numpy/torch defaults.
//! Tensors stored / step() output is `f32`-castable for parity testing.

use flame_core::{Error, Result, Tensor};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeShiftType {
    Linear,
    Exponential,
}

#[derive(Clone, Debug)]
pub struct HeliosDMDConfig {
    pub num_train_timesteps: usize,
    pub shift: f64,
    pub stages: usize,
    pub stage_range: Vec<f64>, // length = stages + 1
    pub gamma: f64,
    pub use_flow_sigmas: bool,
    pub use_dynamic_shifting: bool,
    pub time_shift_type: TimeShiftType,
}

impl HeliosDMDConfig {
    /// Distilled config from
    /// `~/.cache/huggingface/hub/models--BestWishYsh--Helios-Distilled/snapshots/.../scheduler/scheduler_config.json`.
    pub fn distilled_default() -> Self {
        Self {
            num_train_timesteps: 1000,
            shift: 1.0,
            stages: 3,
            stage_range: vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
            gamma: 1.0 / 3.0,
            use_flow_sigmas: true,
            use_dynamic_shifting: true,
            time_shift_type: TimeShiftType::Linear,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HeliosDMDState {
    pub config: HeliosDMDConfig,

    /// Full schedule from `init_sigmas()` — length `num_train_timesteps`.
    pub full_sigmas: Vec<f64>,
    pub full_timesteps: Vec<f64>,

    /// Per-stage scalars from `init_sigmas_for_each_stage()`.
    pub start_sigmas: Vec<f64>,     // length = stages, post γ-renoise correction
    pub end_sigmas: Vec<f64>,       // length = stages
    pub ori_start_sigmas: Vec<f64>, // length = stages, pre-correction
    pub timestep_ratios: Vec<(f64, f64)>, // length = stages
    pub timesteps_per_stage: Vec<Vec<f64>>, // [stage][N]
    pub sigmas_per_stage: Vec<Vec<f64>>,    // [stage][N] — stage-invariant linspace(0.999, 0, N+1)[:-1]

    /// Per-call (after `set_timesteps`): length `num_inference_steps` and
    /// `num_inference_steps + 1` respectively (sigmas has trailing 0).
    pub timesteps: Vec<f64>,
    pub sigmas: Vec<f64>,
}

impl HeliosDMDState {
    pub fn new(config: HeliosDMDConfig) -> Self {
        let mut s = Self {
            config,
            full_sigmas: Vec::new(),
            full_timesteps: Vec::new(),
            start_sigmas: Vec::new(),
            end_sigmas: Vec::new(),
            ori_start_sigmas: Vec::new(),
            timestep_ratios: Vec::new(),
            timesteps_per_stage: Vec::new(),
            sigmas_per_stage: Vec::new(),
            timesteps: Vec::new(),
            sigmas: Vec::new(),
        };
        s.init_sigmas_for_each_stage();
        s
    }

    /// `init_sigmas`:
    /// ```text
    /// alphas = linspace(1, 1/N, N+1)
    /// sigmas = 1 - alphas
    /// sigmas = flip(shift * sigmas / (1 + (shift-1)*sigmas))[:-1]
    /// timesteps = sigmas * N
    /// ```
    pub fn init_sigmas(&mut self) {
        let n = self.config.num_train_timesteps;
        let shift = self.config.shift;

        // alphas[i] = 1 - i*(1 - 1/N)/N for i in 0..=N → length N+1
        let alphas: Vec<f64> = (0..=n)
            .map(|i| 1.0 - (i as f64) * (1.0 - 1.0 / n as f64) / n as f64)
            .collect();
        let sigmas_pre: Vec<f64> = alphas.iter().map(|&a| 1.0 - a).collect();
        let shifted: Vec<f64> = sigmas_pre
            .iter()
            .map(|&s| shift * s / (1.0 + (shift - 1.0) * s))
            .collect();
        // flip then drop last → length N
        let mut flipped: Vec<f64> = shifted.iter().rev().copied().collect();
        flipped.pop(); // drop last (= old first = 0.0)
        let timesteps: Vec<f64> = flipped.iter().map(|&s| s * n as f64).collect();

        self.full_sigmas = flipped;
        self.full_timesteps = timesteps;
        // Mirror the per-call vectors to the full schedule until set_timesteps reseeds.
        self.sigmas = self.full_sigmas.clone();
        self.timesteps = self.full_timesteps.clone();
    }

    /// `init_sigmas_for_each_stage` — compute per-stage scalars + arrays.
    pub fn init_sigmas_for_each_stage(&mut self) {
        self.init_sigmas();

        let n = self.config.num_train_timesteps;
        let stages = self.config.stages;
        let stage_range = &self.config.stage_range;
        let gamma = self.config.gamma;

        self.start_sigmas = vec![0.0; stages];
        self.end_sigmas = vec![0.0; stages];
        self.ori_start_sigmas = vec![0.0; stages];
        let mut stage_distance = vec![0.0f64; stages];

        for is_ in 0..stages {
            let start_indice = ((stage_range[is_] * n as f64) as i64).max(0) as usize;
            let end_indice_raw = (stage_range[is_ + 1] * n as f64) as i64;
            let end_indice = end_indice_raw.min(n as i64) as usize;
            let start_sigma = self.full_sigmas[start_indice];
            let end_sigma = if end_indice < n {
                self.full_sigmas[end_indice]
            } else {
                0.0
            };
            self.ori_start_sigmas[is_] = start_sigma;

            let start_sigma = if is_ != 0 {
                let ori_sigma = 1.0 - start_sigma;
                let corrected =
                    (1.0 / ((1.0 + 1.0 / gamma).sqrt() * (1.0 - ori_sigma) + ori_sigma)) * ori_sigma;
                1.0 - corrected
            } else {
                start_sigma
            };

            stage_distance[is_] = start_sigma - end_sigma;
            self.start_sigmas[is_] = start_sigma;
            self.end_sigmas[is_] = end_sigma;
        }

        let tot_distance: f64 = stage_distance.iter().sum();
        self.timestep_ratios = (0..stages)
            .map(|is_| {
                let start_ratio = if is_ == 0 {
                    0.0
                } else {
                    stage_distance[..is_].iter().sum::<f64>() / tot_distance
                };
                let end_ratio = if is_ == stages - 1 {
                    0.9999999999999999
                } else {
                    stage_distance[..=is_].iter().sum::<f64>() / tot_distance
                };
                (start_ratio, end_ratio)
            })
            .collect();

        self.timesteps_per_stage = Vec::with_capacity(stages);
        self.sigmas_per_stage = Vec::with_capacity(stages);

        for is_ in 0..stages {
            let (lo, hi) = self.timestep_ratios[is_];
            let max_idx = ((lo * n as f64) as i64).max(0) as usize;
            let timestep_max = self.full_timesteps[max_idx].min(999.0);
            let min_idx = ((hi * n as f64) as i64).min((n - 1) as i64) as usize;
            let timestep_min = self.full_timesteps[min_idx];

            // np.linspace(timestep_max, timestep_min, N+1)[:-1] → length N
            let stage_ts: Vec<f64> = linspace(timestep_max, timestep_min, n + 1)
                .into_iter()
                .take(n)
                .collect();
            self.timesteps_per_stage.push(stage_ts);

            // sigmas_per_stage = np.linspace(0.999, 0, N+1)[:-1] — stage-invariant
            let stage_sg: Vec<f64> = linspace(0.999, 0.0, n + 1).into_iter().take(n).collect();
            self.sigmas_per_stage.push(stage_sg);
        }
    }

    /// `set_timesteps(num, stage_idx, mu, is_amplify_first_chunk)`:
    /// builds per-call `self.timesteps` (length num) and `self.sigmas`
    /// (length num + 1, last is 0). Applies `time_shift(mu, 1.0, sigmas)`
    /// when `use_dynamic_shifting`.
    pub fn set_timesteps(
        &mut self,
        num_inference_steps: usize,
        stage_index: usize,
        mu: Option<f64>,
        is_amplify_first_chunk: bool,
    ) {
        let nis = if is_amplify_first_chunk {
            num_inference_steps * 2 + 1
        } else {
            num_inference_steps + 1
        };

        // Reset full schedule (matches diffusers's `self.init_sigmas()` call).
        self.init_sigmas();

        let mut sigmas: Vec<f64>;
        let mut timesteps: Vec<f64>;

        if self.config.stages == 1 {
            // No external sigmas override path supported here — Distilled never
            // hits stages=1 with sigmas=None per the pipeline.
            sigmas = linspace(1.0, 1.0 / self.config.num_train_timesteps as f64, nis + 1)
                .into_iter()
                .take(nis)
                .collect();
            if self.config.shift != 1.0 {
                assert!(
                    !self.config.use_dynamic_shifting,
                    "shift != 1.0 incompatible with use_dynamic_shifting"
                );
                let sh = self.config.shift;
                sigmas = sigmas
                    .iter()
                    .map(|&s| sh * s / (1.0 + (sh - 1.0) * s))
                    .collect();
            }
            timesteps = sigmas
                .iter()
                .map(|&s| s * self.config.num_train_timesteps as f64)
                .collect();
        } else {
            let stage_ts = &self.timesteps_per_stage[stage_index];
            let stage_sg = &self.sigmas_per_stage[stage_index];
            timesteps = linspace(*stage_ts.first().unwrap(), *stage_ts.last().unwrap(), nis);
            sigmas = linspace(*stage_sg.first().unwrap(), *stage_sg.last().unwrap(), nis);
        }

        // self.timesteps = from_numpy(timesteps); self.sigmas = cat([sigmas, [0]])
        // Then drop last timestep, drop second-to-last sigma (keep trailing 0).
        let mut sigmas_with_zero = sigmas.clone();
        sigmas_with_zero.push(0.0);

        // self.timesteps = timesteps[:-1]
        timesteps.pop();
        // self.sigmas = cat([sigmas_with_zero[:-2], sigmas_with_zero[-1:]])
        //             = remove element at index len-2
        let last = *sigmas_with_zero.last().unwrap();
        sigmas_with_zero.truncate(sigmas_with_zero.len() - 2);
        sigmas_with_zero.push(last);

        // Dynamic shifting (Distilled has use_dynamic_shifting=true, shift=1.0).
        if self.config.use_dynamic_shifting {
            assert!(
                (self.config.shift - 1.0).abs() < 1e-12,
                "use_dynamic_shifting requires shift==1.0"
            );
            let mu = mu.expect("mu required when use_dynamic_shifting=true");
            let shifted: Vec<f64> = sigmas_with_zero
                .iter()
                .map(|&t| self.time_shift(mu, 1.0, t))
                .collect();
            sigmas_with_zero = shifted;

            if self.config.stages == 1 {
                let n = self.config.num_train_timesteps as f64;
                timesteps = sigmas_with_zero[..sigmas_with_zero.len() - 1]
                    .iter()
                    .map(|&s| s * n)
                    .collect();
            } else {
                let stage_ts = &self.timesteps_per_stage[stage_index];
                let stage_min = stage_ts.iter().cloned().fold(f64::INFINITY, f64::min);
                let stage_max = stage_ts.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                timesteps = sigmas_with_zero[..sigmas_with_zero.len() - 1]
                    .iter()
                    .map(|&s| stage_min + s * (stage_max - stage_min))
                    .collect();
            }
        }

        self.timesteps = timesteps;
        self.sigmas = sigmas_with_zero;
    }

    pub fn time_shift(&self, mu: f64, sigma: f64, t: f64) -> f64 {
        match self.config.time_shift_type {
            TimeShiftType::Linear => time_shift_linear(mu, sigma, t),
            TimeShiftType::Exponential => time_shift_exponential(mu, sigma, t),
        }
    }

    /// `add_noise(original_samples, noise, timestep, sigmas, timesteps)`:
    /// `sigma = sigmas[argmin(|timesteps - timestep|)]`
    /// `sample = (1 - sigma) * original + sigma * noise`
    pub fn add_noise(
        &self,
        original_samples: &Tensor,
        noise: &Tensor,
        timestep: f64,
        sigmas: &[f64],
        timesteps: &[f64],
    ) -> Result<Tensor> {
        let id = argmin_abs_diff(timesteps, timestep);
        let sigma = sigmas[id];
        let one_minus = original_samples.mul_scalar((1.0 - sigma) as f32)?;
        let scaled_noise = noise.mul_scalar(sigma as f32)?;
        one_minus.add(&scaled_noise)
    }

    /// `convert_flow_pred_to_x0(flow_pred, xt, timestep, sigmas, timesteps)`:
    /// `sigma_t = sigmas[argmin(|timesteps - timestep|)]`
    /// `x0 = xt - sigma_t * flow_pred`
    /// (Diffusers does this in F64 for stability; we do BF16-in/BF16-out
    /// with f32 scalar — within BF16 noise.)
    pub fn convert_flow_pred_to_x0(
        &self,
        flow_pred: &Tensor,
        xt: &Tensor,
        timestep: f64,
        sigmas: &[f64],
        timesteps: &[f64],
    ) -> Result<Tensor> {
        let id = argmin_abs_diff(timesteps, timestep);
        let sigma_t = sigmas[id];
        let scaled = flow_pred.mul_scalar(sigma_t as f32)?;
        xt.sub(&scaled)
    }

    /// `step(model_output, timestep, sample, dmd_noisy_tensor, dmd_sigmas,
    ///       dmd_timesteps, all_timesteps, cur_sampling_step) -> prev_sample`:
    /// 1. x0 = convert_flow_pred_to_x0(model_output, sample, timestep, dmd_sigmas, dmd_timesteps)
    /// 2. if cur < len(all_timesteps) - 1:
    ///       prev = add_noise(x0, dmd_noisy_tensor, all_timesteps[cur+1], dmd_sigmas, dmd_timesteps)
    ///    else:
    ///       prev = x0
    pub fn step(
        &self,
        model_output: &Tensor,
        timestep: f64,
        sample: &Tensor,
        dmd_noisy_tensor: &Tensor,
        dmd_sigmas: &[f64],
        dmd_timesteps: &[f64],
        all_timesteps: &[f64],
        cur_sampling_step: usize,
    ) -> Result<Tensor> {
        if all_timesteps.is_empty() {
            return Err(Error::InvalidInput(
                "step: all_timesteps must be non-empty".into(),
            ));
        }
        let x0 = self.convert_flow_pred_to_x0(
            model_output,
            sample,
            timestep,
            dmd_sigmas,
            dmd_timesteps,
        )?;
        if cur_sampling_step < all_timesteps.len() - 1 {
            let next_t = all_timesteps[cur_sampling_step + 1];
            self.add_noise(&x0, dmd_noisy_tensor, next_t, dmd_sigmas, dmd_timesteps)
        } else {
            Ok(x0)
        }
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// `numpy.linspace(start, stop, num)` — inclusive endpoints, length `num`.
/// Matches `np.linspace` exactly: step = (stop-start) / (num-1) for num>=2.
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let step = (stop - start) / (num as f64 - 1.0);
    (0..num).map(|i| start + (i as f64) * step).collect()
}

/// `argmin(|timesteps[i] - target|)` — pure-CPU scalar lookup.
fn argmin_abs_diff(timesteps: &[f64], target: f64) -> usize {
    let mut best_i = 0usize;
    let mut best_d = f64::INFINITY;
    for (i, &t) in timesteps.iter().enumerate() {
        let d = (t - target).abs();
        if d < best_d {
            best_d = d;
            best_i = i;
        }
    }
    best_i
}

fn time_shift_linear(mu: f64, sigma: f64, t: f64) -> f64 {
    if t == 0.0 {
        // (1/0 - 1)^sigma = inf for sigma > 0; mu / (mu + inf) = 0.
        return 0.0;
    }
    mu / (mu + (1.0 / t - 1.0).powf(sigma))
}

fn time_shift_exponential(mu: f64, sigma: f64, t: f64) -> f64 {
    if t == 0.0 {
        return 0.0;
    }
    mu.exp() / (mu.exp() + (1.0 / t - 1.0).powf(sigma))
}

/// Convenience: also exposed as a free fn so callers can use the time shift
/// without instantiating a state.
pub fn time_shift(kind: TimeShiftType, mu: f64, sigma: f64, t: f64) -> f64 {
    match kind {
        TimeShiftType::Linear => time_shift_linear(mu, sigma, t),
        TimeShiftType::Exponential => time_shift_exponential(mu, sigma, t),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaDevice;
    use flame_core::serialization::load_file;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn take(map: &mut HashMap<String, Tensor>, key: &str) -> Tensor {
        map.remove(key)
            .unwrap_or_else(|| panic!("fixture missing key: {key}"))
    }

    fn vec_max_mean(a: &[f64], b: &[f32]) -> (f64, f64) {
        assert_eq!(a.len(), b.len(), "len mismatch");
        let mut max = 0.0f64;
        let mut sum = 0.0f64;
        for (x, &y) in a.iter().zip(b.iter()) {
            let d = (x - y as f64).abs();
            if d > max {
                max = d;
            }
            sum += d;
        }
        (max, sum / a.len() as f64)
    }

    fn tensor_max_mean(got: &[f32], expected: &[f32]) -> (f32, f32) {
        assert_eq!(got.len(), expected.len(), "len mismatch");
        let mut max = 0.0f32;
        let mut sum = 0.0f64;
        for (&g, &e) in got.iter().zip(expected.iter()) {
            let d = (g - e).abs();
            if d > max {
                max = d;
            }
            sum += d as f64;
        }
        (max, (sum / got.len() as f64) as f32)
    }

    #[test]
    fn linspace_endpoints_and_length() {
        let xs = linspace(0.999, 0.0, 1001);
        assert_eq!(xs.len(), 1001);
        assert!((xs[0] - 0.999).abs() < 1e-15);
        assert!(xs.last().unwrap().abs() < 1e-15);
    }

    #[test]
    fn helios_dmd_parity_vs_pytorch() {
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/pytorch_fixtures/helios/scheduler_reference.safetensors");
        if !fixture.exists() {
            eprintln!("fixture missing — generate via scripts/generate_helios_scheduler.py");
            return;
        }
        let device = CudaDevice::new(0).expect("cuda dev 0");
        let device: Arc<CudaDevice> = device;
        let mut map = load_file(&fixture, &device).expect("load fixture");

        let cfg = HeliosDMDConfig::distilled_default();
        let mut state = HeliosDMDState::new(cfg.clone());

        // ---------- init invariants ----------
        let exp_full_sigmas = take(&mut map, "init.full_sigmas").to_vec_f32().unwrap();
        let exp_full_timesteps = take(&mut map, "init.full_timesteps").to_vec_f32().unwrap();
        let (m, mn) = vec_max_mean(&state.full_sigmas, &exp_full_sigmas);
        eprintln!("full_sigmas: max={m:.3e} mean={mn:.3e}");
        assert!(m < 1e-6, "full_sigmas max diff {m} exceeds 1e-6");

        let (m, mn) = vec_max_mean(&state.full_timesteps, &exp_full_timesteps);
        eprintln!("full_timesteps: max={m:.3e} mean={mn:.3e}");
        assert!(m < 5e-4, "full_timesteps max diff {m} exceeds 5e-4");

        // ---------- per-stage scalars ----------
        let exp_start = take(&mut map, "stage.start_sigmas").to_vec_f32().unwrap();
        let exp_end = take(&mut map, "stage.end_sigmas").to_vec_f32().unwrap();
        let exp_ori = take(&mut map, "stage.ori_start_sigmas").to_vec_f32().unwrap();
        let exp_lo = take(&mut map, "stage.timestep_ratios_lo").to_vec_f32().unwrap();
        let exp_hi = take(&mut map, "stage.timestep_ratios_hi").to_vec_f32().unwrap();

        let (m, _) = vec_max_mean(&state.start_sigmas, &exp_start);
        eprintln!("start_sigmas: max={m:.3e}  rust={:?} py={:?}", state.start_sigmas, exp_start);
        assert!(m < 1e-6, "start_sigmas max {m}");
        let (m, _) = vec_max_mean(&state.end_sigmas, &exp_end);
        eprintln!("end_sigmas:   max={m:.3e}  rust={:?} py={:?}", state.end_sigmas, exp_end);
        assert!(m < 1e-6, "end_sigmas max {m}");
        let (m, _) = vec_max_mean(&state.ori_start_sigmas, &exp_ori);
        assert!(m < 1e-6, "ori_start_sigmas max {m}");

        let lo: Vec<f64> = state.timestep_ratios.iter().map(|&(l, _)| l).collect();
        let hi: Vec<f64> = state.timestep_ratios.iter().map(|&(_, h)| h).collect();
        let (m, _) = vec_max_mean(&lo, &exp_lo);
        assert!(m < 1e-6, "timestep_ratios.lo max {m}");
        let (m, _) = vec_max_mean(&hi, &exp_hi);
        assert!(m < 1e-6, "timestep_ratios.hi max {m}");

        // ---------- per-stage arrays ----------
        for s in 0..cfg.stages {
            let exp_ts = take(&mut map, &format!("stage.timesteps_per_stage_{s}"))
                .to_vec_f32()
                .unwrap();
            let exp_sg = take(&mut map, &format!("stage.sigmas_per_stage_{s}"))
                .to_vec_f32()
                .unwrap();
            let (m, _) = vec_max_mean(&state.timesteps_per_stage[s], &exp_ts);
            eprintln!("timesteps_per_stage[{s}]: max={m:.3e}");
            assert!(m < 5e-4, "timesteps_per_stage[{s}] max {m}");
            let (m, _) = vec_max_mean(&state.sigmas_per_stage[s], &exp_sg);
            eprintln!("sigmas_per_stage[{s}]:   max={m:.3e}");
            assert!(m < 1e-6, "sigmas_per_stage[{s}] max {m}");
        }

        // ---------- set_timesteps per stage ----------
        for s in 0..cfg.stages {
            state.set_timesteps(2, s, Some(0.5), false);
            let exp_ts = take(&mut map, &format!("set.call_timesteps_s{s}"))
                .to_vec_f32()
                .unwrap();
            let exp_sg = take(&mut map, &format!("set.call_sigmas_s{s}"))
                .to_vec_f32()
                .unwrap();
            let (m, _) = vec_max_mean(&state.timesteps, &exp_ts);
            eprintln!(
                "set_timesteps stage={s}: ts max={m:.3e} rust={:?} py={:?}",
                state.timesteps, exp_ts
            );
            assert!(m < 5e-4, "set call timesteps stage {s} max {m}");
            let (m, _) = vec_max_mean(&state.sigmas, &exp_sg);
            eprintln!(
                "set_timesteps stage={s}: sg max={m:.3e} rust={:?} py={:?}",
                state.sigmas, exp_sg
            );
            assert!(m < 1e-6, "set call sigmas stage {s} max {m}");
        }

        // ---------- step() — non-final ----------
        // Re-set for stage 0 since we want dmd_sigmas/timesteps from that call.
        state.set_timesteps(2, 0, Some(0.5), false);

        let model_out = take(&mut map, "step_inputs.model_output");
        let sample = take(&mut map, "step_inputs.sample");
        let dmd_noise = take(&mut map, "step_inputs.dmd_noisy_tensor");
        let dmd_sigmas_t = take(&mut map, "step_inputs.dmd_sigmas").to_vec_f32().unwrap();
        let dmd_ts_t = take(&mut map, "step_inputs.dmd_timesteps").to_vec_f32().unwrap();
        let all_ts_t = take(&mut map, "step_inputs.all_timesteps").to_vec_f32().unwrap();
        let timestep_t = take(&mut map, "step_inputs.timestep").to_vec_f32().unwrap();
        let _cur = take(&mut map, "step_inputs.cur_sampling_step")
            .to_vec_f32()
            .unwrap()[0] as usize;
        let expected = take(&mut map, "step_expected.prev_sample");

        let dmd_sigmas: Vec<f64> = dmd_sigmas_t.iter().map(|&x| x as f64).collect();
        let dmd_ts: Vec<f64> = dmd_ts_t.iter().map(|&x| x as f64).collect();
        let all_ts: Vec<f64> = all_ts_t.iter().map(|&x| x as f64).collect();
        let timestep = timestep_t[0] as f64;

        let got = state
            .step(&model_out, timestep, &sample, &dmd_noise, &dmd_sigmas, &dmd_ts, &all_ts, 0)
            .expect("step");
        let g = got.to_vec_f32().unwrap();
        let e = expected.to_vec_f32().unwrap();
        let (max_abs, mean_abs) = tensor_max_mean(&g, &e);
        eprintln!("step(cur=0) prev_sample: max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}");
        assert!(mean_abs < 1e-3, "step mean_abs {mean_abs} exceeds 1e-3");
        assert!(max_abs < 1e-2, "step max_abs {max_abs} exceeds 1e-2");

        // ---------- step() — last (cur = len-1) → just x0_pred ----------
        let timestep_last_t = take(&mut map, "step_inputs.timestep_last")
            .to_vec_f32()
            .unwrap();
        let cur_last = take(&mut map, "step_inputs.cur_sampling_step_last")
            .to_vec_f32()
            .unwrap()[0] as usize;
        let expected_last = take(&mut map, "step_expected.prev_sample_last");
        let got_last = state
            .step(
                &model_out,
                timestep_last_t[0] as f64,
                &sample,
                &dmd_noise,
                &dmd_sigmas,
                &dmd_ts,
                &all_ts,
                cur_last,
            )
            .expect("step last");
        let gl = got_last.to_vec_f32().unwrap();
        let el = expected_last.to_vec_f32().unwrap();
        let (max_abs, mean_abs) = tensor_max_mean(&gl, &el);
        eprintln!("step(cur=last) prev_sample: max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}");
        assert!(mean_abs < 1e-3, "step-last mean_abs {mean_abs} exceeds 1e-3");
        assert!(max_abs < 1e-2, "step-last max_abs {max_abs} exceeds 1e-2");
    }
}
