# SDXL Black Image — Root Cause Analysis

## Three bugs in `src/bin/sdxl_infer.rs`, all in `build_sdxl_schedule()` and init

### Bug 1: Wrong beta schedule (CRITICAL)
**File**: `src/bin/sdxl_infer.rs:35-37`
**Current (wrong)**:
```rust
// Linear beta schedule
let betas: Vec<f64> = (0..num_train_steps)
    .map(|i| beta_start + (beta_end - beta_start) * i as f64 / (num_train_steps - 1) as f64)
    .collect();
```
**Correct**: SDXL uses `scaled_linear` — `linspace(sqrt(beta_start), sqrt(beta_end), N)^2`:
```rust
let betas: Vec<f64> = (0..num_train_steps)
    .map(|i| {
        let v = beta_start.sqrt() + (beta_end.sqrt() - beta_start.sqrt()) * i as f64 / (num_train_steps - 1) as f64;
        v * v
    })
    .collect();
```
**Impact**: sigma_max becomes 14.61 (correct) instead of 25.15 (wrong). 1.7x error.

### Bug 2: Wrong timestep spacing (CRITICAL)
**File**: `src/bin/sdxl_infer.rs:47-52`
**Current (wrong)**:
```rust
let step_ratio = num_train_steps as f64 / num_steps as f64;
for i in 0..num_steps {
    let t = (num_train_steps as f64 - 1.0 - i as f64 * step_ratio).round() as usize;
    // Produces [999, 499] for 2 steps
```
**Correct**: SDXL uses `leading` spacing with `steps_offset=1`:
```rust
let step_ratio = num_train_steps / num_steps;  // integer division
let mut timestep_indices: Vec<usize> = (0..num_steps)
    .map(|i| i * step_ratio + 1)  // steps_offset = 1
    .collect();
timestep_indices.reverse();
// Produces [501, 1] for 2 steps
```
**Impact**: With 2 steps, Rust uses timesteps [999, 499] instead of correct [501, 1]. Combined with Bug 1, sigma_max is 25.15 instead of 1.62 — a **15.5x** error in initial noise scaling.

### Bug 3: Wrong init_noise_sigma (MODERATE)
**File**: `src/bin/sdxl_infer.rs:134-136`
**Current (wrong)**:
```rust
// Initialize x = noise * sigma_max
...mul_scalar(sigmas[0])?;
```
**Correct**: `init_noise_sigma = sqrt(sigma_max^2 + 1)`:
```rust
let init_sigma = (sigmas[0] * sigmas[0] + 1.0).sqrt();
...mul_scalar(init_sigma)?;
```
**Impact**: With correct schedule (sigma_max=1.62), Rust feeds `noise*0.85` into step 0 instead of `noise*1.0`. ~15% scaling error. At higher step counts the error is smaller since sigma_max is larger.

## What is NOT broken
- Timestep embedding formula: matches LDM reference exactly
- UNet architecture (block layout, skip connections, attention): key format matches checkpoint
- Weight loading: all 1680 keys load correctly with correct shapes
- Euler step formula: `x = x + eps * (sigma_next - sigma)` matches diffusers exactly
- Conv2d, GroupNorm, attention: NOT tested in isolation but Python UNet at same resolution produces correct output

## Fix Priority
1. Fix beta schedule to `scaled_linear` (line 35-37)
2. Fix timestep spacing to `leading` with `steps_offset=1` (line 47-52)  
3. Fix init_noise_sigma to `sqrt(sigma^2 + 1)` (line 134-136)

All three fixes are in `src/bin/sdxl_infer.rs`. The UNet code (`src/models/sdxl_unet.rs`) does NOT need changes.
