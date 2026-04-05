# Quality Parity: FLAME vs PyTorch LTX-2

## Priority: QUALITY FIRST. Speed and pipeline come later.

## Step 1: Generate PyTorch reference at IDENTICAL settings
- Match EVERY parameter to what Rust uses
- Run official pipeline with block-swap (Stagehand or manual)
- Save: frames, video, final latent (before VAE)

## Step 2: Compare
- Visual side-by-side
- Numerical latent comparison (max diff, mean diff, correlation)

## Step 3: If quality differs, find where
Check in order: sigma schedule, Euler step, RoPE frequencies, attention scaling,
timestep embedding, CFG application point

## Step 4: Fix each discrepancy one at a time

## Step 5: Better prompt test with detailed cinematic prompt

## Notes
- Zeros negative embedding works. Official empty-string negative doesn't (too similar
  to positive after connector normalization → CFG produces near-zero velocity).
- The LTX-2 repo git pull (2026-03-30) may affect VAE decode. Pin to ae855f8 if needed.
- Fused ops were reverted for quality baseline. Re-enable after parity confirmed.
