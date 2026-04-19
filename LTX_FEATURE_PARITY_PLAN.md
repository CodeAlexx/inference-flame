# LTX-2 Feature Parity Plan

Goal: close the gap between our pure-Rust LTX-2 inference and what
Lightricks ships in [LTX-Video](https://github.com/Lightricks/LTX-Video)
+ [LTX-Desktop](https://github.com/Lightricks/LTX-Desktop). Each item
has a PyTorch parity test as the gate before implementation.

## Workflow per item

```
1. Write PyTorch/diffusers reference (emit safetensors with expected outputs)
2. Write Rust parity bin (loads ref, runs Rust impl, reports cos_sim)
3. Builder agent: implement
4. Skeptic pass: review (same agent, self-critique)
5. Bug-fixer: address findings
6. Re-run parity — must pass before commit
```

## Current state

| # | Item | Parity test | Rust impl | Notes |
|---|------|-------------|-----------|-------|
| 1 | **LoRA fusion (math)** | ✅ PASS 12/12 @ 0.999999 | ✅ `lora_loader.rs` | Landed session 13 |
| 2 | **LoRA wiring into `ltx2_generate`** | — | 🟡 in progress (builder agent) | blocks 3,4 |
| 3 | **Negative prompts in LTX-2 bins** | pending | pending | needs uncond embed encode |
| 4 | **Spatial + temporal upscaler wired into gen** | pending | pending | upscalers exist as test bins |
| 5 | **STG (Spatiotemporal Guidance)** | pending | pending | skip-layer perturbation |
| 6 | **PAG (Perturbed Attention Guidance)** | pending | pending | similar machinery to STG |
| 7 | **IC-LoRA reference conditioning** | pending | pending | depth/pose/canny/detailer (4 variants); ~~LoRA fusion~~ is just one piece |
| 8 | **Multi-keyframe conditioning + V-extension** | pending | pending | forward/backward extension |
| 9 | **Prompt enhancement** | pending | pending | which model enhances, what prompts |
| 10 | **TeaCache** | pending | pending | step-cache acceleration |

## Priority order

- 🟢 **Phase A — foundational, unlocks demos**
  1. LoRA wiring (in progress)
  2. Negative prompts
  3. Spatial + temporal upscaler wired in
- 🟡 **Phase B — quality upgrades**
  4. STG
  5. PAG
- 🟠 **Phase C — conditioning breadth**
  6. IC-LoRA reference conditioning (depth/pose/canny/detailer)
  7. Multi-keyframe + video extension
- 🔵 **Phase D — nice-to-have**
  8. Prompt enhancement
  9. TeaCache

## References under investigation

Research agent `ltx-researcher` (running 2026-04-19) is reading
Lightricks's repo to produce `LTX_FEATURE_PARITY.md` with exact file:line
references for every feature. Each subsequent phase will cite from that
doc so nobody has to re-derive formulas.

## Session-13 checkpoint

Delivered:
- LoRA fusion parity (commit `9430e51`): 12/12 keys at 0.999999 on
  LTX-2 distilled LoRA.
- Qwen2.5-VL encoder parity (commit `7b29976`): three real bugs
  fixed; final_hidden cos_sim 0.994 on real tokens; full E2E
  qwenimage_gen coherent.
- QwenImage VAE decoder (commit `3c93718`): first Rust path for
  qwen-image-2512 VAE, parity cos_sim 0.999994.

In flight:
- `ltx-researcher` agent — LTX-Video API feature catalog.
- `lora-wiring-builder` agent — LTX2StreamingModel LoRA integration.
