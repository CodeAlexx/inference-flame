//! ComboBox filename → absolute disk path resolution.
//!
//! The Base ComboBox in the Model section surfaces filenames like
//! `flux1-dev-Q4_K_M.gguf` or `sdxl-base-1.0.safetensors`. Each worker keeps
//! a hardcoded default path (safetensors) baked into a `const *_PATH`; when
//! the user picks a different file from the ComboBox we must resolve the
//! filename to an absolute path and pass it to the worker via
//! `GenerateJob::path`.
//!
//! Convention:
//!   - `Some(path)` → worker uses this path, overriding its hardcoded default.
//!   - `None` → worker uses its hardcoded default (backward-compat).
//!
//! AGENT-DEFAULT: we map each model family to a single base directory (the
//! same one the worker's existing hardcoded `const` points at). For models
//! whose default IS a directory/HF-snapshot path (Z-Image base, SD 1.5,
//! Stable Cascade, Qwen-Image, Chroma, ERNIE), we only resolve when the
//! filename is explicitly a `.gguf`; other cases return `None` so the worker
//! uses its native default (which preserves the existing sharded-safetensors
//! loading path). For models with a single .safetensors default file
//! (FLUX, Klein, SD3, SDXL, Anima), we resolve both .gguf and .safetensors
//! selections so users can drop alternates alongside the canonical file.
//!
//! If the resolved path doesn't exist on disk, we let the worker surface its
//! existing "not found at <path>" error — no pre-validation in the UI.

use super::ModelKind;

/// Map a ComboBox filename string to an absolute disk path. Returns `None`
/// to signal "worker should use its hardcoded default" (backward-compat for
/// variant-based dispatch and HF-snapshot directory layouts).
pub fn resolve_image_model_path(filename: &str, kind: ModelKind) -> Option<String> {
    // The `.serenity/models/checkpoints` base holds every model family whose
    // canonical .safetensors lives there. GGUF alternates conventionally sit
    // next to their safetensors sibling, so the same base applies.
    const SERENITY_CKPT: &str = "/home/alex/.serenity/models/checkpoints";
    const ERIDIFFUSION_CKPT: &str = "/home/alex/EriDiffusion/Models/checkpoints";
    const ERNIE_TRANSFORMER_DIR: &str = "/home/alex/models/ERNIE-Image/transformer";
    const ANIMA_DIT_DIR: &str =
        "/home/alex/EriDiffusion/Models/anima/split_files/diffusion_models";

    let is_gguf = filename.to_ascii_lowercase().ends_with(".gguf");

    match kind {
        ModelKind::Mock => None,
        // FLUX 1 Dev: default .safetensors lives at
        // /home/alex/.serenity/models/checkpoints/flux1-dev.safetensors.
        // Both .gguf alternates and explicit .safetensors selections resolve
        // to the same dir.
        ModelKind::FluxDev => Some(format!("{SERENITY_CKPT}/{filename}")),

        // Z-Image base: default is a DIRECTORY of sharded safetensors
        // (zimage_base/transformer). GGUF alternates are single files that
        // conventionally live under `checkpoints/`. Return `None` for the
        // safetensors case so the existing dir-scan path stays intact.
        ModelKind::ZImageBase => {
            if is_gguf {
                Some(format!("{SERENITY_CKPT}/{filename}"))
            } else {
                None
            }
        }
        // Z-Image turbo: default is a single safetensors file under
        // `checkpoints/z_image_turbo_bf16.safetensors`. GGUF alternate sits
        // alongside. Keep None for the canonical safetensors so the hardcoded
        // filename (which may differ from the ComboBox entry) stays authoritative.
        ModelKind::ZImageTurbo => {
            if is_gguf {
                Some(format!("{SERENITY_CKPT}/{filename}"))
            } else {
                None
            }
        }

        // Klein 4B / 9B: defaults under /home/alex/EriDiffusion/Models/checkpoints/
        // Filenames include the size token (klein-4b-Q4_K_M.gguf), so we resolve
        // to that dir directly. Override both .gguf and .safetensors — the
        // ComboBox controls which file gets loaded.
        ModelKind::Klein4B | ModelKind::Klein9B => {
            Some(format!("{ERIDIFFUSION_CKPT}/{filename}"))
        }

        // Chroma: default is a SHARDED set of HF-snapshot safetensors. A GGUF
        // alternate would be a single file; convention places it under the
        // serenity checkpoints dir (same as FLUX).
        ModelKind::Chroma => {
            if is_gguf {
                Some(format!("{SERENITY_CKPT}/{filename}"))
            } else {
                None
            }
        }

        // SD3.5 Medium: default is a single safetensors under .serenity/checkpoints/
        // (stablediffusion35_medium.safetensors). Both .gguf and .safetensors
        // alternates resolve to the same dir.
        ModelKind::Sd35 => Some(format!("{SERENITY_CKPT}/{filename}")),

        // Qwen-Image: default is SHARDED (9 shards under HF-snapshot dir). A
        // GGUF alternate is a single file under checkpoints/.
        ModelKind::QwenImage => {
            if is_gguf {
                Some(format!("{SERENITY_CKPT}/{filename}"))
            } else {
                None
            }
        }

        // ERNIE: default is a DIRECTORY scan
        // (/home/alex/models/ERNIE-Image/transformer). The worker's existing
        // dir-scan logic already handles both .gguf and .safetensors files
        // dropped into that dir — that's the one case where GGUF was already
        // reachable without path plumbing. For a direct GGUF file override
        // (user picked from ComboBox), resolve to a specific path inside that
        // same dir so the worker can be taught to accept a single-file override.
        //
        // AGENT-DEFAULT: keep None for safetensors (preserves existing scan).
        // For GGUF, resolve to the transformer dir + filename. The worker's
        // dir-scan will then find the file if it exists. Effective behavior
        // for the user: dropping the file anywhere under that dir works.
        ModelKind::ErnieImage => {
            if is_gguf {
                Some(format!("{ERNIE_TRANSFORMER_DIR}/{filename}"))
            } else {
                None
            }
        }

        // Anima: default is a single safetensors under
        // /home/alex/EriDiffusion/Models/anima/split_files/diffusion_models/.
        // Resolve all ComboBox selections to that same dir.
        ModelKind::Anima => Some(format!("{ANIMA_DIT_DIR}/{filename}")),

        // SDXL: default is a pre-extracted BF16 safetensors under
        // /home/alex/EriDiffusion/Models/checkpoints/. Resolve both .gguf
        // and .safetensors alternates to the same dir.
        ModelKind::Sdxl => Some(format!("{ERIDIFFUSION_CKPT}/{filename}")),

        // SD 1.5: default is an HF-snapshot subpath
        // (.../unet/diffusion_pytorch_model.safetensors). A GGUF alternate is
        // typically a single file dropped into the checkpoints dir — resolve
        // there. Keep None for safetensors so the HF-snapshot default path
        // (which includes the "/unet/" subdirectory and canonical filename)
        // stays authoritative.
        ModelKind::Sd15 => {
            if is_gguf {
                Some(format!("{SERENITY_CKPT}/{filename}"))
            } else {
                None
            }
        }

        // Stable Cascade: default is a three-stage HF-snapshot layout
        // (stage_c_bf16.safetensors, stage_b_bf16.safetensors, stage_a.safetensors).
        // A single-file ComboBox override can't meaningfully replace three
        // stages — return None so the worker uses its stage-path helpers.
        // (GGUF Cascade is supported by load_cascade_unet per-stage but needs
        // three separate overrides which the current UI doesn't surface.)
        ModelKind::Cascade => None,
    }
}
