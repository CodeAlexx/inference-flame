pub mod ldm_decoder;
pub use ldm_decoder::LdmVAEDecoder;
pub mod ldm_encoder;
pub use ldm_encoder::LdmVAEEncoder;
pub mod ltx2_vae;
pub use ltx2_vae::LTX2VaeDecoder;
pub mod ltx2_encoder;
pub use ltx2_encoder::LTX2VaeEncoder;
pub mod klein_vae;
pub use klein_vae::KleinVaeDecoder;
pub mod lens_vae_wrapper;
pub use lens_vae_wrapper::LensVaeWrapper;
pub mod wan21_vae;
pub use wan21_vae::Wan21VaeDecoder;
pub mod wan21_encoder;
pub use wan21_encoder::Wan21VaeEncoder;
pub mod wan22_vae;
pub use wan22_vae::Wan22VaeDecoder;
// Wan2.2 VAE encoder (z_dim=48) lives at crate::models::wan::encoder per
// the existing Wan2.2 TI2V-5B module organization. Re-exported here so
// the canonical import path is `inference_flame::vae::Wan22VaeEncoder`,
// matching the convention used by other VAEs in this module.
pub use crate::models::wan::Wan22VaeEncoder;
pub mod ltx2_audio_vae;
pub use ltx2_audio_vae::{LTX2AudioVaeDecoder, LTX2AudioVaeEncoder};
pub mod ltx2_vocoder;
pub use ltx2_vocoder::{LTX2Vocoder, LTX2VocoderWithBWE};
pub mod nava_audio_wrap;
pub use nava_audio_wrap::nava_decode_audio;
pub mod hunyuan_vae;
pub use hunyuan_vae::HunyuanVaeDecoder;
pub mod qwenimage_encoder;
pub use qwenimage_encoder::QwenImageVaeEncoder;
pub mod qwenimage_decoder;
pub use qwenimage_decoder::QwenImageVaeDecoder;
pub mod acestep_vae;
pub use acestep_vae::OobleckVaeDecoder;
pub mod oklab;
pub use oklab::{decode_planar as oklab_decode_planar, encode_planar as oklab_encode_planar};
