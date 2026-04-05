//! Dump timestep embedding values for comparison against Python reference.
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use inference_flame::models::ltx2_model::{LTX2Config, LTX2StreamingModel};

const MODEL_PATH: &str = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors";

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let device = global_cuda_device();

    let config = LTX2Config::default();
    // Load globals — connector may fail, that's OK for this test
    let model = match LTX2StreamingModel::load_globals(MODEL_PATH, &config) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Warning: load_globals failed ({e}), but we only need time_embed");
            // Can't proceed without the model. Let's try loading just what we need.
            std::process::exit(1);
        }
    };

    // sigma=1.0, 270 tokens, timestep_scale_multiplier=1000
    let sigma = 1.0f32;
    let num_tokens = 270;
    let inner_dim = config.inner_dim(); // 4096

    // Replicate timestep computation from forward_video_only
    let timestep = Tensor::from_f32_to_bf16(
        vec![sigma], Shape::from_dims(&[1]), device.clone(),
    )?;
    let ts_expanded = timestep.unsqueeze(1)?.expand(&[1, num_tokens])?;
    let ts_scaled = ts_expanded.mul_scalar(config.timestep_scale_multiplier as f32)?;
    let ts_flat = ts_scaled.reshape(&[num_tokens])?;

    let num_mod_params = model.time_embed.num_mod_params;
    let (v_timestep, v_embedded) = model.time_embed.forward(&ts_flat)?;

    let v_timestep = v_timestep.reshape(&[1, num_tokens, num_mod_params * inner_dim])?;
    let v_embedded = v_embedded.reshape(&[1, num_tokens, inner_dim])?;

    println!("timestep_embed: {:?} {:?}", v_timestep.shape().dims(), v_timestep.dtype());
    println!("embedded_timestep: {:?} {:?}", v_embedded.shape().dims(), v_embedded.dtype());

    let temb_data = v_timestep.to_vec()?;
    let emb_data = v_embedded.to_vec()?;

    println!("  temb first 8: {:?}", &temb_data[..8]);
    println!("  embedded first 8: {:?}", &emb_data[..8]);

    let temb_mean: f32 = temb_data.iter().sum::<f32>() / temb_data.len() as f32;
    let emb_mean: f32 = emb_data.iter().sum::<f32>() / emb_data.len() as f32;
    println!("  temb mean={:.6}", temb_mean);
    println!("  embedded mean={:.6}", emb_mean);

    Ok(())
}
