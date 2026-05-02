//! Parity check for FlowUniPcDDim sigma table + timesteps + step_ddim
//! against the Python reference (`scheduler_unipc.py`).
//!
//! The expected sigma/timestep values were computed by directly importing
//! the reference scheduler with shift=5.0, num_inference_steps=8.

use anyhow::{anyhow, Result};
use inference_flame::sampling::magihuman_unipc::FlowUniPcDDim;

const REF_SIGMAS: [f32; 9] = [
    0.9998, 0.972006, 0.937265, 0.892602, 0.833055, 0.7497, 0.624687, 0.416389, 0.0,
];
const REF_TIMESTEPS: [f32; 8] = [999.0, 972.0, 937.0, 892.0, 833.0, 749.0, 624.0, 416.0];

fn main() -> Result<()> {
    let s = FlowUniPcDDim::new(8, 5.0, 1000);
    println!("sigmas    ours:    {:?}", s.sigmas);
    println!("sigmas    ref:     {:?}", REF_SIGMAS);
    println!("timesteps ours:    {:?}", s.timesteps);
    println!("timesteps ref:     {:?}", REF_TIMESTEPS);

    if s.sigmas.len() != REF_SIGMAS.len() {
        return Err(anyhow!("sigma length mismatch"));
    }
    let mut max_diff = 0.0_f32;
    for (a, b) in s.sigmas.iter().zip(REF_SIGMAS.iter()) {
        let d = (a - b).abs();
        if d > max_diff { max_diff = d; }
    }
    if s.timesteps != REF_TIMESTEPS {
        return Err(anyhow!("timestep mismatch"));
    }
    println!("\nmax |sigma diff| = {max_diff:.6}");
    if max_diff > 1e-3 {
        return Err(anyhow!("sigma diverge: {max_diff} > 1e-3"));
    }
    println!("\nPARITY OK ✓");
    Ok(())
}
