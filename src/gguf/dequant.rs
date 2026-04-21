//! CPU-side dequantization from GGML quant formats to `Vec<f32>`.
//!
//! Port notes:
//! - Math ported from `ggml/src/ggml-quants.c` in llama.cpp (canonical C
//!   reference) and cross-checked against `gguf-py/gguf/quants.py` (numpy
//!   reference). Function-to-function mapping:
//!   - `dequant_q8_0_block`   ← `dequantize_row_q8_0`
//!   - `dequant_q4_k_block`   ← `dequantize_row_q4_K`
//!   - `dequant_q5_k_block`   ← `dequantize_row_q5_K`
//!   - `dequant_q6_k_block`   ← `dequantize_row_q6_K`
//! - All block layouts are little-endian, exactly as stored on disk.
//! - K-quant 6-bit scale/min packing: two 6-bit values packed as a pair into
//!   12 bytes (8 sub-blocks × 6 bits each for sc + 8 × 6 bits for m = 96 bits
//!   = 12 bytes). Unpack logic mirrors llama.cpp `get_scale_min_k4`.
//!
//! Outputs are always `Vec<f32>` — the upload path converts to BF16 via
//! `Tensor::from_f32_to_bf16`, matching what the rest of inference-flame does.
//!
//! No rayon: one-shot cost at load time, keeps the dep surface minimal.
//! Rayon can be added later if benchmarks demand it.

use anyhow::{bail, Result};

use super::reader::GgufQuantType;

/// Dequantize a tensor's raw bytes into `Vec<f32>` of `n_elements` values.
pub fn dequantize_to_f32(
    quant: GgufQuantType,
    bytes: &[u8],
    n_elements: usize,
) -> Result<Vec<f32>> {
    match quant {
        GgufQuantType::F32 => dequant_f32(bytes, n_elements),
        GgufQuantType::F16 => dequant_f16(bytes, n_elements),
        GgufQuantType::BF16 => dequant_bf16(bytes, n_elements),
        GgufQuantType::Q8_0 => dequant_q8_0(bytes, n_elements),
        GgufQuantType::Q4_K => dequant_q4_k(bytes, n_elements),
        GgufQuantType::Q5_K => dequant_q5_k(bytes, n_elements),
        GgufQuantType::Q6_K => dequant_q6_k(bytes, n_elements),
        GgufQuantType::Q4_0 => bail!(
            "GGUF: Q4_0 dequant not implemented (rare in image models; file a bug if needed)"
        ),
        GgufQuantType::Q2_K => bail!("GGUF: Q2_K dequant not implemented"),
        GgufQuantType::Q3_K => bail!("GGUF: Q3_K dequant not implemented"),
        GgufQuantType::Unsupported(code) => {
            bail!("GGUF: unsupported ggml_type {code}")
        }
    }
}

// ---------------------------------------------------------------------------
// Trivial formats
// ---------------------------------------------------------------------------

fn dequant_f32(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if bytes.len() != n * 4 {
        bail!("F32: byte len {} != 4 * {}", bytes.len(), n);
    }
    let mut out = vec![0f32; n];
    for (o, c) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    Ok(out)
}

fn dequant_f16(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if bytes.len() != n * 2 {
        bail!("F16: byte len {} != 2 * {}", bytes.len(), n);
    }
    let mut out = vec![0f32; n];
    for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
        let bits = u16::from_le_bytes([c[0], c[1]]);
        *o = half::f16::from_bits(bits).to_f32();
    }
    Ok(out)
}

fn dequant_bf16(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if bytes.len() != n * 2 {
        bail!("BF16: byte len {} != 2 * {}", bytes.len(), n);
    }
    let mut out = vec![0f32; n];
    for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
        let bits = u16::from_le_bytes([c[0], c[1]]);
        *o = half::bf16::from_bits(bits).to_f32();
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Q8_0: 32-element blocks — { d: f16, qs: [i8; 32] }
// 34 bytes per block. Ported from `dequantize_row_q8_0` in ggml-quants.c.
// ---------------------------------------------------------------------------

const Q8_0_BLOCK_SIZE: usize = 34;
const Q8_0_ELEMS: usize = 32;

fn dequant_q8_0(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % Q8_0_ELEMS != 0 {
        bail!("Q8_0: n_elements {n} not divisible by 32");
    }
    let n_blocks = n / Q8_0_ELEMS;
    let expected = n_blocks * Q8_0_BLOCK_SIZE;
    if bytes.len() != expected {
        bail!(
            "Q8_0: byte len {} != expected {} ({} blocks)",
            bytes.len(),
            expected,
            n_blocks
        );
    }

    let mut out = vec![0f32; n];
    for (bi, block) in bytes.chunks_exact(Q8_0_BLOCK_SIZE).enumerate() {
        let out_slice = &mut out[bi * Q8_0_ELEMS..(bi + 1) * Q8_0_ELEMS];
        dequant_q8_0_block(block, out_slice);
    }
    Ok(out)
}

#[inline]
fn dequant_q8_0_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q8_0_BLOCK_SIZE);
    debug_assert_eq!(out.len(), Q8_0_ELEMS);
    let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    for i in 0..Q8_0_ELEMS {
        // The qs bytes start at offset 2 and are stored as signed i8.
        let q = block[2 + i] as i8;
        out[i] = d * q as f32;
    }
}

// ---------------------------------------------------------------------------
// K-quant shared: unpack 6-bit scale/min pairs from 12 bytes.
// Mirrors `get_scale_min_k4` from llama.cpp `ggml-quants.c`.
//
// The 12 bytes encode, for 8 sub-blocks (j=0..8):
//   sc[j] (6 bits) and m[j] (6 bits).
//
// Packing (llama.cpp reference):
//   For j < 4:
//     sc[j] = q[j]     & 63
//     m [j] = q[j+4]   & 63
//   For j >= 4:
//     sc[j] = (q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4)
//     m [j] = (q[j+4] >>   4) | ((q[j  ] >> 6) << 4)
// ---------------------------------------------------------------------------

#[inline]
fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

// ---------------------------------------------------------------------------
// Q4_K: 256-element superblocks, 8 sub-blocks of 32.
// Block layout (144 bytes):
//   d      : f16           (2 B)
//   dmin   : f16           (2 B)
//   scales : [u8; 12]      (scale/min 6-bit packed for 8 sub-blocks)
//   qs     : [u8; 128]     (4-bit weights; 2 weights per byte, 8 sub-blocks × 32 elems)
//
// Per sub-block j (0..8):
//   (sc_j, m_j) = get_scale_min_k4(j)
//   d_j     = d    * sc_j
//   dmin_j  = dmin * m_j
//   The 32 4-bit weights for sub-block j are packed across 16 bytes.
//   Within one byte: low nibble is weight in sub-block j (first half,
//   j & 1 == 0), high nibble is weight in sub-block j+1 (second half,
//   j & 1 == 1). See `dequantize_row_q4_K` in ggml-quants.c.
//
// Reference pseudocode from ggml-quants.c:
//   for each superblock:
//     for j in 0..256 step 64:
//       (sc0, m0) = get_scale_min_k4(is  , scales)
//       (sc1, m1) = get_scale_min_k4(is+1, scales)
//       d0 = d*sc0; m0f = dmin*m0
//       d1 = d*sc1; m1f = dmin*m1
//       for l in 0..32:
//         y[j + l]      = d0 * (qs[l] & 0x0F) - m0f
//         y[j + l + 32] = d1 * (qs[l] >>   4) - m1f
//       qs += 32
//       is += 2
// ---------------------------------------------------------------------------

const QK_K: usize = 256;
const Q4_K_BLOCK_SIZE: usize = 144; // 2 + 2 + 12 + 128

fn dequant_q4_k(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK_K != 0 {
        bail!("Q4_K: n_elements {n} not divisible by 256");
    }
    let n_blocks = n / QK_K;
    let expected = n_blocks * Q4_K_BLOCK_SIZE;
    if bytes.len() != expected {
        bail!(
            "Q4_K: byte len {} != expected {} ({} blocks)",
            bytes.len(),
            expected,
            n_blocks
        );
    }

    let mut out = vec![0f32; n];
    for (bi, block) in bytes.chunks_exact(Q4_K_BLOCK_SIZE).enumerate() {
        let out_slice = &mut out[bi * QK_K..(bi + 1) * QK_K];
        dequant_q4_k_block(block, out_slice);
    }
    Ok(out)
}

fn dequant_q4_k_block(block: &[u8], y: &mut [f32]) {
    debug_assert_eq!(block.len(), Q4_K_BLOCK_SIZE);
    debug_assert_eq!(y.len(), QK_K);

    let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    let dmin = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();

    let mut scales = [0u8; 12];
    scales.copy_from_slice(&block[4..16]);
    let qs = &block[16..16 + 128];

    // Process 256 elements in groups of 64: two sub-blocks (32 + 32) per
    // 32-byte chunk of qs.
    let mut is = 0usize; // sub-block index (0..8)
    let mut y_ofs = 0usize;
    let mut q_ofs = 0usize;
    while is < 8 {
        let (sc0, m0) = get_scale_min_k4(is, &scales);
        let (sc1, m1) = get_scale_min_k4(is + 1, &scales);
        let d0 = d * sc0 as f32;
        let m0f = dmin * m0 as f32;
        let d1 = d * sc1 as f32;
        let m1f = dmin * m1 as f32;

        for l in 0..32 {
            let q = qs[q_ofs + l];
            y[y_ofs + l] = d0 * (q & 0x0F) as f32 - m0f;
            y[y_ofs + l + 32] = d1 * (q >> 4) as f32 - m1f;
        }

        y_ofs += 64;
        q_ofs += 32;
        is += 2;
    }
}

// ---------------------------------------------------------------------------
// Q5_K: 256-element superblocks. 176 bytes per block.
// Layout:
//   d      : f16         (2 B)
//   dmin   : f16         (2 B)
//   scales : [u8; 12]    (same 6-bit packing as Q4_K)
//   qh     : [u8; 32]    (one extra high bit per weight; 256 bits = 32 bytes)
//   qs     : [u8; 128]   (low 4 bits per weight, same packing as Q4_K)
//
// For each weight: q = ((qh_bit) << 4) | q_nibble  (5-bit, 0..31)
// Then: y = d_j * q - dmin_j  (same per-sub-block scales/mins as Q4_K)
//
// Reference: `dequantize_row_q5_K` in ggml-quants.c.
// The qh bit for element at "column" l in the current 64-element window
// uses bit (is/2) from qh[l] for low half, bit (is/2 + 1) for high half.
// Concretely, within the 32-byte qh array, bit `u` of each byte gives the
// high bit for sub-block pair (2u, 2u+1).
// ---------------------------------------------------------------------------

const Q5_K_BLOCK_SIZE: usize = 176;

fn dequant_q5_k(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK_K != 0 {
        bail!("Q5_K: n_elements {n} not divisible by 256");
    }
    let n_blocks = n / QK_K;
    let expected = n_blocks * Q5_K_BLOCK_SIZE;
    if bytes.len() != expected {
        bail!(
            "Q5_K: byte len {} != expected {} ({} blocks)",
            bytes.len(),
            expected,
            n_blocks
        );
    }

    let mut out = vec![0f32; n];
    for (bi, block) in bytes.chunks_exact(Q5_K_BLOCK_SIZE).enumerate() {
        let out_slice = &mut out[bi * QK_K..(bi + 1) * QK_K];
        dequant_q5_k_block(block, out_slice);
    }
    Ok(out)
}

fn dequant_q5_k_block(block: &[u8], y: &mut [f32]) {
    debug_assert_eq!(block.len(), Q5_K_BLOCK_SIZE);
    debug_assert_eq!(y.len(), QK_K);

    let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    let dmin = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();

    let mut scales = [0u8; 12];
    scales.copy_from_slice(&block[4..16]);
    let qh = &block[16..16 + 32];
    let qs = &block[16 + 32..16 + 32 + 128];

    let mut is = 0usize;
    let mut y_ofs = 0usize;
    let mut q_ofs = 0usize;
    // Each pair (is, is+1) of sub-blocks shares one qh bit-plane pair.
    // u = is / 2 = 0..4. In llama.cpp the mask is (1 << u) for the low
    // half of the pair, (1 << (u+1)) wait — let's mirror the canonical
    // code structure directly:
    //
    //   u1 = 1; u2 = 2;
    //   for each sub-block-pair:
    //     ...
    //     for l in 0..32:
    //       y[j+l]    = d0 * ((qs[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m0f
    //       y[j+l+32] = d1 * ((qs[l] >>   4) + ((qh[l] & u2) ? 16 : 0)) - m1f
    //     u1 <<= 2; u2 <<= 2;
    //
    // So we track u1, u2 through iteration.
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;
    while is < 8 {
        let (sc0, m0) = get_scale_min_k4(is, &scales);
        let (sc1, m1) = get_scale_min_k4(is + 1, &scales);
        let d0 = d * sc0 as f32;
        let m0f = dmin * m0 as f32;
        let d1 = d * sc1 as f32;
        let m1f = dmin * m1 as f32;

        for l in 0..32 {
            let lo = qs[q_ofs + l] & 0x0F;
            let hi = qs[q_ofs + l] >> 4;
            let hb0 = if (qh[l] & u1) != 0 { 16u8 } else { 0 };
            let hb1 = if (qh[l] & u2) != 0 { 16u8 } else { 0 };
            y[y_ofs + l] = d0 * (lo + hb0) as f32 - m0f;
            y[y_ofs + l + 32] = d1 * (hi + hb1) as f32 - m1f;
        }

        y_ofs += 64;
        q_ofs += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

// ---------------------------------------------------------------------------
// Q6_K: 256-element superblocks. 210 bytes per block.
// Layout (note: d is at the *end*, unlike Q4_K/Q5_K):
//   ql     : [u8; 128]   (4 low bits per weight, 2 weights per byte × 256)
//   qh     : [u8; 64]    (2 high bits per weight, 4 weights per byte × 256)
//   scales : [i8; 16]    (signed 8-bit per-sub-block scale; 16 sub-blocks of 16)
//   d      : f16         (2 B, at tail)
//
// 6-bit weight = ((qh_bits << 4) | ql_nibble), zero-centered at 32:
//   y[i] = d * scales[sub] * (q - 32)
//
// Reference: `dequantize_row_q6_K` in ggml-quants.c.
// The canonical kernel groups 128 elements at a time, pulling 2 sub-block
// scales (scales[is .. is+8]) per iteration.
// ---------------------------------------------------------------------------

const Q6_K_BLOCK_SIZE: usize = 210;

fn dequant_q6_k(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK_K != 0 {
        bail!("Q6_K: n_elements {n} not divisible by 256");
    }
    let n_blocks = n / QK_K;
    let expected = n_blocks * Q6_K_BLOCK_SIZE;
    if bytes.len() != expected {
        bail!(
            "Q6_K: byte len {} != expected {} ({} blocks)",
            bytes.len(),
            expected,
            n_blocks
        );
    }

    let mut out = vec![0f32; n];
    for (bi, block) in bytes.chunks_exact(Q6_K_BLOCK_SIZE).enumerate() {
        let out_slice = &mut out[bi * QK_K..(bi + 1) * QK_K];
        dequant_q6_k_block(block, out_slice);
    }
    Ok(out)
}

fn dequant_q6_k_block(block: &[u8], y: &mut [f32]) {
    debug_assert_eq!(block.len(), Q6_K_BLOCK_SIZE);
    debug_assert_eq!(y.len(), QK_K);

    let ql = &block[0..128];
    let qh = &block[128..128 + 64];
    // Scales are stored as 16 signed bytes; index each one as i8 inline
    // (the C reference uses `const int8_t * sc = x[i].scales`, we don't
    // need a reinterpret — the runtime cost is zero; this runs once per
    // 256-element block, not per element).
    let scales = &block[192..192 + 16];
    let d = half::f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();

    // Mirror the canonical kernel: process 128 elements at a time, twice.
    // For each 128-element chunk we use 4 scales (8 sub-blocks × 16 elems =
    // 128 elems; 4 scales per chunk to cover 4 sub-blocks × 32 elems wait —
    // let's just follow the ggml-quants.c reference literally:
    //
    //   for each superblock:
    //     ql_ptr = ql; qh_ptr = qh;
    //     sc_ptr = scales;
    //     for _ in 0..2 {                   // two 128-element halves
    //       for l in 0..32 {
    //         is = l / 16;                   // 0 or 1 within this half
    //         q1 = ((ql_ptr[l]       & 0xF) | ((qh_ptr[l] >> 0) & 3) << 4) - 32
    //         q2 = ((ql_ptr[l + 32]  & 0xF) | ((qh_ptr[l] >> 2) & 3) << 4) - 32
    //         q3 = ((ql_ptr[l]       >> 4)  | ((qh_ptr[l] >> 4) & 3) << 4) - 32
    //         q4 = ((ql_ptr[l + 32]  >> 4)  | ((qh_ptr[l] >> 6) & 3) << 4) - 32
    //         y[l]       = d * sc_ptr[is+0] * q1
    //         y[l+32]    = d * sc_ptr[is+2] * q2
    //         y[l+64]    = d * sc_ptr[is+4] * q3
    //         y[l+96]    = d * sc_ptr[is+6] * q4
    //       }
    //       y      += 128
    //       ql_ptr += 64
    //       qh_ptr += 32
    //       sc_ptr += 8
    //     }
    let mut y_ofs = 0usize;
    let mut ql_ofs = 0usize;
    let mut qh_ofs = 0usize;
    let mut sc_ofs = 0usize;

    for _ in 0..2 {
        for l in 0..32 {
            let is = l / 16; // 0 or 1
            let ql_a = ql[ql_ofs + l];
            let ql_b = ql[ql_ofs + l + 32];
            let qh_byte = qh[qh_ofs + l];

            let q1 = ((ql_a & 0x0F) | (((qh_byte >> 0) & 3) << 4)) as i32 - 32;
            let q2 = ((ql_b & 0x0F) | (((qh_byte >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql_a >> 4) | (((qh_byte >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql_b >> 4) | (((qh_byte >> 6) & 3) << 4)) as i32 - 32;

            y[y_ofs + l] = d * scales[sc_ofs + is] as i8 as f32 * q1 as f32;
            y[y_ofs + l + 32] = d * scales[sc_ofs + is + 2] as i8 as f32 * q2 as f32;
            y[y_ofs + l + 64] = d * scales[sc_ofs + is + 4] as i8 as f32 * q3 as f32;
            y[y_ofs + l + 96] = d * scales[sc_ofs + is + 6] as i8 as f32 * q4 as f32;
        }
        y_ofs += 128;
        ql_ofs += 64;
        qh_ofs += 32;
        sc_ofs += 8;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn put_f16(out: &mut Vec<u8>, v: f32) {
        let bits = half::f16::from_f32(v).to_bits();
        out.extend_from_slice(&bits.to_le_bytes());
    }

    /// Sanity check F16/BF16 passthroughs.
    #[test]
    fn f16_passthrough() {
        let mut bytes = Vec::new();
        for v in [0.0f32, 1.0, -1.0, 2.5, -3.75] {
            put_f16(&mut bytes, v);
        }
        let out = dequant_f16(&bytes, 5).unwrap();
        assert_eq!(out.len(), 5);
        for (got, expected) in out.iter().zip([0.0, 1.0, -1.0, 2.5, -3.75].iter()) {
            assert!((got - expected).abs() < 1e-3, "got {got} expected {expected}");
        }
    }

    #[test]
    fn bf16_passthrough() {
        let mut bytes = Vec::new();
        for v in [0.0f32, 1.0, -1.0, 2.5, -3.75] {
            let bits = half::bf16::from_f32(v).to_bits();
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        let out = dequant_bf16(&bytes, 5).unwrap();
        assert_eq!(out.len(), 5);
        for (got, expected) in out.iter().zip([0.0, 1.0, -1.0, 2.5, -3.75].iter()) {
            assert!((got - expected).abs() < 1e-2);
        }
    }

    /// Q8_0: one block with d=2.0 and qs = [0, 1, -1, 2, -2, ... pattern].
    /// Expected output is straightforward: y[i] = d * qs[i].
    #[test]
    fn dequant_q8_0_known_values() {
        let mut block = Vec::with_capacity(34);
        // d = 2.0 as f16
        put_f16(&mut block, 2.0);
        // qs = -16..16 (32 signed i8 values)
        for q in -16i8..16 {
            block.push(q as u8);
        }
        assert_eq!(block.len(), 34);

        let out = dequant_q8_0(&block, 32).unwrap();
        for (i, q) in (-16i8..16).enumerate() {
            let expected = 2.0 * q as f32;
            assert!(
                (out[i] - expected).abs() < 1e-3,
                "i={i} got={} expected={}",
                out[i],
                expected
            );
        }
    }

    /// Q4_K: construct a superblock where the math is hand-traceable.
    ///
    /// Setup:
    ///   d    = 1.0
    ///   dmin = 0.0           (mins don't affect output)
    ///   scales bytes: all zero EXCEPT sc[0] = 1 (via q[0] = 1), others 0.
    ///     → get_scale_min_k4(0) = (1, 0); get_scale_min_k4(1..8) = (0, 0)
    ///   qs: first 32 bytes (sub-blocks 0+1) encode nibble = 7 in both halves.
    ///     byte = 0x77 → low=7, high=7.
    ///
    /// Expected output:
    ///   y[0..32]    = 1.0 * 1 * 7 = 7.0 (sub-block 0, uses sc0=1)
    ///   y[32..64]   = 1.0 * 0 * 7 = 0.0 (sub-block 1, uses sc1=0)
    ///   y[64..256]  = 0 (sub-blocks 2..8 all have sc=0)
    #[test]
    fn dequant_q4_k_known_values() {
        let mut block = Vec::with_capacity(Q4_K_BLOCK_SIZE);
        put_f16(&mut block, 1.0); // d
        put_f16(&mut block, 0.0); // dmin
        // scales[12]: sc[0]=1 at q[0], rest zero.
        let mut scales = [0u8; 12];
        scales[0] = 1;
        block.extend_from_slice(&scales);
        // qs[128]: first 32 bytes = 0x77, rest zero.
        let mut qs = vec![0u8; 128];
        for b in qs[0..32].iter_mut() {
            *b = 0x77;
        }
        block.extend_from_slice(&qs);
        assert_eq!(block.len(), Q4_K_BLOCK_SIZE);

        let out = dequant_q4_k(&block, QK_K).unwrap();
        for i in 0..32 {
            assert!(
                (out[i] - 7.0).abs() < 1e-4,
                "y[{i}] = {} expected 7.0",
                out[i]
            );
        }
        for i in 32..64 {
            assert!(out[i].abs() < 1e-4, "y[{i}] = {} expected 0", out[i]);
        }
        for i in 64..256 {
            assert!(out[i].abs() < 1e-4, "y[{i}] = {} expected 0", out[i]);
        }
    }

    /// Q5_K: same trick as Q4_K. Set sc[0]=1, others 0, dmin=0.
    /// For the first 32 elements, qs[l]=0x00 (low nibble 0) and qh[l]=0xFF
    /// → u1=1 mask is set → hb0=16. So weight = 0 + 16 = 16. y = 1*1*16 = 16.
    /// For the 2nd half (elements 32..64): high nibble of qs[l] is 0, qh[l]&2
    /// = 2 ≠ 0 → hb1=16 → weight=16. But sc1=0 → y=0.
    #[test]
    fn dequant_q5_k_known_values() {
        let mut block = Vec::with_capacity(Q5_K_BLOCK_SIZE);
        put_f16(&mut block, 1.0); // d
        put_f16(&mut block, 0.0); // dmin
        let mut scales = [0u8; 12];
        scales[0] = 1;
        block.extend_from_slice(&scales);
        // qh: all 0xFF for the first 32 bytes (which back the first 64 elems)
        let mut qh = vec![0u8; 32];
        for b in qh.iter_mut() {
            *b = 0xFF;
        }
        block.extend_from_slice(&qh);
        // qs: zeros. High-bit from qh does the work.
        block.extend_from_slice(&vec![0u8; 128]);
        assert_eq!(block.len(), Q5_K_BLOCK_SIZE);

        let out = dequant_q5_k(&block, QK_K).unwrap();
        for i in 0..32 {
            assert!(
                (out[i] - 16.0).abs() < 1e-4,
                "y[{i}] = {} expected 16.0",
                out[i]
            );
        }
        for i in 32..64 {
            assert!(out[i].abs() < 1e-4, "y[{i}] = {} expected 0 (sc1=0)", out[i]);
        }
    }

    /// Q6_K: construct a known-answer block.
    ///
    /// d = 1.0
    /// scales[0..16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    /// ql: all 0x20 → low nibble = 0 for all elements.
    ///     Wait: need to think carefully. Let's make it simple:
    /// ql: all 0x00, qh: all 0x00 → raw q = 0 → q - 32 = -32.
    /// For the first-half first-subblock (is=0, sc_ofs=0) we multiply by
    /// scales[0+0] = 1 → y = 1 * (-32) = -32.
    /// For indices within the first 16 of that half (is=0), we still use
    /// scales[0], so y[0..16] = -32.
    /// For indices 16..32 (is=1), we use scales[0+1] = 0 → y = 0.
    #[test]
    fn dequant_q6_k_known_values() {
        let mut block = vec![0u8; Q6_K_BLOCK_SIZE];
        // ql = 128 bytes of 0 at offset 0
        // qh = 64 bytes of 0 at offset 128
        // scales at offset 192 (16 bytes): scales[0] = 1
        block[192] = 1i8 as u8;
        // d at offset 208
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        block[208] = d_bits[0];
        block[209] = d_bits[1];

        let out = dequant_q6_k(&block, QK_K).unwrap();
        // First 16 elements: scale[0] = 1, q = 0-32 = -32
        for i in 0..16 {
            assert!(
                (out[i] - (-32.0)).abs() < 1e-4,
                "y[{i}] = {} expected -32",
                out[i]
            );
        }
        // 16..32: scale[1] = 0 → y = 0
        for i in 16..32 {
            assert!(out[i].abs() < 1e-4, "y[{i}] = {} expected 0", out[i]);
        }
    }

    // -----------------------------------------------------------------------
    // Reference-validated dequant tests (SKEPTIC_GGUF.md P1-1).
    //
    // These byte arrays and expected outputs were generated by linking
    // against llama.cpp's compiled libggml and calling the canonical
    // `dequantize_row_q*` kernels. They exercise state-space regions the
    // earlier self-consistency tests didn't:
    //   - Q4_K distinct-scales: non-zero scales AND mins on all 8 sub-blocks,
    //     covering both j<4 and j>=4 branches of get_scale_min_k4, with
    //     dmin != 0.
    //   - Q5_K all-high-bits: qh = 0xFF validates all 4 shift positions of
    //     the u1,u2 <<= 2 progression.
    //   - Q5_K alternating-qh: qh = 0xAA alternates hit/miss across each
    //     pair, directly validating each individual shift step.
    //   - Q6_K four-shifts: qh = 0xE4 extracts {0,1,2,3} at shifts {0,2,4,6},
    //     validating all four 2-bit qh positions.
    //
    // Regenerate via tools/gguf_ref/gen_ref.c if the ggml format ever
    // changes (it has been stable since v2 / K-quants release).
    // -----------------------------------------------------------------------

    /// d=2.0, qs = -16..15. Output is y = 2 * qs.
    const Q8_0_BLOCK_BASIC: [u8; 34] = [
        0x00, 0x40, 0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC,
        0xFD, 0xFE, 0xFF, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        0x0C, 0x0D, 0x0E, 0x0F,
    ];
    const Q8_0_REF_BASIC: [f32; 32] = [
        -32.0, -30.0, -28.0, -26.0, -24.0, -22.0, -20.0, -18.0, -16.0, -14.0, -12.0, -10.0, -8.0,
        -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0,
        26.0, 28.0, 30.0,
    ];

    #[test]
    fn dequant_q8_0_reference_basic() {
        let out = dequant_q8_0(&Q8_0_BLOCK_BASIC, 32).unwrap();
        for (i, (&got, &expected)) in out.iter().zip(Q8_0_REF_BASIC.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-4,
                "i={i} got={got} expected={expected}"
            );
        }
    }

    /// Q4_K with all 8 sub-blocks having distinct non-zero scales and mins.
    /// d=1.0, dmin=0.5. sc = {3,5,7,11,13,17,19,23}, m = {1,2,4,8,16,32,33,48}.
    /// qs bytes all 0x3B → low nibble 11, high nibble 3.
    /// Per-group output (constant within each 32-elem sub-block):
    ///   grp 0 (lo, sc=3,  m=1 ) = 3 *11 - 0.5*1  =  32.5
    ///   grp 1 (hi, sc=5,  m=2 ) = 5 * 3 - 0.5*2  =  14.0
    ///   grp 2 (lo, sc=7,  m=4 ) = 7 *11 - 0.5*4  =  75.0
    ///   grp 3 (hi, sc=11, m=8 ) = 11* 3 - 0.5*8  =  29.0
    ///   grp 4 (lo, sc=13, m=16) = 13*11 - 0.5*16 = 135.0
    ///   grp 5 (hi, sc=17, m=32) = 17* 3 - 0.5*32 =  35.0  (j>=4 branch)
    ///   grp 6 (lo, sc=19, m=33) = 19*11 - 0.5*33 = 192.5  (j>=4 branch)
    ///   grp 7 (hi, sc=23, m=48) = 23* 3 - 0.5*48 =  45.0  (j>=4 branch)
    #[rustfmt::skip]
    const Q4_K_BLOCK_DISTINCT: [u8; 144] = [
        0x00, 0x3C, 0x00, 0x38, 0x03, 0x45, 0x47, 0x4B, 0x41, 0x82, 0x84, 0xC8, 0x0D, 0x01, 0x13, 0x07,
        0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B,
        0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B,
        0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B,
        0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B,
        0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B,
        0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B,
        0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B,
        0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B,
    ];
    const Q4_K_REF_GROUPS: [f32; 8] = [32.5, 14.0, 75.0, 29.0, 135.0, 35.0, 192.5, 45.0];

    #[test]
    fn dequant_q4_k_reference_distinct_scales() {
        let out = dequant_q4_k(&Q4_K_BLOCK_DISTINCT, 256).unwrap();
        for grp in 0..8 {
            let expected = Q4_K_REF_GROUPS[grp];
            for l in 0..32 {
                let i = grp * 32 + l;
                assert!(
                    (out[i] - expected).abs() < 1e-3,
                    "grp={grp} i={i} got={} expected={}",
                    out[i],
                    expected
                );
            }
        }
    }

    /// Q5_K, qh = all 0xFF. Every element = 16.0 because every bit of qh
    /// tests as set regardless of which u1,u2 mask is in play. If the
    /// `u1,u2 <<= 2` progression were buggy only later sub-blocks would
    /// differ from 16 — this test catches any such bug.
    #[rustfmt::skip]
    const Q5_K_BLOCK_ALL_HIGH_BITS: [u8; 176] = [
        0x00, 0x3C, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    #[test]
    fn dequant_q5_k_reference_all_high_bits() {
        let out = dequant_q5_k(&Q5_K_BLOCK_ALL_HIGH_BITS, 256).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!((v - 16.0).abs() < 1e-4, "i={i} got={v} expected 16.0");
        }
    }

    /// Q5_K with qh = 0xAA (0b10101010). This alternating pattern validates
    /// the u1,u2 <<= 2 shift progression directly: at each iteration,
    /// u1 hits a '0' bit (output 0) and u2 hits a '1' bit (output 16),
    /// yielding sub-blocks 0,2,4,6 → 0 and 1,3,5,7 → 16.
    #[rustfmt::skip]
    const Q5_K_BLOCK_ALTERNATING_QH: [u8; 176] = [
        0x00, 0x3C, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01,
        0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
        0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    #[test]
    fn dequant_q5_k_reference_alternating_qh() {
        let out = dequant_q5_k(&Q5_K_BLOCK_ALTERNATING_QH, 256).unwrap();
        for sub in 0..8 {
            let expected = if sub % 2 == 0 { 0.0 } else { 16.0 };
            for l in 0..32 {
                let i = sub * 32 + l;
                assert!(
                    (out[i] - expected).abs() < 1e-4,
                    "sub={sub} i={i} got={} expected={}",
                    out[i],
                    expected
                );
            }
        }
    }

    /// Q6_K with qh = 0xE4 (0b11100100) → shifts {0,2,4,6} extract {0,1,2,3}.
    /// With ql=0 and all scales=1, the four 32-element groups are:
    ///   shift 0 bits '00' → hi=0  → -32
    ///   shift 2 bits '01' → hi=16 → -16
    ///   shift 4 bits '10' → hi=32 →   0
    ///   shift 6 bits '11' → hi=48 →  16
    /// Exercised for BOTH 128-elem halves (sc_ofs progression).
    #[rustfmt::skip]
    const Q6_K_BLOCK_FOUR_SHIFTS: [u8; 210] = [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4,
        0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4,
        0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4,
        0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4,
        0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x00, 0x3C,
    ];

    #[test]
    fn dequant_q6_k_reference_four_shifts() {
        let out = dequant_q6_k(&Q6_K_BLOCK_FOUR_SHIFTS, 256).unwrap();
        let groups = [-32.0f32, -16.0, 0.0, 16.0];
        for half in 0..2 {
            for (g, &expected) in groups.iter().enumerate() {
                for l in 0..32 {
                    let i = half * 128 + g * 32 + l;
                    assert!(
                        (out[i] - expected).abs() < 1e-4,
                        "half={half} g={g} i={i} got={} expected={}",
                        out[i],
                        expected
                    );
                }
            }
        }
    }
}
