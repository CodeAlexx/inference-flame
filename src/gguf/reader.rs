//! GGUF binary format reader.
//!
//! Parses the GGUF v2/v3 header + tensor-info table into `GgufHeader`.
//! Tensor payloads are accessed via the memory-mapped byte range
//! `[data_offset, data_offset + tensor_bytes)` — dequant lives in
//! `crate::gguf::dequant`.
//!
//! Spec reference: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>
//!
//! Binary layout:
//! ```text
//! Magic       : "GGUF" (4 bytes, 0x46554747 little-endian)
//! Version     : u32 (2 or 3 supported)
//! N_TENSORS   : u64
//! N_METADATA  : u64
//! Metadata    : N_METADATA × (key: String, type: u32, value: variable)
//! TensorInfos : N_TENSORS × (name: String, n_dims: u32, dims: [u64], type: u32, offset: u64)
//! (pad to general.alignment bytes; default 32)
//! Tensor data : contiguous blob; per-tensor offsets are relative to the data blob start
//! ```

use anyhow::{anyhow, bail, Context, Result};
use std::path::Path;

/// GGML tensor type enum (subset we care about for weight loading).
///
/// Values are the on-disk `ggml_type` u32 codes. Unsupported types are
/// represented by [`GgufQuantType::Unsupported`] carrying the raw code so
/// the error surfaces at the tensor that requires it, not at parse time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgufQuantType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    BF16,
    Unsupported(u32),
}

impl GgufQuantType {
    pub fn from_code(code: u32) -> Self {
        match code {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            8 => Self::Q8_0,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            30 => Self::BF16,
            other => Self::Unsupported(other),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q8_0 => "Q8_0",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::BF16 => "BF16",
            Self::Unsupported(_) => "UNSUPPORTED",
        }
    }

    /// Compute the on-disk byte size for `n_elements` of this type.
    /// Returns `None` for unsupported types OR when `n_elements` is not
    /// divisible by the type's block size (quantized formats are strictly
    /// block-aligned; a non-multiple means either a malformed file or a
    /// bug upstream — surface it here rather than silently rounding down).
    pub fn byte_size(&self, n_elements: u64) -> Option<u64> {
        // Helper for block-aligned types.
        fn block_size(n: u64, block_elems: u64, block_bytes: u64) -> Option<u64> {
            if n % block_elems == 0 {
                Some((n / block_elems) * block_bytes)
            } else {
                None
            }
        }
        match self {
            Self::F32 => Some(n_elements * 4),
            Self::F16 | Self::BF16 => Some(n_elements * 2),
            // Q8_0 / Q4_0: 32-element blocks.
            Self::Q8_0 => block_size(n_elements, 32, 34),
            Self::Q4_0 => block_size(n_elements, 32, 18),
            // K-quants: 256-element superblocks.
            Self::Q4_K => block_size(n_elements, 256, 144), // 2+2+12+128
            Self::Q5_K => block_size(n_elements, 256, 176), // 2+2+12+32+128
            Self::Q6_K => block_size(n_elements, 256, 210), // 128+64+16+2
            Self::Q2_K => block_size(n_elements, 256, 84),  // 16+2+2+64
            Self::Q3_K => block_size(n_elements, 256, 110), // 32+64+12+2
            Self::Unsupported(_) => None,
        }
    }
}

/// One tensor entry from the GGUF tensor-info table.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Key/name as stored in the file (pre-remap).
    pub name: String,
    /// Shape, row-major. GGUF stores dims reversed relative to numpy/torch;
    /// we reverse them on parse so `dims` matches the usual `[out, in, ...]`.
    pub dims: Vec<usize>,
    pub quant: GgufQuantType,
    /// Absolute byte offset in the file where this tensor's data begins
    /// (post-alignment; caller can `&mmap[offset .. offset + byte_size()]`).
    pub data_offset: u64,
    /// On-disk byte size. `None` for unsupported quant types.
    pub byte_size: Option<u64>,
}

impl GgufTensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().fold(1u64, |a, &d| a * d as u64)
    }
}

/// Parsed GGUF header (metadata + tensor table) minus the payload.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub alignment: u64,
    /// The architecture string from `general.architecture` metadata (if present).
    pub architecture: Option<String>,
    pub tensors: Vec<GgufTensorInfo>,
}

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;

/// Metadata value type codes (GGUF spec §"gguf_metadata_value_type").
const GGUF_TYPE_U8: u32 = 0;
const GGUF_TYPE_I8: u32 = 1;
const GGUF_TYPE_U16: u32 = 2;
const GGUF_TYPE_I16: u32 = 3;
const GGUF_TYPE_U32: u32 = 4;
const GGUF_TYPE_I32: u32 = 5;
const GGUF_TYPE_F32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_U64: u32 = 10;
const GGUF_TYPE_I64: u32 = 11;
const GGUF_TYPE_F64: u32 = 12;

/// A zero-copy little-endian byte cursor. Keeps everything on the parse
/// side small and allocation-free.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn read_exact(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            bail!(
                "GGUF: unexpected EOF at pos={} wanted={} remaining={}",
                self.pos,
                n,
                self.remaining()
            );
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_exact(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_exact(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_exact(len)?;
        // GGUF strings are UTF-8, not null-terminated.
        String::from_utf8(bytes.to_vec())
            .map_err(|e| anyhow!("GGUF: invalid UTF-8 in string: {e}"))
    }

    /// Advance past a metadata value of the given type without materializing
    /// it. Used for keys we don't care about (most of them).
    fn skip_value(&mut self, vtype: u32) -> Result<()> {
        match vtype {
            GGUF_TYPE_U8 | GGUF_TYPE_I8 | GGUF_TYPE_BOOL => {
                self.read_exact(1)?;
            }
            GGUF_TYPE_U16 | GGUF_TYPE_I16 => {
                self.read_exact(2)?;
            }
            GGUF_TYPE_U32 | GGUF_TYPE_I32 | GGUF_TYPE_F32 => {
                self.read_exact(4)?;
            }
            GGUF_TYPE_U64 | GGUF_TYPE_I64 | GGUF_TYPE_F64 => {
                self.read_exact(8)?;
            }
            GGUF_TYPE_STRING => {
                let _ = self.read_string()?;
            }
            GGUF_TYPE_ARRAY => {
                let elem_type = self.read_u32()?;
                let n = self.read_u64()?;
                for _ in 0..n {
                    self.skip_value(elem_type)?;
                }
            }
            other => bail!("GGUF: unknown metadata value type {other}"),
        }
        Ok(())
    }
}

/// Parse the GGUF header + tensor table from a byte buffer.
///
/// The returned `GgufHeader.tensors[i].data_offset` is the **absolute**
/// file offset where the i-th tensor's bytes start (post-alignment).
pub fn parse_header(buf: &[u8]) -> Result<GgufHeader> {
    let mut cur = Cursor::new(buf);

    // Magic
    let magic = cur.read_exact(4).context("reading magic")?;
    if magic != GGUF_MAGIC {
        bail!(
            "GGUF: bad magic {:?} (expected {:?})",
            magic,
            &GGUF_MAGIC[..]
        );
    }

    let version = cur.read_u32().context("reading version")?;
    if version != 2 && version != 3 {
        bail!(
            "GGUF: unsupported version {version} (only v2 and v3 are supported)"
        );
    }

    let n_tensors = cur.read_u64().context("reading n_tensors")?;
    let n_metadata = cur.read_u64().context("reading n_metadata")?;

    // Parse metadata. We capture two keys:
    //   - general.alignment     (u32 or u64; defaults to 32)
    //   - general.architecture  (string; informational)
    let mut alignment = DEFAULT_ALIGNMENT;
    let mut architecture: Option<String> = None;

    for i in 0..n_metadata {
        let key = cur
            .read_string()
            .with_context(|| format!("reading metadata key {i}"))?;
        let vtype = cur
            .read_u32()
            .with_context(|| format!("reading metadata[{key}] type"))?;

        match key.as_str() {
            "general.alignment" => match vtype {
                GGUF_TYPE_U32 => {
                    alignment = cur.read_u32()? as u64;
                }
                GGUF_TYPE_U64 => {
                    alignment = cur.read_u64()?;
                }
                GGUF_TYPE_I32 => {
                    alignment = cur.read_u32()? as u64;
                }
                GGUF_TYPE_I64 => {
                    alignment = cur.read_i64()? as u64;
                }
                _ => {
                    // Unexpected type for alignment; skip and keep default.
                    cur.skip_value(vtype)?;
                }
            },
            "general.architecture" => {
                if vtype == GGUF_TYPE_STRING {
                    architecture = Some(cur.read_string()?);
                } else {
                    cur.skip_value(vtype)?;
                }
            }
            _ => cur.skip_value(vtype)?,
        }
    }

    if alignment == 0 {
        alignment = DEFAULT_ALIGNMENT;
    }

    // Parse tensor info table.
    let mut infos = Vec::with_capacity(n_tensors as usize);
    for i in 0..n_tensors {
        let name = cur
            .read_string()
            .with_context(|| format!("reading tensor[{i}] name"))?;
        let n_dims = cur
            .read_u32()
            .with_context(|| format!("reading tensor[{name}] n_dims"))?;
        if n_dims > 8 {
            bail!("GGUF: tensor[{name}] has implausible n_dims={n_dims}");
        }
        // GGUF stores dims in "ggml order" (fastest-moving first). Reverse
        // so shape matches numpy/safetensors conventions ([out, in, ...]).
        let mut dims_ggml = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims_ggml.push(cur.read_u64()? as usize);
        }
        let dims: Vec<usize> = dims_ggml.into_iter().rev().collect();

        let ggml_type = cur
            .read_u32()
            .with_context(|| format!("reading tensor[{name}] type"))?;
        let data_offset_rel = cur
            .read_u64()
            .with_context(|| format!("reading tensor[{name}] offset"))?;

        let quant = GgufQuantType::from_code(ggml_type);
        let n_elems = dims.iter().fold(1u64, |a, &d| a * d as u64);
        let byte_size = quant.byte_size(n_elems);

        infos.push(GgufTensorInfo {
            name,
            dims,
            quant,
            // We fill absolute offsets after we know the aligned data base.
            data_offset: data_offset_rel,
            byte_size,
        });
    }

    // Align the cursor to `alignment` bytes; that is the start of the
    // tensor-data blob. Per-tensor offsets are relative to this base.
    let header_end = cur.pos as u64;
    let data_base = align_up(header_end, alignment);

    // Re-absolute the offsets.
    for info in infos.iter_mut() {
        info.data_offset = data_base + info.data_offset;
    }

    Ok(GgufHeader {
        version,
        alignment,
        architecture,
        tensors: infos,
    })
}

/// Parse header from a file path. Memory-maps the file.
pub fn parse_header_from_file(path: &Path) -> Result<(GgufHeader, memmap2::Mmap)> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening GGUF file {}", path.display()))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .with_context(|| format!("mmapping GGUF file {}", path.display()))?;
    let header = parse_header(&mmap)
        .with_context(|| format!("parsing GGUF header {}", path.display()))?;
    Ok((header, mmap))
}

#[inline]
fn align_up(x: u64, a: u64) -> u64 {
    if a == 0 {
        return x;
    }
    (x + a - 1) / a * a
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny synthetic GGUF blob in memory (version 3, 0 tensors,
    /// one string-valued metadata entry) and round-trip it through the parser.
    #[test]
    fn header_parse_synthetic() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_metadata

        // Metadata entry: key="general.architecture", type=STRING, value="flux"
        let key = b"general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        let val = b"flux";
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val);

        let header = parse_header(&buf).expect("parse");
        assert_eq!(header.version, 3);
        assert_eq!(header.tensors.len(), 0);
        assert_eq!(header.architecture.as_deref(), Some("flux"));
        assert_eq!(header.alignment, DEFAULT_ALIGNMENT);
    }

    #[test]
    fn align_up_basic() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }

    /// P1-3 synthetic: verify GGUF dim reversal for a 3D F32 tensor.
    ///
    /// GGUF stores dims "fastest-moving-first". The reader reverses them so
    /// the Rust-side `dims` match the usual `[out, in, ...]` / NCHW
    /// convention. A wrong-order-but-same-product tensor would load
    /// silently and multiply without error, producing garbage that's hard
    /// to diagnose. This test writes GGUF-order [3, 5, 7] and asserts
    /// the post-parse dims are the reverse [7, 5, 3].
    #[test]
    fn dim_reversal_synthetic_3d() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_metadata

        let name = b"t";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&3u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&3u64.to_le_bytes()); // GGUF dim[0]
        buf.extend_from_slice(&5u64.to_le_bytes()); // GGUF dim[1]
        buf.extend_from_slice(&7u64.to_le_bytes()); // GGUF dim[2]
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // data_offset_rel

        while buf.len() % (DEFAULT_ALIGNMENT as usize) != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&vec![0u8; 3 * 5 * 7 * 4]);

        let header = parse_header(&buf).expect("parse");
        assert_eq!(header.tensors.len(), 1);
        let info = &header.tensors[0];
        assert_eq!(info.name, "t");
        assert_eq!(info.quant, GgufQuantType::F32);
        assert_eq!(
            info.dims,
            vec![7usize, 5, 3],
            "dims must be reversed from GGUF storage order"
        );
        assert_eq!(info.n_elements(), 3 * 5 * 7);
    }

    /// Linear weight convention: GGUF [IN, OUT] → Rust [OUT, IN].
    /// This is the common case for every linear layer in FLUX/SD3/Chroma.
    #[test]
    fn dim_reversal_synthetic_linear_weight() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        let name = b"linear.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&4u64.to_le_bytes()); // GGUF IN
        buf.extend_from_slice(&6u64.to_le_bytes()); // GGUF OUT
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        buf.extend_from_slice(&0u64.to_le_bytes());
        while buf.len() % (DEFAULT_ALIGNMENT as usize) != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&vec![0u8; 4 * 6 * 4]);

        let header = parse_header(&buf).expect("parse");
        let info = &header.tensors[0];
        assert_eq!(
            info.dims,
            vec![6usize, 4],
            "linear weight must be [OUT, IN] after reversal"
        );
    }

    /// P2-2: `byte_size` must return None (not silently round down) when
    /// n_elements isn't divisible by the block size of a quantized format.
    #[test]
    fn byte_size_rejects_misaligned_block_counts() {
        // Q4_K requires n % 256 == 0.
        assert_eq!(GgufQuantType::Q4_K.byte_size(256), Some(144));
        assert_eq!(GgufQuantType::Q4_K.byte_size(512), Some(288));
        assert_eq!(
            GgufQuantType::Q4_K.byte_size(300),
            None,
            "malformed 300-elem Q4_K must not silently round"
        );
        // Q8_0 requires n % 32 == 0.
        assert_eq!(GgufQuantType::Q8_0.byte_size(32), Some(34));
        assert_eq!(GgufQuantType::Q8_0.byte_size(64), Some(68));
        assert_eq!(GgufQuantType::Q8_0.byte_size(33), None);
        // Q5_K, Q6_K same block size as Q4_K.
        assert_eq!(GgufQuantType::Q5_K.byte_size(300), None);
        assert_eq!(GgufQuantType::Q6_K.byte_size(300), None);
        // F32 and friends don't have a block constraint.
        assert_eq!(GgufQuantType::F32.byte_size(1), Some(4));
        assert_eq!(GgufQuantType::F32.byte_size(300), Some(1200));
    }

    #[test]
    fn quant_type_roundtrip() {
        assert_eq!(GgufQuantType::from_code(0), GgufQuantType::F32);
        assert_eq!(GgufQuantType::from_code(1), GgufQuantType::F16);
        assert_eq!(GgufQuantType::from_code(8), GgufQuantType::Q8_0);
        assert_eq!(GgufQuantType::from_code(12), GgufQuantType::Q4_K);
        assert_eq!(GgufQuantType::from_code(13), GgufQuantType::Q5_K);
        assert_eq!(GgufQuantType::from_code(14), GgufQuantType::Q6_K);
        assert_eq!(GgufQuantType::from_code(30), GgufQuantType::BF16);
        assert!(matches!(
            GgufQuantType::from_code(999),
            GgufQuantType::Unsupported(999)
        ));
    }
}
