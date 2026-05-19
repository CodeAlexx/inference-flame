//! Lance tokenizer dump — parity probe for Python ↔ Rust tokenization.
//!
//! Reads the same `Lance_3B/tokenizer.json`, encodes the fairy prompt in
//! the same three forms as `/tmp/parity_tok.py`, prints JSON to stdout.
//! Compare side-by-side.

use std::path::PathBuf;

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

const TOK_PATH: &str = "/home/alex/.serenity/models/lance/Lance_3B/tokenizer.json";

const FAIRY: &str = "Depict a tiny fairy near the fire at the end of match in a macro, shallow-focus scene. The fairy is heating her hands on a fire. She appears fragile and doll-like, with a large head and soft translucent wings. Woman's fingers holding the match has black nail polish. Lighting is intimate and diffused, Snow on the background dissolving into blur. Still, tender, and quietly whimsical.";

const T2I_SYS: &str = "Describe the image by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background:";

fn main() -> Result<()> {
    let tok = Tokenizer::from_file(PathBuf::from(TOK_PATH))
        .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;

    let raw: Vec<u32> = tok
        .encode(FAIRY, false)
        .map_err(|e| anyhow::anyhow!("encode raw: {e}"))?
        .get_ids()
        .to_vec();

    let mut simple: Vec<u32> = Vec::with_capacity(raw.len() + 2);
    simple.push(151644);
    simple.extend(raw.iter().copied());
    simple.push(151645);

    let template_str = format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        T2I_SYS, FAIRY
    );
    let template: Vec<u32> = tok
        .encode(template_str.clone(), false)
        .map_err(|e| anyhow::anyhow!("encode template: {e}"))?
        .get_ids()
        .to_vec();

    let json = serde_json::json!({
        "raw_len": raw.len(),
        "simple_len": simple.len(),
        "template_len": template.len(),
        "raw_first_30": &raw[..raw.len().min(30)],
        "simple_first_30": &simple[..simple.len().min(30)],
        "template_first_30": &template[..template.len().min(30)],
        "template_last_10": &template[template.len().saturating_sub(10)..],
        "template_full_string": &template_str[..template_str.len().min(200)],
    });
    println!("{}", serde_json::to_string_pretty(&json).context("json")?);
    Ok(())
}
