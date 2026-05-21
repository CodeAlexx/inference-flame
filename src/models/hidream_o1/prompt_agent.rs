//! HiDream-O1 prompt-rewrite agent.
//!
//! Port of `HiDream-O1-Image/prompt_agent.py`. Wraps a Gemma-4-31B-it
//! text decoder with the SCALIST system prompt; rewrites raw user
//! input into a JSON-wrapped, knowledge-resolved, layout-anchored
//! English creative-director's-brief that HiDream-O1's Qwen3-VL text
//! encoder expects. Without this step, raw prompts produce
//! mush — the model was trained against agent-rewritten prompts.
//!
//! Negative prompt for the unconditional branch is hardcoded to
//! `" "` (single space) — see
//! `HiDream-O1-Image/models/pipeline.py:160` and `:232`. The agent
//! does NOT rewrite the negative; only the positive user prompt.
//!
//! Reference: `HiDream-O1-Image/prompt_agent.py`.
//! Runtime: text-only decode, all GPU (no CPU fallback at execution).

use std::borrow::Cow;

/// Verbatim Chinese SCALIST system prompt from
/// `prompt_agent.py:6-50`. **Do not edit casually** — this is what
/// the model was trained to follow and any rewording risks degrading
/// the rewrite quality. The trailing `Final output: JSON only,
/// no other text` is what makes the response parseable.
pub const REWRITE_SYSTEM_PROMPT: &str = "你是专业的AI图像生成Prompt工程师的Prompt Engineering Engine,也是一名拥有百科知识和视觉导演能力的创意总监.你的任务是分析用户的原始图像需求,推理出隐含知识和最佳视觉方案,并改写成一个**明确,详细,可直接用于图像生成的英文prompt**.

## 核心目标

图像生成模型只能执行直接的视觉描述,不能自行补全背景知识,逻辑关系或文字内容.因此,你必须提前完成知识解析,空间规划和视觉导演,把结果显式写入prompt中.

使用 SCALIST 框架扩写每个画面:
- **Subject**: 主体的身份,外观,颜色,材质,纹理,动作,表情,服饰.
- **Composition**: 镜头景别,视角,主体位置,前景/中景/背景层次,留白和视觉焦点.
- **Action**: 主体正在做什么,动作方向,姿态,互动关系.
- **Location**: 场景地点,室内/室外,时代,天气,时间段,环境细节.
- **Image style**: photorealistic, cinematic, oil painting, watercolor, anime, 3D render 等,并匹配合适的光线和色彩氛围.
- **Specs**: 摄影/渲染参数,如 85mm lens, low-angle shot, shallow depth of field, soft diffused light, dramatic backlighting, matte texture, sharp focus.
- **Text rendering**: 如果用户要求文字,必须把准确文字放在英文双引号中,并说明字体风格,颜色,大小,材质和精确位置.

1. **知识解析与显式化**: 凡是诗词,歌词,名言,公式,历史人物,科学概念,地标,名画,文化符号,历史事件,UI布局或现实世界对象,都要先解析出具体答案和可见特征,再写入prompt.不要只写 \"Mona Lisa\",\"Dunkirk evacuation\",\"freedom\" 这类需要模型自行理解的词.
2. **空间与逻辑锚定**: 把模糊关系改写为明确布局,例如 top left corner, centered in the foreground, slightly behind the main subject, background out of focus, text aligned along the bottom edge.不要使用\"旁边\"\"一些\"\"好看\"等含糊表达.
3. **文字排版精度**: 中文,英文,公式,多语言文本都必须逐字保留在引号中,例如 \"床前明月光,疑是地上霜.举头望明月,低头思故乡.\" 或 \"E = mc²\";同时指定字体(calligraphy, serif, sans-serif, handwritten),颜色,材质和位置.
4. **真实世界落地**: 如果用户要求事实准确的内容,例如历史文物,天气现象,人物肖像,建筑,仪表盘或应用界面,要使用你的内部知识补全准确视觉细节.
5. **抽象概念具象化**: 把\"自由,孤独,未来感,治愈\"等抽象词转成可见场景,符号和氛围,例如飞鸟,断裂锁链,辽阔天空,冷色霓虹,柔和晨光等.

## 示例合并学习

- 用户说\"李白的静夜思写在墙上\",prompt 应写出完整中文诗句,并指定它以优雅中国书法写在古旧石墙的哪个位置.
- 用户说\"三大力学的奠基人\"或\"爱因斯坦写质能方程\",prompt 应解析出 Isaac Newton 或 Albert Einstein,并描述人物外貌,时代服饰,黑板,公式 \"E = mc²\" 等可见内容.
- 用户说\"蒙娜丽莎\"\"比萨斜塔\"\"福字\"\"敦刻尔克大撤退\",prompt 应描述对应画面特征: 神秘微笑与交叠双手,倾斜白色大理石钟楼与拱廊,红底金色/黑色书法 \"福\",1940年海滩上等待撤离的士兵和海面船只.

## 输出prompt要求

- prompt 必须是一个英文的,连贯自然的单段落,像 Creative Director's Brief,而不是关键词堆砌或 tag soup.
- 长度通常为 80-220 词;简单需求可以更短,复杂画面可以更长.
- 最重要的主体和画面意图放在开头,然后自然展开构图,动作,地点,风格,技术参数和文字渲染.
- 使用完整句子,丰富但准确的形容词,摄影/绘画/设计术语.
- 不要包含任何需要图像模型继续推理才能理解的表达.
- prompt 必须自包含,仅凭prompt本身就能准确生成图片.

## 执行步骤

1. **Analyze**: 识别核心主体,用户意图,文字要求,参考限制和需要解析的隐含知识.
2. **Reason**: 选择最适合画面的光线,镜头,角度,纹理,风格,空间布局和事实细节.
3. **Rewrite**: 输出最终增强后的英文单段落prompt.

只输出JSON,不加任何其他文字:
{\"prompt\": \"英文单段落prompt\", \"reasoning\": \"你的推理和知识解析过程(中文简述)\", \"resolved_knowledge\": \"你解析了哪些隐含知识(中文,如果没有隐含知识写'无')\"}";

/// Hardcoded unconditional/negative prompt for HiDream-O1, per
/// `HiDream-O1-Image/models/pipeline.py:160` and `:232`. Single
/// space character — the model was trained with this exact uncond
/// caption. Power users can override at the CLI level if needed
/// but the default MUST be `" "` for parity with reference output.
pub const HIDREAM_O1_HARDCODED_NEGATIVE: &str = " ";

/// Output of the rewrite agent. Mirrors the JSON contract from
/// `prompt_agent.py`'s `_wrap_result`. `prompt` is the field
/// HiDream-O1 actually consumes; `reasoning` and
/// `resolved_knowledge` are diagnostic.
#[derive(Debug, Clone)]
pub struct RewriteResult {
    /// English creative-director's-brief — feed this to HiDream-O1's
    /// Qwen3-VL text encoder, NOT the raw user input.
    pub prompt: String,
    /// Chinese reasoning trace from the model. Empty if parse fell back.
    pub reasoning: String,
    /// Chinese summary of resolved knowledge ("无" if none).
    pub resolved_knowledge: String,
    /// Raw model output, set when JSON parse failed and we used the
    /// fallback wrap. None on clean parse.
    pub raw_fallback: Option<String>,
}

/// Find the outermost balanced `{ ... }` block in `text`, ignoring
/// braces inside JSON-style strings. Direct port of
/// `prompt_agent.py::_extract_json_block`.
fn extract_json_block(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let mut depth: i32 = 0;
    let mut start: Option<usize> = None;
    let mut in_string = false;
    let mut escape_next = false;
    for (i, &ch) in bytes.iter().enumerate() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == b'\\' {
            escape_next = true;
            continue;
        }
        if ch == b'"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if ch == b'{' {
            if depth == 0 {
                start = Some(i);
            }
            depth += 1;
        } else if ch == b'}' {
            depth -= 1;
            if depth == 0 {
                if let Some(s) = start {
                    return Some(&text[s..=i]);
                }
            }
        }
    }
    None
}

/// Replace unescaped newlines inside string literals with `\n`. Direct
/// port of `prompt_agent.py::_fix_unescaped_newlines`. The Gemma model
/// sometimes emits raw newlines inside its JSON string values, which
/// makes the JSON technically invalid; this lets us recover without
/// asking for a regenerate.
fn fix_unescaped_newlines(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_string = false;
    let mut escape_next = false;
    for ch in text.chars() {
        if escape_next {
            out.push(ch);
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            out.push(ch);
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            out.push(ch);
            continue;
        }
        if in_string && ch == '\n' {
            out.push_str("\\n");
            continue;
        }
        if in_string && ch == '\r' {
            continue;
        }
        out.push(ch);
    }
    out
}

/// Strip a ```json ... ``` fence if present, taking the inner content.
fn strip_code_fence(text: &str) -> Cow<'_, str> {
    if !text.contains("```") {
        return Cow::Borrowed(text);
    }
    // Find first ``` (optionally followed by `json`), capture until next ```.
    let bytes = text.as_bytes();
    if let Some(start) = text.find("```") {
        let after_open = start + 3;
        // Skip optional language tag (`json` etc) and one newline.
        let mut content_start = after_open;
        while content_start < bytes.len() && bytes[content_start] != b'\n' {
            content_start += 1;
        }
        if content_start < bytes.len() {
            content_start += 1; // skip the newline
        }
        if let Some(end_rel) = text[content_start..].find("```") {
            let content_end = content_start + end_rel;
            return Cow::Owned(text[content_start..content_end].trim().to_string());
        }
    }
    Cow::Borrowed(text)
}

/// Parse the model's raw output into a structured result.
/// Tries (1) the raw text, (2) the brace-block extraction, (3) the
/// unescaped-newline-fixed version. Returns None if none succeed.
fn parse_json(text: &str) -> Option<serde_json::Value> {
    let text = text.trim();
    let stripped = strip_code_fence(text);
    let block: &str = extract_json_block(&stripped).unwrap_or(&stripped);
    for candidate in [Cow::Borrowed(block), Cow::Owned(fix_unescaped_newlines(block))] {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&candidate) {
            return Some(v);
        }
    }
    None
}

/// Wrap the raw model output into a `RewriteResult`. On parse
/// failure, falls back to the user's original input augmented with
/// quality keywords — matches `prompt_agent.py::_wrap_result`.
pub fn wrap_result(raw: &str, user_input: &str) -> RewriteResult {
    if let Some(v) = parse_json(raw) {
        let prompt = v.get("prompt").and_then(|x| x.as_str()).map(str::to_string);
        if let Some(prompt) = prompt {
            return RewriteResult {
                prompt,
                reasoning: v
                    .get("reasoning")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
                resolved_knowledge: v
                    .get("resolved_knowledge")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
                raw_fallback: None,
            };
        }
    }
    RewriteResult {
        prompt: format!(
            "{user_input}, highly detailed, masterpiece, best quality, sharp focus"
        ),
        reasoning: "解析失败,使用原始描述并添加质量关键词".to_string(),
        resolved_knowledge: "无".to_string(),
        raw_fallback: Some(raw.to_string()),
    }
}

/// Render the Gemma-4 chat template for a (system, user) turn pair
/// with the generation prompt at the end, matching
/// `processor.apply_chat_template([{"role":"system",...},
/// {"role":"user",...}], enable_thinking=False,
/// add_generation_prompt=True)`. Verified against
/// `/home/alex/models/Gemma-4-31B-it/chat_template.jinja`
/// (lines 175-205 emit the system turn, 215-353 emit the message
/// loop, 356-363 emit the model generation prompt with the
/// thinking-off channel marker).
///
/// Output shape:
/// ```text
/// <bos><|turn>system
/// {system}<turn|>
/// <|turn>user
/// {user}<turn|>
/// <|turn>model
/// <|channel>thought
/// <channel|>
/// ```
///
/// Notable: Gemma-4 uses `<|turn>role` / `<turn|>` delimiters, NOT
/// Gemma-3's `<start_of_turn>` / `<end_of_turn>`. The
/// `<|channel>thought\n<channel|>` suffix is what `enable_thinking=False`
/// emits — it tells the model to skip reasoning mode and go straight
/// to output. Without it the model often emits a long thinking
/// block before the JSON, which our parser still handles but wastes
/// tokens.
pub fn render_chat_template(system: &str, user: &str) -> String {
    let mut out = String::with_capacity(system.len() + user.len() + 128);
    out.push_str("<bos>");
    if !system.is_empty() {
        out.push_str("<|turn>system\n");
        out.push_str(system.trim());
        out.push_str("<turn|>\n");
    }
    out.push_str("<|turn>user\n");
    out.push_str(user);
    out.push_str("<turn|>\n");
    out.push_str("<|turn>model\n");
    // enable_thinking=False sentinel — required for the model to
    // skip its thinking channel and emit the JSON directly.
    out.push_str("<|channel>thought\n<channel|>");
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API: invoked from HiDream-O1 pipeline.
// ─────────────────────────────────────────────────────────────────────────────

/// Trait abstracting "a thing that takes a chat-template string and
/// returns generated text". Concrete impl in
/// `inference_flame::models::gemma4::Gemma4TextDecoder`. Lets tests
/// inject a stub backend without dragging in the 31B model.
pub trait RewriteBackend {
    /// Generate completion text from a fully-rendered chat-template
    /// prompt. `max_new_tokens` and `temperature` mirror the Python
    /// agent's args (4096, 1.0 respectively in the reference).
    fn generate(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<String, anyhow::Error>;
}

/// Rewrite a raw user prompt via the given backend.
///
/// Replicates `prompt_agent.py::rewrite_prompt_local`:
/// 1. Render chat template with SCALIST system + user input.
/// 2. Generate up to `max_new_tokens` at `temperature` (do_sample=true when T>0).
/// 3. Parse JSON; fall back to user_input + quality keywords on parse failure.
pub fn rewrite_prompt<B: RewriteBackend>(
    backend: &mut B,
    user_input: &str,
    max_new_tokens: usize,
    temperature: f32,
) -> Result<RewriteResult, anyhow::Error> {
    let prompt = render_chat_template(REWRITE_SYSTEM_PROMPT, user_input);
    let raw = backend.generate(&prompt, max_new_tokens, temperature)?;
    Ok(wrap_result(&raw, user_input))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_block_basic() {
        let s = r#"some preamble {"a": 1, "b": "x"} trailing"#;
        let block = extract_json_block(s).unwrap();
        assert_eq!(block, r#"{"a": 1, "b": "x"}"#);
    }

    #[test]
    fn extract_json_block_ignores_braces_in_strings() {
        let s = r#"junk {"a": "value with { brace"} trailing"#;
        let block = extract_json_block(s).unwrap();
        assert_eq!(block, r#"{"a": "value with { brace"}"#);
    }

    #[test]
    fn fix_unescaped_newlines_inside_string() {
        let s = "{\"a\": \"line1\nline2\"}";
        let fixed = fix_unescaped_newlines(s);
        assert_eq!(fixed, "{\"a\": \"line1\\nline2\"}");
    }

    #[test]
    fn wrap_result_clean_parse() {
        let raw = r#"{"prompt": "P", "reasoning": "R", "resolved_knowledge": "无"}"#;
        let r = wrap_result(raw, "u");
        assert_eq!(r.prompt, "P");
        assert_eq!(r.reasoning, "R");
        assert!(r.raw_fallback.is_none());
    }

    #[test]
    fn wrap_result_fallback_on_garbage() {
        let r = wrap_result("not json at all", "a cat on a chair");
        assert!(r.prompt.contains("a cat on a chair"));
        assert!(r.prompt.contains("masterpiece"));
        assert!(r.raw_fallback.is_some());
    }

    #[test]
    fn wrap_result_handles_codefence() {
        let raw = "```json\n{\"prompt\": \"P\", \"reasoning\": \"R\", \"resolved_knowledge\": \"无\"}\n```";
        let r = wrap_result(raw, "u");
        assert_eq!(r.prompt, "P");
    }

    #[test]
    fn chat_template_includes_system_user_and_thinking_off_marker() {
        let t = render_chat_template("SYS", "USR");
        assert!(t.starts_with("<bos>"));
        assert!(t.contains("<|turn>system\nSYS<turn|>\n"));
        assert!(t.contains("<|turn>user\nUSR<turn|>\n"));
        // enable_thinking=False — generation prompt ends with the
        // thought-channel-closed sentinel, not a bare role marker.
        assert!(t.ends_with("<|turn>model\n<|channel>thought\n<channel|>"));
    }

    #[test]
    fn chat_template_no_system_skips_system_block() {
        let t = render_chat_template("", "USR");
        assert!(!t.contains("<|turn>system"));
        assert!(t.contains("<|turn>user\nUSR<turn|>\n"));
    }

    #[test]
    fn negative_prompt_is_single_space() {
        assert_eq!(HIDREAM_O1_HARDCODED_NEGATIVE, " ");
    }
}
