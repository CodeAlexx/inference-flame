#!/usr/bin/env python3
"""Transcribe audio (WAV or MP4) via Whisper.

Defaults to whisper-tiny on CPU — fast (~5s for 1-min audio) and small (~39 MB).
For higher accuracy: `--model base` (74 MB), `--model small` (244 MB),
`--model medium` (769 MB).

Usage:
    python tools/transcribe.py /path/to/audio_or_video
    python tools/transcribe.py video.mp4 --model base
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# Default to CPU so it doesn't compete with magihuman_infer for GPU memory.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="WAV, MP3, MP4, or any ffmpeg-readable audio source")
    ap.add_argument("--model", default="tiny", choices=["tiny", "base", "small", "medium", "large"],
                    help="Whisper model size (default: tiny)")
    ap.add_argument("--language", default=None, help="Force a language code (en, fr, ...). Default: auto-detect")
    args = ap.parse_args()

    import whisper  # imported here to keep --help fast

    print(f"[whisper] loading model: {args.model}", file=sys.stderr)
    model = whisper.load_model(args.model)
    print(f"[whisper] transcribing {args.input}", file=sys.stderr)
    kwargs = {"language": args.language} if args.language else {}
    result = model.transcribe(args.input, fp16=False, **kwargs)

    text = result.get("text", "").strip()
    lang = result.get("language", "?")
    segs = result.get("segments", [])
    print(f"\n[lang={lang}]")
    print(f"[duration ≈ {segs[-1]['end']:.2f}s]" if segs else "")
    print()
    if segs:
        for s in segs:
            t0, t1, txt = s["start"], s["end"], s["text"].strip()
            print(f"  [{t0:6.2f} → {t1:6.2f}]  {txt}")
        print()
    print("FULL TRANSCRIPT:")
    print(f"  {text!r}")


if __name__ == "__main__":
    main()
