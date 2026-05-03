#!/usr/bin/env python3
"""Render a waveform PNG from an audio file or video file.

Accepts WAV, MP3, MP4, or anything ffmpeg can read. Output is a PNG you
can `Read` to view: shows the audio envelope (top) and an RMS energy
trace (bottom) over time.

Usage:
    python tools/waveform.py speech.wav [--out wave.png]
    python tools/waveform.py video.mp4 --out video_audio.png
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile


def extract_to_wav(src: Path) -> Path:
    """If src is already a WAV, return it as-is. Otherwise extract audio
    via ffmpeg into a temp WAV (16-bit, 44.1 kHz, mono)."""
    if src.suffix.lower() == ".wav":
        return src
    tmp = Path(tempfile.gettempdir()) / f"_wave_extract_{os.getpid()}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(src),
         "-ac", "1", "-ar", "44100", "-vn",
         "-c:a", "pcm_s16le",
         str(tmp)],
        check=True,
    )
    return tmp


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    rate, data = wavfile.read(path)
    # Convert to mono float32 in [-1, 1]
    if data.dtype == np.int16:
        f = data.astype(np.float32) / np.iinfo(np.int16).max
    elif data.dtype == np.int32:
        f = data.astype(np.float32) / np.iinfo(np.int32).max
    elif data.dtype == np.float32 or data.dtype == np.float64:
        f = data.astype(np.float32)
    elif data.dtype == np.uint8:
        # 8-bit PCM unsigned 0..255 with center 128
        f = (data.astype(np.float32) - 128.0) / 127.0
    else:
        raise ValueError(f"unsupported dtype: {data.dtype}")
    if f.ndim == 2:
        f = f.mean(axis=1)
    return f, rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default=None,
                    help="Output PNG path. Default: <input stem>_wave.png in the same dir.")
    ap.add_argument("--width", type=int, default=14, help="figure width (inches)")
    ap.add_argument("--height", type=int, default=4, help="figure height (inches)")
    args = ap.parse_args()

    src = Path(args.input).resolve()
    if not src.exists():
        sys.exit(f"not found: {src}")

    out = Path(args.out) if args.out else src.with_name(src.stem + "_wave.png")

    wav_path = extract_to_wav(src)
    samples, rate = load_mono(wav_path)
    n = samples.size
    dur = n / rate
    print(f"loaded {wav_path.name}: {dur:.3f}s, {rate} Hz, {n} samples")

    # Figure: top = waveform, bottom = RMS energy in 20 ms windows
    t = np.arange(n) / rate
    win = max(1, int(rate * 0.02))  # 20 ms
    n_full = (n // win) * win
    rms = np.sqrt((samples[:n_full].reshape(-1, win) ** 2).mean(axis=1))
    rms_t = (np.arange(rms.size) + 0.5) * win / rate

    fig, axes = plt.subplots(2, 1, figsize=(args.width, args.height), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(t, samples, linewidth=0.5)
    axes[0].set_ylabel("amplitude")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_title(f"{src.name} — {dur:.2f}s @ {rate} Hz")
    axes[0].grid(alpha=0.3)
    axes[1].plot(rms_t, rms, color="darkorange", linewidth=1.0)
    axes[1].fill_between(rms_t, rms, alpha=0.4, color="darkorange")
    axes[1].set_ylabel("RMS")
    axes[1].set_xlabel("time (s)")
    axes[1].set_xlim(0, dur)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")

    # Cleanup tmp if we created one
    if wav_path != src and wav_path.parent == Path(tempfile.gettempdir()):
        try:
            os.remove(wav_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
