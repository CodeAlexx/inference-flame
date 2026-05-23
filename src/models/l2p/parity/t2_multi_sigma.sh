#!/bin/bash
# T2 — Forward parity at 5 sigmas.
#
# Re-runs the existing forward-parity infrastructure at 5 sigma values to
# confirm DiT-body math is invariant to sigma. Pairs Python `sigma` with
# Rust `v_in = 1 - sigma` so both hit the same internal t_emb input
# (Python `t*1000` vs Rust `(1-v_in)*1000`).
#
# PASS: cos >= 0.999 on every capture point across all 5 sigmas.

set -e
set -u

OUT_DIR="/tmp/l2p_thorough_parity/t2_multi_sigma"
mkdir -p "$OUT_DIR"

PARITY_DIR="$(dirname "$0")"
RUST_BIN="/home/alex/EriDiffusion/inference-flame/target/release/l2p_parity_capture"
SAMPLE="/home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_l2p_512/10.safetensors"
WEIGHTS="/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors"

# LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib breaks PyTorch's bundled cuDNN
# (libcudnn_graph.so.9 there references symbols missing from PyTorch's cudnn).
# So we DO NOT export it globally — set it ONLY for the Rust binary inline,
# and run Python steps with the inherited (clean) env.
RUST_LD_PATH="/home/alex/libs/libtorch/lib"
export FLAME_ALLOC_POOL=0
export FLAME_NO_CUDNN_SDPA_BWD=1

SIGMAS=(0.05 0.25 0.50 0.75 0.95)
PASS_ALL=1

for SIGMA in "${SIGMAS[@]}"; do
    # Rust v_in = 1 - python_sigma
    RUST_VIN=$(env -u LD_LIBRARY_PATH python3 -c "print(round(1.0 - $SIGMA, 6))")

    PY_OUT="$OUT_DIR/py_sigma${SIGMA}.safetensors"
    RU_OUT="$OUT_DIR/ru_sigma${SIGMA}.safetensors"

    echo "=== sigma_python=$SIGMA  v_in_rust=$RUST_VIN ==="

    echo "  [python] capturing..."
    env -u LD_LIBRARY_PATH python3 "$PARITY_DIR/python_capture.py" --sigma "$SIGMA" --out "$PY_OUT" 2>&1 | tail -3
    if [ ! -f "$PY_OUT" ]; then
        echo "  FAIL at sigma=$SIGMA (python capture produced no output)"
        PASS_ALL=0
        continue
    fi

    echo "  [rust]   capturing..."
    LD_LIBRARY_PATH="$RUST_LD_PATH:$LD_LIBRARY_PATH" "$RUST_BIN" --model "$WEIGHTS" --sample "$SAMPLE" --sigma "$RUST_VIN" --out "$RU_OUT" 2>&1 | tail -3
    if [ ! -f "$RU_OUT" ]; then
        echo "  FAIL at sigma=$SIGMA (rust capture produced no output)"
        PASS_ALL=0
        continue
    fi

    echo "  [diff]"
    if env -u LD_LIBRARY_PATH python3 "$PARITY_DIR/diff_parity.py" --python "$PY_OUT" --rust "$RU_OUT" --threshold 0.999 2>&1 | tail -45 | tee "$OUT_DIR/diff_sigma${SIGMA}.txt" | grep -E "first divergence|all-clean"; then
        :
    fi

    if grep -q "all-clean" "$OUT_DIR/diff_sigma${SIGMA}.txt"; then
        echo "  PASS at sigma=$SIGMA"
    else
        echo "  FAIL at sigma=$SIGMA"
        PASS_ALL=0
    fi
done

echo
if [ $PASS_ALL -eq 1 ]; then
    echo "[T2] ALL 5 sigmas PASS"
    exit 0
else
    echo "[T2] FAIL — some sigma failed"
    exit 1
fi
