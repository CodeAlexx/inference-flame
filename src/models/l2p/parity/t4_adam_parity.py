#!/usr/bin/env python3
"""T4 — Adam optimizer step parity.

Compares the EriDiffusion flame-core Adam formula (see flame_core/src/adam.rs:55-99)
against PyTorch's torch.optim.Adam.

Both implement the canonical Kingma-Ba Adam:
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    p_t = p_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)

This is a formula-equivalence test — not GPU bit-exactness.
Both stacks use F32 moments and F32 arithmetic for the update.

PASS: param values after 3 fixed grad steps differ by < 1e-5 vs PyTorch reference.
"""
import sys
import torch


def adam_step_flamecore(p, g, m, v, t, lr, beta1, beta2, eps, weight_decay=0.0):
    """Pure-Python mirror of flame_core::adam::adam_fused_kernel.

    p, g, m, v are F32 torch tensors. Updates in place, returns nothing.
    Matches the kernel at flame-core/src/adam.rs:65-99 except wd applies
    AFTER the moment update (vs PyTorch's decoupled "AdamW" which subtracts
    wd*p BEFORE the moment update). flame-core's `adam.rs` is plain Adam
    (no AdamW); weight_decay path adds to grad (standard Adam-with-wd).

    Mirror is exact for weight_decay=0 (our test case).
    """
    m.mul_(beta1).add_(g, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    m_hat = m / bc1
    v_hat = v / bc2
    step = m_hat / (v_hat.sqrt() + eps)
    p.add_(step, alpha=-lr)
    if weight_decay > 0.0:
        # See flame-core/src/adam.rs:92 — p -= lr * weight_decay * p
        # AFTER the gradient update. This is the "post-multiplicative-decay"
        # variant. PyTorch's plain Adam adds wd*p to the GRADIENT first.
        # These diverge once wd != 0.
        p.mul_(1.0 - lr * weight_decay)


def main():
    torch.manual_seed(42)
    device = "cpu"

    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight_decay = 0.0  # avoid the documented Adam-vs-AdamW divergence

    # Small tensor: 5x5
    shape = (5, 5)
    p_init = torch.randn(shape, device=device, dtype=torch.float32)
    g_list = [torch.randn(shape, device=device, dtype=torch.float32) for _ in range(3)]

    # --- PyTorch reference ---
    p_torch = p_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([p_torch], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    p_torch_history = []
    m_torch_history = []
    v_torch_history = []
    for t in range(1, 4):
        p_torch.grad = g_list[t - 1].clone()
        opt.step()
        st = opt.state[p_torch]
        p_torch_history.append(p_torch.detach().clone())
        m_torch_history.append(st["exp_avg"].clone())
        v_torch_history.append(st["exp_avg_sq"].clone())

    # --- flame-core formula mirror ---
    p_flame = p_init.clone()
    m_flame = torch.zeros_like(p_flame)
    v_flame = torch.zeros_like(p_flame)
    p_flame_history = []
    m_flame_history = []
    v_flame_history = []
    for t in range(1, 4):
        g = g_list[t - 1].clone()
        adam_step_flamecore(p_flame, g, m_flame, v_flame, t, lr, beta1, beta2, eps, weight_decay)
        p_flame_history.append(p_flame.clone())
        m_flame_history.append(m_flame.clone())
        v_flame_history.append(v_flame.clone())

    # Compare.
    print(f"[T4] Adam: lr={lr}, betas=({beta1},{beta2}), eps={eps}, wd={weight_decay}")
    print()
    print(f"{'step':>4}  {'param max_abs':>13}  {'m max_abs':>11}  {'v max_abs':>11}")
    print("-" * 50)
    worst_p = 0.0
    worst_m = 0.0
    worst_v = 0.0
    for t in range(3):
        dp = (p_torch_history[t] - p_flame_history[t]).abs().max().item()
        dm = (m_torch_history[t] - m_flame_history[t]).abs().max().item()
        dv = (v_torch_history[t] - v_flame_history[t]).abs().max().item()
        worst_p = max(worst_p, dp)
        worst_m = max(worst_m, dm)
        worst_v = max(worst_v, dv)
        print(f"{t+1:>4}  {dp:>13.3e}  {dm:>11.3e}  {dv:>11.3e}")
    print()
    print(f"Worst max_abs: param={worst_p:.3e}, m={worst_m:.3e}, v={worst_v:.3e}")
    print()

    THRESHOLD = 1e-5
    if max(worst_p, worst_m, worst_v) < THRESHOLD:
        print(f"[T4] PASS (worst diff < {THRESHOLD})")
        return 0
    else:
        print(f"[T4] FAIL (worst diff >= {THRESHOLD})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
