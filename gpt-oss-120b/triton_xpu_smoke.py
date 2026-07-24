#!/usr/bin/env python
"""Minimal Triton-XPU JIT smoke (must be a real .py file for @triton.jit)."""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def main():
    print("torch", torch.__version__, "xpu_count", torch.xpu.device_count())
    print("triton", triton.__version__)
    n = 1024
    x = torch.ones(n, device="xpu", dtype=torch.float32)
    y = torch.ones(n, device="xpu", dtype=torch.float32)
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(n, 128),)](x, y, out, n, BLOCK=128)
    torch.xpu.synchronize()
    s = float(out.sum())
    print("triton_xpu_smoke_ok", float(out[0]), s)
    assert abs(s - 2 * n) < 1e-3, s


if __name__ == "__main__":
    main()
