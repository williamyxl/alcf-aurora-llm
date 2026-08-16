import torch, triton, triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m), mask=m)


def main():
    print("torch", torch.__version__, "triton", triton.__version__,
          "dev", torch.xpu.get_device_properties(0).name, flush=True)
    n = 4096
    x = torch.randn(n, device="xpu"); y = torch.randn(n, device="xpu"); o = torch.empty_like(x)
    grid = (triton.cdiv(n, 256),)
    print("launching triton kernel...", flush=True)
    add_kernel[grid](x, y, o, n, BLOCK=256)
    torch.xpu.synchronize()
    print("TRITON ADD OK, err=", float((o - (x + y)).abs().max()), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
