import torch
torch.xpu.init()
s = torch.xpu.current_stream()
q = s.sycl_queue
print("sycl_queue type:", type(q), "repr:", repr(q)[:120], flush=True)
print("is int?", isinstance(q, int), flush=True)
try:
    print("as int:", int(q), flush=True)
except Exception as e:
    print("int() fails:", e, flush=True)
# PyCapsule?
print("capsule?", "PyCapsule" in str(type(q)), flush=True)
print("DONE", flush=True)
