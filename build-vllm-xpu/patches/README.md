# Patches for Aurora self-built stack

See parent `../README.md` and `../VERSIONS.md`.

| File | Component | Purpose |
|------|-----------|---------|
| `triton_intel_driver_opencl_optional.txt` | Triton 3.6 `driver.c` | OpenCL twin probe try/catch + `TRITON_INTEL_DEVICE_EXTENSIONS` (AuroraBug#102) |
| `triton_intel_driver.c.aurora-opencl-optional` | same | Saved patched `driver.c` copy |
| `mem_info.cpp.aurora-ze-fallback` | vllm_xpu_kernels | ZE memory-info fallback for Aurora build |
| `block_table_slot_mapping_torch_fallback.txt` | vLLM | Torch slot-mapping when Triton unavailable |

Re-apply after `pip install --force-reinstall` of triton / vllm / kernels.
