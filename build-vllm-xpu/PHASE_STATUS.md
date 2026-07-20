# Phase status

## Summary (CLOSED)

| Phase | Status | Closed |
|-------|--------|--------|
| 0 Prep | PASS | 2026-07-16 |
| 1 llvm + triton | PASS | 2026-07-16 (runtime later swapped to Triton 3.6) |
| 2 torch 2.10 XPU | PASS | 2026-07-17 (`2.10.0a0+git449b176`; 2.14 attempt invalidated) |
| 3 IPEX + oneCCL | PASS | 2026-07-17 |
| 4 vllm + kernels | PASS | 2026-07-17 |
| 5 Inference | PASS | 2026-07-18 — `SUCCESS_INFER.md` (job 8680184) |
| 6 LoRA 1 epoch | PASS | 2026-07-18 — `SUCCESS_TRAIN.md` (job 8680277) |

**Project closed:** infra + gpt-oss-120b inference + LoRA train smoke.

Canonical docs: `../README.md`, `VERSIONS.md`, `SUCCESS_INFER.md`, `SUCCESS_TRAIN.md`.

---

## Chronological log

phase=0 status=PASS date=2026-07-16T20:19:40+00:00
notes: ci-lib.sh patched (oneAPI after reset; KEEP_BUILD_DIR persistent build-src); conda env /lus/flare/projects/MOFA/xiaoliyan/workdir/llm/gpt-oss-120b/build-vllm-xpu/env py3.12 with uv/jq; pins.env non-empty branch refs (SHA lock after Phase 4).

phase=1 status=IN_PROGRESS date=2026-07-16T21:38:03+00:00 notes: resubmit after exporting pins.env (unbound FRAMEWORKS_* in job 8676828)

phase=1 status=IN_PROGRESS date=2026-07-16T22:36:23+00:00 notes: llvm PASS; triton FAIL distutils/uv-pip-into-conda; patched setup_uv_venv; resubmitting

phase=1 status=PASS date=2026-07-16T22:51:57+00:00 triton_wheel=triton-3.8.0+gita4fdf97a-cp312-cp312-linux_x86_64.whl llvm=llvm-850a2b1b975c061ae0fc982ba68064d305485cb2

phase=2 status=IN_PROGRESS date=2026-07-16T22:52:05+00:00 notes: first torch PBS submit after Phase 1 CLOSE

phase=2 status=IN_PROGRESS date=2026-07-17T00:24:10+00:00 notes: torch wheel built 2.6G; pip/xpu verify pending (walltime killed install in 8677180)

phase=2 status=PASS date=2026-07-17T00:39:37+00:00 torch=2.14.0a0+git946334e xpu_count=6

phase=3a status=IN_PROGRESS date=2026-07-17T00:39:44+00:00

phase=3a status=FAIL date=2026-07-17T01:32:28+00:00 notes: IPEX xpu-main requires Torch 2.10; built torch was 2.14 from main. Re-pin FRAMEWORKS_TORCH_VERSION=v2.10.0; reopen Phase 2.

phase=2 status=REOPENED date=2026-07-17T01:32:28+00:00 notes: rebuild torch at v2.10.0 to match IPEX; prior PASS invalidated

phase=2 status=IN_PROGRESS date=2026-07-17T01:32:28+00:00 notes: building torch v2.10.0

phase=2 status=PASS date=2026-07-17T02:41:59+00:00 torch=2.10.0a0+git449b176 xpu_count=6 pin=v2.10.0

phase=3a status=IN_PROGRESS date=2026-07-17T02:42:00+00:00 notes: retry IPEX against torch 2.10

phase=3a status=IN_PROGRESS date=2026-07-17T04:22:50+00:00 notes: walltime mid bdist_wheel job 8677463; KEEP_BUILD_DIR resubmit

phase=3a status=IN_PROGRESS date=2026-07-17T05:23:38+00:00 notes: switched to capacity queue walltime=04:00:00 (qdel debug 8677538); KEEP_BUILD_DIR resume

phase=3a status=IN_PROGRESS date=2026-07-17T05:24:20+00:00 notes: capacity walltime=10:00:00 resubmit after qalter blocked

phase=3a status=IN_PROGRESS date=2026-07-17T05:32:54+00:00 notes: back to debug walltime=00:59:59; KEEP_BUILD_DIR chain

phase=3a status=IN_PROGRESS date=2026-07-17T06:47:45+00:00 notes: wheel built 2.4G; pip/import pending (walltime killed install in 8677588)

phase=3a status=PASS date=2026-07-17T07:08:59+00:00 ipex=2.10.10+gitd0f992f

phase=3b status=PASS date=2026-07-17 (oneccl_bind_pt 2.8.0+xpu)

phase=4 status=PASS date=2026-07-17 vllm + vllm_xpu_kernels; VERSIONS.md started

phase=5 status=IN_PROGRESS date=2026-07-17T13:47:00+00:00 notes: Triton 3.8 JIT broken (FuncOp assert even on smoke). Swapped to frameworks triton-3.6.0+git225cdbde wheel; retesting smoke+infer

phase=5 status=IN_PROGRESS date=2026-07-17T14:18:51+00:00 notes: triton 3.6 needs ONEAPI_DEVICE_SELECTOR=level_zero:gpu; resubmitting infer

phase=5 status=IN_PROGRESS date=2026-07-17T14:48:11+00:00 notes: Triton JIT broken on PVC (3.8 FuncOp; 3.6 SYCL no-device). Uninstalled triton so HAS_TRITON=False; retry infer via xpu_kernels/native

phase=5 status=IN_PROGRESS date=2026-07-17T15:13:51+00:00 notes: no-triton hit inductor TritonMissing on embed; retry TORCHDYNAMO_DISABLE=1

phase=5 status=IN_PROGRESS date=2026-07-17T15:29:20+00:00 notes: past dynamo; segfault in CCL allreduce PersistentDeviceCodeCache. unset SYCL_CACHE_PERSISTENT; CCL_WORKER_COUNT=1

phase=5 status=IN_PROGRESS date=2026-07-17T15:47:30+00:00 notes: past KV cache alloc; missing torchvision. Installed frameworks torchvision-0.25.0+8ac84ee --no-deps; resubmit

phase=5 status=IN_PROGRESS date=2026-07-17T16:16:18+00:00 notes: KV alloc OK; kernel_warmup TypeError without Triton. Patched block_table compute_slot_mapping torch fallback; resubmit

phase=5 status=IN_PROGRESS date=2026-07-17T16:48:55+00:00 notes: past slot_mapping; OOM sampler warmup 256 dummy. Set gpu_memory_utilization=0.82 max_num_seqs=16; resubmit

phase=5 status=IN_PROGRESS date=2026-07-17T17:21:39+00:00 notes: job 8679391 got METRICS_JSON but reply was all bangs and decode_tok_s bogus (V1 engine.step). Rewrote one_chat to llm.generate + print warmup; temp=0; resubmit

phase=5 status=IN_PROGRESS date=2026-07-17T17:59:26+00:00 notes: confirmed all token_id=0 (!!!). Known XPU FLASH_ATTN decode accuracy bug. Trying FLEX_ATTENTION + VLLM_XPU_FUSED_MOE_USE_MXFP4_FP8=1

phase=5 status=IN_PROGRESS date=2026-07-17T18:21:44+00:00 notes: FLEX_ATTENTION rejected on XPU. Reinstalled triton-3.6.0+225cdbde; force attention_backend=TRITON_ATTN + MXFP4_FP8; resubmit

phase=5 status=IN_PROGRESS date=2026-07-17T18:50:10+00:00 notes: TRITON_ATTN selected but SYCL No-device at JIT. Revert FLASH_ATTN; try VLLM_XPU_FUSED_MOE_USE_REF=1 + MXFP4_FP8 to isolate MoE vs attn

phase=5 status=IN_PROGRESS date=2026-07-17T19:32:51+00:00 notes: Triton installed breaks FLASH_ATTN init (SYCL No-device). Uninstalled Triton again; FLASH_ATTN + REF MoE + MXFP4_FP8; resubmit

phase=5 status=IN_PROGRESS date=2026-07-17T20:16:15+00:00 notes: AuroraBug#102 — ONEAPI_DEVICE_SELECTOR=level_zero:gpu breaks Triton. Set *:gpu; reinstall triton-3.6; TRITON_ATTN + smoke; drop REF MoE

phase=5 status=IN_PROGRESS date=2026-07-17T20:45:37+00:00 notes: *:gpu set; inline jit smoke failed (needs .py). one_chat SEGV early with TRITON_ATTN. Gate via triton_xpu_smoke.py then one_chat

phase=5 status=IN_PROGRESS date=2026-07-17T21:24:59+00:00 notes: Triton smoke PASS with *:gpu. one_chat SEGV at LLM init. Retry unset ONEAPI_DEVICE_SELECTOR + faulthandler + TRITON_ATTN

phase=5 status=IN_PROGRESS date=2026-07-17T21:53:04+00:00 notes: unset ONEAPI still SEGV at LLM+TRITON_ATTN after smoke PASS. Submit TP=1 diag_triton_attn.pbs

phase=5 status=IN_PROGRESS date=2026-07-17T22:19:21+00:00 notes: TP=1 TRITON_ATTN also SEGV after max_model_len. Triton smoke OK. Submit FLASH_ATTN control with Triton installed

phase=5 status=IN_PROGRESS date=2026-07-17T22:45:37+00:00 notes: FLASH_ATTN also SEGV with Triton+unset ONEAPI. Root cause is selector unset/*:gpu vs vLLM MP. Try level_zero:gpu;opencl:gpu dual selector

phase=5 status=IN_PROGRESS date=2026-07-17T23:01:56+00:00 notes: OpenCL in selector SEGVs vLLM; L0-only throws in Triton init_devices. Patched driver.c try/catch + TRITON_INTEL_DEVICE_EXTENSIONS; submit diag_l0_triton

phase=5 status=IN_PROGRESS date=2026-07-17T23:21:11+00:00 notes: L0+patched-driver smoke PASS (8680059). diag failed missing __main__ guard (not SEGV). Fixed diag; resubmit L0+TRITON_ATTN; updated infer_chat.pbs recipe

phase=5 status=IN_PROGRESS date=2026-07-17T23:39:41+00:00 notes: 8680085 TRITON_ATTN under L0 no SEGV; TP=1 OOM at KV (expected). Submit infer_chat TP=8 with L0+TRITON_INTEL_DEVICE_EXTENSIONS

phase=5 status=IN_PROGRESS date=2026-07-18T00:24:58+00:00 notes: 8680110 TRITON_ATTN+L0 SUCCESS (no SEGV) but all token_id=0 !!!. Confirmed Using Triton backend + XPU Mxfp4 MoE. Next: +VLLM_XPU_FUSED_MOE_USE_REF=1

phase=5 status=PASS date=2026-07-18T01:03:54+00:00 notes: job 8680184 TRITON_ATTN+L0+TRITON_INTEL_DEVICE_EXTENSIONS+VLLM_XPU_FUSED_MOE_USE_REF=1 coherent Type-I isotherm reply + METRICS_JSON. SUCCESS_INFER.md written. Phase 6 unblocked.

phase=6 status=IN_PROGRESS date=2026-07-18T01:09:31+00:00 notes: Phase5 PASS. Installed peft/trl/accelerate (torch unchanged). Submit train_lora_smoke with Mxfp4Config(dequantize=True) + LoRA q/v epochs=1

phase=6 status=IN_PROGRESS date=2026-07-18T01:38:21+00:00 notes: 8680241 loaded dequant+LoRA then TRL chunked_nll partial __func__ crash. Set loss_type=nll; resubmit

phase=6 status=PASS date=2026-07-18T02:16:12+00:00 notes: job 8680277 LoRA/SFT epochs=1 on XPU complete; TRAIN_JSON ok; adapter at checkpoints/lora-smoke/adapter. SUCCESS_TRAIN.md written. Full project closed (infra+infer+train).

phase=project status=CLOSED date=2026-07-18T02:30:00+00:00 notes: Docs refreshed (README.md, VERSIONS.md, PHASE_STATUS summary, patches, plan todos marked complete).

phase=perf status=PAUSED date=2026-07-20T18:00:00+00:00 notes: S2–S5 closed (~0.37 warm tok/s quality recipe). P7 code landed (disable_log_stats=False) but validation not finished (debug queue starved). Standing rule: every future metric campaign = TP=2/4/8 + P7 fields. Session recovery doc: build-vllm-xpu/RESUME.md (full recipe, OOM/P7 root causes, job ledger, commands). Local commits f60a2bb+f6d174d may need git push from authed host.
