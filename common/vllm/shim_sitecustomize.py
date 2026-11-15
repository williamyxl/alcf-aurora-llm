# Direct, one-shot compat patches for vllm 0.26.1 + transformers>=5.22 (Beam/Muse NoPE rope None entries).
# No __import__ hook (that stalled vLLM startup). Imports the 3 vLLM modules once and patches them.
def _d(rp):
    return {k: v for k, v in rp.items() if v is not None} if isinstance(rp, dict) and any(v is None for v in rp.values()) else rp
def _go():
    try:
        import vllm.transformers_utils.config as c
        _o = c.patch_legacy_rope_type; c.patch_legacy_rope_type = lambda rp: _o(_d(rp))
    except Exception: pass
    try:
        import vllm.config.model as m
        _l = m._get_and_verify_max_len
        def v(*a, **k):
            hf = k.get("hf_config") or (a[0] if a else None)
            rp = getattr(hf, "rope_parameters", None) if hf is not None else None
            if isinstance(rp, dict) and any(x is None for x in rp.values()):
                sv = dict(rp); hf.rope_parameters = _d(rp)
                try: return _l(*a, **k)
                finally: hf.rope_parameters = sv
            return _l(*a, **k)
        m._get_and_verify_max_len = v
    except Exception: pass
    try:
        import vllm.model_executor.models.transformers.utils as tu
        _c = tu.can_enable_torch_compile
        def ce(vc):
            tc = getattr(getattr(vc, "model_config", None), "hf_config", None); tc = getattr(tc, "text_config", tc) if tc is not None else None
            rp = getattr(tc, "rope_parameters", None) if tc is not None else None
            if isinstance(rp, dict) and any(x is None for x in rp.values()):
                s = dict(rp); tc.rope_parameters = _d(rp)
                try: return _c(vc)
                finally: tc.rope_parameters = s
            return _c(vc)
        tu.can_enable_torch_compile = ce
    except Exception: pass
_go()
