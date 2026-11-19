#!/usr/bin/env python3
"""August 2026 (last recipe) vs November 2026 (this session), per model: four panels.

  1. prefill tok/s            (single stream)
  2. generation tok/s          (single stream, decode)
  3. time to first token, ms   (single stream, lower is better)
  4. high-concurrency aggregate generation tok/s

Every value is in DATA below with its source. A missing value is drawn as an empty bar labelled "N/A";
nothing is estimated.

    python3 common/tools/plot_then_vs_now.py   ->  docs/figures/then_vs_now.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "docs" / "figures" / "then_vs_now.png"
N = None

# label, then=(prefill, gen, ttft_ms), now=(prefill, gen, ttft_ms), conc=(then, now) aggregate gen tok/s, source note
DATA = [
    ("gpt-oss-120b MXFP4\nllama.cpp, 1 tile MoE->CPU",
     (41.4, 41.6, 411), (48.6, 42.4, N), (417, 194.4),
     "single: Aug llamacpp/MEASUREMENTS.md (ctx 4096); Nov llama_f4hbm_ctx4096 log (prompt eval, decode; TTFT not logged). "
     "conc: Aug 2-tile batched-bench peak 417; Nov 2-tile batched NP=64 194.4. Tile counts differ."),
    ("gpt-oss-120b MXFP4\nvLLM, 1 x TP4 eager",
     (N, N, N), (1977, 26.3, 44), (N, N),
     "Nov only: fw26 PERF_JSON warm2 (eager). Aug vLLM concurrency was 3 x TP4 / 6 x TP2 layouts, not this one."),
    ("gpt-oss-120b MXFP4\nvLLM, 1 x TP4 XPU graph",
     (N, N, N), (2887, 60.0, 30), (N, N),
     "Nov only: boost_xpugraph PERF_JSON warm2 (XPU graph on)."),
    ("gpt-oss-120b MXFP4\nvLLM, node-level data parallel",
     (N, N, N), (N, N, N), (4565, 11859.9),
     "conc: Aug 3 x TP4 node aggregate ~4565 (fw2025, 128 seqs/engine, 35.7 req/s); Nov 6 x TP2 node aggregate 11859.9 (92.7 req/s). "
     "Same engine family, different layout."),
    ("Laguna S 2.1 BF16\nvLLM, TP8 XPU graph",
     (N, N, N), (1069, 36.6, 58), (N, 2341.7),
     "Nov only: serve_laguna_graph (warm single; 128-way, 512 req). No Aug run."),
    ("Nemotron-3-Ultra UD-IQ2_M\nllama.cpp, 1 tile MoE->CPU",
     (11.9, 7.1, 2670), (N, 7.8, N), (N, N),
     "single: Aug nemotron-3-ultra/llamacpp/MEASUREMENTS.md; Nov single-tile llama sweep c=1 gen 7.1 (TTFT 2679 ms, prefill not logged)."),
    ("Nemotron-3-Ultra UD-IQ2_M\nllama.cpp, 3 x 4 tiles DP",
     (N, N, N), (N, N, N), (15.6, 15.7),
     "conc at c=8, same layout: Aug 15.6 (0.13 req/s), Nov 15.7 (0.132 req/s)."),
    ("Nemotron-3-Ultra INT4\nvLLM, TP8 eager",
     (N, N, N), (194.5, 7.8, 185), (N, 115.0),
     "Nov only: vLLM INT4 with patched top-k=22 kernel, warm single; conc 16-way, 32 req."),
    ("Nemotron-4-340B i1-Q4_K_M\nllama.cpp, 6 tiles",
     (11.0, 2.3, 2390), (N, 2.3, 2375), (9.9, 6.6),
     "single: Aug nemotron-4-340b/llamacpp/MEASUREMENTS.md; Nov single.out c=1 (TTFT 2375, prefill not logged). "
     "conc: Aug 3 x 4-tile DP 9.9; Nov 2 x 6-tile DP c=8 6.6. Layouts differ."),
    ("Inkling UD-IQ1_S\nllama.cpp, 12 tiles",
     (8.70, 6.73, 2300), (6.19, 6.57, N), (N, 17.1),
     "single: Aug inkling/llamacpp/MEASUREMENTS.md PG10; Nov 17-token prompt (prefill not like-for-like), TTFT not logged. "
     "conc: Nov batched-bench 128 seqs (15.3 at 64); no Aug batched run."),
    ("Nemotron-4-340B FP8-Dynamic\nvLLM, TP8 eager",
     (N, N, N), (N, N, N), (N, N),
     "Nov: the run completed but output is INVALID (ok_frac 0, garbage text). Its numbers (10.0 gen, 381 conc, 167 prefill) are not reported as results."),
]

def _fmt(v, f):
    return "N/A" if v is None else f.format(v)

def _cell(v, big):
    if v is None:
        return "N/A"
    return f"{v:,.0f}" if big else f"{v:,.1f}"

def readme_table():
    rows = ["| Model — engine, layout | Prefill tok/s (Aug / Nov) | Generation tok/s (Aug / Nov) | TTFT ms (Aug / Nov) | High-concurrency gen tok/s (Aug / Nov) |",
            "|---|---:|---:|---:|---:|"]
    for r in DATA:
        name = r[0].replace("\n", " — ")
        pf = f"{_cell(r[1][0], r[1][0] is None or r[1][0] >= 100)} / {_cell(r[2][0], r[2][0] is None or r[2][0] >= 100)}"
        gn = f"{_cell(r[1][1], False)} / {_cell(r[2][1], False)}"
        tt = f"{_cell(r[1][2], True)} / {_cell(r[2][2], True)}"
        cc = f"{_cell(r[3][0], r[3][0] is not None and r[3][0] >= 100)} / {_cell(r[3][1], r[3][1] is not None and r[3][1] >= 100)}"
        rows.append(f"| {name} | {pf} | {gn} | {tt} | {cc} |")
    return "\n".join(rows)

def main():
    labels = [r[0] for r in DATA]
    x = np.arange(len(labels))
    w = 0.38
    panels = [
        ("Prefill, single stream (tok/s)", lambda r, s: r[1 if s == "then" else 2][0], "{:,.0f}", "higher is better"),
        ("Generation, single stream (tok/s)", lambda r, s: r[1 if s == "then" else 2][1], "{:,.1f}", "higher is better"),
        ("Time to first token, single stream (ms)", lambda r, s: r[1 if s == "then" else 2][2], "{:,.0f}", "lower is better"),
        ("High-concurrency aggregate generation (tok/s)", lambda r, s: r[3][0 if s == "then" else 1], "{:,.1f}", "higher is better"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(15, 21), sharex=True)
    for ax, (title, get, fmt, note) in zip(axes, panels):
        for side, off, colour, name in [("then", -w / 2, "#9aa5b1", "Aug 2026 (last recipe)"),
                                         ("now", w / 2, "#1f77b4", "Nov 2026 (this session)")]:
            vals = [get(r, side) for r in DATA]
            bars = ax.bar(x + off, [v or 0 for v in vals], w, color=colour, label=name, edgecolor="black", linewidth=0.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v or 0, _fmt(v, fmt), ha="center", va="bottom",
                        fontsize=8, color="#333" if v is not None else "#888")
        ax.set_title(f"{title}  -  {note}", loc="left", fontsize=11)
        ax.set_yscale("log")
        ax.grid(axis="y", which="major", alpha=0.3)
        ax.set_ylabel("log scale")
    axes[0].legend(loc="upper right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, fontsize=8)
    fig.suptitle("Aurora LLM serving: August vs November 2026", fontsize=13, y=0.995)
    fig.text(0.01, 0.005, "Single-stream: prefill, decode, TTFT. Concurrency: aggregate generation at high concurrency; configs "
             "differ across periods, see the table printed by this script. N/A = not measured. Log scale.", fontsize=8, color="#444")
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("wrote", OUT)
    print(f"{'model / engine':50} {'single then':18} {'single now':18} {'conc then / now':18}")
    for r in DATA:
        t = lambda v: "N/A" if v is None else v
        print(f"{r[0].replace(chr(10),' '):50} {str(r[1]):18} {str(r[2]):18} {str(tuple(t(v) for v in r[3])):18}")
    print()
    for r in DATA:
        print(f"- {r[0].replace(chr(10),' ')}: {r[4]}")
    if "--readme" in __import__("sys").argv:
        update_readme()

def update_readme():
    readme = Path(__file__).resolve().parents[2] / "README.md"
    txt = readme.read_text()
    start = txt.index("<!-- then-vs-now:start -->"); end = txt.index("<!-- then-vs-now:end -->")
    block = ("<!-- then-vs-now:start -->\n" + readme_table() + "\n<!-- then-vs-now:end -->")
    readme.write_text(txt[:start] + block + txt[end + len("<!-- then-vs-now:end -->"):])
    print("updated", readme)

if __name__ == "__main__":
    main()
