#!/usr/bin/env python3
"""Poster plots for gpt-oss-120b MXFP4 serving on one Aurora node.

Numbers are the quality-OK points in CONCURRENCY_RESULTS.md (2026-08-16):
prompt ~93 tokens, max_tokens=128.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "gpt-oss-120b" / "vllm"

INK = "#1c2833"
MUTED = "#5d6d7e"
BLUE = "#1f4e79"
SLATE = "#7f8c8d"
ORANGE = "#b85c38"
GRID = "#d5d8dc"


def _style(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=10)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def _fmt_tok(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f}"
    return f"{value:.1f}"


def _fmt_ttft(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.1f} s"


def plot_peak_deployment(path: Path) -> None:
    # Top-to-bottom. tok/s is peak aggregate generation on one node.
    # ttft_s is None when that run did not record time to first token.
    # gpt-oss and Inkling concurrency runs did not log TTFT; those markers
    # are the single-stream TTFT of the same engine. Nemotron TTFT is p50
    # at the same concurrency as the tok/s bar.
    rows = [
        ("gpt-oss-120B\nvLLM 3× TP=4, 12 tiles", 4565, 0.052),
        ("gpt-oss-120B\nvLLM 1× TP=8, 8 tiles", 3597, 0.047),
        ("gpt-oss-120B\nllama.cpp 2-tile GPU", 417, 0.558),
        ("gpt-oss-120B\nllama.cpp 6× MoE→CPU", 110, None),
        ("Inkling-975B\nllama.cpp 12-tile batched", 16.8, 2.54),
        ("Nemotron-3-Ultra-550B\n3× 4-tile full-XPU", 15.6, 12.5),
        ("Nemotron-4-340B\n3× 4-tile full-XPU", 9.9, 10.6),
    ]
    labels = [row[0] for row in rows]
    tok_s = [row[1] for row in rows]
    ttft_s = [row[2] for row in rows]

    fig, ax = plt.subplots(figsize=(8.6, 6.6), dpi=300)
    fig.patch.set_facecolor("white")
    y = [len(rows) - 1 - i for i in range(len(rows))]
    y_tok = [yi + 0.16 for yi in y]
    y_ttft = [yi - 0.16 for yi in y]

    ttft_y = [yi for yi, t in zip(y_ttft, ttft_s) if t is not None]
    ttft_v = [t for t in ttft_s if t is not None]
    ax.barh(ttft_y, ttft_v, color=ORANGE, height=0.30, zorder=3, label="Time to first token")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Time to first token (s)")
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    _style(ax)
    ax.spines["bottom"].set_color(ORANGE)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=ORANGE)
    ax.xaxis.label.set_color(ORANGE)

    ax_tok = ax.twiny()
    ax_tok.barh(y_tok, tok_s, color=BLUE, height=0.30, zorder=3, label="Generation throughput")
    ax_tok.set_xlim(0, 6400)
    ax_tok.set_xlabel("Aggregate generation throughput (tok/s)")
    ax_tok.set_facecolor("none")
    for spine in ("left", "right", "bottom"):
        ax_tok.spines[spine].set_visible(False)
    ax_tok.spines["top"].set_color(BLUE)
    ax_tok.tick_params(axis="x", colors=BLUE, labelsize=10)
    ax_tok.xaxis.label.set_color(BLUE)

    for yi, value in zip(y_tok, tok_s):
        ax_tok.text(
            value + 80,
            yi,
            _fmt_tok(value),
            va="center",
            ha="left",
            color=BLUE,
            fontsize=8,
        )
    for yi, seconds in zip(y_ttft, ttft_s):
        if seconds is None:
            continue
        ax.text(
            seconds + 0.25,
            yi,
            _fmt_ttft(seconds),
            va="center",
            ha="left",
            color=ORANGE,
            fontsize=8,
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="Time to first token (bottom axis)"),
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="Generation throughput (top axis)"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        fontsize=8,
    )

    # ax.set_title(
    #     "LLMs on one Aurora node — peak generation and time to first token",
    #     loc="left",
    #     color=INK,
    #     fontsize=12,
    #     pad=28,
    # )
    # fig.text(
    #     0.0,
    #     -0.18,
    #     "Throughput is peak aggregate generation on one node. Nemotron TTFT is p50 at that same concurrency.\n"
    #     "gpt-oss and Inkling concurrency runs did not record TTFT; those bars are single-stream "
    #     "(vLLM TP=4 52 ms, TP=8 47 ms, llama.cpp 2-tile 558 ms, Inkling PG12 2.54 s).\n"
    #     "gpt-oss 6× MoE→CPU has no TTFT. Sources: CONCURRENCY_RESULTS.md, VLLM_RESULTS.md, RESULTS.md, "
    #     "INKLING_RESULTS.md, nemotron/BEST_RECIPES.md, BEST_RECIPES_N4.md.",
    #     color=MUTED,
    #     fontsize=7.5,
    #     transform=fig.transFigure,
    # )
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    fig.savefig(path, bbox_inches="tight", pad_inches=0.35, facecolor="white")
    plt.close(fig)


def plot_concurrency(path: Path) -> None:
    concurrency = [64, 128, 256]
    vllm = [1031, 2049, 3597]  # best TP: TP=4 at 64-way, TP=8 at 128 and 256
    llama = [194, 277, 417]  # 2-tile pure GPU, llama-batched-bench

    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=160)
    fig.patch.set_facecolor("white")
    ax.plot(
        concurrency,
        vllm,
        color=BLUE,
        marker="o",
        markersize=7,
        linewidth=2.2,
        label="vLLM, best TP (one engine)",
        zorder=3,
    )
    ax.plot(
        concurrency,
        llama,
        color=SLATE,
        marker="s",
        markersize=6,
        linewidth=2.2,
        label="llama.cpp, 2-tile pure GPU",
        zorder=3,
    )
    for x, y in zip(concurrency, vllm):
        ax.annotate(
            f"{y:,}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            color=BLUE,
            fontsize=9,
        )
    for x, y in zip(concurrency, llama):
        ax.annotate(
            f"{y:,}",
            (x, y),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            color=SLATE,
            fontsize=9,
        )

    ax.set_xticks(concurrency, ["64-way", "128-way", "256-way"])
    ax.set_xlim(40, 280)
    ax.set_ylim(0, 4200)
    ax.set_xlabel("Concurrent sequences")
    ax.set_ylabel("Aggregate generation throughput (tok/s)")
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    ax.set_title(
        "gpt-oss-120b MXFP4 — aggregate generation vs concurrency",
        loc="left",
        color=INK,
        fontsize=12,
        pad=12,
    )
    fig.text(
        0.01,
        0.01,
        "Source: CONCURRENCY_RESULTS.md head-to-head · 2026-08-16 · "
        "vLLM TP=4 at 64-way, TP=8 at 128- and 256-way · "
        "4565 tok/s (3× TP=4) is a different layout and is not on this curve",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    peak = OUT / "gptoss_peak_deployment.png"
    scaling = OUT / "gptoss_concurrency_scaling.png"
    plot_peak_deployment(peak)
    plot_concurrency(scaling)
    print(peak)
    print(scaling)


if __name__ == "__main__":
    main()
