#!/usr/bin/env python3
"""Regenerate the figures and result tables from the raw logs.

Reads logs/*.log and writes figures/*.png plus analysis/results_table*.md.
Every reported number is derived here from the logs. Needs only matplotlib.

    python3 analysis/make_figures.py
"""
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Palette: reference categorical slots (validated for colour-blind separation).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
C_PUBLISHED = "#2a78d6"   # slot 1: published checkpoint
C_FINETUNED = "#eb6834"   # slot 2: after our fine-tune
C_MATCH = "#1baf7a"       # slot 3: matching time (fig 4); run 4 (fig 5)
C_GPU = "#eda100"         # slot 4: pre-decoder GPU time

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

# Shots per basis for each evaluation log (not recorded in the log's summary line).
SHOTS = {}
for name in ("before_si1000_p002 before_si1000_p003 before_dep_p003 after1_si1000_p002 after1_si1000_p003 "
             "after1_dep_p003 acc13_after3_si1000_p003 acc13_after3_si1000_p002 acc13_after3_dep_p003 "
             "acc13_before_si1000_p002 acc13_before_dep_p003 d13_before_si1000_p002 d13_after1_si1000_p002 "
             "d13_before_dep_p003 d13_after1_dep_p003 acc13_after4_si1000_p002 acc13_after4_si1000_p003 "
             "acc13_after4_dep_p003").split():
    SHOTS[name] = 1048576
for name in "d13_before_si1000_p003 d13_after1_si1000_p003 acc13_before_si1000_p003".split():
    SHOTS[name] = 524288
for name in ("d17_before_si1000_p003 d17_after1_si1000_p003 r26_acc_before_dep_p003 r26_fast_before_dep_p003 "
             "r52_acc_before_dep_p003 r52_fast_before_dep_p003 r52_acc_before_si1000_p002 r52_acc_after3_si1000_p002 "
             "r52_fast_before_si1000_p002 r52_fast_after1_si1000_p002 r52_acc_after3_dep_p003 r52_fast_after1_dep_p003 "
             "r52_acc_after4_si1000_p002 r52_acc_after4_dep_p003").split():
    SHOTS[name] = 262144
for name in ("r52_acc_before_si1000_p003 r52_acc_after3_si1000_p003 r52_fast_before_si1000_p003 "
             "r52_fast_after1_si1000_p003 d17r34_acc_before_si1000_p003 d17r34_acc_after3_si1000_p003 "
             "r52_acc_after4_si1000_p003").split():
    SHOTS[name] = 131072


def load(name):
    """Return dict(alone, with_pre, ratio, se, shots) for one evaluation log, or None if absent."""
    path = LOGS / f"{name}.log"
    if not path.exists():
        return None
    lines = [l for l in path.read_text(errors="replace").splitlines() if "Inference Summary" in l]
    if not lines:
        return None
    d = json.loads(lines[-1][lines[-1].index("{"):])
    a, b = d["ler"]["avg_no_predecoder"], d["ler"]["avg_after_predecoder"]
    n = 2 * SHOTS[name]  # X and Z bases
    se_a, se_b = math.sqrt(a * (1 - a) / n), math.sqrt(b * (1 - b) / n)
    ratio = a / b
    se = ratio * math.sqrt((se_a / a) ** 2 + (se_b / b) ** 2)
    lat = d.get("pymatching_latency_us_per_round", {})
    return dict(alone=a, with_pre=b, ratio=ratio, se=se, shots=SHOTS[name],
                lat_alone=lat.get("avg_no_predecoder"), lat_after=lat.get("avg_after_predecoder"))


def epoch_curve(logname):
    text = (LOGS / logname).read_text(errors="replace")
    return [float(x) for x in re.findall(r"LER reduction factor: ([\d.]+)x", text)]


# (row label, published log, fine-tuned log)
SI1000_ROWS = [
    ("Accurate model (R = 13)", None, None),
    ("d=13, 52 rounds, p=0.003", "r52_acc_before_si1000_p003", "r52_acc_after3_si1000_p003"),
    ("d=17, 34 rounds, p=0.003", "d17r34_acc_before_si1000_p003", "d17r34_acc_after3_si1000_p003"),
    ("d=13, 13 rounds, p=0.003", "acc13_before_si1000_p003", "acc13_after3_si1000_p003"),
    ("d=13, 52 rounds, p=0.002", "r52_acc_before_si1000_p002", "r52_acc_after3_si1000_p002"),
    ("d=13, 13 rounds, p=0.002", "acc13_before_si1000_p002", "acc13_after3_si1000_p002"),
    ("Fast model (R = 9)", None, None),
    ("d=13, 52 rounds, p=0.003", "r52_fast_before_si1000_p003", "r52_fast_after1_si1000_p003"),
    ("d=17, 17 rounds, p=0.003", "d17_before_si1000_p003", "d17_after1_si1000_p003"),
    ("d=13, 13 rounds, p=0.003", "d13_before_si1000_p003", "d13_after1_si1000_p003"),
    ("d=9, 9 rounds, p=0.003", "before_si1000_p003", "after1_si1000_p003"),
    ("d=13, 52 rounds, p=0.002", "r52_fast_before_si1000_p002", "r52_fast_after1_si1000_p002"),
    ("d=13, 13 rounds, p=0.002", "d13_before_si1000_p002", "d13_after1_si1000_p002"),
    ("d=9, 9 rounds, p=0.002", "before_si1000_p002", "after1_si1000_p002"),
]
DEP_ROWS = [
    ("Accurate model (R = 13)", None, None),
    ("d=13, 52 rounds", "r52_acc_before_dep_p003", "r52_acc_after3_dep_p003"),
    ("d=13, 13 rounds", "acc13_before_dep_p003", "acc13_after3_dep_p003"),
    ("Fast model (R = 9)", None, None),
    ("d=13, 52 rounds", "r52_fast_before_dep_p003", "r52_fast_after1_dep_p003"),
    ("d=13, 13 rounds", "d13_before_dep_p003", "d13_after1_dep_p003"),
    ("d=9, 9 rounds", "before_dep_p003", "after1_dep_p003"),
]


def usable(rows):
    """Drop rows where either measurement is missing, and headers left with no rows."""
    kept = []
    for label, b, a in rows:
        if b is None:
            kept.append((label, None, None))
        else:
            rb, ra = load(b), load(a)
            if rb and ra:
                kept.append((label, rb, ra))
    out = []
    for i, row in enumerate(kept):
        is_header = row[1] is None
        next_is_data = i + 1 < len(kept) and kept[i + 1][1] is not None
        if not is_header or next_is_data:
            out.append(row)
    return out


def dumbbell(rows, title, subtitle, fname, xlim):
    rows = usable(rows)
    n = len(rows)
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * n + 1.9), dpi=200)
    ax.set_xlim(*xlim)
    ax.set_ylim(n - 0.4, -0.8)
    ax.axvline(1.0, color=AXIS, lw=1, zorder=1)
    ax.text(1.0, -0.62, "parity with PyMatching alone", ha="center", va="center", fontsize=8.5, color=INK2,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2.5), zorder=5)
    for y, (label, rb, ra) in enumerate(rows):
        if rb is None:
            ax.text(xlim[0], y, label, ha="left", va="center", fontsize=10, fontweight="bold", color=INK,
                    transform=ax.transData)
            continue
        ax.plot([rb["ratio"], ra["ratio"]], [y, y], color=AXIS, lw=2, zorder=2, solid_capstyle="round")
        for r, c in ((rb, C_PUBLISHED), (ra, C_FINETUNED)):
            ax.plot([r["ratio"] - r["se"], r["ratio"] + r["se"]], [y, y], color=INK2, lw=1, zorder=3)
            ax.plot(r["ratio"], y, "o", ms=9, color=c, mec=SURFACE, mew=2, zorder=4)
        left, right = (rb, ra) if rb["ratio"] <= ra["ratio"] else (ra, rb)
        ax.text(left["ratio"] - left["se"] - 0.012, y, f"{left['ratio']:.2f}", ha="right", va="center", fontsize=8.5, color=INK2)
        ax.text(right["ratio"] + right["se"] + 0.012, y, f"{right['ratio']:.2f}", ha="left", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks([y for y, r in enumerate(rows) if r[1] is not None])
    ax.set_yticklabels([r[0] for r in rows if r[1] is not None], fontsize=9)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("LER improvement factor:  LER(PyMatching alone) ÷ LER(pre-decoder + PyMatching)", fontsize=9)
    handles = [Line2D([], [], marker="o", ls="", ms=9, color=C_PUBLISHED, mec=SURFACE, mew=2, label="Published checkpoint"),
               Line2D([], [], marker="o", ls="", ms=9, color=C_FINETUNED, mec=SURFACE, mew=2, label="Fine-tuned checkpoint"),
               Line2D([], [], color=INK2, lw=1, label="± 1 standard error")]
    H = fig.get_figheight()  # header is laid out in inches from the top, so it holds at any figure height
    fig.suptitle(title, x=0.012, y=1 - 0.08 / H, ha="left", va="top", fontsize=12.5, fontweight="bold")
    fig.text(0.012, 1 - 0.37 / H, subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 1 - 0.56 / H), ncol=3, frameon=False, fontsize=9,
               handletextpad=0.4, columnspacing=1.6)
    top = 1 - (1.05 / H)
    fig.subplots_adjust(left=0.26, right=0.965, top=top, bottom=0.62 / fig.get_figheight() + 0.02)
    fig.savefig(OUT / fname)
    plt.close(fig)
    return rows


def learning_curves(fname):
    runs = [
        ("Fast model, d = 9, 9 rounds (run 1)", "ft1_train.log", load("before_si1000_p003"), 8388608 / 1e6),
        ("Accurate model, d = 13, 13 rounds (run 3)", "ft3_train_b512.log", load("acc13_before_si1000_p003"), 1048576 / 1e6),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), dpi=200, sharey=True)
    for ax, (name, log, before, step) in zip(axes, runs):
        ys = epoch_curve(log)
        xs = [step * (i + 1) for i in range(len(ys))]
        ax.axhline(1.0, color=AXIS, lw=1, zorder=1)
        ax.plot([0] + xs, [before["ratio"]] + ys, color=C_FINETUNED, lw=2, zorder=2, solid_joinstyle="round")
        ax.plot(xs, ys, "o", ms=6, color=C_FINETUNED, mec=SURFACE, mew=1.5, zorder=3)
        ax.plot(0, before["ratio"], "o", ms=9, color=C_PUBLISHED, mec=SURFACE, mew=2, zorder=4)
        ax.annotate(f"published: {before['ratio']:.2f}", (0, before["ratio"]), xytext=(11, -4), textcoords="offset points",
                    fontsize=8.5, color=INK2)
        last = sum(ys[-4:]) / 4
        ax.annotate(f"last 4 epochs: {last:.2f}", (xs[-1], ys[-1]), xytext=(-4, 12), textcoords="offset points",
                    fontsize=8.5, color=INK2, ha="right")
        ax.set_title(name, loc="left", fontsize=10, color=INK, pad=8)
        ax.set_xlabel("Training samples (millions)", fontsize=9)
        ax.yaxis.grid(True, color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.03 * xs[-1], xs[-1] * 1.03)
    axes[0].set_ylabel("LER improvement factor", fontsize=9)
    axes[0].set_ylim(0.74, 1.42)
    axes[1].tick_params(axis="y", length=0)
    axes[1].spines["left"].set_visible(False)
    handles = [Line2D([], [], marker="o", ls="", ms=9, color=C_PUBLISHED, mec=SURFACE, mew=2, label="Published checkpoint (before fine-tuning)"),
               Line2D([], [], marker="o", ms=6, lw=2, color=C_FINETUNED, mec=SURFACE, mew=1.5, label="After each fine-tuning epoch")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.9), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("LER improvement during fine-tuning", x=0.012, y=0.985, ha="left", fontsize=12.5, fontweight="bold")
    fig.text(0.012, 0.925, "SI1000, p = 0.003. Each point: 131,072 shots per basis (standard error about 0.03).",
             ha="left", va="top", fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.72, bottom=0.135, wspace=0.06)
    fig.savefig(OUT / fname)
    plt.close(fig)


def bench(name):
    text = (LOGS / f"bench_{name}.log").read_text(errors="replace")
    def grab(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None
    per_batch = grab(r"GPU stage per batch \(incl\. copies\)\s+mean\s+([\d.]+)")
    per_shot = grab(r"GPU stage per shot \(batch / size\)\s+mean\s+([\d.]+)")
    return dict(alone=grab(r"decode time per shot\s+mean\s+([\d.]+)"),
                match=grab(r"matching per shot \(residual\)\s+mean\s+([\d.]+)"),
                gpu=per_shot if per_shot is not None else per_batch)


def timing(fname):
    b1, b64 = bench("large_after_b1"), bench("large_after_b64")
    rows = [("PyMatching alone", b1["alone"], 0.0),
            ("With pre-decoder,\nbatch size 1", b1["match"], b1["gpu"]),
            ("With pre-decoder,\nbatch size 64", b64["match"], b64["gpu"])]
    fig, ax = plt.subplots(figsize=(8.6, 3.3), dpi=200)
    h = 0.42
    for y, (label, match, gpu) in enumerate(rows):
        ax.barh(y, match, height=h, color=C_MATCH, zorder=3)
        if gpu:
            ax.barh(y, gpu - 0.7, left=match + 0.7, height=h, color=C_GPU, zorder=3)
        ax.text(match + gpu + 2.5, y, f"{match + gpu:.0f} µs", va="center", ha="left", fontsize=9.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, 165)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.xaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("Decoding time per shot, microseconds (13 rounds)", fontsize=9)
    handles = [Patch(color=C_MATCH, label="Matching (CPU)"), Patch(color=C_GPU, label="Pre-decoder step (GPU, incl. copies)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.865), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Decoding time per shot: PyMatching alone vs pre-decoder + PyMatching", x=0.012, y=0.985,
                 ha="left", fontsize=12.5, fontweight="bold")
    fig.text(0.012, 0.91, "Fine-tuned Accurate model, d = 13, 13 rounds, SI1000 p = 0.003, RTX PRO 4000, TensorRT FP32, 50,000 shots.",
             ha="left", va="top", fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.2, right=0.975, top=0.72, bottom=0.17)
    fig.savefig(OUT / fname)
    plt.close(fig)
    return rows

# (row label, published log, run 3 log, run 4 log); large model, distance 13
RUN4_ROWS = [
    ("SI1000", None, None, None),
    ("52 rounds, p=0.003", "r52_acc_before_si1000_p003", "r52_acc_after3_si1000_p003", "r52_acc_after4_si1000_p003"),
    ("13 rounds, p=0.003", "acc13_before_si1000_p003", "acc13_after3_si1000_p003", "acc13_after4_si1000_p003"),
    ("52 rounds, p=0.002", "r52_acc_before_si1000_p002", "r52_acc_after3_si1000_p002", "r52_acc_after4_si1000_p002"),
    ("13 rounds, p=0.002", "acc13_before_si1000_p002", "acc13_after3_si1000_p002", "acc13_after4_si1000_p002"),
    ("Circuit-level depolarizing, p=0.003", None, None, None),
    ("52 rounds", "r52_acc_before_dep_p003", "r52_acc_after3_dep_p003", "r52_acc_after4_dep_p003"),
    ("13 rounds", "acc13_before_dep_p003", "acc13_after3_dep_p003", "acc13_after4_dep_p003"),
]


def three_way(fname, xlim):
    """Published vs run 3 (fine-tuned at p=0.003) vs run 4 (fine-tuned at p=0.002), large model."""
    rows = [(lab, None) if b is None else (lab, [load(b), load(r3), load(r4)]) for lab, b, r3, r4 in RUN4_ROWS]
    n = len(rows)
    series = [("Published checkpoint", C_PUBLISHED), ("Fine-tuned at p = 0.003 (run 3)", C_FINETUNED),
              ("Fine-tuned at p = 0.002 (run 4)", C_MATCH)]
    fig, ax = plt.subplots(figsize=(8.6, 0.74 * n + 1.9), dpi=200)
    ax.set_xlim(*xlim)
    ax.set_ylim(n - 0.45, -0.85)
    ax.axvline(1.0, color=AXIS, lw=1, zorder=1)
    ax.text(1.0, -0.66, "parity with PyMatching alone", ha="center", va="center", fontsize=8.5, color=INK2,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2.5), zorder=5)
    for y, (label, rs) in enumerate(rows):
        if rs is None:
            ax.text(xlim[0], y + 0.12, label, ha="left", va="center", fontsize=10, fontweight="bold", color=INK)
            continue
        for (name, c), r, dy in zip(series, rs, (-0.27, 0.0, 0.27)):
            ax.plot([r["ratio"] - r["se"], r["ratio"] + r["se"]], [y + dy, y + dy], color=INK2, lw=1, zorder=3)
            ax.plot(r["ratio"], y + dy, "o", ms=8, color=c, mec=SURFACE, mew=1.8, zorder=4)
            ax.text(r["ratio"] + r["se"] + 0.014, y + dy, f"{r['ratio']:.2f}", ha="left", va="center", fontsize=8.5, color=INK2)
    data_y = [y for y, r in enumerate(rows) if r[1] is not None]
    ax.set_yticks(data_y)
    ax.set_yticklabels([rows[y][0] for y in data_y], fontsize=9)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("LER improvement factor:  LER(PyMatching alone) ÷ LER(pre-decoder + PyMatching)", fontsize=9)
    handles = [Line2D([], [], marker="o", ls="", ms=8, color=c, mec=SURFACE, mew=1.8, label=name) for name, c in series]
    handles.append(Line2D([], [], color=INK2, lw=1, label="± 1 standard error"))
    H = fig.get_figheight()
    fig.suptitle("Fine-tuning at p = 0.003 (run 3) vs p = 0.002 (run 4)",
                 x=0.012, y=1 - 0.08 / H, ha="left", va="top", fontsize=12.5, fontweight="bold")
    fig.text(0.012, 1 - 0.37 / H, "Accurate model, d = 13. Runs 3 and 4 differ only in the SI1000 noise strength used for fine-tuning.",
             ha="left", va="top", fontsize=9.5, color=INK2)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 1 - 0.56 / H), ncol=2, frameon=False, fontsize=9,
               handletextpad=0.4, columnspacing=1.6)
    fig.subplots_adjust(left=0.2, right=0.965, top=1 - 1.25 / H, bottom=0.62 / H + 0.02)
    fig.savefig(OUT / fname)
    plt.close(fig)
    lines = ["| Noise | Setting (Accurate model, d=13) | Shots per basis | Published | Run 3 (fine-tuned at p=0.003) | Run 4 (fine-tuned at p=0.002) |",
             "|---|---|---|---|---|---|"]
    noise = ""
    for label, rs in rows:
        if rs is None:
            noise = "SI1000" if "SI1000" in label else "Depolarizing p=0.003"
            continue
        shots = sorted({r["shots"] for r in rs})
        lines.append(f"| {noise} | {label} | {' / '.join(f'{s:,}' for s in shots)} | "
                     + " | ".join(f"{r['ratio']:.2f} ± {r['se']:.2f}" for r in rs) + " |")
    (ROOT / "analysis" / "results_table_run4.md").write_text("\n".join(lines) + "\n")
    return lines


def table(rows_si, rows_dep):
    lines = ["| Noise | Model | Setting | Shots per basis | LER, PyMatching alone | Published | Fine-tuned |",
             "|---|---|---|---|---|---|---|"]
    for noise, rows in (("SI1000", rows_si), ("Depolarizing p=0.003", rows_dep)):
        model = ""
        for label, rb, ra in rows:
            if rb is None:
                model = label.replace(" model", "")
                continue
            shots = f"{rb['shots']:,}" if rb["shots"] == ra["shots"] else f"{rb['shots']:,} / {ra['shots']:,}"
            lines.append(f"| {noise} | {model} | {label} | {shots} | {rb['alone']:.3e} | "
                         f"{rb['ratio']:.2f} ± {rb['se']:.2f} | {ra['ratio']:.2f} ± {ra['se']:.2f} |")
    (ROOT / "analysis" / "results_table.md").write_text("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    si = dumbbell(SI1000_ROWS, "SI1000 noise: published vs fine-tuned checkpoints",
                  "Values above 1: the pre-decoder lowers the logical error rate relative to PyMatching alone.",
                  "fig1_si1000_before_after.png", (0.40, 1.86))
    dep = dumbbell(DEP_ROWS, "Circuit-level depolarizing noise, p = 0.003: published vs fine-tuned checkpoints",
                   "The noise model the published checkpoints were trained on.",
                   "fig2_depolarizing_before_after.png", (0.40, 1.86))
    learning_curves("fig3_learning_curves.png")
    t = timing("fig4_timing.png")
    lines = table(si, dep)
    print("\n".join(lines))
    print("\n".join(three_way("fig5_run3_vs_run4.png", (0.55, 1.86))))
    print("run 4 per-epoch checks:", epoch_curve("ft4_train_b512.log"))
    print("timing rows:", [(r[0].replace("\n", " "), round(r[1], 1), round(r[2], 1)) for r in t])
