"""Figures. Only aggregated quantities are plotted; person-level lines are
drawn without identifiers and in random order."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

WITHIN = "#2a78d6"  # categorical slot 1 (blue)
BETWEEN = "#eb6834"  # categorical slot 2 (orange)
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def icc_plot(icc_table: pd.DataFrame, path):
    t = icc_table.sort_values("icc")
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(t) + 1.2))
    y = np.arange(len(t))
    ax.barh(y, t["icc"], color=BETWEEN, height=0.55, label="between-person share (ICC)")
    ax.barh(y, 1 - t["icc"], left=t["icc"], color=WITHIN, height=0.55, label="within-person share")
    for yi, v in zip(y, t["icc"]):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8, color="white" if v > 0.9 else INK)
    ax.set_yticks(y, t["label"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of total variance")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def coef_plot(coefs: pd.DataFrame, path, title: str = ""):
    """``coefs`` columns: outcome, predictor, level ('within'/'between'),
    estimate, ci_low, ci_high. One panel per outcome."""

    outcomes = list(dict.fromkeys(coefs["outcome"]))
    preds = list(dict.fromkeys(coefs["predictor"]))
    fig, axes = plt.subplots(1, len(outcomes), figsize=(4.2 * len(outcomes), 0.55 * len(preds) + 1.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, out in zip(axes, outcomes):
        c = coefs[coefs.outcome == out]
        for k, (lvl, col, off) in enumerate((("within", WITHIN, -0.14), ("between", BETWEEN, 0.14))):
            cc = c[c.level == lvl].set_index("predictor").reindex(preds)
            y = np.arange(len(preds)) + off
            ax.errorbar(cc["estimate"], y, xerr=[cc["estimate"] - cc["ci_low"], cc["ci_high"] - cc["estimate"]],
                        fmt="o", color=col, ms=5, lw=1.6, capsize=0, label=f"{lvl}-person")
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(np.arange(len(preds)), preds)
        ax.invert_yaxis()
        ax.set_title(out, fontsize=10, color=INK)
        ax.set_xlabel("change in outcome (95% CI)", fontsize=9)
        _style(ax)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5 * len(outcomes), -0.22), ncol=2, frameon=False, fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def spaghetti_plot(fixed_int, fixed_slope, person_int, person_slope, x_range, path, xlabel, ylabel, seed=0):
    """Person-specific lines from random-slope BLUPs (no IDs), plus the fixed
    within-person line. Person offsets are relative to the fixed intercept."""

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(person_int))
    x = np.linspace(*x_range, 20)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for i in order:
        ax.plot(x, fixed_int + person_int[i] + (fixed_slope + person_slope[i]) * x, color=MUTED, alpha=0.35, lw=1)
    ax.plot(x, fixed_int + fixed_slope * x, color=WITHIN, lw=2.5, label="average within-person slope")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def compliance_curve(curve: pd.DataFrame, path, min_persons: int = 20):
    c = curve[curve["n_persons"] >= min_persons]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.fill_between(c["study_day"], c["ci_low"], c["ci_high"], color=WITHIN, alpha=0.18, lw=0)
    ax.plot(c["study_day"], c["mean"], color=WITHIN, lw=2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of prompts answered\n(mean over persons, 95% CI)", fontsize=9)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax2.bar(c["study_day"], c["n_persons"], color=MUTED, width=0.8)
    ax2.set_ylabel("persons", fontsize=9)
    ax2.set_xlabel("study day (since first scheduled prompt)")
    _style(ax2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
