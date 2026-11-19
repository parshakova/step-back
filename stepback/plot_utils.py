"""
Plotting and loading utilities for step-back experiments.

Plotting style follows Crawshaw et al. (arXiv 2510.09827).
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from .record import Record

# ── LaTeX rendering and high-DPI defaults ──
matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── Display names for legend (capitalize first letter) ──
DISPLAY_NAMES = {
    "muon":                "Muon",
    "spectral-clip":       "Spectral-Clip",
    "spectral-clip-avg":   "Spectral-Clip-Avg",
    "spectral-clip-1":     "Spectral-Clip-1",
    "spectral-inv":        "Spectral-Inv",
    "spectral-inv-pseudo": "Spectral-Inv-Pseudo",
    "spectral-inv-damp":   "Spectral-Inv-Damp",
    "muon-bot-p":          "Muon-Bot-P",
    "muon-bot-p90":        "Muon-Bot-P90",
    "muon-bot-p50":        "Muon-Bot-P50",
    "muon-bot-p99":        "Muon-Bot-P99",
    "spectral-inv-shift-35": "Spectral-Inv-Shift-35",
    "spectral-clip-ns":    "Spectral-Clip-NS",
    "adamw":               "AdamW",
    "adam":                 "Adam",
    "sgd-m":               "SGD-M",
    "sgd":                 "SGD",
}

def _display_name(name):
    return DISPLAY_NAMES.get(name, name.capitalize())

# ── Default color palette (variants share the base method color) ──
COLORS = {
    "muon":                "#000000",
    "spectral-clip":       "#8A2BE2",
    "spectral-clip-avg":   "#8A2BE2",
    "spectral-clip-1":     "#8A2BE2",
    "adamw":               "#FF6B35",
    "adam":                 "#FF6B35",
    "sgd-m":               "#2CA02C",
    "spectral-inv":        "#D62728",
    "spectral-inv-pseudo": "#D62728",
    "spectral-inv-damp":   "#D62728",
    "spectral-inv-shift-35": "#E377C2",
    "spectral-clip-ns":    "#17BECF",
    "sgd":                 "#f68427",
    "muon-bot-p":          "#005F9E",
    "muon-bot-p90":        "#005F9E",
    "muon-bot-p50":        "#003560",
    "muon-bot-p99":        "#1A8FCD",
}

# ── Line styles (variants of the same method differ by linestyle) ──
LINESTYLES = {
    "muon":                "-",
    "spectral-clip":       "-",
    "spectral-clip-avg":   "-",
    "spectral-clip-1":     "--",
    "adamw":               "-",
    "adam":                 "-",
    "sgd-m":               "-",
    "spectral-inv":        "-",
    "spectral-inv-pseudo": "-",
    "spectral-inv-damp":   "--",
    "spectral-inv-shift-35": "-.",
    "spectral-clip-ns":    "--",
    "sgd":                 "-",
    "muon-bot-p":          "--",
    "muon-bot-p90":        "-",
    "muon-bot-p50":        "--",
    "muon-bot-p99":        "-.",
}


def load_sweep(exp_id, output_dir="output/"):
    """Load per-job output files ``<exp_id>-NNN.json`` into a Record.

    Parameters
    ----------
    exp_id : str
        Experiment identifier (config name without ``.json``).
    output_dir : str
        Directory containing the output JSON files.

    Returns
    -------
    Record
        A merged Record with colors registered for known optimizers.
    """
    pattern = os.path.join(output_dir, f"{exp_id}-*.json")
    exp_ids = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(pattern)
    )
    if not exp_ids:
        raise FileNotFoundError(f"No files matching {pattern}")

    R = Record(exp_ids, output_dir=output_dir)

    for name, color in COLORS.items():
        R.aes[name] = {"color": color, "linestyle": LINESTYLES.get(name, "-"), "markevery": 10, "zorder": 10}

    return R


def plot_val_acc_vs_lr(R, cutoff=None, figsize=(4, 3), filename=None, skip=("sgd-m",), ylim=None):
    """Plot final validation accuracy vs learning rate.

    Parameters
    ----------
    R : Record
        A Record object (from ``load_sweep`` or constructed manually).
    cutoff : tuple or None
        Epoch range ``(start, end)`` to average over.  Default: last epoch only.
    figsize : tuple
        Figure size.
    filename : str or None
        If given, save PDF to this path.
    skip : tuple of str
        Optimizer names (lowercase) to skip. Default: ``("sgd-m",)``.

    Returns
    -------
    fig, ax
    """
    if isinstance(skip, str):
        skip = (skip,)

    # Only group by name + lr; ignore other hyperparameter columns so that
    # each method produces a single curve even when id_df has extra columns.
    ignore = [c for c in R.id_df.columns if c not in ("name", "lr")]
    sweep_df = R.build_sweep_df(score="val_score", xaxis="lr", cutoff=cutoff,
                                ignore_columns=ignore)

    fig, ax = plt.subplots(figsize=figsize)

    for idx in sweep_df.index.unique():
        sub = sweep_df.loc[idx].sort_values("lr") if isinstance(sweep_df.loc[idx], pd.DataFrame) else sweep_df.loc[[idx]]
        # extract optimizer name from the (possibly multi-level) index
        name = idx[0] if isinstance(idx, tuple) else idx
        if name in skip:
            continue
        aes = R.aes.get(name, R.aes["default"])
        color = aes["color"]
        ls = aes.get("linestyle", "-")

        ax.plot(
            sub["lr"], sub["val_score"],
            color=color, linestyle=ls,
            linewidth=2, alpha=0.85, label=_display_name(name), zorder=10,
        )
        if "val_score_std" in sub.columns:
            y = sub["val_score"].values
            ystd = sub["val_score_std"].values
            ax.fill_between(sub["lr"], y - ystd, y + ystd,
                            color=color, alpha=0.12, zorder=0)

    ax.set_xscale("log")
    ax.set_xlabel("Learning rate", fontsize=10)
    ax.set_ylabel("Validation accuracy", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(axis="both", lw=0.2, ls="--", zorder=0)
    if ylim is not None:
        ax.set_ylim(ylim)

    # deduplicate legend entries, place outside plot on top
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    n = len(by_label)
    if n > 0:
        ax.legend(by_label.values(), by_label.keys(),
                  loc="lower center", bbox_to_anchor=(0.5, 1.02),
                  ncol=3, fontsize=9,
                  frameon=False, columnspacing=1.0)

    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.17, right=0.95)

    if filename:
        fig.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)

    return fig, ax


def plot_val_acc_vs_epoch_seeds(R, figsize=(4, 3), filename=None, ylim=None):
    """Plot validation accuracy over epochs for the best LR of each optimizer, with mean +/- std shading.

    Parameters
    ----------
    R : Record
    figsize : tuple
    filename : str or None
    ylim : tuple or None

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name in R.id_df["name"].unique():
        sub = R.base_df[R.base_df["name"] == name]
        final = sub[sub["epoch"] == sub["epoch"].max()]
        best_row = final.loc[final["val_score"].idxmax()]
        best_id = best_row["id"]

        raw = R.raw_df[R.raw_df["id"] == best_id].sort_values(["epoch", "run_id"])
        grouped = raw.groupby("epoch")["val_score"]
        epochs = grouped.mean().index.values
        mean = grouped.mean().values
        std = grouped.std().values

        aes = R.aes.get(name, R.aes["default"])
        color = aes["color"]
        ls = aes.get("linestyle", "-")

        ax.plot(epochs, mean, color=color, linestyle=ls, linewidth=2, alpha=0.85,
                label=_display_name(name), zorder=10)
        ax.fill_between(epochs, mean - std, mean + std,
                        color=color, alpha=0.15, zorder=0)

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Validation accuracy", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(axis="both", lw=0.2, ls="--", zorder=0)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, fontsize=9, frameon=False, columnspacing=1.0)
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.17, right=0.95)

    if filename:
        fig.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)

    return fig, ax


def plot_val_acc_vs_epoch(R, figsize=(4, 3), filename=None, ylim=None):
    """Plot validation accuracy over epochs for the best LR of each optimizer.

    Parameters
    ----------
    R : Record
    figsize : tuple
    filename : str or None

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name in R.id_df["name"].unique():
        sub = R.base_df[R.base_df["name"] == name]
        final = sub[sub["epoch"] == sub["epoch"].max()]
        best_row = final.loc[final["val_score"].idxmax()]
        best_id = best_row["id"]

        curve = R.base_df[R.base_df["id"] == best_id].drop_duplicates(subset=["epoch"], keep="first")
        conf = dict(item.split("=") for item in best_id.split(":"))
        aes = R.aes.get(name, R.aes["default"])
        color = aes["color"]
        ls = aes.get("linestyle", "-")

        ax.plot(
            curve["epoch"], curve["val_score"],
            color=color, linestyle=ls, linewidth=2, alpha=0.85,
            # label=f'{_display_name(name)} (lr={conf["lr"]})',
            label=f'{_display_name(name)}', zorder=10,
        )

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Validation accuracy", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(axis="both", lw=0.2, ls="--", zorder=0)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, fontsize=9,
              frameon=False, columnspacing=1.0)
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.17, right=0.95)

    if filename:
        fig.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)

    return fig, ax
