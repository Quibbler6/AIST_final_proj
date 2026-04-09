import os
import re
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_step_scores(eval_dir: str, benchmark: str = "aime"):
    """
    Load all <step>_<benchmark>.json files and compute mean score per step.

    Args:
        eval_dir: directory containing evaluation json files
        benchmark: benchmark name, e.g. "aime"

    Returns:
        steps: List[int]
        scores: List[float]
    """
    eval_path = Path(eval_dir)
    pattern = re.compile(rf"^(\d+)_{re.escape(benchmark)}\.json$")

    step_to_score = {}

    for file in eval_path.iterdir():
        if not file.is_file():
            continue

        match = pattern.match(file.name)
        if match is None:
            continue

        step = int(match.group(1))

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Each file is a list of samples; each sample has "scores": [x]
        sample_scores = []
        for item in data:
            if "scores" not in item:
                continue
            scores = item["scores"]
            if isinstance(scores, list) and len(scores) > 0:
                sample_scores.append(float(scores[0]))

        if len(sample_scores) == 0:
            print(f"[Warning] No valid scores found in {file}")
            continue

        step_score = float(np.mean(sample_scores))
        step_to_score[step] = step_score

    steps = sorted(step_to_score.keys())
    scores = [step_to_score[s] for s in steps]
    return steps, scores


def plot_curve(
    steps,
    scores,
    save_prefix="aime_step_score",
    title="AIME Step Score",
):
    """
    Plot a publication-style training curve.
    """
    # Publication-ish style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        steps,
        scores,
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        label="AIME",
    )

    # Highlight best point
    best_idx = int(np.argmax(scores))
    best_step = steps[best_idx]
    best_score = scores[best_idx]
    ax.scatter([best_step], [best_score], s=55, zorder=5)
    ax.annotate(
        f"Best: {best_score:.3f} @ {best_step}",
        xy=(best_step, best_score),
        xytext=(10, 8),
        textcoords="offset points",
        fontsize=12,
    )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Step Score")
    ax.set_title(title)

    # AIME score is usually in [0, 1]
    ax.set_ylim(-0.02, 1.02)

    # Clean paper style
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Optional: slightly fewer x ticks for neatness
    if len(steps) > 10:
        xticks = np.linspace(min(steps), max(steps), num=7, dtype=int)
        ax.set_xticks(sorted(set(xticks.tolist())))

    ax.legend(frameon=False)
    fig.tight_layout()

    pdf_path = f"{save_prefix}.pdf"
    png_path = f"{save_prefix}.png"
    # fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # print(f"Saved figure to: {pdf_path}")
    print(f"Saved figure to: {png_path}")


def main():
    # eval_dir = "oat-output/qwen2.5-Math-1.5b-r1-zero_0327T20:34:58/eval_results"
    eval_dir = "oat-output/qwen2.5-Math-1.5b-drgrpo-qwenmathtemplate_0401T10:57:43/eval_results"
    steps, scores = load_step_scores(eval_dir, benchmark="aime")

    if len(steps) == 0:
        raise ValueError(f"No AIME json files found in {eval_dir}")

    print(f"Loaded {len(steps)} points.")
    print(f"Step range: {steps[0]} -> {steps[-1]}")
    print(f"Best score: {max(scores):.4f} at step {steps[int(np.argmax(scores))]}")

    plot_curve(
        steps,
        scores,
        save_prefix="aime_step_score_curve_qwen_template",
        title="AIME Step Score over Training",
    )


if __name__ == "__main__":
    main()