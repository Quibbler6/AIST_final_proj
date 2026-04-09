import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_step_scores(eval_dir: str, benchmark: str = "minerva"):
    """
    Load all <step>_<benchmark>.json files and compute mean score per step.

    Args:
        eval_dir: directory containing evaluation json files
        benchmark: benchmark name, e.g. "minerva"

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

        step_to_score[step] = float(np.mean(sample_scores))

    steps = sorted(step_to_score.keys())
    scores = [step_to_score[s] for s in steps]
    return steps, scores


def moving_average(values, window=5):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def plot_curve(
    steps,
    scores,
    save_path="minerva_step_score_curve.png",
    title="Minerva Step Score over Training",
):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.0,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    smoothed = moving_average(scores, window=5)

    # Raw curve
    ax.plot(
        steps,
        scores,
        linewidth=1.3,
        marker="o",
        markersize=3.5,
        alpha=0.35,
        label="Raw",
    )

    # Smoothed curve
    ax.plot(
        steps,
        smoothed,
        linewidth=2.8,
        label="Smoothed",
    )

    # Best point on raw curve
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
    ax.set_ylim(-0.02, 1.02)

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if len(steps) > 10:
        xticks = np.linspace(min(steps), max(steps), num=7, dtype=int)
        ax.set_xticks(sorted(set(xticks.tolist())))

    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {save_path}")


def main():
    eval_dir = "oat-output/qwen2.5-Math-1.5b-drgrpo-qwenmathtemplate_0401T10:57:43/eval_results"
    steps, scores = load_step_scores(eval_dir, benchmark="minerva")

    if len(steps) == 0:
        raise ValueError(f"No Minerva json files found in {eval_dir}")

    print(f"Loaded {len(steps)} points.")
    print(f"Step range: {steps[0]} -> {steps[-1]}")
    print(f"Best score: {max(scores):.4f} at step {steps[int(np.argmax(scores))]}")

    plot_curve(
        steps,
        scores,
        save_path="minerva_step_score_curve_qwen_template.png",
        title="Minerva Step Score over Training",
    )


if __name__ == "__main__":
    main()