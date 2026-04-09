import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


BENCHMARKS = ["aime", "math", "minerva", "olympiad_bench", "amc"]
MAX_STEP = 416   # ⭐ 核心控制参数


def load_step_scores(eval_dir: str, benchmark: str, max_step: int):
    eval_path = Path(eval_dir)
    pattern = re.compile(rf"^(\d+)_{re.escape(benchmark)}\.json$")

    step_to_score = {}

    for file in eval_path.iterdir():
        match = pattern.match(file.name)
        if match is None:
            continue

        step = int(match.group(1))

        # ⭐ 关键过滤
        if step > max_step:
            continue

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        scores = []
        for item in data:
            if "scores" in item and len(item["scores"]) > 0:
                scores.append(float(item["scores"][0]))

        if len(scores) > 0:
            step_to_score[step] = np.mean(scores)

    steps = sorted(step_to_score.keys())
    values = [step_to_score[s] for s in steps]

    return steps, values


def moving_average(values, window=5):
    values = np.asarray(values)
    if len(values) < window:
        return values

    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def plot_all(eval_dir):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.0,
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        "aime": "#1f77b4",
        "math": "#ff7f0e",
        "minerva": "#2ca02c",
        "olympiad_bench": "#d62728",
        "amc": "#9467bd",
    }

    for bench in BENCHMARKS:
        steps, scores = load_step_scores(eval_dir, bench, MAX_STEP)

        if len(steps) == 0:
            continue

        smoothed = moving_average(scores, window=5)

        # raw
        ax.plot(
            steps,
            scores,
            linewidth=1.0,
            alpha=0.25,
            color=colors[bench]
        )

        # smoothed
        ax.plot(
            steps,
            smoothed,
            linewidth=2.5,
            label=bench.upper(),
            color=colors[bench]
        )

        # best
        best_idx = int(np.argmax(scores))
        ax.scatter(
            steps[best_idx],
            scores[best_idx],
            s=40,
            color=colors[bench],
            zorder=5
        )

    # ===== 核心修改 =====
    ax.set_xlim(0, MAX_STEP)   # ⭐ 强制统一 x 轴

    # 可选：统一刻度（推荐）
    xticks = np.linspace(0, MAX_STEP, 5, dtype=int)
    ax.set_xticks(xticks)

    # ===== 其他设置 =====
    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.02)

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig("five_benchmarks_single_plot_stepno.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved: five_benchmarks_single_plot_stepno.png")


def main():
    # eval_dir = "oat-output/qwen2.5-Math-1.5b-drgrpo-qwenmathtemplate_0401T10:57:43/eval_results"
    # eval_dir = "oat-output/qwen2.5-Math-1.5b-r1-zero_0327T20:34:58/eval_results"
    eval_dir = "oat-output/qwen2.5-Math-1.5b-drgrpo-NOtemplate_0402T14:37:37/eval_results"
    # eval_dir = "oat-output/qwen2.5-Math-1.5b-grpo-qwenmathtemplate_0404T11:07:10/eval_results"
    plot_all(eval_dir)


if __name__ == "__main__":
    main()