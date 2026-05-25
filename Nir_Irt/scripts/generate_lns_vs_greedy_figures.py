from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
PROJECT_DIR = ROOT / "Nir_Irt"
IMAGES_DIR = PROJECT_DIR / "images"
TABLES_DIR = PROJECT_DIR / "tables"

GREEDY_DIR = RESULTS_DIR / "greedy_algorithm"
LNS_DIR = RESULTS_DIR / "lns_solver"

COMMON_DATASETS = [
    "dataset_100.json",
    "dataset_200_2.json",
    "dataset_benchmark.json",
]

DATASET_LABELS = {
    "dataset_100.json": "dataset_100",
    "dataset_200_2.json": "dataset_200_2",
    "dataset_benchmark.json": "dataset_benchmark",
}

METRICS = [
    "final_score",
    "volume_utilization",
    "item_coverage",
    "fragility_score",
    "time_score",
]

METRIC_LABELS = {
    "final_score": "Final score",
    "volume_utilization": "Volume utilization",
    "item_coverage": "Item coverage",
    "fragility_score": "Fragility score",
    "time_score": "Time score",
}


def load_results(solver_dir: Path, dataset_name: str) -> list[dict]:
    with open(solver_dir / dataset_name, "r", encoding="utf-8") as fh:
        return json.load(fh)


def valid_tasks(tasks: list[dict]) -> list[dict]:
    return [task for task in tasks if task.get("metrics", {}).get("valid") is True]


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_dataset(dataset_name: str) -> dict:
    greedy_tasks = valid_tasks(load_results(GREEDY_DIR, dataset_name))
    lns_tasks = valid_tasks(load_results(LNS_DIR, dataset_name))
    if len(greedy_tasks) != len(lns_tasks):
        raise ValueError(f"Dataset size mismatch for {dataset_name}")

    row = {
        "dataset": dataset_name,
        "dataset_label": DATASET_LABELS.get(dataset_name, dataset_name),
        "tasks": len(greedy_tasks),
    }

    for solver_name, tasks in (("greedy", greedy_tasks), ("lns", lns_tasks)):
        for metric in METRICS:
            row[f"{solver_name}_{metric}"] = average(
                [task["metrics"][metric] for task in tasks]
            )
        row[f"{solver_name}_solve_time_ms"] = average(
            [task["solve_time_ms"] for task in tasks]
        )

    wins = 0
    ties = 0
    losses = 0
    for greedy_task, lns_task in zip(greedy_tasks, lns_tasks):
        greedy_score = greedy_task["metrics"]["final_score"]
        lns_score = lns_task["metrics"]["final_score"]
        if lns_score > greedy_score + 1e-9:
            wins += 1
        elif greedy_score > lns_score + 1e-9:
            losses += 1
        else:
            ties += 1

    row["lns_wins"] = wins
    row["ties"] = ties
    row["lns_losses"] = losses
    return row


def write_summary_csv(rows: list[dict]) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "lns_vs_greedy_summary.csv"
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_score(rows: list[dict]) -> None:
    labels = [row["dataset_label"] for row in rows]
    greedy_scores = [row["greedy_final_score"] for row in rows]
    lns_scores = [row["lns_final_score"] for row in rows]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.bar(x - width / 2, greedy_scores, width, label="Greedy", color="#4c78a8")
    ax.bar(x + width / 2, lns_scores, width, label="LNS", color="#f58518")

    ax.set_ylim(0.6, 0.78)
    ax.set_ylabel("Average score")
    ax.set_title("Average final score: LNS vs Greedy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for offset, values in ((-width / 2, greedy_scores), (width / 2, lns_scores)):
        for idx, value in enumerate(values):
            ax.text(x[idx] + offset, value + 0.003, f"{value:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "lns_vs_greedy_score.png", dpi=200)
    plt.close(fig)


def plot_metrics(rows: list[dict]) -> None:
    labels = [row["dataset_label"] for row in rows]
    metric_order = [
        "volume_utilization",
        "item_coverage",
        "fragility_score",
        "time_score",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes = axes.flatten()
    x = np.arange(len(labels))
    width = 0.34

    for ax, metric in zip(axes, metric_order):
        greedy_values = [row[f"greedy_{metric}"] for row in rows]
        lns_values = [row[f"lns_{metric}"] for row in rows]
        ax.bar(x - width / 2, greedy_values, width, label="Greedy", color="#4c78a8")
        ax.bar(x + width / 2, lns_values, width, label="LNS", color="#f58518")
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylim(0.0 if metric == "item_coverage" else 0.45, 1.05)
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(loc="lower right")
    fig.suptitle("Metric comparison on common datasets", fontsize=14)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "lns_vs_greedy_metrics.png", dpi=200)
    plt.close(fig)


def plot_runtime_and_wins(rows: list[dict]) -> None:
    labels = [row["dataset_label"] for row in rows]
    greedy_times = [row["greedy_solve_time_ms"] for row in rows]
    lns_times = [row["lns_solve_time_ms"] for row in rows]
    wins = [row["lns_wins"] for row in rows]
    ties = [row["ties"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    x = np.arange(len(labels))
    width = 0.36

    axes[0].bar(x - width / 2, greedy_times, width, label="Greedy", color="#4c78a8")
    axes[0].bar(x + width / 2, lns_times, width, label="LNS", color="#f58518")
    axes[0].set_title("Average solve time")
    axes[0].set_ylabel("Milliseconds")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=0)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x, wins, width=0.55, label="LNS wins", color="#54a24b")
    axes[1].bar(x, ties, width=0.55, bottom=wins, label="Ties", color="#bab0ab")
    axes[1].set_title("Per-task comparison against Greedy")
    axes[1].set_ylabel("Number of tasks")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=0)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "lns_vs_greedy_runtime_wins.png", dpi=200)
    plt.close(fig)


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    rows = [summarize_dataset(dataset_name) for dataset_name in COMMON_DATASETS]
    write_summary_csv(rows)
    plot_score(rows)
    plot_metrics(rows)
    plot_runtime_and_wins(rows)
    print("Saved figures to", IMAGES_DIR)


if __name__ == "__main__":
    main()
