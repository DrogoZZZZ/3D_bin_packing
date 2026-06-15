from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
IMAGES = ROOT / "images"

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 11,
    }
)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def save(fig, name):
    fig.savefig(IMAGES / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def dataset_composition():
    rows = read_csv(TABLES / "dataset_summary.csv")
    scenarios = []
    parsed = []
    for row in rows:
        values = {}
        for pair in row["scenarios"].split("; "):
            name, count = pair.split(": ")
            values[name] = int(count)
            if name not in scenarios:
                scenarios.append(name)
        parsed.append(values)

    labels = [Path(row["dataset"]).stem.replace("dataset_benchmark", "benchmark") for row in rows]
    colors = ["#66c2a5", "#fc8d62", "#e78ac3", "#a6d854", "#e5c494", "#b3b3b3"]
    fig, ax = plt.subplots(figsize=(11.2, 5.8), constrained_layout=True)
    bottom = np.zeros(len(rows))
    for name, color in zip(scenarios, colors):
        values = np.array([item.get(name, 0) for item in parsed])
        ax.bar(labels, values, bottom=bottom, label=name, color=color)
        bottom += values
    ax.set_ylabel("Number of tasks")
    ax.set_xlabel("Dataset")
    ax.set_title("Scenario composition of generated datasets")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, loc="upper left")
    save(fig, "dataset_composition.png")


def benchmark_charts():
    rows = read_csv(TABLES / "benchmark_summary.csv")
    names = {
        "greedy_algorithm": "Greedy",
        "conditions": "MaxRects",
        "gan_ga_solver": "GAN+GA",
        "lns_solver": "LNS",
    }
    datasets = ["dataset_100.json", "dataset_200_2.json", "dataset_benchmark.json"]
    labels = ["dataset_100", "dataset_200_2", "benchmark"]
    solvers = ["greedy_algorithm", "conditions", "gan_ga_solver", "lns_solver"]
    lookup = {(row["solver"], row["dataset"]): row for row in rows}

    x = np.arange(len(datasets))
    width = 0.19
    fig, ax = plt.subplots(figsize=(11.2, 6.1), constrained_layout=True)
    for index, solver in enumerate(solvers):
        values = [float(lookup[solver, dataset]["score"]) for dataset in datasets]
        ax.bar(x + (index - 1.5) * width, values, width, label=names[solver])
    ax.set_xticks(x, labels)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Average final score")
    ax.set_ylim(0.62, 0.76)
    ax.set_title("Average final score by solver and dataset")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    save(fig, "all_solvers_score.png")

    metrics = [
        ("volume", "Volume utilization"),
        ("coverage", "Item coverage"),
        ("fragility", "Fragility score"),
        ("time_score", "Time score"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2), constrained_layout=True)
    benchmark = "dataset_benchmark.json"
    for ax, (field, title) in zip(axes.flat, metrics):
        values = [float(lookup[solver, benchmark][field]) for solver in solvers]
        ax.bar([names[solver] for solver in solvers], values, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
        ax.set_title(title)
        ax.set_xlabel("Solver")
        ax.set_ylabel("Score")
        ax.set_ylim(0.45, 1.05)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Metric components on benchmark dataset", fontsize=14)
    save(fig, "benchmark_components.png")


def lns_greedy_charts():
    rows = read_csv(Path(__file__).resolve().parents[2] / "Nir_Irt" / "tables" / "lns_vs_greedy_summary.csv")
    labels = ["dataset_100", "dataset_200_2", "benchmark"]
    x = np.arange(len(rows))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.4, 5.8), constrained_layout=True)
    greedy = [float(row["greedy_final_score"]) for row in rows]
    lns = [float(row["lns_final_score"]) for row in rows]
    ax.bar(x - width / 2, greedy, width, label="Greedy", color="#4c78a8")
    ax.bar(x + width / 2, lns, width, label="LNS", color="#f58518")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Average final score")
    ax.set_ylim(0.62, 0.77)
    ax.set_title("Average final score: LNS vs Greedy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save(fig, "lns_vs_greedy_score.png")

    metrics = [
        ("volume_utilization", "Volume utilization"),
        ("item_coverage", "Item coverage"),
        ("fragility_score", "Fragility score"),
        ("time_score", "Time score"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), constrained_layout=True)
    for ax, (field, title) in zip(axes.flat, metrics):
        greedy = [float(row[f"greedy_{field}"]) for row in rows]
        lns = [float(row[f"lns_{field}"]) for row in rows]
        ax.bar(x - width / 2, greedy, width, label="Greedy", color="#4c78a8")
        ax.bar(x + width / 2, lns, width, label="LNS", color="#f58518")
        ax.set_xticks(x, labels)
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.set_ylim(0.45, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("Metric comparison on common datasets", fontsize=14)
    save(fig, "lns_vs_greedy_metrics.png")


if __name__ == "__main__":
    dataset_composition()
    benchmark_charts()
    lns_greedy_charts()
