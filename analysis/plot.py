from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from util import (
    DATA_BASE_PATH,
    DIM_MARKERS,
    DIMENSIONALITIES,
    CONFIG_PARAMS,
    load_and_preprocess,
    info,
    get_config_label,
    add_percentile_lines,
)

# Define globals
OUTPUT_DIR = Path("./output")
FIG_SIZE = (12, 8)
DIM_COLORS = {
    50: "#4C72B0",
    100: "#55A868",
    200: "#C44E52",
}
CONFIG_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
}
DEFAULT_PERCENTILES = [0.50, 0.75, 0.90, 0.95]

# Define plot styling
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
    }
)

plt.rcParams["font.size"] = 18

def plot_index_construction_times_by_config(run_data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for dim in DIMENSIONALITIES:
        dim_data = run_data[run_data["dim"] == dim]
        sns.scatterplot(
            data=dim_data,
            x="config_id",
            y="index_construction_time",
            marker=DIM_MARKERS[dim],
            s=100,
            label=f"dim={dim}",
            ax=ax,
        )

        # Trend Line
        x = dim_data["config_id"].values
        y = dim_data["index_construction_time"].values
        z = np.polyfit(x, np.log10(y), deg=1) # Log scale due to log axis
        p = np.poly1d(z)
        ax.plot(
            x,
            10**p(x),
            linestyle="--",
            color=DIM_COLORS[dim],
            alpha=0.7,
        )


    ax.set_yscale("log")
    ax.set_xlabel("Index Configuration")
    ax.set_ylabel("Index Construction Time (seconds, log scale)")
    ax.set_title("Index Construction Time by Configuration")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"Config {i}\n{CONFIG_PARAMS[i]['M']}/{CONFIG_PARAMS[i]['efConstruction']}"
                       for i in [1, 2, 3]])
    ax.legend(title="Dimensionality")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "index-time-trend-by-dim.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    info(f"Saved: {output_path}")


def plot_index_construction_times_by_dim(run_data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for config_id in range(1, 4):
        config_data = run_data[run_data["config_id"] == config_id]
        sns.scatterplot(
            data=config_data,
            x="dim",
            y="index_construction_time",
            s=100,
            label=f"Config: {get_config_label(config_id)}",
            ax=ax,
        )

        # Trend Line
        x = config_data["dim"].values
        y = config_data["index_construction_time"].values
        z = np.polyfit(x, np.log10(y), deg=1) # Log scale due to log axis
        p = np.poly1d(z)
        ax.plot(
            x,
            10**p(x),
            linestyle="--",
            color=CONFIG_COLORS[config_id],
            alpha=0.7,
        )


    ax.set_yscale("log")
    ax.set_xlabel("Dimensionality")
    ax.set_ylabel("Index Construction Time (seconds, log scale)")
    ax.set_title("Index Construction Time by Configuration")
    ax.set_xticks([50, 100, 200])
    ax.legend(title="Configuration")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "index-time-trend-by-dim.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    info(f"Saved: {output_path}")


def cost_benefit(run_data: pd.DataFrame):
    # Add an averaged point for each config/dim
    avg_times = run_data.groupby(["config_id", "dim"]).agg({
        "index_construction_time": "mean",
        "Recall": "mean",
        "config_label": "first",
    }).reset_index()

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for config_id in range(1, 4):
        for dim in DIMENSIONALITIES:
            runs = run_data[
                (run_data["config_id"] == config_id) &
                (run_data["dim"] == dim)
            ]

            ax.scatter(
                x=runs["index_construction_time"],
                y=runs["Recall"],
                marker=DIM_MARKERS[dim],
                color=CONFIG_COLORS[config_id],
                s=70,
                alpha=0.7,
            )
            # Plot average point
            avg_point = avg_times[
                (avg_times["config_id"] == config_id) &
                (avg_times["dim"] == dim)
            ]
            ax.scatter(
                x=avg_point["index_construction_time"],
                y=avg_point["Recall"],
                marker=DIM_MARKERS[dim],
                color=CONFIG_COLORS[config_id],
                s=200,
                edgecolor="black",
                label=f"Config {config_id}, dim {dim}",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Index Construction Time (seconds, log scale)")
    ax.set_ylabel("Recall")
    ax.set_title("Cost-Benefit Analysis of Index Configurations\n(Small markers: individual runs, Large markers: averages)")
    ax.grid(True, alpha=0.3, which="both")

    # Customize legend
    dim_handles = [
        plt.Line2D([0], [0], marker=DIM_MARKERS[dim], color="w", label=f"Dim {dim}",
                   markerfacecolor="gray", markersize=10)
        for dim in DIMENSIONALITIES
    ]
    config_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=f"Config {config_id}",
                   markerfacecolor=CONFIG_COLORS[config_id], markersize=10)
        for config_id in range(1, 4)
    ]
    first_legend = ax.legend(handles=dim_handles, title="Dimensionality", loc="lower right")
    ax.add_artist(first_legend)
    ax.legend(handles=config_handles, title="Configuration", loc="lower left")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "cost-benefit-analysis.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    info(f"Saved: {output_path}")


def plot_job_vs_session_recall_cdf(df: pd.DataFrame):
    runs = df["run_id"].unique()

    for run_id in runs:
        run_data = df[df["run_id"] == run_id]

        plt.figure(figsize=FIG_SIZE)
        session_data = run_data[run_data["is_session"]]
        simple_job_data = run_data[~run_data["is_session"]]

        ax = sns.ecdfplot(
                session_data["Recall"],
                label=f"Session Recall\n(n={len(session_data)})",
                stat="proportion",
                color="darkorange",
        )
        add_percentile_lines(ax, session_data["Recall"], DEFAULT_PERCENTILES, "darkorange")

        ax = sns.ecdfplot(
                simple_job_data["Recall"],
                label=f"Job Recall\n(n={len(simple_job_data)})",
                stat="proportion",
                color="steelblue",
        )
        add_percentile_lines(ax, session_data["Recall"], DEFAULT_PERCENTILES, "steelblue")

        plt.xlabel("Recall")
        plt.ylabel("Proportion")
        plt.title(f"Recall CDF for Run {run_id}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = OUTPUT_DIR / f"recall-cdf-run-{run_id}.pdf"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        info(f"Saved: {output_path}")


def multiplot_job_vs_session_recall_cdf(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 3, figsize=FIG_SIZE, sharex=True, sharey=True)

    for row, config_id in enumerate(range(1, 4)):
        for col, dim in enumerate(DIMENSIONALITIES):
            ax = axes[row, col]
            run_data = df[
                (df["config_id"] == config_id) &
                (df["dim"] == dim)
            ]

            session_data = run_data[run_data["is_session"]]
            simple_job_data = run_data[~run_data["is_session"]]

            sns.ecdfplot(
                data=session_data["Recall"],
                stat="proportion",
                color="darkorange",
                ax=ax,
                label="Session Recall",
            )
            add_percentile_lines(ax, session_data["Recall"], DEFAULT_PERCENTILES, "darkorange")

            sns.ecdfplot(
                data=simple_job_data["Recall"],
                stat="proportion",
                color="steelblue",
                ax=ax,
                label="Job Recall",
            )
            add_percentile_lines(ax, simple_job_data["Recall"], DEFAULT_PERCENTILES, "steelblue")

            ax.set_title(f"Config {config_id}, Dim {dim}")
            ax.grid(True, alpha=0.3)

    fig.suptitle("Recall CDF by Configuration and Dimensionality", y=1.02)
    plt.tight_layout()
    output_path = OUTPUT_DIR / "recall-cdf-multiplot.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    info(f"Saved: {output_path}")


def plot_latency_cdf(df: pd.DataFrame):
    runs = df["run_id"].unique()

    for run_id in runs:
        run_data = df[df["run_id"] == run_id]
        session_data = run_data[run_data["is_session"]]
        simple_job_data = run_data[~run_data["is_session"]]
        config_id = run_data["config_id"].iloc[0]

        plt.figure(figsize=FIG_SIZE)

        # Overall
        ax = sns.ecdfplot(
                run_data["latency_ms"],
                label=f"Overall Latency CDF\n(n={len(run_data)})",
                stat="proportion",
                color=CONFIG_COLORS[config_id],
        )
        add_percentile_lines(ax, run_data["latency_ms"], DEFAULT_PERCENTILES, CONFIG_COLORS[config_id])

        # Sessions
        ax = sns.ecdfplot(
                session_data["latency_ms"],
                label=f"Session Latency CDF\n(n={len(session_data)})",
                stat="proportion",
                color="darkorange",
        )
        add_percentile_lines(ax, session_data["latency_ms"], DEFAULT_PERCENTILES, "darkorange")

        # Simple Jobs
        ax = sns.ecdfplot(
                simple_job_data["latency_ms"],
                label=f"Job Latency CDF\n(n={len(simple_job_data)})",
                stat="proportion",
                color="steelblue",
        )
        add_percentile_lines(ax, simple_job_data["latency_ms"], DEFAULT_PERCENTILES, "steelblue")

        plt.xlabel("Latency (ms)")
        plt.ylabel("Proportion")
        plt.title(f"Latency CDF for Run {run_id}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = OUTPUT_DIR / f"latency-cdf-run-{run_id}.pdf"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        info(f"Saved: {output_path}")


def multiplot_latency_cdf(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 3, figsize=FIG_SIZE, sharex=True, sharey=True)

    for row, config_id in enumerate(range(1, 4)):
        for col, dim in enumerate(DIMENSIONALITIES):
            ax = axes[row, col]
            run_data = df[
                (df["config_id"] == config_id) &
                (df["dim"] == dim)
            ]
            session_data = run_data[run_data["is_session"]]
            simple_job_data = run_data[~run_data["is_session"]]

            # Overall
            sns.ecdfplot(
                data=run_data["latency_ms"],
                stat="proportion",
                color=CONFIG_COLORS[config_id],
                ax=ax,
                label="Overall Latency",
            )
            add_percentile_lines(ax, run_data["latency_ms"], DEFAULT_PERCENTILES, CONFIG_COLORS[config_id])

            # Sessions
            sns.ecdfplot(
                data=session_data["latency_ms"],
                label=f"Session Latency CDF\n(n={len(session_data)})",
                stat="proportion",
                color="darkorange",
                ax=ax,
            )
            add_percentile_lines(ax, session_data["latency_ms"], DEFAULT_PERCENTILES, "darkorange")

            # Simple Jobs
            sns.ecdfplot(
                data=simple_job_data["latency_ms"],
                label=f"Job Latency CDF\n(n={len(simple_job_data)})",
                stat="proportion",
                color="steelblue",
                ax=ax,
            )
            add_percentile_lines(ax, simple_job_data["latency_ms"], DEFAULT_PERCENTILES, "steelblue")

            ax.set_title(f"Config {config_id}, Dim {dim}")
            ax.grid(True, alpha=0.3)

    fig.suptitle("Latency CDF by Configuration and Dimensionality", y=1.02)
    plt.tight_layout()
    output_path = OUTPUT_DIR / "latency-cdf-multiplot.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    info(f"Saved: {output_path}")


def plot_recall_over_session(df: pd.DataFrame):
    runs = df["run_id"].unique()

    for run_id in runs:
        run_data = df[df["run_id"] == run_id]

        plt.figure(figsize=FIG_SIZE)
        session_data = run_data[run_data["is_session"]]

        # Aggregate to plot mean and p25/p75 bands
        p25 = session_data.groupby("step")["Recall"].quantile(0.25)
        p75 = session_data.groupby("step")["Recall"].quantile(0.75)
        mean_recall = session_data.groupby("step")["Recall"].mean()

        plt.fill_between(
            mean_recall.index,
            p25,
            p75,
            color=CONFIG_COLORS[run_data["config_id"].iloc[0]],
            alpha=0.3,
            label="25th-75th Percentile",
        )
        sns.lineplot(
            x=mean_recall.index,
            y=mean_recall.values,
            color=CONFIG_COLORS[run_data["config_id"].iloc[0]],
            label="Mean Recall",
        )

        plt.xlabel("Session Number")
        plt.ylabel("Recall")
        plt.title(f"Recall Over Sessions for Run {run_id}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = OUTPUT_DIR / f"recall-over-sessions-run-{run_id}.pdf"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        info(f"Saved: {output_path}")


def multainplot_recall_over_session(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 3, figsize=FIG_SIZE, sharex=True, sharey=True)

    for row, config_id in enumerate(range(1, 4)):
        for col, dim in enumerate(DIMENSIONALITIES):
            ax = axes[row, col]
            run_data = df[
                (df["config_id"] == config_id) &
                (df["dim"] == dim) &
                (df["is_session"])
            ]

            # Aggregate to plot mean and p25/p75 bands
            p25 = run_data.groupby("step")["Recall"].quantile(0.25)
            p75 = run_data.groupby("step")["Recall"].quantile(0.75)
            mean_recall = run_data.groupby("step")["Recall"].mean()

            ax.fill_between(
                mean_recall.index,
                p25,
                p75,
                color=CONFIG_COLORS[config_id],
                alpha=0.3,
                label="25th-75th Percentile",
            )
            sns.lineplot(
                x=mean_recall.index,
                y=mean_recall.values,
                color=CONFIG_COLORS[config_id],
                label="Mean Recall",
                ax=ax,
            )

            ax.set_title(f"Config {config_id}, Dim {dim}")
            ax.grid(True, alpha=0.3)

    fig.suptitle("Recall Over Sessions by Configuration and Dimensionality", y=1.02)
    plt.tight_layout()
    output_path = OUTPUT_DIR / "recall-over-sessions-multiplot.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    info(f"Saved: {output_path}")


def plot_latency_vs_recall_hexbin(df: pd.DataFrame):
    runs = df["run_id"].unique()

    for run_id in runs:
        run_data = df[df["run_id"] == run_id]

        plt.figure(figsize=FIG_SIZE)

        hb = plt.hexbin(
            run_data["latency_ms"],
            run_data["Recall"],
            gridsize=30,
            cmap="Blues",
            mincnt=1,
        )
        plt.colorbar(hb, label="Count")

        add_percentile_lines(plt.gca(), run_data["latency_ms"], DEFAULT_PERCENTILES, "red")

        plt.xlabel("Latency (ms)")
        plt.ylabel("Recall")
        plt.title(f"Latency vs Recall Hexbin for Run {run_id}")
        plt.grid(True, alpha=0.3)

        output_path = OUTPUT_DIR / f"latency-vs-recall-hexbin-run-{run_id}.pdf"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        info(f"Saved: {output_path}")


def main():
    df = load_and_preprocess(DATA_BASE_PATH)
    # Group by run to get individual run data points
    run_data = df.groupby(["config_id", "dim", "run_number"]).agg({
        "index_construction_time": "first",
        "config_label": "first",
        "Recall": "mean",
    }).reset_index()

    plot_index_construction_times_by_config(run_data)
    plot_index_construction_times_by_dim(run_data)
    cost_benefit(run_data)

    plot_job_vs_session_recall_cdf(df)
    multiplot_job_vs_session_recall_cdf(df)
    plot_latency_cdf(df)
    multiplot_latency_cdf(df)

if __name__ == "__main__":
    main()
