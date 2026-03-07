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
            color=DIM_COLORS[dim],
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
    output_path = OUTPUT_DIR / "index-time-trend-by-config.pdf"
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
    ax.set_title("Index Construction Time by Dimensionality")
    ax.set_xticks([50, 100, 200])
    ax.legend(title="Configuration")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "index-time-trend-by-dim.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    info(f"Saved: {output_path}")


def cost_benefit(run_data: pd.DataFrame, recall_aggregation: str = "mean"):
    aggregation_func = None
    match recall_aggregation:
        case "mean": aggregation_func = lambda x: x.mean()
        case "median": aggregation_func = lambda x: x.median()
        case "p1": aggregation_func = lambda x: x.quantile(0.01)
        case "p5": aggregation_func = lambda x: x.quantile(0.05)
        case "p10": aggregation_func = lambda x: x.quantile(0.10)
        case "p50": aggregation_func = lambda x: x.quantile(0.50)
        case "p95": aggregation_func = lambda x: x.quantile(0.95)
        case "p97": aggregation_func = lambda x: x.quantile(0.97)
        case "p99": aggregation_func = lambda x: x.quantile(0.99)
        case _: aggregation_func = lambda x: x.mean()

    # Add an averaged point for each config/dim
    avg_times = run_data.groupby(["config_id", "dim"]).agg({
        "index_construction_time": "mean",
        "Recall": aggregation_func,
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
    ax.set_ylabel(f"Recall ({recall_aggregation})")
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
    output_path = OUTPUT_DIR / f"cost-benefit-analysis-{recall_aggregation}.pdf"
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
                color="darkviolet",
        )
        add_percentile_lines(ax, session_data["Recall"], DEFAULT_PERCENTILES, "darkviolet")

        ax = sns.ecdfplot(
                simple_job_data["Recall"],
                label=f"Job Recall\n(n={len(simple_job_data)})",
                stat="proportion",
                color="mediumturquoise",
        )
        add_percentile_lines(ax, session_data["Recall"], DEFAULT_PERCENTILES, "mediumturquoise")

        plt.xlabel("Recall")
        plt.ylabel("Proportion")
        plt.title(f"Recall CDF for Run {run_id}")
        plt.legend(loc="lower right")
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
                color="darkviolet",
                ax=ax,
                label="Session Recall",
            )

            sns.ecdfplot(
                data=simple_job_data["Recall"],
                stat="proportion",
                color="mediumturquoise",
                ax=ax,
                label="Job Recall",
            )

            ax.set_title(f"Config {config_id}, Dim {dim}")
            ax.grid(True, alpha=0.3)

    # Add a single legend for the entire figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.99))

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

        # Sessions
        ax = sns.ecdfplot(
                session_data["latency_ms"],
                label=f"Session Latency CDF\n(n={len(session_data)})",
                stat="proportion",
                color="darkviolet",
        )

        # Simple Jobs
        ax = sns.ecdfplot(
                simple_job_data["latency_ms"],
                label=f"Job Latency CDF\n(n={len(simple_job_data)})",
                stat="proportion",
                color="mediumturquoise",
        )

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

            # Sessions
            sns.ecdfplot(
                data=session_data["latency_ms"],
                label=f"Session Latency CDF\n(n={len(session_data)})",
                stat="proportion",
                color="darkviolet",
                ax=ax,
            )

            # Simple Jobs
            sns.ecdfplot(
                data=simple_job_data["latency_ms"],
                label=f"Job Latency CDF\n(n={len(simple_job_data)})",
                stat="proportion",
                color="mediumturquoise",
                ax=ax,
            )

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


def plot_session_recall_evolution_per_run(df: pd.DataFrame, output_dir: Path):
    # Plot recall evolution over the course of sessions.
    runs = df["run_id"].unique()
    
    for run_id in sorted(runs):
        run_data = df[df["run_id"] == run_id]
        config_id = run_data["config_id"].iloc[0]
        dim = run_data["dim"].iloc[0]
        run_num = run_data["run_number"].iloc[0]
        
        # Filter for session queries
        session_data = run_data[run_data["is_session"]].copy()
        
        if len(session_data) == 0:
            info(f"No session data for {run_id}, skipping recall evolution plot")
            continue
        
        # Get unique sessions
        sessions = session_data["session_id"].unique()
        n_sessions = len(sessions)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        step_recalls = {}
        
        for session_id in sessions:
            sess = session_data[session_data["session_id"] == session_id].sort_values("step")
            steps = sess["step"].values
            recalls = sess["Recall"].values
            
            # Collect for aggregation
            for step, recall in zip(steps, recalls):
                if step not in step_recalls:
                    step_recalls[step] = []
                step_recalls[step].append(recall)
        
        steps_sorted = sorted(step_recalls.keys())
        mean_recalls = [np.mean(step_recalls[s]) for s in steps_sorted]
        p25_recalls = [np.percentile(step_recalls[s], 25) for s in steps_sorted]
        p75_recalls = [np.percentile(step_recalls[s], 75) for s in steps_sorted]
        
        # Confidence band (25th-75th percentile)
        ax.fill_between(steps_sorted, p25_recalls, p75_recalls, 
                       color=CONFIG_COLORS[config_id], alpha=0.3, label="P25-P75 range")
        
        # Mean line
        ax.plot(steps_sorted, mean_recalls, color=CONFIG_COLORS[config_id], 
               linewidth=2.5, label="Mean recall", marker="o", markersize=4)
        
        ax.set_xlabel("Step within Session")
        ax.set_ylabel("Recall")
        ax.set_title(f"Recall Evolution Over Sessions\n"
                    f"Config {config_id} dim={dim}, {run_num}\n"
                    f"{n_sessions:,} sessions, {len(session_data):,} queries")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        # y-axis  padding
        y_min = min(min(p25_recalls), 0) - 0.05
        y_max = min(max(p75_recalls) + 0.1, 1.05)
        ax.set_ylim(y_min, y_max)
        
        # Add summary stats
        overall_mean = session_data["Recall"].mean()
        overall_std = session_data["Recall"].std()
        ax.text(0.02, 0.02, f"Overall: mean={overall_mean:.3f}, std={overall_std:.3f}",
               transform=ax.transAxes, fontsize=9, va="bottom",
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
        
        plt.tight_layout()
        output_path = output_dir / f"session_recall_evolution_{run_id}.pdf"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
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
    cost_benefit(run_data, recall_aggregation="mean")
    cost_benefit(run_data, recall_aggregation="median")
    cost_benefit(run_data, recall_aggregation="p1")
    cost_benefit(run_data, recall_aggregation="p5")
    cost_benefit(run_data, recall_aggregation="p10")
    cost_benefit(run_data, recall_aggregation="p50")
    cost_benefit(run_data, recall_aggregation="p95")
    cost_benefit(run_data, recall_aggregation="p97")
    cost_benefit(run_data, recall_aggregation="p99")
    plot_job_vs_session_recall_cdf(df)
    multiplot_job_vs_session_recall_cdf(df)
    plot_latency_cdf(df)
    multiplot_latency_cdf(df)
    plot_session_recall_evolution_per_run(df, OUTPUT_DIR)

if __name__ == "__main__":
    main()
