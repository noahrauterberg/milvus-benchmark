import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Define Constants

DATA_BASE_PATH = "../data"
PARQUET_FILE = "enhanced-results.parquet"
PREP_LOG_FILE = "prepare-log.txt"
OUTPUT_DIR = "./output"

DIMENSIONALITIES = [50, 100, 200]
INDEX_CONFIGURATIONS = [1, 2, 3]

# Index configuration parameters (M, efConstruction)
CONFIG_PARAMS = {
    1: {"M": 15, "efConstruction": 180},
    2: {"M": 30, "efConstruction": 360},
    3: {"M": 60, "efConstruction": 720},
}

# Color palette for configurations
CONFIG_COLORS = {
    1: "#1f77b4",  # blue
    2: "#ff7f0e",  # orange
    3: "#2ca02c",  # green
}

# Default output directory
OUTPUT_DIR = "./output"

LOG_LEVEL = 1  # 0: debug, 1: info, 2: error

# Markers for dimensionalities
DIM_MARKERS = {
    50: "o",
    100: "s",
    200: "^",
}

# Logging Helpers

def debug(msg):
    if LOG_LEVEL <= 0:
        print(f"[DEBUG] {msg}")


def info(msg):
    if LOG_LEVEL <= 1:
        print(f"[INFO] {msg}")


def error(msg):
    if LOG_LEVEL <= 2:
        print(f"[ERROR] {msg}")


def parse_time_sec(raw_time: str) -> float:
    """
    Parses a Go duration string (e.g., "1h20m44.700351043s") to seconds.
    """
    total = 0.0
    matches = re.findall(r"(\d+\.?\d*)([hms])", raw_time)
    for value, unit in matches:
        value = float(value)
        if unit == "h":
            total += value * 3600
        elif unit == "m":
            total += value * 60
        elif unit == "s":
            total += value
        else:
            error(f"Unrecognized unit: {unit} in time string: {raw_time}")

    debug(f"Parsed time string '{raw_time}' to {total} seconds")
    return total


def format_time(seconds: float) -> str:
    """
    Parses seconds back into appropriate time unit string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_config_label(config_id: int) -> str:
    """
    Create a string label for the index configuration.
    """
    params = CONFIG_PARAMS.get(config_id, {})
    if params:
        return f"M={params['M']}, ef={params['efConstruction']}"
    return f"Config {config_id}"


def load_run(run_path: str, config_id: int, dim: int, run_number: str) -> pd.DataFrame | None:
    """
    Load a single benchmark run from the specified directory.
    """
    parquet_path = os.path.join(run_path, PARQUET_FILE)
    prep_log_path = os.path.join(run_path, PREP_LOG_FILE) # For index construction time

    if not os.path.exists(parquet_path):
        debug(f"Parquet file not found: {parquet_path}")
        return None

    df = pd.read_parquet(parquet_path)

    # Extract index construction time from prepare-log.txt
    index_construction_time = None
    if os.path.exists(prep_log_path):
        with open(prep_log_path, "r") as file:
            for line in file:
                if "Index constructed in" in line:
                    parts = line.split("Index constructed in")
                    if len(parts) > 1:
                        index_construction_time = parse_time_sec(parts[1].strip())
                        break

    if index_construction_time is None:
        error(f"Could not extract index construction time from {prep_log_path}")
        index_construction_time = np.nan

    # Add metadata columns
    df["dim"] = dim
    df["config_id"] = config_id
    df["run_number"] = run_number
    df["index_construction_time"] = index_construction_time

    # Add config parameters
    params = CONFIG_PARAMS.get(config_id, {})
    df["M"] = params.get("M", np.nan)
    df["efConstruction"] = params.get("efConstruction", np.nan)
    df["config_label"] = get_config_label(config_id)

    # Create unique run identifier
    df["run_id"] = f"config{config_id}-dim{dim}-{run_number}"

    info(f"Loaded {len(df)} records from {run_path}")
    return df


def load_all_runs(base_path: str = DATA_BASE_PATH) -> list[pd.DataFrame]:
    """
    Load all benchmark runs from the data directory.
    Directory structure expected:
        {base_path}/config{N}-dim{D}/run{M}/output-config{N}-dim{D}/
    """
    debug(f"Loading runs from {base_path}")
    runs = []

    for dim in DIMENSIONALITIES:
        for config_id in INDEX_CONFIGURATIONS:
            config_dim_path = os.path.join(base_path, f"config{config_id}-dim{dim}")

            if not os.path.exists(config_dim_path):
                debug(f"Directory not found: {config_dim_path}")
                continue

            # Find all run subdirectories
            subdirs = [f for f in os.scandir(config_dim_path) if f.is_dir()]
            for subdir in subdirs:
                run_name = os.path.basename(subdir.path)
                output_dir = os.path.join(
                    subdir.path, f"output-config{config_id}-dim{dim}"
                )

                if not os.path.exists(output_dir):
                    debug(f"Output directory not found: {output_dir}")
                    continue

                df = load_run(output_dir, config_id, dim, run_name)
                if df is not None:
                    runs.append(df)

    info(f"Loaded {len(runs)} runs total")
    return runs


def combine_runs(runs: list[pd.DataFrame]) -> pd.DataFrame:
    if not runs:
        raise ValueError("No runs to combine")
    return pd.concat(runs, ignore_index=True)


# Preprocessing Functions
def parse_job_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse Job IDs to extract session information.

    Job ID formats:
        - "J-{index}" for independent jobs
        - "S-{sessionId}-{step}" for session queries

    Adds columns:
        - is_session: Boolean indicating if query is part of a session
        - session_id: Session ID (NaN for independent jobs)
        - step: Step within session (NaN for independent jobs)
    """
    df = df.copy()

    def parse_id(job_id):
        if isinstance(job_id, str) and job_id.startswith("S-"):
            parts = job_id.split("-")
            if len(parts) == 3:
                try:
                    return True, int(parts[1]), int(parts[2])
                except ValueError:
                    pass
        return False, None, None

    parsed = df["Id"].apply(parse_id)
    df["is_session"] = parsed.apply(lambda x: x[0])
    df["session_id"] = parsed.apply(lambda x: x[1])
    df["step"] = parsed.apply(lambda x: x[2])

    return df


def convert_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert time units from microseconds to milliseconds.
    """
    df = df.copy()

    if "Latency" in df.columns:
        df["latency_ms"] = df["Latency"] / 1_000_000

    if "SchedulingDelay" in df.columns:
        df["scheduling_delay_ms"] = df["SchedulingDelay"] / 1_000_000

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_job_id(df)
    df = convert_units(df)
    return df


def load_and_preprocess(base_path: str = DATA_BASE_PATH) -> pd.DataFrame:
    runs = load_all_runs(base_path)
    if not runs:
        raise ValueError(f"No valid runs found in {base_path}")

    df = combine_runs(runs)
    df = preprocess(df)
    return df


# Generate summary statistics aggregated across all runs per config/dim or per individual run.
def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    if "latency_ms" not in df.columns:
        df = convert_units(df)

    def agg_stats(group):
        return pd.Series({
            "M": group["M"].iloc[0],
            "efConstruction": group["efConstruction"].iloc[0],
            "config_label": group["config_label"].iloc[0],
            "n_queries": len(group),
            "n_sessions": group["is_session"].sum() if "is_session" in group.columns else 0,
            "index_construction_time_s": group["index_construction_time"].iloc[0],
            "latency_median_ms": group["latency_ms"].median(),
            "latency_p50_ms": group["latency_ms"].quantile(0.50),
            "latency_p90_ms": group["latency_ms"].quantile(0.90),
            "latency_p95_ms": group["latency_ms"].quantile(0.95),
            "latency_p99_ms": group["latency_ms"].quantile(0.99),
            "recall_median": group["Recall"].median(),
            "recall_std": group["Recall"].std(),
            "recall_min": group["Recall"].min(),
            "recall_max": group["Recall"].max(),
            "recall_p50": group["Recall"].quantile(0.50),
            "recall_p90": group["Recall"].quantile(0.90),
            "recall_p95": group["Recall"].quantile(0.95),
            "recall_p99": group["Recall"].quantile(0.99),
        })

    summary = df.groupby(["config_id", "dim", "run_number"]).apply(
        agg_stats, include_groups=False
    )
    return summary.reset_index()


def aggregated_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "latency_ms" not in df.columns:
        df = convert_units(df)

    def agg_stats(group):
        return pd.Series({
            "M": group["M"].iloc[0],
            "efConstruction": group["efConstruction"].iloc[0],
            "config_label": group["config_label"].iloc[0],
            "n_runs": group["run_number"].nunique(),
            "n_queries": len(group),
            "index_construction_time_mean_s": group["index_construction_time"].mean(),
            "index_construction_time_std_s": group["index_construction_time"].std(),
            "latency_mean_ms": group["latency_ms"].mean(),
            "latency_p50_ms": group["latency_ms"].quantile(0.50),
            "latency_p90_ms": group["latency_ms"].quantile(0.90),
            "latency_p95_ms": group["latency_ms"].quantile(0.95),
            "latency_p99_ms": group["latency_ms"].quantile(0.99),
            "recall_mean": group["Recall"].mean(),
            "recall_std": group["Recall"].std(),
            "recall_p50": group["Recall"].quantile(0.50),
            "recall_p90": group["Recall"].quantile(0.90),
            "recall_p95": group["Recall"].quantile(0.95),
            "recall_p99": group["Recall"].quantile(0.99),
        })

    summary = df.groupby(["config_id", "dim"]).apply(agg_stats, include_groups=False)
    return summary.reset_index()


def add_percentile_lines(ax, series: pd.Series, percentiles: list[int], color: str = "gray"):
    quantiles = series.quantile(percentiles)
    for idx, q in enumerate(quantiles):
        ax.axvline(x=q, color=color, linestyle="--", alpha=0.5)
        ax.text(
                q,
                percentiles[idx],
                f"{int(percentiles[idx]*100)}%\n({q:.{2}f})",
                horizontalalignment="center",
                color=color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
        )


def explore_sessions(df: pd.DataFrame, 
                     latency_threshold_pct: float = 0.1,
                     recall_threshold: float = 0.02) -> bool:
    """
    Determine if session-specific analysis reveals interesting patterns.
    
    Args:
        df: DataFrame with is_session column
        latency_threshold_pct: Minimum relative latency difference to be interesting
        recall_threshold: Minimum absolute recall difference to be interesting
    
    Returns:
        True if sessions show meaningfully different behavior from single jobs
    """
    if 'is_session' not in df.columns:
        return False
    
    sessions = df[df['is_session']]
    jobs = df[~df['is_session']]
    
    if len(sessions) == 0 or len(jobs) == 0:
        return False
    
    # Ensure units are converted
    latency_col = 'latency_ms' if 'latency_ms' in df.columns else 'Latency'
    
    # Test 1: Latency difference
    session_latency = sessions[latency_col].median()
    job_latency = jobs[latency_col].median()
    latency_diff = abs(session_latency - job_latency) / job_latency
    
    # Test 2: Recall difference
    recall_diff = abs(sessions['Recall'].mean() - jobs['Recall'].mean())
    
    is_interesting = latency_diff > latency_threshold_pct or recall_diff > recall_threshold
    
    if is_interesting:
        print(f"\nSession analysis is interesting:")
        print(f"  Latency diff: {latency_diff:.1%} (threshold: {latency_threshold_pct:.1%})")
        print(f"  Recall diff: {recall_diff:.3f} (threshold: {recall_threshold:.3f})")
    else:
        print(f"\nSession behavior similar to single jobs:")
        print(f"  Latency diff: {latency_diff:.1%}")
        print(f"  Recall diff: {recall_diff:.3f}")
    
    return is_interesting


def pareto_frontier_mask(points: np.ndarray, minimize_x: bool = True, maximize_y: bool = True) -> np.ndarray:
    """
    Find the Pareto frontier for a set of 2D points.

    Args:
        points: Nx2 array of (x, y) points
        minimize_x: Whether to minimize x (True) or maximize x (False)
        maximize_y: Whether to maximize y (True) or minimize y (False)

    Returns:
        Boolean mask indicating Pareto-optimal points
    """
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # Check if j dominates i
            x_better = (points[j, 0] <= points[i, 0]) if minimize_x else (points[j, 0] >= points[i, 0])
            y_better = (points[j, 1] >= points[i, 1]) if maximize_y else (points[j, 1] <= points[i, 1])

            x_strict = (points[j, 0] < points[i, 0]) if minimize_x else (points[j, 0] > points[i, 0])
            y_strict = (points[j, 1] > points[i, 1]) if maximize_y else (points[j, 1] < points[i, 1])

            if x_better and y_better and (x_strict or y_strict):
                is_pareto[i] = False
                break

    return is_pareto
