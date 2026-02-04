# Analysis

This directory contains python scripts to facilitate the analysis of benchmarking data generated during experiments.
The `plot.py` module provides a collection of data visualization utilities tailored to perform performance analysis of benchmarking over varying configurations, dimensions, and workloads.
The `util.py` on the other hand is a helper module that serves as some functions for data preprocessing logging, and defines some constants.

## Requirements

The analysis script assumes that the benchmark data is stored in the `../data/` directory relative to the execution path of this script.
Further, the script assumes that within the `data/` directory, each configuration, dimeanionality pair has its own subdirectory following the naming convention `config{config_id}-dim{dimensionality}/`.
Each of those subdirectories is in turn expected to follow the following structure:
```
../data/config<config-id>-dim<dimensionality>
├── run1
│   ├── benchmark-metrics.csv
│   ├── milvus-metrics.csv
│   └── output-<config-id>-dim<dimensionality>
│       ├── benchmark-jobs.csv
│       ├── benchmark-log-session.csv
│       ├── benchmark-log.txt
│       ├── Cleanup-log.txt
│       ├── data-rows.gob
│       ├── enhanced-results.parquet
│       ├── main-log.txt
│       ├── prepare-log.txt
│       └── warmup-log.txt
├── run2
│   ├── milvus-metrics.csv
│   └── output-<config-id>-dim<dimensionality>
│       ├── benchmark-jobs.csv
│       ├── benchmark-log-session.csv
│       ├── benchmark-log.txt
│       ├── Cleanup-log.txt
│       ├── data-rows.gob
│       ├── enhanced-results.parquet
│       ├── main-log.txt
│       ├── prepare-log.txt
│       └── warmup-log.txt
└── run3
    ├── benchmark-metrics.csv
    ├── milvus-metrics.csv
    └── output-<config-id>-dim<dimensionality>
        ├── benchmark-jobs.csv
        ├── benchmark-log-session.csv
        ├── benchmark-log.txt
        ├── Cleanup-log.txt
        ├── data-rows.gob
        ├── enhanced-results.parquet
        ├── main-log.txt
        ├── prepare-log.txt
        └── warmup-log.txt
```

## Key Functions

- `plot_index_construction_times_by_config(run_data: pd.DataFrame)`
  Plots the relationship between index construction time (log scale) and configuration ID by dimensionality.  
  - Generates scatter plots with markers representing dimensions.
  - Overlays trend lines for visualizing the log-scale relationship.
  - Saves the output to `output/index-time-trend-by-dim.pdf`.

- `plot_index_construction_times_by_dim(run_data: pd.DataFrame)`
  Analyzes index construction time across dimensionalities while iterating configurations.  
  - Provides insight into dimension's impact on construction time.
  - Saves the plot to `output/index-time-trend-by-dim.pdf`.

- `cost_benefit(run_data: pd.DataFrame)`
  Produces a cost-benefit chart showcasing recall performance against index construction time for each configuration and dimension.  
  - Includes markers for individual runs and aggregate averages.
  - Outputs file to `output/cost-benefit-analysis.pdf`.

- `plot_job_vs_session_recall_cdf(df: pd.DataFrame)`
  Compares recall distributions (CDF) for jobs vs sessions across unique runs.  
  - Utilizes ECDF plots to quantify the proportions of recall distribution.
  - Saves individual plots for each run.

- `multiplot_job_vs_session_recall_cdf(df: pd.DataFrame)`
  Aggregates recall distributions (CDF) for jobs and sessions across configurations and dimensions.  
  - Saves multi-subplot PDF showcasing configurations and dimensionalities in one file.

- `plot_latency_cdf(df: pd.DataFrame)`
  Generates CDF plots for latency distributions across sessions, jobs, and entire runs.  
  - Highlights performance variations within specific subcategories.

- `multiplot_latency_cdf(df: pd.DataFrame)`
  Same as `plot_latency_cdf`, but aggregates results across varying dimensions and configurations, outputting data as a grid chart.

- `plot_recall_over_session(df: pd.DataFrame)`
  Visualizes the stability of recall over multiple sessions, showing mean trends and p25/p75 bands.  
  - Outputs in `recall-over-sessions-run-{run_id}.pdf`.

- `multainplot_recall_over_session(df: pd.DataFrame)`
  Analyzes session-based recall stability across configurations and dimensions in a consolidated multiplot file.  

- `plot_latency_vs_recall_hexbin(df: pd.DataFrame)`
  Combines latency and recall data to produce hexbin plots for visualizing high-granularity relationships between the two metrics.  
  - Saved as `latency-vs-recall-hexbin-run-{run_id}.pdf`.

