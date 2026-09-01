#!/usr/bin/env python3
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Root directory containing sweep subfolders (e.g., BILSTM_L1, LSTM_L2)
SWEEPS_ROOT = "../../data/lucas/univariate-sweep"

# Regex to match both bilstm and lstm folders and capture hyperparams (layers optional)
pattern = re.compile(
    r"(?P<model>lstm)_seq(?P<seq_len>\d+)_bs(?P<batch_size>\d+)_lr(?P<lr>[\d.eE-]+)_hs(?P<hidden_size>\d+)(?:_layers(?P<num_layers>\d+))?"
)

for sweep_dir in os.listdir(SWEEPS_ROOT):
    base_dir = os.path.join(SWEEPS_ROOT, sweep_dir)
    if not os.path.isdir(base_dir):
        continue

    records = []
    html_file = os.path.join(base_dir, "correlation_heatmap.html")

    # Iterate over runs
    for run_dir in os.listdir(base_dir):
        m = pattern.match(run_dir)
        if not m:
            continue
        params = m.groupdict()
        # cast
        params["seq_len"] = int(params["seq_len"])
        params["batch_size"] = int(params["batch_size"])
        lr_str = params["lr"]
        if lr_str.startswith("."):
            lr_str = "0" + lr_str
        params["lr"] = float(lr_str)
        params["hidden_size"] = int(params["hidden_size"])
        params["num_layers"] = int(params.get("num_layers") or 1)

        # load metrics
        summary_path = os.path.join(base_dir, run_dir, "horizon_summary.csv")
        if os.path.isfile(summary_path):
            df = pd.read_csv(summary_path)
            test_df = df[df["split"] == "test"]
            if not test_df.empty:
                records.append(
                    {
                        **params,
                        "avg_r2": test_df["r2"].mean(),
                        "avg_mae": test_df["mae"].mean(),
                        "avg_rmse": test_df["rmse"].mean(),
                        "avg_mse": test_df["mse"].mean(),
                    }
                )

    # assemble DataFrame
    if not records:
        print(f"No runs found for {sweep_dir}, skipping correlation analysis.")
        continue
    df_all = pd.DataFrame(records)
    print(
        f"\nSweep {sweep_dir}: Collected {len(df_all)} runs for correlation analysis."
    )

    # compute correlation matrix on numeric columns that vary
    df_numeric = df_all.select_dtypes(include=[np.number]).loc[
        :, lambda df: df.nunique() > 1
    ]
    corr_matrix = df_numeric.corr()
    print(f"Correlation matrix for {sweep_dir}:\n", corr_matrix)

    # plot heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="r"),
        )
    )
    fig.update_layout(
        title=f"Correlation Heatmap for {sweep_dir}",
        width=700,
        height=700,
        template="plotly_white",
    )
    fig.write_html(html_file, auto_open=False)
    print(f"Heatmap saved to {html_file}")
