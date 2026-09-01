#!/usr/bin/env python3
import glob
import os

import pandas as pd
import plotly.graph_objects as go

# === Configuration ===
SWEEP_ROOT = "../../data/lucas/univariate-sweep"
BEST_PATTERN = os.path.join(SWEEP_ROOT, "*", "best_configurations.csv")

# Output filenames
SUMMARY_L1 = os.path.join(SWEEP_ROOT, "summary_L1.csv")
SUMMARY_L2 = os.path.join(SWEEP_ROOT, "summary_L2.csv")
HEATMAP_L1 = os.path.join(SWEEP_ROOT, "correlation_L1.html")
HEATMAP_L2 = os.path.join(SWEEP_ROOT, "correlation_L2.html")
COMBINED_TXT = os.path.join(SWEEP_ROOT, "combined_insights.txt")

# Columns for correlation
HYPERPARAMS = ["seq_len", "batch_size", "lr", "hidden_size"]
METRICS = [
    "avg_r2",
    "min_r2",
    "max_r2",
    "avg_mse",
    "min_mse",
    "max_mse",
    "avg_mae",
    "min_mae",
    "max_mae",
    "max_ae",
]
CORR_COLUMNS = HYPERPARAMS + METRICS

# 1) Gather all best configurations
records = []
for path in glob.glob(BEST_PATTERN):
    sweep = os.path.basename(os.path.dirname(path))
    df = pd.read_csv(path)
    if df.empty:
        continue
    df["sweep"] = sweep
    df["model_base"] = df["model"].str.split("_").str[0].str.lower()
    if "num_layers" not in df.columns:
        df["num_layers"] = 1
    records.append(df)
all_df = pd.concat(records, ignore_index=True)

# 2) Write per-layer summary tables
for nl, summary_file in [(1, SUMMARY_L1), (2, SUMMARY_L2)]:
    df_nl = all_df[all_df["num_layers"] == nl]
    if df_nl.empty:
        continue
    summary = (
        df_nl.groupby("model_base")
        .agg(
            avg_r2=("avg_r2", "mean"),
            min_r2=("min_r2", "mean"),
            max_r2=("max_r2", "mean"),
            avg_mse=("avg_mse", "mean"),
            min_mse=("min_mse", "mean"),
            max_mse=("max_mse", "mean"),
            avg_mae=("avg_mae", "mean"),
            min_mae=("min_mae", "mean"),
            max_mae=("max_mae", "mean"),
            max_ae=("max_ae", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_file, index=False)
    print(f"Saved summary L{nl} to {summary_file}")


# 3) Save interactive heatmaps
def save_heatmap(df_nl, path, title):
    cols = [c for c in CORR_COLUMNS if c in df_nl.columns]
    # remove constant columns to avoid NaNs
    cols = [c for c in cols if df_nl[c].nunique() > 1]
    corr_df = df_nl[cols].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.index,
            colorscale="RdBu",
            reversescale=True,
            zmid=0,
        )
    )
    fig.update_layout(
        title=title, width=700, height=600, coloraxis_colorbar=dict(title="r")
    )
    fig.write_html(path)
    print(f"Saved heatmap to {path}")


for nl, html_path in [(1, HEATMAP_L1), (2, HEATMAP_L2)]:
    df_nl = all_df[all_df["num_layers"] == nl]
    if df_nl.empty:
        continue
    save_heatmap(df_nl, html_path, f"Correlation Heatmap for L{nl} Models")

# 4) Generate combined insights
lines = ["=== Combined Summaries & Insights ==="]
# References
for nl, fpath in [(1, SUMMARY_L1), (2, SUMMARY_L2)]:
    if os.path.isfile(fpath):
        lines.append(f"Summary table for L{nl} models: {fpath}")
for nl, fpath in [(1, HEATMAP_L1), (2, HEATMAP_L2)]:
    if os.path.isfile(fpath):
        lines.append(f"Interactive heatmap for L{nl} models: {fpath}")
lines.append("")


# Correlation classification
def classify(r):
    if abs(r) < 0.1:
        return "no"
    if abs(r) < 0.4:
        return "minimal"
    return "major"


# Interpret correlations for each hyperparameter and metric
for nl in [1, 2]:
    df_nl = all_df[all_df["num_layers"] == nl]
    if df_nl.empty:
        continue
    corr_df = df_nl[CORR_COLUMNS].corr()
    lines.append(f"-- Insights for L{nl} Models --")
    # filter hyperparameters with variance
    hps = [hp for hp in HYPERPARAMS if df_nl[hp].nunique() > 1]
    for hp in hps:
        for met in METRICS:
            if met not in corr_df.columns:
                continue
            r = corr_df.at[hp, met]
            strength = classify(r)
            if strength == "no":
                lines.append(
                    f"Parameter '{hp}' shows no effect on '{met}' (r={r:.2f})."
                )
            elif strength == "minimal":
                lines.append(
                    f"Parameter '{hp}' has minimal effect on '{met}' (r={r:.2f})."
                )
            else:
                sign = "increase" if r > 0 else "decrease"
                lines.append(
                    f"Increasing '{hp}' has a major {sign} effect on '{met}' (r={r:.2f})."
                )
    lines.append("")

# 5) Write combined insights file
with open(COMBINED_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"✅ Combined insights saved to {COMBINED_TXT}")
