#!/usr/bin/env python3
import os

import numpy as np
import pandas as pd
import plotly.express as px

HYPERPARAMS = ["seq_len", "batch_size", "lr", "hidden_size", "num_layers"]
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

SWEEP_ROOT = "../../data/lucas/univariate-sweep"
PLOTS_DIR = os.path.join(SWEEP_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_bar_violin_strip_plots(df, sweep_name, nl):
    for metric in METRICS:
        if metric not in df.columns:
            continue
        for hparam in HYPERPARAMS:
            if hparam not in df.columns or df[hparam].nunique() < 2:
                continue

            # MSE plots (original behavior)
            bar_data = df.groupby(hparam)[metric].mean().reset_index()
            fig_bar = px.bar(
                bar_data,
                x=hparam,
                y=metric,
                title=f"{sweep_name} L{nl}: Mean {metric} vs {hparam}",
            )
            fig_bar.write_html(
                os.path.join(PLOTS_DIR, f"L{nl}_{metric}_bar_{hparam}.html")
            )

            fig_violin = px.violin(
                df,
                y=metric,
                x=hparam,
                box=True,
                points="all",
                title=f"{sweep_name} L{nl}: {metric} distribution by {hparam} (Violin)",
            )
            fig_violin.write_html(
                os.path.join(PLOTS_DIR, f"L{nl}_{metric}_violin_{hparam}.html")
            )

            fig_strip = px.strip(
                df,
                y=metric,
                x=hparam,
                title=f"{sweep_name} L{nl}: {metric} distribution by {hparam} (Strip)",
            )
            fig_strip.write_html(
                os.path.join(PLOTS_DIR, f"L{nl}_{metric}_strip_{hparam}.html")
            )

            print(f"✅ Plots for {metric} vs {hparam} saved (L{nl})")

            # RMSE versions (for mse columns only)
            if "mse" in metric:
                rmse_metric = metric.replace("mse", "rmse")
                df[rmse_metric] = np.sqrt(df[metric])
                bar_data_rmse = df.groupby(hparam)[rmse_metric].mean().reset_index()

                fig_bar_rmse = px.bar(
                    bar_data_rmse,
                    x=hparam,
                    y=rmse_metric,
                    title=f"{sweep_name} L{nl}: Mean {rmse_metric} vs {hparam}",
                )
                fig_bar_rmse.write_html(
                    os.path.join(PLOTS_DIR, f"L{nl}_{rmse_metric}_bar_{hparam}.html")
                )

                fig_violin_rmse = px.violin(
                    df,
                    y=rmse_metric,
                    x=hparam,
                    box=True,
                    points="all",
                    title=f"{sweep_name} L{nl}: {rmse_metric} distribution by {hparam} (Violin)",
                )
                fig_violin_rmse.write_html(
                    os.path.join(PLOTS_DIR, f"L{nl}_{rmse_metric}_violin_{hparam}.html")
                )

                fig_strip_rmse = px.strip(
                    df,
                    y=rmse_metric,
                    x=hparam,
                    title=f"{sweep_name} L{nl}: {rmse_metric} distribution by {hparam} (Strip)",
                )
                fig_strip_rmse.write_html(
                    os.path.join(PLOTS_DIR, f"L{nl}_{rmse_metric}_strip_{hparam}.html")
                )

                print(f"📈 RMSE Plots for {rmse_metric} vs {hparam} saved (L{nl})")


def process_layer_summaries():
    dfs = []
    for layer_dir in ["LSTM_L1", "LSTM_L2"]:
        layer_path = os.path.join(SWEEP_ROOT, layer_dir)
        csv_path = os.path.join(layer_path, "best_configurations.csv")
        if not os.path.isfile(csv_path):
            print(f"❌ Missing: {csv_path}")
            continue

        nl = 1 if "L1" in layer_dir else 2
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                print(f"⚠️ Empty: {csv_path}")
                continue
            df["num_layers"] = nl
            dfs.append(df)
            save_bar_violin_strip_plots(df, sweep_name=layer_dir, nl=nl)
        except Exception as e:
            print(f"❌ Error processing {csv_path}: {e}")

    return dfs


def aggregate_and_plot_by_layer(dfs):
    if not dfs:
        print("No data to aggregate.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    save_bar_violin_strip_plots(combined_df, sweep_name="AllModels", nl="ALL")


if __name__ == "__main__":
    dfs = process_layer_summaries()
    aggregate_and_plot_by_layer(dfs)
