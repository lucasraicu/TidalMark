#!/usr/bin/env python3
import glob
import os

import pandas as pd

SWEEPS_ROOT = "../../data/lucas/univariate-sweep"


def classify(r):
    if abs(r) < 0.1:
        return "no"
    if abs(r) < 0.4:
        return "minimal"
    return "major"


HYPERPARAMS = ["seq_len", "batch_size", "lr", "hidden_size", "num_layers"]
METRICS = [
    "score",
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

for sweep_dir in os.listdir(SWEEPS_ROOT):
    base_dir = os.path.join(SWEEPS_ROOT, sweep_dir)
    if not os.path.isdir(base_dir):
        continue

    output_csv = os.path.join(base_dir, "best_configurations.csv")
    summary_file = os.path.join(base_dir, "best_summary.txt")

    records = []
    for run_dir in os.listdir(base_dir):
        run_path = os.path.join(base_dir, run_dir)
        summary_path = os.path.join(run_path, "horizon_summary.csv")
        error_path = os.path.join(run_path, "horizon_errors.csv")
        if not (os.path.isdir(run_path) and os.path.isfile(summary_path)):
            continue

        df = pd.read_csv(summary_path)
        df_test = df[df["split"] == "test"]

        avg_r2 = df_test["r2"].mean()
        min_r2 = df_test["r2"].min()
        max_r2 = df_test["r2"].max()

        avg_mse = df_test["mse"].mean()
        min_mse = df_test["mse"].min()
        max_mse = df_test["mse"].max()

        avg_mae = df_test["mae"].mean()
        min_mae = df_test["mae"].min()
        max_mae = df_test["mae"].max()

        max_ae = None
        if os.path.isfile(error_path):
            err_df = pd.read_csv(error_path)
            if "test" in err_df.columns:
                ae_vals = err_df["test"].dropna().values
                if ae_vals.size > 0:
                    max_ae = ae_vals.max()

        score = (1 - avg_r2) + avg_mse

        parts = run_dir.split("_")
        model_name = parts[0]
        seq_len = int(next(p[3:] for p in parts if p.startswith("seq")))
        batch_size = int(next(p[2:] for p in parts if p.startswith("bs")))
        lr_str = next(p[2:] for p in parts if p.startswith("lr"))
        if lr_str.startswith("."):
            lr_str = "0" + lr_str
        lr = float(lr_str)
        hidden_size = int(next(p[2:] for p in parts if p.startswith("hs")))
        nl_part = next(
            (p for p in parts if p.startswith("nl") or p.startswith("layers")), None
        )
        num_layers = (
            int(nl_part[2:] if nl_part and nl_part.startswith("nl") else nl_part[6:])
            if nl_part
            else 1
        )

        records.append(
            {
                "model": model_name,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "lr": lr,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "avg_r2": avg_r2,
                "min_r2": min_r2,
                "max_r2": max_r2,
                "avg_mse": avg_mse,
                "min_mse": min_mse,
                "max_mse": max_mse,
                "avg_mae": avg_mae,
                "min_mae": min_mae,
                "max_mae": max_mae,
                "max_ae": max_ae,
                "score": score,
                "run_dir": run_dir,
            }
        )

    if not records:
        print(f"No valid runs found under '{sweep_dir}'. Skipping.")
        continue

    df_all = pd.DataFrame(records)
    df_sorted = df_all.sort_values(by="score", ascending=True).reset_index(drop=True)
    df_sorted.to_csv(output_csv, index=False)
    print(f"Saved ranked configurations to: {output_csv}")

    progressions = {
        "Sequence length": df_all.groupby("seq_len")["avg_r2"].mean(),
        "Batch size": df_all.groupby("batch_size")["avg_r2"].mean(),
        "Learning rate": df_all.groupby("lr")["avg_r2"].mean(),
        "Hidden size": df_all.groupby("hidden_size")["avg_r2"].mean(),
        "Num layers": df_all.groupby("num_layers")["avg_r2"].mean(),
        "Avg MAE by seq_len": df_all.groupby("seq_len")["avg_mae"].mean(),
        "Max AE by seq_len": df_all.groupby("seq_len")["max_ae"].mean(),
    }

    corr_cols = [col for col in (HYPERPARAMS + METRICS) if col in df_all.columns]
    corr_df = df_all[corr_cols].corr()
    correlation_insights = ["-- Correlation Impact Summaries --"]
    for hp in HYPERPARAMS:
        if df_all[hp].nunique() < 2:
            continue
        for met in METRICS:
            if met not in corr_df.columns:
                continue
            r = corr_df.at[hp, met]
            strength = classify(r)
            if strength == "no":
                correlation_insights.append(
                    f"Parameter '{hp}' shows no effect on '{met}' (r={r:.2f})."
                )
            elif strength == "minimal":
                correlation_insights.append(
                    f"Parameter '{hp}' has minimal effect on '{met}' (r={r:.2f})."
                )
            else:
                sign = "increase" if r > 0 else "decrease"
                correlation_insights.append(
                    f"Increasing '{hp}' has a major {sign} effect on '{met}' (r={r:.2f})."
                )

    with open(summary_file, "w") as f:
        f.write(f"Best configuration summary for {sweep_dir}\n")
        f.write("=" * 30 + "\n")
        best = df_sorted.iloc[0]
        f.write(f"Model:         {best['model']}\n")
        f.write(f"Seq length:    {best['seq_len']}\n")
        f.write(f"Batch size:    {best['batch_size']}\n")
        f.write(f"Learning rate: {best['lr']}\n")
        f.write(f"Hidden size:   {best['hidden_size']}\n")
        f.write(f"Num layers:    {best['num_layers']}\n")
        f.write(
            f"Avg R²:        {best['avg_r2']:.4f} (min={best['min_r2']:.4f}, max={best['max_r2']:.4f})\n"
        )
        f.write(
            f"Avg MSE:       {best['avg_mse']:.4f} (min={best['min_mse']:.4f}, max={best['max_mse']:.4f})\n"
        )
        f.write(
            f"Avg MAE:       {best['avg_mae']:.4f} (min={best['min_mae']:.4f}, max={best['max_mae']:.4f})\n"
        )
        f.write(
            f"Max AE:        {best['max_ae'] if best['max_ae'] is not None else 'N/A'}\n"
        )
        f.write(f"Score:         {best['score']:.4f}\n\n")
        f.write("How R² changes with each hyperparameter (mean across runs):\n")
        f.write("-" * 50 + "\n")
        for name, series in progressions.items():
            f.write(f"\n{name} vs. mean R²:\n")
            for val, mean_val in series.sort_index().items():
                f.write(f"  {val}: {mean_val:.4f}\n")
        f.write("\n" + "-" * 50 + "\n\n")
        for line in correlation_insights:
            f.write(line + "\n")

    print(
        f"Summary (with progressions and correlation insights) saved to: {summary_file}\n"
    )
