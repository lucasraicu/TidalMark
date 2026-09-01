#!/usr/bin/env python3
import glob
import os

import pandas as pd
from tqdm import tqdm

SWEEP_ROOT = "../../data/lucas/univariate-sweep"
BEST_PATTERN = os.path.join(SWEEP_ROOT, "*", "best_configurations.csv")
OUTPUT_SUM = os.path.join(SWEEP_ROOT, "summary_and_tail_metrics.txt")

STEP_MINUTES = 6
STEPS_PER_DAY = 24 * 60 // STEP_MINUTES
HORIZONS_DAYS = [1, 3, 5, 7]
HORIZONS = [d * STEPS_PER_DAY for d in HORIZONS_DAYS]

HYPERPARAMS = ["seq_len", "batch_size", "lr", "hidden_size", "num_layers"]

records = []
for path in glob.glob(BEST_PATTERN):
    sweep = os.path.basename(os.path.dirname(path))
    df = pd.read_csv(path)
    if df.empty:
        continue
    df["sweep"] = sweep
    df["model_base"] = df["model"].str.split("_").str[0].str.lower()
    records.append(df)

if not records:
    print("No best_configurations.csv found. Exiting.")
    exit(0)

all_df = pd.concat(records, ignore_index=True)

stats = []
for _, row in tqdm(all_df.iterrows(), total=all_df.shape[0], desc="Processing runs"):
    sweep = row["sweep"]
    run_dir = row["run_dir"]
    summary_path = os.path.join(SWEEP_ROOT, sweep, run_dir, "horizon_summary.csv")
    error_path = os.path.join(SWEEP_ROOT, sweep, run_dir, "horizon_errors.csv")
    if not os.path.isfile(summary_path):
        continue

    dfh = pd.read_csv(summary_path)
    err_df = pd.read_csv(error_path) if os.path.isfile(error_path) else None

    for h_step, h_day in zip(HORIZONS, HORIZONS_DAYS):
        test_h = dfh[(dfh["split"] == "test") & (dfh["horizon"] == h_step)]
        if test_h.empty:
            continue
        mae = test_h["mae"].iat[0]
        r2 = test_h["r2"].iat[0]
        mse = test_h["mse"].iat[0]
        max_ae = None
        if err_df is not None and "test" in err_df.columns:
            errs = err_df[err_df["horizon"] == h_step]["test"].dropna().values
            if errs.size > 0:
                max_ae = errs.max()
        stats.append(
            {
                "sweep": sweep,
                "run_dir": run_dir,
                "model": row["model"],
                "model_base": row["model_base"],
                "seq_len": row.get("seq_len"),
                "batch_size": row.get("batch_size"),
                "lr": row.get("lr"),
                "hidden_size": row.get("hidden_size"),
                "num_layers": row.get("num_layers", 1),
                "horizon_days": h_day,
                "mae": mae,
                "r2": r2,
                "mse": mse,
                "max_ae": max_ae,
            }
        )

stats_df = pd.DataFrame(stats)

best_by_r2 = {}
best_by_mae = {}
best_by_maxae = {}
for d in HORIZONS_DAYS:
    dfh = stats_df[stats_df["horizon_days"] == d]
    if dfh.empty:
        continue
    best_by_r2[d] = dfh.loc[dfh["r2"].idxmax()]
    best_by_mae[d] = dfh.loc[dfh["mae"].idxmin()]
    best_by_maxae[d] = dfh.loc[dfh["max_ae"].idxmin()]

best_avg_r2_models = []
for (mb, nl), grp in all_df.groupby(["model_base", "num_layers"]):
    best_avg_r2_models.append(grp.loc[grp["avg_r2"].idxmax()])
best_avg_r2_models = sorted(best_avg_r2_models, key=lambda x: x["avg_r2"], reverse=True)

best_per_model = []
for mb, grp in all_df.groupby("model_base"):
    best_per_model.append(grp.loc[grp["avg_r2"].idxmax()])

overall_best = max(best_per_model, key=lambda x: x["avg_r2"])

with open(OUTPUT_SUM, "w") as f:
    for d in HORIZONS_DAYS:
        f.write(f"=== Horizon: {d}d ===\n")
        for label, s in zip(
            ["Best R²   ", "Best MAE  ", "Best MaxAE"],
            [best_by_r2[d], best_by_mae[d], best_by_maxae[d]],
        ):
            f.write(
                f"{label} | Sweep={s['sweep']} | Model={s['model']} | Horizon={d}d | "
                f"R²={s['r2']:.4f} | MSE={s['mse']:.4f} | MAE={s['mae']:.4f} | MaxAE={s['max_ae']:.4f} | "
                f"seq_len={s['seq_len']} | batch_size={s['batch_size']} | lr={s['lr']} | "
                f"hidden_size={s['hidden_size']} | num_layers={s['num_layers']}\n"
            )
        f.write("\n")

    f.write("=== Best Configurations by R² ===\n")
    for r in best_avg_r2_models:
        f.write(
            f"Sweep={r['sweep']} | Model={r['model']} | Layers={int(r['num_layers'])} | "
            f"seq_len={r['seq_len']} | batch_size={r['batch_size']} | lr={r['lr']} | "
            f"hidden_size={r['hidden_size']} | num_layers={r['num_layers']} | "
            f"Avg R²={r['avg_r2']:.4f} (min={r['min_r2']:.4f}, max={r['max_r2']:.4f}) | "
            f"MSE={r['avg_mse']:.4f} (min={r['min_mse']:.4f}, max={r['max_mse']:.4f}) | "
            f"MAE={r['avg_mae']:.4f} (min={r['min_mae']:.4f}, max={r['max_mae']:.4f}) | "
            f"MaxAE={r['max_ae']:.4f}\n"
        )
    f.write("\n")

    f.write("=== Best Configuration per Model Base ===\n")
    for r in best_per_model:
        f.write(
            f"{r['model_base'].upper():<8} | Sweep={r['sweep']} | Model={r['model']} | "
            f"seq_len={r['seq_len']} | batch_size={r['batch_size']} | lr={r['lr']} | "
            f"hidden_size={r['hidden_size']} | num_layers={r['num_layers']} | "
            f"Avg R²={r['avg_r2']:.4f} (min={r['min_r2']:.4f}, max={r['max_r2']:.4f}) | "
            f"MSE={r['avg_mse']:.4f} (min={r['min_mse']:.4f}, max={r['max_mse']:.4f}) | "
            f"MAE={r['avg_mae']:.4f} (min={r['min_mae']:.4f}, max={r['max_mae']:.4f}) | "
            f"MaxAE={r['max_ae']:.4f}\n"
        )
    f.write("\n")

    b = overall_best
    f.write("=== Best Overall Configuration ===\n")
    f.write(
        f"Sweep={b['sweep']} | Model={b['model']} | "
        f"seq_len={b['seq_len']} | batch_size={b['batch_size']} | lr={b['lr']} | "
        f"hidden_size={b['hidden_size']} | num_layers={b['num_layers']} | "
        f"Avg R²={b['avg_r2']:.4f} (min={b['min_r2']:.4f}, max={b['max_r2']:.4f}) | "
        f"MSE={b['avg_mse']:.4f} (min={b['min_mse']:.4f}, max={b['max_mse']:.4f}) | "
        f"MAE={b['avg_mae']:.4f} (min={b['min_mae']:.4f}, max={b['max_mae']:.4f}) | "
        f"MaxAE={b['max_ae']:.4f}\n"
    )

print(f"✅ Summary written to {OUTPUT_SUM}")
