#!/usr/bin/env python3
"""
Single-layer training & sweep script with:
 - Per-epoch wall-clock timings (train_time_s, val_time_s)
 - Losses stored in scaled space and physical units (m, m^2)
 - Neighbor
 - Train/Val RMSE (meters) plot
 - No GPU util/mem logging
"""
"""
PYTHONUNBUFFERED=1 python3 -u multivariate-lstm-noaa-modularized-revamped.py \
  --station-id 1611400 \
  --model lstm \
  --sweep
"""
from __future__ import annotations

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import os

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
import argparse
import errno
import gc
import itertools
import json
import math
import subprocess  # nosec
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATA_PATH = os.path.normpath(
    os.path.join(
        SCRIPT_DIR,
        "../../data/lucas/model_data_less_than_20_missing_no_igld_no_dups.tsv",
    )
)
DEFAULT_BASE_OUTPUT = os.path.normpath(os.path.join(SCRIPT_DIR, "../../data/lucas"))
STATIONS_FILE = os.path.normpath(
    os.path.join(SCRIPT_DIR, "../../data/lucas/active_stations.jsonl")
)

STATION_SPLIT_SCRIPT = os.path.join(SCRIPT_DIR, "split_stations.py")
NOAA_SPLIT_SCRIPT = os.path.join(SCRIPT_DIR, "split_noaa_predictions.py")

torch.cuda.empty_cache()

STEP_MINUTES = 6
DEFAULT_HORIZONS_DAYS = [1, 3, 5, 7]
DEFAULT_SEQS_DAYS = [14]


# Print system-wide open file handle count
def get_system_open_file_handles() -> int:
    with open("/proc/sys/fs/file-nr", "r") as f:
        allocated, _, _ = map(int, f.readline().split())
    return allocated


# List this process’s open file descriptors
def get_process_open_file_descriptors() -> list[tuple[int, str]]:
    base = "/proc/self/fd"
    fd_map = {}
    for entry in os.listdir(base):
        path = os.path.join(base, entry)
        try:
            target = os.readlink(path)
        except OSError as e:
            if e.errno == errno.ENOENT:
                continue
            raise
        fd_map[int(entry)] = target
    return sorted(fd_map.items(), key=lambda x: x[0])


# Print system and this process file-descriptor status
def print_file_descriptor_status():
    system_count = get_system_open_file_handles()
    print(f"\n>>> System-wide open file handles: {system_count}")
    process_fds = get_process_open_file_descriptors()
    print(f">>> This process has {len(process_fds)} open file descriptors:")
    for fd, target in process_fds:
        print(f"    {fd:3d} → {target}")
    print()


# Convert days to number of time steps
def days_to_steps(days: int) -> int:
    return int(days * 24 * 60 / STEP_MINUTES)


# Parse command-line arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train single-layer model or run full hyperparameter sweep."
    )
    parser.add_argument(
        "--disable-lr-adapt",
        action="store_true",
        help="Disable learning-rate scheduler",
    )
    parser.add_argument("--sweep", action="store_true", help="Run grid sweep")
    parser.add_argument("--data", type=str, default=None, help="TSV data path")
    parser.add_argument(
        "--model",
        choices=["lstm", "bilstm", "conv-lstm", "gru", "attn-lstm"],
        help="Model type",
    )
    parser.add_argument("--seq-len", type=int, help="Sequence length (days)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[64])
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[2],
        help="Number of RNN layers (for sweep or single run)",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=DEFAULT_HORIZONS_DAYS,
        help="Forecast horizons (days)",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--base-output", type=str, default=None, help="Output directory"
    )
    parser.add_argument("--models", nargs="+", type=str, default=["lstm"])
    parser.add_argument("--seqs", nargs="+", type=int, default=DEFAULT_SEQS_DAYS)
    parser.add_argument("--batches", nargs="+", type=int, default=[512])
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-3])
    parser.add_argument(
        "--station-id",
        type=str,
        required=True,
        help="Primary station ID",
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data is None:
        args.data = os.path.normpath(
            os.path.join(
                script_dir,
                "../../data/lucas/model_data_less_than_20_missing_no_igld_no_dups.tsv",
            )
        )
    if args.base_output is None:
        args.base_output = os.path.normpath(
            os.path.join(script_dir, "../../data/lucas")
        )

    if args.seq_len is not None:
        args.seq_len = days_to_steps(args.seq_len)
    args.horizons = [days_to_steps(d) for d in args.horizons]
    args.seqs = [days_to_steps(d) for d in args.seqs]

    if not args.sweep:
        if len(args.layers) != 1:
            parser.error("When not sweeping, specify exactly one --layers")
        args.num_layers = args.layers[0]
    else:
        # for sweep, we'll set args.num_layers per combination
        args.num_layers = None

    return args


CUTOFF_DATE = pd.Timestamp("2024-08-31")  # Update if needed


def extract_station_data(df, station_id: str) -> pd.DataFrame:
    df_station = df[df["station"] == station_id].copy()
    df_station["time"] = pd.to_datetime(df_station["time"])
    df_station = df_station[df_station["time"] <= CUTOFF_DATE]
    df_station = df_station[["time", "value"]].dropna().sort_values("time")
    return df_station


def load_and_preprocess_data(
    path: str, station_id: str, noaa_file: str = "../../data/lucas/merged_noaa_all.tsv"
) -> str:
    print(f"Loading primary and NOAA prediction data for station {station_id} …")

    out_filename = f"{station_id}_with_noaa.tsv"
    out_path = os.path.join(DEFAULT_BASE_OUTPUT, out_filename)
    if os.path.exists(out_path):
        print(f"Using cached merged dataset: {out_path}")
        return out_path

    # Ensure split station file exists or create it
    station_file = os.path.join(DEFAULT_BASE_OUTPUT, f"{station_id}.tsv")
    if not os.path.exists(station_file):
        print(f"{station_file} not found. Running split_stations.py to generate it…")

        cmd = [
            "python3",
            STATION_SPLIT_SCRIPT,
            "--data",
            path,
            "--stations",
            station_id,
            "--out-dir",
            DEFAULT_BASE_OUTPUT,
            "--chunksize",
            "500000",
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, close_fds=True)  # nosec

        print(
            f"  [Debug] FD count after station {len(get_process_open_file_descriptors())}"
        )

        if not os.path.exists(station_file):
            raise RuntimeError(f"Station file generation failed for {station_id}")

    # Load primary station data
    with open(station_file, "r") as f:
        df_primary = pd.read_csv(f, sep="\t", parse_dates=["time"])

    df_primary.rename(columns={"value": "main"}, inplace=True)

    # Load NOAA prediction file and filter by station
    noaa_station_file = os.path.join(DEFAULT_BASE_OUTPUT, f"{station_id}_noaa.tsv")

    if not os.path.exists(noaa_station_file):
        print(
            f"{noaa_station_file} not found. Running split_noaa_predictions.py to generate it…"
        )

        cmd = [
            "python3",
            NOAA_SPLIT_SCRIPT,
            "--data",
            noaa_file,
            "--stations",
            station_id,
            "--out-dir",
            DEFAULT_BASE_OUTPUT,
            "--chunksize",
            "500000",
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, close_fds=True)  # nosec
        print(
            f"  [Debug] FD count after pred {len(get_process_open_file_descriptors())}"
        )

        if not os.path.exists(noaa_station_file):
            raise RuntimeError(f"NOAA file generation failed for {station_id}")

    # Load NOAA prediction file
    with open(noaa_station_file, "r") as f:
        df_noaa = pd.read_csv(f, sep="\t", parse_dates=["time"])
    df_noaa.rename(columns={"value": "noaa"}, inplace=True)

    df_primary["time"] = pd.to_datetime(df_primary["time"]).dt.round("min")
    df_noaa["time"] = pd.to_datetime(df_noaa["time"]).dt.round("min")

    # Merge observed and NOAA predictions on time
    merged = pd.merge(
        df_primary[["time", "main"]], df_noaa[["time", "noaa"]], on="time", how="inner"
    )
    merged.dropna(inplace=True)

    if merged.empty:
        raise ValueError(
            f"No overlapping timestamps between station and NOAA for {station_id}"
        )

    merged.to_csv(out_path, sep="\t", index=False)
    print(f"Saved merged dataset to: {out_path}")
    return out_path


def find_nearest_station(
    station_id: str, data_tsv: str, stations_file: str = STATIONS_FILE
) -> str:
    df = pd.read_csv(data_tsv, sep="\t", usecols=["station"], dtype={"station": str})
    data_ids = set(df["station"])
    if station_id not in data_ids:
        raise RuntimeError(f"Station {station_id} not found in {data_tsv}")

    records = []
    with open(stations_file) as f:
        for line in f:
            d = json.loads(line)
            sid = d.get("details", {}).get("id") or d.get("id")
            lat, lng = d.get("lat"), d.get("lng")
            if sid in data_ids and lat is not None and lng is not None:
                records.append((sid, float(lat), float(lng)))

    if not records:
        raise RuntimeError("No valid neighbor stations in data")

    lat0, lng0 = next((lat, lng) for sid, lat, lng in records if sid == station_id)

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        return 2 * R * math.asin(math.sqrt(a))

    neighbors = [
        (sid, haversine(lat0, lng0, lat, lng))
        for sid, lat, lng in records
        if sid != station_id
    ]
    neighbors.sort(key=lambda x: x[1])
    return neighbors[0][0]


# Dataset for scaled sequences
class ScaledWaterLevelDataset(Dataset):
    def __init__(self, features, targets, seq_length, future_steps, scaler_X, scaler_y):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
        self.future_steps = future_steps
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    def __len__(self) -> int:
        return len(self.features) - self.seq_length - self.future_steps + 1

    def __getitem__(self, idx: int):
        x_raw = self.features[idx : idx + self.seq_length]
        y_raw = self.targets[
            idx + self.seq_length : idx + self.seq_length + self.future_steps
        ]
        x_scaled = self.scaler_X.transform(x_raw)
        y_scaled = self.scaler_y.transform(y_raw.reshape(-1, 1)).flatten()
        return torch.tensor(x_scaled, dtype=torch.float32), torch.tensor(
            y_scaled, dtype=torch.float32
        )


# Build DataLoaders
def build_data_loaders(
    train_dataset, val_dataset, test_dataset, batch_size, num_workers
):
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
    }


# Train one epoch
def train_one_epoch(model, data_loader, optimizer, loss_function, device, epoch):
    model.train()
    total_loss = 0.0
    start_time = time.perf_counter()
    desc = f"Train Epoch {epoch}"
    for X_batch, y_batch in tqdm(data_loader, desc="  Training"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = loss_function(predictions, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    elapsed = time.perf_counter() - start_time
    print(
        f"  [Debug] FD count after training epoch {epoch}: {len(get_process_open_file_descriptors())}"
    )
    return total_loss / len(data_loader.dataset), elapsed


# Validate one epoch
def validate_one_epoch(model, data_loader, loss_function, scaler_y, device, epoch):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_truths = []
    start_time = time.perf_counter()
    desc = f"Val Epoch {epoch}"
    with torch.no_grad():
        for X_batch, y_batch in tqdm(data_loader, desc="  Validation"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            total_loss += loss_function(predictions, y_batch).item() * X_batch.size(0)
            all_preds.append(predictions.cpu().numpy())
            all_truths.append(y_batch.cpu().numpy())
    elapsed = time.perf_counter() - start_time
    avg_loss = total_loss / len(data_loader.dataset)
    preds_arr = np.concatenate(all_preds, axis=0)
    truths_arr = np.concatenate(all_truths, axis=0)
    inv_preds = scaler_y.inverse_transform(preds_arr)
    inv_truths = scaler_y.inverse_transform(truths_arr)
    print(
        f"  [Debug] FD count after validating epoch {epoch}: {len(get_process_open_file_descriptors())}"
    )
    return avg_loss, elapsed, inv_preds, inv_truths


# Full train and evaluation
def train_and_evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Training {args.model.upper()} ===")

    station_id = args.station_id
    print(f"Primary station: {station_id} using NOAA predictions")

    merged_path = load_and_preprocess_data(args.data, station_id)

    with open(merged_path, "r") as f:
        merged_df = pd.read_csv(f, sep="\t")

    features_array = merged_df[["main", "noaa"]].values.astype(np.float32)
    targets_array = merged_df["main"].values.astype(np.float32)

    total_samples = len(features_array)
    training_end_index = int(total_samples * 0.8)
    validation_end_index = int(total_samples * 0.9)

    X_train = features_array[:training_end_index]
    y_train = targets_array[:training_end_index]
    X_val = features_array[training_end_index:validation_end_index]
    y_val = targets_array[training_end_index:validation_end_index]
    X_test = features_array[validation_end_index:]
    y_test = targets_array[validation_end_index:]

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

    future_steps_count = max(args.horizons)
    train_ds = ScaledWaterLevelDataset(
        X_train, y_train, args.seq_len, future_steps_count, scaler_X, scaler_y
    )
    val_ds = ScaledWaterLevelDataset(
        X_val, y_val, args.seq_len, future_steps_count, scaler_X, scaler_y
    )
    test_ds = ScaledWaterLevelDataset(
        X_test, y_test, args.seq_len, future_steps_count, scaler_X, scaler_y
    )

    data_loaders = build_data_loaders(
        train_ds, val_ds, test_ds, args.batch_size, args.num_workers
    )

    model_classes = {
        "lstm": "LSTMModel",
        "bilstm": "BiLSTMModel",
        "conv-lstm": "ConvLSTMModel",
        "gru": "GRUModel",
        "attn-lstm": "AttnLSTMModel",
    }

    model_class = getattr(models, model_classes[args.model])

    model = model_class(
        input_dim=features_array.shape[1],
        hidden_dim=args.hidden_size,
        num_layers=args.num_layers,
        future_steps=future_steps_count,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if not args.disable_lr_adapt:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=args.patience, factor=0.5
        )
    loss_function = nn.MSELoss()

    print_file_descriptor_status()

    seq_days = round(args.seq_len * STEP_MINUTES / 60 / 24)
    run_directory = os.path.join(
        args.base_output,
        f"{args.model}_seq{seq_days}"
        f"_bs{args.batch_size}_lr{args.lr}"
        f"_hs{args.hidden_size}"
        f"_layers{args.num_layers}",
    )
    os.makedirs(run_directory, exist_ok=True)

    with open(os.path.join(run_directory, "scalers.json"), "w") as f:
        json.dump(
            {"mean_y": scaler_y.mean_[0], "std_y": scaler_y.scale_[0]}, f, indent=2
        )

    history_records = []
    best_validation_loss = float("inf")
    patience_counter = args.patience

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_time = train_one_epoch(
            model, data_loaders["train"], optimizer, loss_function, device, epoch
        )
        val_loss, val_time, preds_val, truths_val = validate_one_epoch(
            model, data_loaders["val"], loss_function, scaler_y, device, epoch
        )

        scaled_mse_train = train_loss
        scaled_mse_val = val_loss

        # add these two lines:
        scaled_rmse_train = scaled_mse_train**0.5
        scaled_rmse_val = scaled_mse_val**0.5

        # physical‐unit MSE (m²)
        physical_mse_train = scaled_mse_train * (scaler_y.scale_[0] ** 2)
        physical_mse_val = scaled_mse_val * (scaler_y.scale_[0] ** 2)

        # and your RMSE in meters
        physical_rmse_train = physical_mse_train**0.5
        physical_rmse_val = physical_mse_val**0.5

        print(
            f"\nEpoch {epoch} Train RMSE (m): {physical_rmse_train:.6f}  | scaled RMSE: {scaled_rmse_train:.6f}"
        )
        print(
            f"Epoch {epoch} Val RMSE   (m): {physical_rmse_val:.6f}  | scaled RMSE: {scaled_rmse_val:.6f}"
        )

        current_lr = optimizer.param_groups[0]["lr"]
        for horizon in args.horizons:
            preds_h = preds_val[:, horizon - 1]
            trues_h = truths_val[:, horizon - 1]
            mse_h = mean_squared_error(trues_h, preds_h)
            mae_h = mean_absolute_error(trues_h, preds_h)
            r2_h = r2_score(trues_h, preds_h)
            print(
                f"Epoch {epoch} | Horizon {horizon} | "
                f"RMSE(m): {np.sqrt(mse_h):.6f}, "
                f"MAE(m): {mae_h:.6f}, "
                f"MSE(m^2): {mse_h:.6f}, "
                f"R2: {r2_h:.4f}, "
                f"LR: {current_lr:.1e}"
            )

        if scheduler and epoch > 3:
            scheduler.step(val_loss)
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            patience_counter = args.patience
            torch.save(model.state_dict(), os.path.join(run_directory, "best_model.pt"))
            print("   Saved new best model.")
        else:
            patience_counter -= 1
            print(f"   No improvement; patience left {patience_counter}")
            if patience_counter == 0:
                print("   Early stopping.")
                break

        history_records.append(
            {
                "epoch": epoch,
                "train_mse_scaled": scaled_mse_train,
                "val_mse_scaled": scaled_mse_val,
                "train_mse_m2": physical_mse_train,
                "val_mse_m2": physical_mse_val,
                "train_rmse_m": physical_rmse_train,
                "val_rmse_m": physical_rmse_val,
                "lr": optimizer.param_groups[0]["lr"],
                "train_time_s": train_time,
                "val_time_s": val_time,
            }
        )
        history_df = pd.DataFrame(history_records)
        with open(os.path.join(run_directory, "history.csv"), "w", newline="") as f:
            history_df.to_csv(f, index=False)

    model.load_state_dict(
        torch.load(
            os.path.join(run_directory, "best_model.pt"), map_location=device
        )  # nosec
    )
    return model, data_loaders, scaler_y, args, run_directory, device


# Evaluate any split
def evaluate_split(model: torch.Tensor, data_loader: Dataloader, scaler_y, device):
    model.eval()
    all_preds = []
    all_truths = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            preds = model(X_batch.to(device))
            all_preds.append(preds.cpu().numpy())
            all_truths.append(y_batch.numpy())
    return np.vstack(all_preds), np.vstack(all_truths)


# Save evaluation results to CSV
def save_evaluation_results(model, data_loaders, scaler_y, args, run_directory, device):
    summary_records = []
    error_records = []
    loss_function = nn.MSELoss()

    for split_name, loader in data_loaders.items():
        preds, truths = evaluate_split(model, loader, scaler_y, device)
        for horizon in args.horizons:
            p = preds[:, horizon - 1]
            t = truths[:, horizon - 1]
            summary_records.append(
                {
                    "split": split_name,
                    "horizon": horizon,
                    "mse": mean_squared_error(t, p),
                    "rmse": np.sqrt(mean_squared_error(t, p)),
                    "mae": mean_absolute_error(t, p),
                    "r2": r2_score(t, p),
                }
            )
        overall_mse = loss_function(
            torch.tensor(preds), torch.tensor(truths)
        ).item() * (scaler_y.scale_[0] ** 2)
        overall_rmse = np.sqrt(overall_mse)
        summary_records.append(
            {
                "split": split_name,
                "horizon": -1,
                "mse": overall_mse,
                "rmse": overall_rmse,
                "mae": np.nan,
                "r2": np.nan,
            }
        )

        abs_errors = np.abs(preds - truths)
        for i in range(abs_errors.shape[0]):
            for idx, horizon in enumerate(args.horizons):
                error_records.append(
                    {
                        "split": split_name,
                        "horizon": horizon,
                        "sample_index": i,
                        "error": abs_errors[i, idx],
                    }
                )

    with open(os.path.join(run_directory, "horizon_summary.csv"), "w", newline="") as f:
        pd.DataFrame(summary_records).to_csv(f, index=False)
    err_df = pd.DataFrame(error_records)
    err_pivot = err_df.pivot(
        index=["horizon", "sample_index"], columns="split", values="error"
    ).reset_index()
    with open(os.path.join(run_directory, "horizon_errors.csv"), "w", newline="") as f:
        err_pivot.to_csv(f, index=False)

    print(
        f"\nSaved horizon summary → {os.path.join(run_directory,'horizon_summary.csv')}"
    )
    print(f"Saved horizon errors  → {os.path.join(run_directory,'horizon_errors.csv')}")
    gc.collect()


# Run full hyperparameter sweep
def run_hyperparameter_sweep(args):
    combos = itertools.product(
        args.models, args.seqs, args.batches, args.lrs, args.hidden_sizes, args.layers
    )
    for model_type, seq_length, batch_size, lr_value, hidden_size, num_layers in tqdm(
        combos, desc="Grid Sweep"
    ):
        args.model = model_type
        args.seq_len = seq_length
        args.batch_size = batch_size
        args.lr = lr_value
        args.hidden_size = hidden_size
        args.num_layers = num_layers
        print(
            f"\n--- Sweep {model_type}, seq={seq_length}, bs={batch_size}, lr={lr_value}, hs={hidden_size}, layers={num_layers}"
        )
        mdl, loaders, scl_y, parsed_args, run_dir, dev = train_and_evaluate(args)
        save_evaluation_results(mdl, loaders, scl_y, parsed_args, run_dir, dev)
        plot_training_history(run_dir)
        plot_error_histogram(run_dir)


if __name__ == "__main__":
    print_file_descriptor_status()
    arguments = parse_arguments()
    if arguments.sweep:
        run_hyperparameter_sweep(arguments)
    else:
        mdl, data_loaders, scaler_y, parsed_args, run_dir, dev = train_and_evaluate(
            arguments
        )
        save_evaluation_results(mdl, data_loaders, scaler_y, parsed_args, run_dir, dev)
        plot_training_history(run_dir)
        plot_error_histogram(run_dir)
        gc.collect()
    print_file_descriptor_status()
