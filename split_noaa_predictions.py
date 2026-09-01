#!/usr/bin/env python3
"""
Usage:
  PYTHONUNBUFFERED=1 python3 -u split_noaa_predictions.py \
    --data ../../data/lucas/merged_noaa_all.tsv \
    --stations 1611400 1612340 1612480 1615680 1617433 1617760 1619808 1619910 \
    --out-dir ../../data/lucas \
    --chunksize 500000
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

TIME_COL = "time"
VALUE_COL = "value"
STATION_COL = "station"


def write_headers(handles, stations, out_dir):
    for sid in stations:
        fh = (out_dir / f"{sid}_noaa.tmp.tsv").open("w")
        fh.write(f"{TIME_COL}\t{VALUE_COL}\n")
        handles[sid] = fh


def close_all(handles):
    for fh in handles.values():
        fh.close()


def sort_each_station(out_dir, stations):
    for sid in tqdm(stations, desc="Sorting NOAA files", unit="stn", position=0):
        tmp = out_dir / f"{sid}_noaa.tmp.tsv"
        final = out_dir / f"{sid}_noaa.tsv"
        if not tmp.exists():
            continue
        df = pd.read_csv(tmp, sep="\t", parse_dates=[TIME_COL])
        df = df.sort_values(TIME_COL)
        df.to_csv(final, sep="\t", index=False)
        tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--stations", nargs="+", required=True)
    ap.add_argument(
        "--out-dir", default=".", help="Output directory (default: current dir)"
    )
    ap.add_argument("--chunksize", type=int, default=250000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stations = [str(s) for s in args.stations]
    handles = {}
    write_headers(handles, stations, out_dir)

    chunk_bar = tqdm(desc="Chunks", unit="chunk", dynamic_ncols=True, file=sys.stdout)
    row_bar = tqdm(
        desc="Rows written", unit="row", dynamic_ncols=True, file=sys.stdout, position=1
    )

    try:
        for chunk in pd.read_csv(
            args.data,
            sep="\t",
            chunksize=args.chunksize,
            dtype={STATION_COL: str},
            parse_dates=[TIME_COL],
        ):
            chunk_bar.update(1)
            chunk.dropna(subset=[TIME_COL, VALUE_COL, STATION_COL], inplace=True)
            chunk = chunk[chunk[STATION_COL].isin(stations)]
            if chunk.empty:
                continue

            for sid, sub in chunk.groupby(STATION_COL, sort=False):
                fh = handles.get(sid)
                if fh is None:
                    continue
                times = sub[TIME_COL].astype(str).to_numpy()
                vals = sub[VALUE_COL].to_numpy()
                for t, v in zip(times, vals):
                    fh.write(f"{t}\t{v}\n")
                row_bar.update(len(sub))
    finally:
        close_all(handles)
        chunk_bar.close()
        row_bar.close()

    sort_each_station(out_dir, stations)

    existing = {p.stem.replace("_noaa", "") for p in out_dir.glob("*_noaa.tsv")}
    missing = set(stations) - existing
    if missing:
        print(
            "WARNING: no rows found for:", ", ".join(sorted(missing)), file=sys.stderr
        )


if __name__ == "__main__":
    main()
