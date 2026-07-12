# ~/Projects/joel-nrl-predictor/tools/merge_nrl_csvs.py
from __future__ import annotations
import argparse
from pathlib import Path
import re
import pandas as pd

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args()

def infer_season_phase(p: Path) -> tuple[int|None, str|None]:
    m = re.search(r"nrl_(\d{4})_(regular|finals)\.csv$", p.name)
    if not m: return None, None
    return int(m.group(1)), m.group(2)

REQUIRED_COLS = [
    "season","round","date_header","status","kickoff_local",
    "home_team","away_team","home_score","away_score","venue","match_href",
]

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    # status unify
    s = df["status"].astype("string").str.lower().str.strip()
    s = s.replace({"full-time":"full time","ft":"full time"})
    df["status"] = s
    # numeric scores
    for col in ["home_score","away_score","round","season"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    # parsed datetime (best-effort) from kickoff_local
    # keeps original string too
    try:
        df["kickoff_local_dt"] = pd.to_datetime(df["kickoff_local"], errors="coerce")
    except Exception:
        df["kickoff_local_dt"] = pd.NaT
    # column order
    cols = REQUIRED_COLS + [c for c in df.columns if c not in REQUIRED_COLS]
    return df[cols]

def main() -> None:
    ns = parse_args()
    history = Path(ns.history_dir)
    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(history.glob("nrl_*.csv"))
    if not files:
        print("No inputs."); return

    parts: list[pd.DataFrame] = []
    for f in files:
        season, phase = infer_season_phase(f)
        if season is None: continue
        df = pd.read_csv(f)
        df = normalize(df)
        df.insert(0, "phase", phase)  # "regular" | "finals"
        parts.append(df)

    if not parts:
        print("No valid inputs."); return

    merged = pd.concat(parts, ignore_index=True, sort=False)
    # helpful sort
    sort_cols = [c for c in ["season","phase","round","kickoff_local_dt"] if c in merged.columns]
    merged = merged.sort_values(sort_cols, kind="stable")
    merged.to_csv(out, index=False)
    print(f"Wrote {len(merged)} rows -> {out}")

if __name__ == "__main__":
    main()
