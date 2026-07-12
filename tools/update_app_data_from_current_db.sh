#!/usr/bin/env bash
set -euo pipefail

PREDICTOR="/Users/joelcouchman/Projects/joel-nrl-predictor"
SCRAPER="/Users/joelcouchman/total_nrl_scraper"
DB="$SCRAPER/data/smoke/season_2026_r1_r13/nrl.db"
PYTHON="/Users/joelcouchman/anaconda3/envs/nrl-team-strength/bin/python"

cd "$PREDICTOR"

echo
echo "============================================================"
echo "1. CHECKING REPO AND DATABASE"
echo "============================================================"

git pull --ff-only origin main

if [[ ! -f "$DB" ]]; then
    echo "Database not found:"
    echo "$DB"
    exit 1
fi

echo "Using database:"
echo "$DB"

echo
echo "============================================================"
echo "2. UPDATING APP DATA FROM DATABASE"
echo "============================================================"

DB_PATH="$DB" SCRAPER_PATH="$SCRAPER" "$PYTHON" - <<'PY'
from __future__ import annotations

import csv
import json
import os
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

DB_PATH = Path(os.environ["DB_PATH"])
SCRAPER_PATH = Path(os.environ["SCRAPER_PATH"])

RESULTS_PATH = Path("data/2026/nrl_results.csv")
BYES_PATH = Path("data/2026/nrl_byes.csv")
META_PATH = Path("data/2026/nrl_results.meta.json")

TEAM_ALIASES = {
    "brisbane broncos": "Broncos",
    "broncos": "Broncos",
    "canterbury-bankstown bulldogs": "Bulldogs",
    "canterbury bulldogs": "Bulldogs",
    "bulldogs": "Bulldogs",
    "north queensland cowboys": "Cowboys",
    "cowboys": "Cowboys",
    "dolphins": "Dolphins",
    "the dolphins": "Dolphins",
    "st george illawarra dragons": "Dragons",
    "st. george illawarra dragons": "Dragons",
    "dragons": "Dragons",
    "parramatta eels": "Eels",
    "eels": "Eels",
    "newcastle knights": "Knights",
    "knights": "Knights",
    "penrith panthers": "Panthers",
    "panthers": "Panthers",
    "south sydney rabbitohs": "Rabbitohs",
    "rabbitohs": "Rabbitohs",
    "canberra raiders": "Raiders",
    "raiders": "Raiders",
    "sydney roosters": "Roosters",
    "roosters": "Roosters",
    "manly warringah sea eagles": "Sea Eagles",
    "manly sea eagles": "Sea Eagles",
    "sea eagles": "Sea Eagles",
    "manly": "Sea Eagles",
    "cronulla-sutherland sharks": "Sharks",
    "cronulla sharks": "Sharks",
    "sharks": "Sharks",
    "melbourne storm": "Storm",
    "storm": "Storm",
    "gold coast titans": "Titans",
    "titans": "Titans",
    "new zealand warriors": "Warriors",
    "nz warriors": "Warriors",
    "warriors": "Warriors",
    "wests tigers": "Wests Tigers",
    "tigers": "Wests Tigers",
}

def normalise_team(value):
    text = str(value).strip()
    key = " ".join(text.lower().replace("&", "and").split())
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    raise ValueError(f"Unknown team name: {value!r}")

def normalise_url(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().rstrip("/")

def git_commit(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "Unavailable"

with sqlite3.connect(DB_PATH) as conn:
    cols = {row[1] for row in conn.execute("pragma table_info(matches)")}

    if {"full_time_home", "full_time_away"}.issubset(cols):
        home_score = "m.full_time_home"
        away_score = "m.full_time_away"
    elif {"home_score", "away_score"}.issubset(cols):
        home_score = "m.home_score"
        away_score = "m.away_score"
    else:
        raise SystemExit("Could not find score columns in matches table.")

    datetime_expr = "m.datetime" if "datetime" in cols else "null"
    venue_expr = "m.venue" if "venue" in cols else "null"
    url_expr = "m.url" if "url" in cols else "null"

    completed = pd.read_sql_query(
        f"""
        select
            2026 as season,
            cast(m.round as integer) as round,
            {datetime_expr} as kickoff_local,
            'Full Time' as status,
            ht.name as home_team,
            at.name as away_team,
            cast({home_score} as integer) as home_score,
            cast({away_score} as integer) as away_score,
            {venue_expr} as venue,
            {url_expr} as match_href
        from matches m
        join teams ht on ht.id = m.home_team_id
        join teams at on at.id = m.away_team_id
        where {home_score} is not null
          and {away_score} is not null
        order by m.round, m.id
        """,
        conn,
    )

if completed.empty:
    raise SystemExit("Database returned no completed matches.")

completed["round"] = completed["round"].astype(int)
completed["home_team"] = completed["home_team"].map(normalise_team)
completed["away_team"] = completed["away_team"].map(normalise_team)
completed["home_score"] = completed["home_score"].astype(int)
completed["away_score"] = completed["away_score"].astype(int)
completed["url_key"] = completed["match_href"].map(normalise_url)
completed["team_key"] = list(zip(completed["round"], completed["home_team"], completed["away_team"]))

results = pd.read_csv(RESULTS_PATH)
byes = pd.read_csv(BYES_PATH)

results["round"] = results["round"].astype(int)
results["home_team"] = results["home_team"].map(normalise_team)
results["away_team"] = results["away_team"].map(normalise_team)
results["url_key"] = results["match_href"].map(normalise_url)
results["team_key"] = list(zip(results["round"], results["home_team"], results["away_team"]))

url_to_index = {v: i for i, v in results["url_key"].items() if v}
team_to_index = {v: i for i, v in results["team_key"].items()}

results["status"] = "Upcoming"
results["home_score"] = pd.NA
results["away_score"] = pd.NA

unmatched = []

for row in completed.itertuples(index=False):
    idx = None

    if row.url_key and row.url_key in url_to_index:
        idx = url_to_index[row.url_key]
    elif row.team_key in team_to_index:
        idx = team_to_index[row.team_key]

    if idx is None:
        unmatched.append(
            {
                "round": row.round,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "home_score": row.home_score,
                "away_score": row.away_score,
            }
        )
        continue

    results.at[idx, "status"] = "Full Time"
    results.at[idx, "home_score"] = int(row.home_score)
    results.at[idx, "away_score"] = int(row.away_score)

if unmatched:
    print(pd.DataFrame(unmatched).to_string(index=False))
    raise SystemExit("Some DB matches could not be matched to the app fixture list.")

results = results.drop(columns=["url_key", "team_key"])

completed_mask = results["status"].eq("Full Time")
completed_count = int(completed_mask.sum())
remaining_count = int((~completed_mask).sum())

if completed_count != len(completed):
    raise SystemExit(
        f"Matched {completed_count} rows but DB has {len(completed)} completed matches."
    )

round_summary = (
    results.groupby("round")["status"]
    .agg(
        completed=lambda s: int(s.eq("Full Time").sum()),
        total="size",
    )
    .reset_index()
)

complete_rounds = (
    round_summary.loc[
        round_summary["completed"].eq(round_summary["total"]),
        "round",
    ]
    .astype(int)
    .tolist()
)

partial_rounds = (
    round_summary.loc[
        (round_summary["completed"] > 0)
        & (round_summary["completed"] < round_summary["total"]),
        "round",
    ]
    .astype(int)
    .tolist()
)

last_complete_round = 0
for r in sorted(complete_rounds):
    if all(k in complete_rounds for k in range(1, r + 1)):
        last_complete_round = r

data_cutoff = (
    f"Rounds 1-{last_complete_round} complete"
    if last_complete_round
    else "No complete rounds"
)

begun_rounds = set(
    round_summary.loc[round_summary["completed"] > 0, "round"].astype(int)
)

byes["round"] = byes["round"].astype(int)
byes["team"] = byes["team"].map(normalise_team)
byes["credited"] = byes["round"].map(
    lambda r: "true" if int(r) in begun_rounds else "false"
)

byes["bye_points"] = byes["credited"].map(
    lambda credited: 2 if str(credited).lower() == "true" else 0
)

credited_bye_count = int(
    byes["credited"].astype(str).str.lower().eq("true").sum()
)

results.to_csv(RESULTS_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
byes.to_csv(BYES_PATH, index=False, quoting=csv.QUOTE_MINIMAL)

meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}

meta.update(
    {
        "season": 2026,
        "data_cutoff": data_cutoff,
        "fixture_count": int(len(results)),
        "completed_match_count": completed_count,
        "remaining_fixture_count": remaining_count,
        "credited_bye_count": credited_bye_count,
        "complete_rounds": complete_rounds,
        "partial_rounds": partial_rounds,
        "last_complete_round": int(last_complete_round),
        "database_completed_match_count": int(len(completed)),
        "database_path": str(DB_PATH),
        "generated_at_australia_sydney": datetime.now(
            ZoneInfo("Australia/Sydney")
        ).isoformat(),
        "upstream_scraper_commit": git_commit(SCRAPER_PATH),
        "export_method": "merge current SQLite completed matches onto predictor fixture schedule",
    }
)

META_PATH.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

print("Merged database results into app schedule.")
print(f"Database completed matches: {len(completed)}")
print(f"App fixtures:                {len(results)}")
print(f"Completed matches:           {completed_count}")
print(f"Remaining fixtures:          {remaining_count}")
print(f"Credited byes:               {credited_bye_count}")
print(f"Data cutoff:                 {data_cutoff}")
print(f"Partial rounds:              {partial_rounds}")
PY

echo
echo "============================================================"
echo "3. REVIEWING AND COMMITTING DATA ONLY"
echo "============================================================"

git diff --check

git status -sb
git diff --stat

git add \
    data/2026/nrl_results.csv \
    data/2026/nrl_byes.csv \
    data/2026/nrl_results.meta.json

if git diff --cached --quiet; then
    echo "No app data changed. Nothing to commit."
    exit 0
fi

CUTOFF="$("$PYTHON" - <<'PY'
import json
from pathlib import Path
meta = json.loads(Path("data/2026/nrl_results.meta.json").read_text())
print(meta.get("data_cutoff", "latest database"))
PY
)"

git commit -m "Update predictor data through ${CUTOFF}"
git push origin main

echo
echo "============================================================"
echo "APP DATA UPDATE PUSHED SUCCESSFULLY"
echo "============================================================"

"$PYTHON" - <<'PY'
import json
from pathlib import Path

meta = json.loads(Path("data/2026/nrl_results.meta.json").read_text())

print()
print("Live app should redeploy with:")
print("  Data cutoff:       ", meta.get("data_cutoff"))
print("  Completed matches: ", meta.get("completed_match_count"))
print("  Remaining fixtures:", meta.get("remaining_fixture_count"))
print("  Credited byes:     ", meta.get("credited_bye_count"))
print("  Partial rounds:    ", meta.get("partial_rounds"))
PY
