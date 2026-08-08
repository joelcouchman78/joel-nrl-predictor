from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import re
import subprocess

import pandas as pd


EXPECTED_TEAMS = [
    "Broncos",
    "Bulldogs",
    "Cowboys",
    "Dolphins",
    "Dragons",
    "Eels",
    "Knights",
    "Panthers",
    "Rabbitohs",
    "Raiders",
    "Roosters",
    "Sea Eagles",
    "Sharks",
    "Storm",
    "Titans",
    "Warriors",
    "Wests Tigers",
]

PREDICTOR_ROOT = Path("/Users/joelcouchman/Projects/joel-nrl-predictor")
SCRAPER_ROOT = Path("/Users/joelcouchman/total_nrl_scraper")
ANALYSIS_ROOT = SCRAPER_ROOT / "out/analysis"
OUTPUT_PATH = PREDICTOR_ROOT / "data/2026/app_model_settings.json"


def git_commit(repo: Path) -> str | None:
    process = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode == 0 and process.stdout.strip():
        return process.stdout.strip()
    return None


def model_sort_key(path: Path) -> tuple[int, int]:
    input_path = path / "model_input_matches.csv"
    match_count = -1
    if input_path.exists():
        try:
            match_count = len(pd.read_csv(input_path))
        except Exception:
            match_count = -1

    round_match = re.search(r"r1_r(\d+)", path.name)
    latest_round = int(round_match.group(1)) if round_match else -1
    return match_count, latest_round


def find_latest_model_root() -> Path:
    candidates = [
        path
        for path in ANALYSIS_ROOT.glob("bayesian_team_strength_2026_r1_r*")
        if (
            path.is_dir()
            and (
                path
                / "recency_weighted_raw_strength"
                / "team_strength_posterior_recency_weighted.csv"
            ).exists()
            and (path / "recency_weighted_raw_strength" / "model_parameters.csv").exists()
            and (path / "model_input_matches.csv").exists()
        )
    ]
    if not candidates:
        raise RuntimeError("No valid 2026 Bayesian team-strength model roots found.")

    return max(candidates, key=model_sort_key)


def read_parameter_mean(path: Path, parameter: str, fallback: float) -> float:
    frame = pd.read_csv(path)
    if not {"parameter", "mean"}.issubset(frame.columns):
        return fallback

    rows = frame.loc[frame["parameter"] == parameter]
    if rows.empty:
        return fallback

    return float(rows.iloc[0]["mean"])


def main() -> None:
    model_root = find_latest_model_root()

    strength_path = (
        model_root
        / "recency_weighted_raw_strength"
        / "team_strength_posterior_recency_weighted.csv"
    )
    parameters_path = model_root / "recency_weighted_raw_strength" / "model_parameters.csv"
    volatility_path = (
        model_root
        / "volatility_variability_analysis"
        / "app_strength_uncertainty_mapping.csv"
    )
    input_path = model_root / "model_input_matches.csv"

    strengths = pd.read_csv(strength_path)
    required_strength_columns = {
        "team",
        "recency_weighted_mean_strength",
        "recency_weighted_sd_strength",
    }
    missing_strength_columns = required_strength_columns - set(strengths.columns)
    if missing_strength_columns:
        raise RuntimeError(
            f"Missing strength columns in {strength_path}: "
            f"{sorted(missing_strength_columns)}"
        )

    teams_found = set(strengths["team"].astype(str))
    missing_teams = sorted(set(EXPECTED_TEAMS) - teams_found)
    extra_teams = sorted(teams_found - set(EXPECTED_TEAMS))
    if missing_teams or extra_teams:
        raise RuntimeError(
            f"Team mismatch. Missing={missing_teams}; extra={extra_teams}"
        )

    strengths = strengths.set_index("team").loc[EXPECTED_TEAMS]

    model_inputs = pd.read_csv(input_path)
    completed_match_count = int(len(model_inputs))
    if completed_match_count <= 156:
        raise RuntimeError(
            f"Latest model only has {completed_match_count} matches; "
            "refusing to overwrite the R21 calibration."
        )

    round_match = re.search(r"r1_r(\d+)", model_root.name)
    latest_round = int(round_match.group(1)) if round_match else None
    round_range = (
        f"Rounds 1-{latest_round}"
        if latest_round is not None
        else "Unknown round range"
    )

    home_advantage_points = read_parameter_mean(
        parameters_path,
        "listed_home_advantage",
        fallback=0.0,
    )
    match_randomness_points = read_parameter_mean(
        parameters_path,
        "sigma_margin",
        fallback=19.0,
    )

    team_strength_points = {
        team: round(float(strengths.loc[team, "recency_weighted_mean_strength"]), 3)
        for team in EXPECTED_TEAMS
    }
    team_strength_sd_points = {
        team: round(float(strengths.loc[team, "recency_weighted_sd_strength"]), 3)
        for team in EXPECTED_TEAMS
    }

    if all(abs(value) < 1e-9 for value in team_strength_points.values()):
        raise RuntimeError("Refusing to write all-zero team strengths.")

    if any(value <= 0 for value in team_strength_sd_points.values()):
        raise RuntimeError("Refusing to write non-positive team strength SDs.")

    settings = {
        "schema_version": 1,
        "rating_mode": "points",
        "model_run": model_root.name,
        "source_strength_file": str(strength_path),
        "source_strength_mean_column": "recency_weighted_mean_strength",
        "source_strength_sd_column": "recency_weighted_sd_strength",
        "source_parameters_file": str(parameters_path),
        "source_volatility_file": str(volatility_path) if volatility_path.exists() else None,
        "source_database_path": str(
            SCRAPER_ROOT / "data/smoke/season_2026_r1_r13/nrl.db"
        ),
        "source_scraper_commit": git_commit(SCRAPER_ROOT),
        "source_predictor_commit": git_commit(PREDICTOR_ROOT),
        "season": 2026,
        "round_range": round_range,
        "completed_match_count": completed_match_count,
        "current_strength_half_life_rounds": 6,
        "home_advantage_points": round(home_advantage_points, 3),
        "match_randomness_points": round(match_randomness_points, 3),
        "team_strength_points": team_strength_points,
        "team_strength_sd_points": team_strength_sd_points,
        "team_match_volatility_multipliers": {
            team: 1.0
            for team in EXPECTED_TEAMS
        },
        "generated_at_australia_perth": datetime.now(
            ZoneInfo("Australia/Perth")
        ).isoformat(timespec="seconds"),
    }

    OUTPUT_PATH.write_text(
        json.dumps(settings, indent=2) + "\n",
        encoding="utf-8",
    )

    print("WROTE", OUTPUT_PATH)
    print("model_run:", settings["model_run"])
    print("round_range:", settings["round_range"])
    print("completed_match_count:", settings["completed_match_count"])
    print("home_advantage_points:", settings["home_advantage_points"])
    print("match_randomness_points:", settings["match_randomness_points"])
    print("source_strength_file:", settings["source_strength_file"])
    print()
    print("TEAM STRENGTHS")
    for team, value in sorted(
        settings["team_strength_points"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        sd = settings["team_strength_sd_points"][team]
        print(f"{team:14s} {value:+.3f}   sd={sd:.3f}")


if __name__ == "__main__":
    main()
