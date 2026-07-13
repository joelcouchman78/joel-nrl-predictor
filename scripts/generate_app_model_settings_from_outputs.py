from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json
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

MODEL_ROOT = SCRAPER_ROOT / "out/analysis/bayesian_team_strength_2026_r1_r19"
STRENGTH_PATH = (
    MODEL_ROOT
    / "recency_weighted_raw_strength"
    / "team_strength_posterior_recency_weighted.csv"
)
PARAMETERS_PATH = MODEL_ROOT / "recency_weighted_raw_strength" / "model_parameters.csv"
VOLATILITY_PATH = (
    MODEL_ROOT
    / "volatility_variability_analysis"
    / "app_strength_uncertainty_mapping.csv"
)
INPUT_PATH = MODEL_ROOT / "model_input_matches.csv"

OUTPUT_PATH = PREDICTOR_ROOT / "data/2026/app_model_settings.json"


def git_commit(repo: Path) -> str | None:
    process = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode == 0:
        return process.stdout.strip()
    return None


def read_parameter_mean(path: Path, parameter: str, fallback: float) -> float:
    if not path.exists():
        return fallback

    frame = pd.read_csv(path)
    if not {"parameter", "mean"}.issubset(frame.columns):
        return fallback

    rows = frame.loc[frame["parameter"] == parameter]
    if rows.empty:
        return fallback

    return float(rows.iloc[0]["mean"])


def validate_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def main() -> None:
    validate_file(STRENGTH_PATH)
    validate_file(PARAMETERS_PATH)
    validate_file(VOLATILITY_PATH)
    validate_file(INPUT_PATH)

    strengths = pd.read_csv(STRENGTH_PATH)
    required_strength_columns = {
        "team",
        "recency_weighted_mean_strength",
        "recency_weighted_sd_strength",
    }
    missing_strength_columns = required_strength_columns - set(strengths.columns)
    if missing_strength_columns:
        raise RuntimeError(
            f"Missing strength columns in {STRENGTH_PATH}: "
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

    model_inputs = pd.read_csv(INPUT_PATH)
    completed_match_count = int(len(model_inputs))
    if completed_match_count != 140:
        raise RuntimeError(
            f"Expected 140 R1-R19 model-input matches, got {completed_match_count}"
        )

    home_advantage_points = read_parameter_mean(
        PARAMETERS_PATH,
        "listed_home_advantage",
        fallback=0.0,
    )
    match_randomness_points = read_parameter_mean(
        PARAMETERS_PATH,
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
        "model_run": "bayesian_team_strength_2026_r1_r19",
        "source_strength_file": str(STRENGTH_PATH),
        "source_strength_mean_column": "recency_weighted_mean_strength",
        "source_strength_sd_column": "recency_weighted_sd_strength",
        "source_parameters_file": str(PARAMETERS_PATH),
        "source_volatility_file": str(VOLATILITY_PATH),
        "source_database_path": str(SCRAPER_ROOT / "data/smoke/season_2026_r1_r13/nrl.db"),
        "source_scraper_commit": git_commit(SCRAPER_ROOT),
        "source_predictor_commit": git_commit(PREDICTOR_ROOT),
        "season": 2026,
        "round_range": "Rounds 1-19",
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
    print("source_strength_file:", settings["source_strength_file"])
    print("source_strength_mean_column:", settings["source_strength_mean_column"])
    print("source_strength_sd_column:", settings["source_strength_sd_column"])
    print("home_advantage_points:", settings["home_advantage_points"])
    print("match_randomness_points:", settings["match_randomness_points"])
    print("completed_match_count:", settings["completed_match_count"])
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
