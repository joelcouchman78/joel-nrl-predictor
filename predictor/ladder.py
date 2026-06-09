from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


COMPLETED_STATUSES = {
    "full time",
    "ft",
    "final",
    "finished",
    "complete",
}

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

RESULT_COLUMNS = {
    "season",
    "round",
    "status",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
}

BYE_COLUMNS = {
    "season",
    "round",
    "team",
    "credited",
    "bye_points",
}

LADDER_COLUMNS = [
    "Team",
    "GP",
    "W",
    "D",
    "L",
    "PF",
    "PA",
    "Diff",
    "Byes",
    "CompPts",
]


def normalise_status(value: object) -> str:
    return str(value or "").strip().lower()


def status_is_completed(value: object) -> bool:
    return normalise_status(value) in COMPLETED_STATUSES


def parse_boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value or "").strip().lower()

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n", ""}:
        return False

    raise ValueError(
        f"Cannot interpret boolean value: {value!r}"
    )


def validate_required_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(frame.columns))

    if missing:
        raise ValueError(
            f"{label} is missing required columns: "
            f"{missing}"
        )


def load_results_csv(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)

    validate_required_columns(
        frame,
        RESULT_COLUMNS,
        "results CSV",
    )

    return frame


def load_byes_csv(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)

    validate_required_columns(
        frame,
        BYE_COLUMNS,
        "byes CSV",
    )

    return frame


def completed_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    validate_required_columns(
        results,
        RESULT_COLUMNS,
        "results data",
    )

    completed = results.loc[
        results["status"].map(
            status_is_completed
        )
    ].copy()

    completed["home_score"] = pd.to_numeric(
        completed["home_score"],
        errors="raise",
    ).astype(int)

    completed["away_score"] = pd.to_numeric(
        completed["away_score"],
        errors="raise",
    ).astype(int)

    if (
        completed["home_team"].isna().any()
        or completed["away_team"].isna().any()
    ):
        raise ValueError(
            "Completed results contain a missing team"
        )

    if (
        completed["home_score"].lt(0).any()
        or completed["away_score"].lt(0).any()
    ):
        raise ValueError(
            "Completed results contain a negative score"
        )

    identity_columns = [
        "round",
        "home_team",
        "away_team",
    ]

    duplicate_mask = completed.duplicated(
        subset=identity_columns,
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = completed.loc[
            duplicate_mask,
            identity_columns,
        ].to_dict("records")

        raise ValueError(
            "Completed results contain duplicate "
            f"fixtures: {duplicates[:10]}"
        )

    return completed


def credited_byes(
    byes: pd.DataFrame,
) -> pd.DataFrame:
    validate_required_columns(
        byes,
        BYE_COLUMNS,
        "bye data",
    )

    frame = byes.copy()

    frame["credited"] = frame[
        "credited"
    ].map(parse_boolean)

    frame["bye_points"] = pd.to_numeric(
        frame["bye_points"],
        errors="raise",
    ).astype(int)

    credited = frame.loc[
        frame["credited"]
    ].copy()

    if credited["team"].isna().any():
        raise ValueError(
            "Credited byes contain a missing team"
        )

    if credited["bye_points"].lt(0).any():
        raise ValueError(
            "Credited byes contain negative points"
        )

    duplicate_mask = credited.duplicated(
        subset=["round", "team"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = credited.loc[
            duplicate_mask,
            ["round", "team"],
        ].to_dict("records")

        raise ValueError(
            "Credited byes contain duplicates: "
            f"{duplicates[:10]}"
        )

    return credited


def sort_ladder(
    ladder: pd.DataFrame,
) -> pd.DataFrame:
    """
    Interim compatibility ordering.

    The existing application uses competition
    points, differential and points for. The exact
    2026 NRL tertiary countback will be verified
    before the simulation engine is finalised.
    """
    return (
        ladder.sort_values(
            by=[
                "CompPts",
                "Diff",
                "PF",
                "Team",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def compute_ladder(
    results: pd.DataFrame,
    byes: pd.DataFrame,
    teams: Iterable[str] = EXPECTED_TEAMS,
) -> pd.DataFrame:
    team_list = list(teams)

    if len(team_list) != len(set(team_list)):
        raise ValueError(
            "Team list contains duplicates"
        )

    team_set = set(team_list)

    stats = {
        team: {
            "Team": team,
            "GP": 0,
            "W": 0,
            "D": 0,
            "L": 0,
            "PF": 0,
            "PA": 0,
            "Diff": 0,
            "Byes": 0,
            "CompPts": 0,
        }
        for team in team_list
    }

    completed = completed_results(results)

    result_teams = set(
        completed["home_team"]
    ) | set(
        completed["away_team"]
    )

    unknown_result_teams = sorted(
        result_teams - team_set
    )

    if unknown_result_teams:
        raise ValueError(
            "Completed results contain unknown "
            f"teams: {unknown_result_teams}"
        )

    for row in completed.itertuples(
        index=False
    ):
        home = row.home_team
        away = row.away_team

        home_score = int(row.home_score)
        away_score = int(row.away_score)

        stats[home]["GP"] += 1
        stats[away]["GP"] += 1

        stats[home]["PF"] += home_score
        stats[home]["PA"] += away_score

        stats[away]["PF"] += away_score
        stats[away]["PA"] += home_score

        if home_score > away_score:
            stats[home]["W"] += 1
            stats[away]["L"] += 1
            stats[home]["CompPts"] += 2

        elif away_score > home_score:
            stats[away]["W"] += 1
            stats[home]["L"] += 1
            stats[away]["CompPts"] += 2

        else:
            stats[home]["D"] += 1
            stats[away]["D"] += 1
            stats[home]["CompPts"] += 1
            stats[away]["CompPts"] += 1

    credited = credited_byes(byes)

    unknown_bye_teams = sorted(
        set(credited["team"]) - team_set
    )

    if unknown_bye_teams:
        raise ValueError(
            "Credited byes contain unknown "
            f"teams: {unknown_bye_teams}"
        )

    for row in credited.itertuples(
        index=False
    ):
        team = row.team

        stats[team]["Byes"] += 1
        stats[team]["CompPts"] += int(
            row.bye_points
        )

    for team in team_list:
        stats[team]["Diff"] = (
            stats[team]["PF"]
            - stats[team]["PA"]
        )

        if (
            stats[team]["W"]
            + stats[team]["D"]
            + stats[team]["L"]
            != stats[team]["GP"]
        ):
            raise AssertionError(
                f"W/D/L invariant failed for {team}"
            )

    ladder = pd.DataFrame(
        stats.values(),
        columns=LADDER_COLUMNS,
    )

    if int(ladder["GP"].sum()) != (
        2 * len(completed)
    ):
        raise AssertionError(
            "Total GP does not equal twice "
            "the completed match count"
        )

    if int(ladder["W"].sum()) != int(
        ladder["L"].sum()
    ):
        raise AssertionError(
            "Total wins do not equal total losses"
        )

    if int(ladder["PF"].sum()) != int(
        ladder["PA"].sum()
    ):
        raise AssertionError(
            "Total PF does not equal total PA"
        )

    if int(ladder["Diff"].sum()) != 0:
        raise AssertionError(
            "Total differential is not zero"
        )

    return sort_ladder(ladder)


def build_ladder_from_files(
    results_path: Path | str,
    byes_path: Path | str,
    teams: Iterable[str] = EXPECTED_TEAMS,
) -> pd.DataFrame:
    return compute_ladder(
        load_results_csv(results_path),
        load_byes_csv(byes_path),
        teams,
    )
