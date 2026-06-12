from pathlib import Path

import pandas as pd

from predictor.ladder import (
    EXPECTED_TEAMS,
    build_ladder_from_files,
    compute_ladder,
    credited_byes,
    load_byes_csv,
    load_results_csv,
)


ROOT = Path(
    "/Users/joelcouchman/Projects/"
    "joel-nrl-predictor"
)

RESULTS_PATH = (
    ROOT
    / "data"
    / "2026"
    / "nrl_results.csv"
)

BYES_PATH = (
    ROOT
    / "data"
    / "2026"
    / "nrl_byes.csv"
)

META_PATH = (
    ROOT
    / "data"
    / "2026"
    / "nrl_results.meta.json"
)


def test_2026_input_contract() -> None:
    results = load_results_csv(
        RESULTS_PATH
    )

    byes = load_byes_csv(
        BYES_PATH
    )

    assert len(results) == 204
    assert len(byes) == 51
    assert META_PATH.exists()

    assert (
        results["status"]
        .value_counts()
        .to_dict()
        == {
            "Upcoming": 95,
            "Full Time": 109,
        }
    )

    assert set(
        results["home_team"]
    ) | set(
        results["away_team"]
    ) == set(EXPECTED_TEAMS)


def test_credited_bye_contract() -> None:
    byes = load_byes_csv(
        BYES_PATH
    )

    credited = credited_byes(byes)

    assert len(credited) == 22
    assert int(
        credited["bye_points"].sum()
    ) == 44

    credited_counts = (
        credited["team"]
        .value_counts()
        .to_dict()
    )

    assert credited_counts == {
        "Roosters": 2,
        "Rabbitohs": 2,
        "Dolphins": 2,
        "Sharks": 2,
        "Wests Tigers": 2,
        "Panthers": 1,
        "Warriors": 2,
        "Knights": 1,
        "Sea Eagles": 1,
        "Broncos": 1,
        "Raiders": 1,
        "Bulldogs": 1,
        "Eels": 1,
        "Titans": 2,
        "Dragons": 1,
    }


def test_round_15_partial_ladder_snapshot() -> None:
    ladder = build_ladder_from_files(
        RESULTS_PATH,
        BYES_PATH,
    )

    expected_order = [
        "Panthers",
        "Warriors",
        "Dolphins",
        "Roosters",
        "Sea Eagles",
        "Sharks",
        "Knights",
        "Rabbitohs",
        "Cowboys",
        "Wests Tigers",
        "Storm",
        "Broncos",
        "Bulldogs",
        "Raiders",
        "Titans",
        "Eels",
        "Dragons",
    ]

    assert ladder["Team"].tolist() == (
        expected_order
    )

    expected_points = {
        "Panthers": 26,
        "Warriors": 22,
        "Roosters": 20,
        "Sea Eagles": 18,
        "Dolphins": 20,
        "Sharks": 18,
        "Knights": 18,
        "Rabbitohs": 16,
        "Cowboys": 16,
        "Wests Tigers": 16,
        "Storm": 12,
        "Broncos": 12,
        "Bulldogs": 12,
        "Raiders": 12,
        "Titans": 10,
        "Eels": 10,
        "Dragons": 4,
    }

    actual_points = dict(
        zip(
            ladder["Team"],
            ladder["CompPts"],
        )
    )

    assert actual_points == expected_points

    assert int(ladder["GP"].sum()) == 218
    assert int(ladder["PF"].sum()) == int(
        ladder["PA"].sum()
    )
    assert int(ladder["Diff"].sum()) == 0
    assert int(ladder["W"].sum()) == int(
        ladder["L"].sum()
    )


def test_bye_adds_points_not_game_played() -> None:
    results = pd.DataFrame(
        [
            {
                "season": 2026,
                "round": 1,
                "status": "Full Time",
                "home_team": "Panthers",
                "away_team": "Dragons",
                "home_score": 20,
                "away_score": 10,
            }
        ]
    )

    byes = pd.DataFrame(
        [
            {
                "season": 2026,
                "round": 1,
                "team": "Warriors",
                "credited": True,
                "bye_points": 2,
            }
        ]
    )

    ladder = compute_ladder(
        results,
        byes,
        teams=[
            "Panthers",
            "Dragons",
            "Warriors",
        ],
    ).set_index("Team")

    assert ladder.loc[
        "Warriors",
        "GP",
    ] == 0

    assert ladder.loc[
        "Warriors",
        "Byes",
    ] == 1

    assert ladder.loc[
        "Warriors",
        "CompPts",
    ] == 2


def test_away_win_updates_scores_and_diff() -> None:
    results = pd.DataFrame(
        [
            {
                "season": 2026,
                "round": 1,
                "status": "Full Time",
                "home_team": "Dragons",
                "away_team": "Panthers",
                "home_score": 8,
                "away_score": 20,
            }
        ]
    )

    byes = pd.DataFrame(
        columns=[
            "season",
            "round",
            "team",
            "credited",
            "bye_points",
        ]
    )

    ladder = compute_ladder(
        results,
        byes,
        teams=[
            "Dragons",
            "Panthers",
        ],
    ).set_index("Team")

    assert ladder.loc[
        "Dragons",
        "PF",
    ] == 8

    assert ladder.loc[
        "Dragons",
        "PA",
    ] == 20

    assert ladder.loc[
        "Dragons",
        "Diff",
    ] == -12

    assert ladder.loc[
        "Panthers",
        "PF",
    ] == 20

    assert ladder.loc[
        "Panthers",
        "PA",
    ] == 8

    assert ladder.loc[
        "Panthers",
        "Diff",
    ] == 12

    assert ladder.loc[
        "Panthers",
        "CompPts",
    ] == 2
