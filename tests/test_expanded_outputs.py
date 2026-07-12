from __future__ import annotations

import numpy as np
import pandas as pd

from predictor.outputs import (
    fixture_probability_table,
    monte_carlo_error_table,
    position_probability_table,
    scenario_mask,
    scenario_team_summary,
    team_fixture_leverage_table,
    team_wins_needed_table,
)
from predictor.simulation import run_simulations


def synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    teams = [f"Team {index}" for index in range(1, 11)]
    rows = []

    completed_pairs = [
        ("Team 1", "Team 2", 24, 12),
        ("Team 3", "Team 4", 18, 16),
        ("Team 5", "Team 6", 10, 20),
        ("Team 7", "Team 8", 14, 14),
        ("Team 9", "Team 10", 22, 8),
    ]
    for round_number, (home, away, home_score, away_score) in enumerate(
        completed_pairs, start=1
    ):
        rows.append(
            {
                "season": 2026,
                "round": round_number,
                "status": "Full Time",
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
            }
        )

    future_pairs = [
        ("Team 1", "Team 3"),
        ("Team 2", "Team 4"),
        ("Team 5", "Team 7"),
        ("Team 6", "Team 8"),
        ("Team 9", "Team 1"),
        ("Team 10", "Team 2"),
        ("Team 3", "Team 5"),
        ("Team 4", "Team 6"),
        ("Team 7", "Team 9"),
        ("Team 8", "Team 10"),
    ]
    for offset, (home, away) in enumerate(future_pairs, start=6):
        rows.append(
            {
                "season": 2026,
                "round": offset,
                "status": "Upcoming",
                "home_team": home,
                "away_team": away,
                "home_score": np.nan,
                "away_score": np.nan,
            }
        )

    results = pd.DataFrame(rows)
    byes = pd.DataFrame(
        columns=["season", "round", "team", "credited", "bye_points"]
    )
    return results, byes, teams


def test_expanded_result_contract_and_reproducibility() -> None:
    results, byes, teams = synthetic_inputs()
    first = run_simulations(
        results,
        byes,
        simulation_count=300,
        seed=1234,
        teams=teams,
    )
    second = run_simulations(
        results,
        byes,
        simulation_count=300,
        seed=1234,
        teams=teams,
    )

    assert first.orders == second.orders
    assert first.fixture_margins.shape == (300, 10)
    assert np.array_equal(first.fixture_margins, second.fixture_margins)
    assert first.top8_cutoff_points.shape == (300,)

    required_summary_columns = {
        "Top 2 %",
        "Top 4 %",
        "5th-8th %",
        "Top 8 %",
        "Bottom 4 %",
        "Modal Pos",
        "Position SD",
        "Position Variance",
        "Pos P10",
        "Pos P90",
        "Mean CompPts",
        "CompPts P10",
        "CompPts P90",
        "Mean Diff",
        "Mean Future Wins",
    }
    assert required_summary_columns.issubset(first.summary.columns)

    for team in teams:
        assert len(first.positions[team]) == 300
        assert len(first.competition_points[team]) == 300
        assert len(first.differentials[team]) == 300
        assert len(first.future_wins[team]) == 300
        assert np.array_equal(
            first.future_wins[team],
            first.final_wins[team] - first.current_wins[team],
        )

    for simulation_index, order in enumerate(first.orders):
        eighth_team = order[7]
        assert (
            first.top8_cutoff_points[simulation_index]
            == first.competition_points[eighth_team][simulation_index]
        )


def test_position_probabilities_and_fixture_probabilities_sum_to_100() -> None:
    results, byes, teams = synthetic_inputs()
    result = run_simulations(
        results,
        byes,
        simulation_count=400,
        seed=99,
        teams=teams,
    )

    positions = position_probability_table(result)
    assert positions.shape == (10, 11)
    row_sums = positions.drop(columns="Team").sum(axis=1)
    assert np.allclose(row_sums.to_numpy(), 100.0)

    fixtures = fixture_probability_table(result)
    outcome_sums = (
        fixtures["Home Win %"]
        + fixtures["Draw %"]
        + fixtures["Away Win %"]
    )
    assert np.allclose(outcome_sums.to_numpy(), 100.0)


def test_conditional_outputs_and_scenario_filter() -> None:
    results, byes, teams = synthetic_inputs()
    result = run_simulations(
        results,
        byes,
        simulation_count=1000,
        seed=2026,
        teams=teams,
    )

    wins = team_wins_needed_table(result, "Team 1")
    assert not wins.empty
    assert wins["Samples"].sum() == 1000
    assert wins["Simulation Share %"].sum() == pytest_approx(100.0)

    leverage = team_fixture_leverage_table(result, "Team 1")
    assert len(leverage) == 2
    assert set(leverage["Venue"]) == {"Home", "Away"}

    mask = scenario_mask(result, {0: "Home win", 4: "Home win"})
    assert mask.dtype == bool
    assert mask.shape == (1000,)
    assert np.all(result.fixture_margins[mask, 0] > 0)
    assert np.all(result.fixture_margins[mask, 4] > 0)

    summary = scenario_team_summary(result, "Team 1", mask)
    assert summary["Samples"] == int(mask.sum())
    assert 0.0 <= summary["Top 8 %"] <= 100.0


def test_monte_carlo_error_table_is_finite() -> None:
    results, byes, teams = synthetic_inputs()
    result = run_simulations(
        results,
        byes,
        simulation_count=250,
        seed=7,
        teams=teams,
    )
    table = monte_carlo_error_table(result)
    assert len(table) == len(teams)
    assert np.isfinite(table["Top 4 MC SE (pp)"]).all()
    assert np.isfinite(table["Top 8 MC SE (pp)"]).all()


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, abs=1e-9)
