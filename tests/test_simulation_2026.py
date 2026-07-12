from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from predictor.ladder import (
    parse_boolean,
    completed_results,
    EXPECTED_TEAMS,
    compute_ladder,
    load_byes_csv,
    load_results_csv,
)

from predictor.simulation import (
    SimulationParameters,
    apply_match_result,
    build_point_team_priors,
    build_team_priors,
    differential_per_game,
    estimate_total_points,
    future_bye_counts,
    future_fixtures,
    initialise_final_state,
    run_simulations,
    scores_from_total_and_margin,
    simulate_once,
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


def load_inputs():
    return (
        load_results_csv(RESULTS_PATH),
        load_byes_csv(BYES_PATH),
    )


def test_future_schedule_contract() -> None:
    results, byes = load_inputs()

    fixtures = future_fixtures(
        results
    )

    future_byes = future_bye_counts(
        byes
    )

    assert len(fixtures) + len(completed_results(results)) == len(results)
    credited_bye_count = int(
        byes["credited"].map(parse_boolean).sum()
    )
    assert sum(future_byes.values()) == len(byes) - credited_bye_count

    all_bye_counts = (
        byes["team"]
        .value_counts()
        .to_dict()
    )

    assert all_bye_counts == {
        team: 3
        for team in EXPECTED_TEAMS
    }


def test_strength_baseline_uses_diff_per_game() -> None:
    ladder = pd.DataFrame(
        [
            {
                "Team": "A",
                "GP": 2,
                "Diff": 20,
            },
            {
                "Team": "B",
                "GP": 10,
                "Diff": 20,
            },
            {
                "Team": "C",
                "GP": 0,
                "Diff": 0,
            },
        ]
    )

    rates = differential_per_game(
        ladder
    )

    assert rates["A"] == 10.0
    assert rates["B"] == 2.0
    assert rates["C"] == 0.0


def test_point_based_priors_use_margin_points_directly() -> None:
    strengths = {
        team: 0.0
        for team in EXPECTED_TEAMS
    }
    strengths["Panthers"] = 12.0
    strengths["Wests Tigers"] = -8.0

    sds = {
        team: 4.0
        for team in EXPECTED_TEAMS
    }

    priors = build_point_team_priors(
        team_strength_points=strengths,
        team_strength_sd_points=sds,
    )

    assert priors["Panthers"].mean == 12.0
    assert priors["Wests Tigers"].mean == -8.0
    assert priors["Panthers"].std == 4.0


def test_score_conversion_preserves_margin() -> None:
    assert scores_from_total_and_margin(
        40,
        -12,
    ) == (14, 26)

    assert scores_from_total_and_margin(
        41,
        0,
    ) == (21, 21)

    assert scores_from_total_and_margin(
        10,
        20,
    ) == (20, 0)

    for total, margin in [
        (42, 7),
        (39, -8),
        (0, 0),
        (5, -12),
    ]:
        home, away = (
            scores_from_total_and_margin(
                total,
                margin,
            )
        )

        assert home >= 0
        assert away >= 0
        assert (
            home - away
            == int(round(margin))
        )


def test_away_win_updates_signed_state() -> None:
    state = {
        "Home": {
            "Team": "Home",
            "GP": 0,
            "W": 0,
            "D": 0,
            "L": 0,
            "PF": 0,
            "PA": 0,
            "Diff": 0,
            "Byes": 0,
            "CompPts": 0,
        },
        "Away": {
            "Team": "Away",
            "GP": 0,
            "W": 0,
            "D": 0,
            "L": 0,
            "PF": 0,
            "PA": 0,
            "Diff": 0,
            "Byes": 0,
            "CompPts": 0,
        },
    }

    apply_match_result(
        state,
        "Home",
        "Away",
        8,
        20,
    )

    state["Home"]["Diff"] = (
        state["Home"]["PF"]
        - state["Home"]["PA"]
    )

    state["Away"]["Diff"] = (
        state["Away"]["PF"]
        - state["Away"]["PA"]
    )

    assert state["Home"]["Diff"] == -12
    assert state["Away"]["Diff"] == 12
    assert state["Home"]["L"] == 1
    assert state["Away"]["W"] == 1
    assert state["Away"]["CompPts"] == 2


def test_single_simulation_final_invariants() -> None:
    results, byes = load_inputs()

    current = compute_ladder(
        results,
        byes,
    )

    fixtures = future_fixtures(
        results
    )

    outstanding_byes = (
        future_bye_counts(byes)
    )

    parameters = SimulationParameters()

    priors = build_team_priors(
        current,
        strength_adjustment_scale=(
            parameters
            .strength_adjustment_scale
        ),
    )

    total_mean, total_sigma = (
        estimate_total_points(
            results
        )
    )

    final = simulate_once(
        current_ladder=current,
        fixtures=fixtures,
        future_byes=(
            outstanding_byes
        ),
        priors=priors,
        rng=np.random.default_rng(
            20260609
        ),
        parameters=parameters,
        expected_total=total_mean,
        total_sigma=total_sigma,
    )

    assert set(final["GP"]) == {24}
    assert set(final["Byes"]) == {3}

    assert int(
        final["CompPts"].sum()
    ) == 510

    assert int(final["PF"].sum()) == int(
        final["PA"].sum()
    )

    assert int(final["Diff"].sum()) == 0

    assert int(final["W"].sum()) == int(
        final["L"].sum()
    )

    assert (
        final["W"]
        + final["D"]
        + final["L"]
        == final["GP"]
    ).all()


def test_initial_state_adds_future_byes_only() -> None:
    results, byes = load_inputs()

    current = compute_ladder(
        results,
        byes,
    )

    outstanding = future_bye_counts(
        byes
    )

    state = initialise_final_state(
        current,
        outstanding,
    )

    assert {
        row["Byes"]
        for row in state.values()
    } == {3}

    current_points = int(
        current["CompPts"].sum()
    )

    final_start_points = sum(
        row["CompPts"]
        for row in state.values()
    )

    assert (
        final_start_points
        - current_points
        == 2 * sum(outstanding.values())
    )


def test_reproducible_seed() -> None:
    results, byes = load_inputs()

    first = run_simulations(
        results=results,
        byes=byes,
        simulation_count=25,
        seed=12345,
    )

    second = run_simulations(
        results=results,
        byes=byes,
        simulation_count=25,
        seed=12345,
    )

    assert first.orders == second.orders
    assert first.summary.equals(
        second.summary
    )

    for team in EXPECTED_TEAMS:
        assert np.array_equal(
            first.positions[team],
            second.positions[team],
        )


def test_position_distribution_occupancy() -> None:
    results, byes = load_inputs()

    simulations = 40

    result = run_simulations(
        results=results,
        byes=byes,
        simulation_count=simulations,
        seed=67890,
    )

    assert result.fixture_count + len(completed_results(results)) == len(results)
    assert result.simulation_count == (
        simulations
    )

    occupancy = Counter()

    for team in EXPECTED_TEAMS:
        values = result.positions[team]

        assert len(values) == simulations
        assert values.min() >= 1
        assert values.max() <= 17

        occupancy.update(
            values.tolist()
        )

    assert occupancy == Counter(
        {
            position: simulations
            for position in range(1, 18)
        }
    )

    assert round(
        result.summary[
            "Minor Prem. %"
        ].sum(),
        10,
    ) == 100.0

    assert round(
        result.summary[
            "Wooden Spoon %"
        ].sum(),
        10,
    ) == 100.0
