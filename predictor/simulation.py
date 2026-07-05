from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from predictor.ladder import (
    BYE_COLUMNS,
    EXPECTED_TEAMS,
    LADDER_COLUMNS,
    RESULT_COLUMNS,
    completed_results,
    compute_ladder,
    parse_boolean,
    sort_ladder,
    status_is_completed,
    validate_required_columns,
)


@dataclass(frozen=True)
class Fixture:
    round: int
    home: str
    away: str


@dataclass(frozen=True)
class TeamPrior:
    mean: float
    std: float
    differential_per_game: float


@dataclass(frozen=True)
class SimulationParameters:
    home_advantage: float = 0.1
    alpha: float = 10.0
    margin_sigma: float = 16.0
    total_sigma: float | None = None
    strength_adjustment_scale: float = 1.5


@dataclass(frozen=True)
class SimulationResult:
    orders: tuple[tuple[str, ...], ...]
    positions: dict[str, np.ndarray]
    summary: pd.DataFrame
    seed: int | None
    fixture_count: int
    simulation_count: int
    estimated_total_mean: float
    estimated_total_sigma: float


def future_fixtures(
    results: pd.DataFrame,
    teams: Iterable[str] = EXPECTED_TEAMS,
) -> tuple[Fixture, ...]:
    validate_required_columns(
        results,
        RESULT_COLUMNS,
        "results data",
    )

    team_set = set(teams)

    future = results.loc[
        ~results["status"].map(
            status_is_completed
        )
    ].copy()

    future["round"] = pd.to_numeric(
        future["round"],
        errors="raise",
    ).astype(int)

    if (
        future["home_team"].isna().any()
        or future["away_team"].isna().any()
    ):
        raise ValueError(
            "Future fixtures contain a missing team"
        )

    fixture_teams = set(
        future["home_team"]
    ) | set(
        future["away_team"]
    )

    unknown = sorted(
        fixture_teams - team_set
    )

    if unknown:
        raise ValueError(
            "Future fixtures contain unknown "
            f"teams: {unknown}"
        )

    duplicate_mask = future.duplicated(
        subset=[
            "round",
            "home_team",
            "away_team",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = future.loc[
            duplicate_mask,
            [
                "round",
                "home_team",
                "away_team",
            ],
        ].to_dict("records")

        raise ValueError(
            "Future fixtures contain duplicates: "
            f"{duplicates[:10]}"
        )

    return tuple(
        Fixture(
            round=int(row.round),
            home=str(row.home_team),
            away=str(row.away_team),
        )
        for row in future.itertuples(
            index=False
        )
    )


def future_bye_counts(
    byes: pd.DataFrame,
    teams: Iterable[str] = EXPECTED_TEAMS,
) -> Counter[str]:
    validate_required_columns(
        byes,
        BYE_COLUMNS,
        "bye data",
    )

    team_set = set(teams)
    frame = byes.copy()

    frame["credited"] = frame[
        "credited"
    ].map(parse_boolean)

    if frame["team"].isna().any():
        raise ValueError(
            "Bye schedule contains a missing team"
        )

    unknown = sorted(
        set(frame["team"]) - team_set
    )

    if unknown:
        raise ValueError(
            "Bye schedule contains unknown "
            f"teams: {unknown}"
        )

    duplicate_mask = frame.duplicated(
        subset=["round", "team"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = frame.loc[
            duplicate_mask,
            ["round", "team"],
        ].to_dict("records")

        raise ValueError(
            "Bye schedule contains duplicates: "
            f"{duplicates[:10]}"
        )

    future = frame.loc[
        ~frame["credited"]
    ]

    return Counter(
        str(team)
        for team in future["team"]
    )


def differential_per_game(
    ladder: pd.DataFrame,
) -> pd.Series:
    required = {
        "Team",
        "GP",
        "Diff",
    }

    validate_required_columns(
        ladder,
        required,
        "ladder",
    )

    indexed = ladder.set_index(
        "Team"
    )

    games = pd.to_numeric(
        indexed["GP"],
        errors="raise",
    ).astype(float)

    differential = pd.to_numeric(
        indexed["Diff"],
        errors="raise",
    ).astype(float)

    values = np.divide(
        differential.to_numpy(),
        games.to_numpy(),
        out=np.zeros(
            len(indexed),
            dtype=float,
        ),
        where=games.to_numpy() > 0,
    )

    return pd.Series(
        values,
        index=indexed.index,
        name="DiffPerGame",
        dtype=float,
    )


def _rating_value(
    ratings: Mapping[str, float] | None,
    team: str,
    default: float,
    label: str,
) -> float:
    if ratings is None:
        value = default
    else:
        value = float(
            ratings.get(
                team,
                default,
            )
        )

    if not 0.0 <= value <= 10.0:
        raise ValueError(
            f"{label} for {team} must be "
            f"between 0 and 10, found {value}"
        )

    return value


def build_team_priors(
    ladder: pd.DataFrame,
    strength_ratings: (
        Mapping[str, float] | None
    ) = None,
    variability_ratings: (
        Mapping[str, float] | None
    ) = None,
    strength_adjustment_scale: float = 1.5,
) -> dict[str, TeamPrior]:
    rates = differential_per_game(
        ladder
    )

    mean_rate = float(rates.mean())
    std_rate = float(
        rates.std(ddof=0)
    )

    if not np.isfinite(std_rate):
        std_rate = 0.0

    if std_rate <= 0.0:
        std_rate = 1.0

    priors = {}

    for team, rate in rates.items():
        strength_rating = _rating_value(
            strength_ratings,
            team,
            5.0,
            "strength rating",
        )

        variability_rating = _rating_value(
            variability_ratings,
            team,
            5.0,
            "variability rating",
        )

        base = (
            float(rate) - mean_rate
        ) / std_rate

        strength_adjustment = (
            (strength_rating - 5.0)
            / 5.0
            * strength_adjustment_scale
        )

        prior_std = (
            0.5
            + variability_rating
            / 10.0
            * 1.5
        )

        priors[str(team)] = TeamPrior(
            mean=(
                base
                + strength_adjustment
            ),
            std=prior_std,
            differential_per_game=float(
                rate
            ),
        )

    return priors


def estimate_total_points(
    results: pd.DataFrame,
) -> tuple[float, float]:
    completed = completed_results(
        results
    )

    if completed.empty:
        return 42.0, 14.0

    totals = (
        completed["home_score"]
        + completed["away_score"]
    ).astype(float)

    mean = float(totals.mean())
    sigma = float(
        totals.std(ddof=0)
    )

    if not np.isfinite(mean):
        mean = 42.0

    if (
        not np.isfinite(sigma)
        or sigma <= 0.0
    ):
        sigma = 14.0

    return mean, sigma


def scores_from_total_and_margin(
    total: float | int,
    margin: float | int,
) -> tuple[int, int]:
    total_integer = max(
        0,
        int(round(float(total))),
    )

    margin_integer = int(
        round(float(margin))
    )

    if abs(margin_integer) > total_integer:
        total_integer = abs(
            margin_integer
        )

    if (
        total_integer
        + margin_integer
    ) % 2 != 0:
        total_integer += 1

    home_score = (
        total_integer
        + margin_integer
    ) // 2

    away_score = (
        total_integer
        - margin_integer
    ) // 2

    if home_score < 0 or away_score < 0:
        raise AssertionError(
            "Generated a negative score"
        )

    if (
        home_score
        - away_score
        != margin_integer
    ):
        raise AssertionError(
            "Generated scores do not preserve "
            "the requested margin"
        )

    return home_score, away_score


def simulate_match_scores(
    rng: np.random.Generator,
    expected_margin: float,
    margin_sigma: float,
    expected_total: float,
    total_sigma: float,
) -> tuple[int, int]:
    if margin_sigma <= 0:
        raise ValueError(
            "margin_sigma must be positive"
        )

    if total_sigma <= 0:
        raise ValueError(
            "total_sigma must be positive"
        )

    sampled_margin = int(
        np.rint(
            rng.normal(
                expected_margin,
                margin_sigma,
            )
        )
    )

    sampled_total = int(
        np.rint(
            rng.normal(
                expected_total,
                total_sigma,
            )
        )
    )

    return scores_from_total_and_margin(
        sampled_total,
        sampled_margin,
    )


def apply_match_result(
    state: dict[str, dict[str, int]],
    home: str,
    away: str,
    home_score: int,
    away_score: int,
) -> None:
    if home not in state or away not in state:
        raise ValueError(
            f"Unknown simulated fixture: "
            f"{home} v {away}"
        )

    if home == away:
        raise ValueError(
            "A team cannot play itself"
        )

    home_score = int(home_score)
    away_score = int(away_score)

    if home_score < 0 or away_score < 0:
        raise ValueError(
            "Simulated scores cannot be negative"
        )

    state[home]["GP"] += 1
    state[away]["GP"] += 1

    state[home]["PF"] += home_score
    state[home]["PA"] += away_score

    state[away]["PF"] += away_score
    state[away]["PA"] += home_score

    if home_score > away_score:
        state[home]["W"] += 1
        state[away]["L"] += 1
        state[home]["CompPts"] += 2

    elif away_score > home_score:
        state[away]["W"] += 1
        state[home]["L"] += 1
        state[away]["CompPts"] += 2

    else:
        state[home]["D"] += 1
        state[away]["D"] += 1
        state[home]["CompPts"] += 1
        state[away]["CompPts"] += 1


def initialise_final_state(
    current_ladder: pd.DataFrame,
    future_byes: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    state = {}

    for row in current_ladder.itertuples(
        index=False
    ):
        state[str(row.Team)] = {
            "Team": str(row.Team),
            "GP": int(row.GP),
            "W": int(row.W),
            "D": int(row.D),
            "L": int(row.L),
            "PF": int(row.PF),
            "PA": int(row.PA),
            "Diff": int(row.Diff),
            "Byes": int(row.Byes),
            "CompPts": int(row.CompPts),
        }

    for team, count in future_byes.items():
        if team not in state:
            raise ValueError(
                f"Future bye has unknown team: "
                f"{team}"
            )

        count = int(count)

        if count < 0:
            raise ValueError(
                "Future bye count cannot "
                "be negative"
            )

        state[team]["Byes"] += count
        state[team]["CompPts"] += (
            2 * count
        )

    return state


def _validate_final_ladder(
    ladder: pd.DataFrame,
) -> None:
    if int(ladder["PF"].sum()) != int(
        ladder["PA"].sum()
    ):
        raise AssertionError(
            "Final simulated PF does not "
            "equal PA"
        )

    if int(ladder["Diff"].sum()) != 0:
        raise AssertionError(
            "Final simulated differential "
            "does not sum to zero"
        )

    if int(ladder["W"].sum()) != int(
        ladder["L"].sum()
    ):
        raise AssertionError(
            "Final simulated wins do not "
            "equal losses"
        )

    if not (
        ladder["W"]
        + ladder["D"]
        + ladder["L"]
        == ladder["GP"]
    ).all():
        raise AssertionError(
            "Final simulated W/D/L does not "
            "equal GP"
        )


def simulate_once(
    current_ladder: pd.DataFrame,
    fixtures: Sequence[Fixture],
    future_byes: Mapping[str, int],
    priors: Mapping[str, TeamPrior],
    rng: np.random.Generator,
    parameters: SimulationParameters,
    expected_total: float,
    total_sigma: float,
) -> pd.DataFrame:
    if parameters.alpha <= 0:
        raise ValueError(
            "alpha must be positive"
        )

    if parameters.margin_sigma <= 0:
        raise ValueError(
            "margin_sigma must be positive"
        )

    teams = current_ladder[
        "Team"
    ].astype(str).tolist()

    missing_priors = sorted(
        set(teams) - set(priors)
    )

    if missing_priors:
        raise ValueError(
            "Missing team priors: "
            f"{missing_priors}"
        )

    sampled_strengths = {
        team: float(
            rng.normal(
                priors[team].mean,
                priors[team].std,
            )
        )
        for team in teams
    }

    state = initialise_final_state(
        current_ladder,
        future_byes,
    )

    for fixture in fixtures:
        expected_margin = (
            parameters.alpha
            * (
                sampled_strengths[
                    fixture.home
                ]
                + parameters.home_advantage
                - sampled_strengths[
                    fixture.away
                ]
            )
        )

        home_score, away_score = (
            simulate_match_scores(
                rng=rng,
                expected_margin=(
                    expected_margin
                ),
                margin_sigma=(
                    parameters.margin_sigma
                ),
                expected_total=expected_total,
                total_sigma=total_sigma,
            )
        )

        apply_match_result(
            state,
            fixture.home,
            fixture.away,
            home_score,
            away_score,
        )

    for team in teams:
        state[team]["Diff"] = (
            state[team]["PF"]
            - state[team]["PA"]
        )

    ladder = pd.DataFrame(
        state.values(),
        columns=LADDER_COLUMNS,
    )

    ladder = sort_ladder(
        ladder
    )

    _validate_final_ladder(
        ladder
    )

    return ladder


def run_simulations(
    results: pd.DataFrame,
    byes: pd.DataFrame,
    simulation_count: int,
    strength_ratings: (
        Mapping[str, float] | None
    ) = None,
    variability_ratings: (
        Mapping[str, float] | None
    ) = None,
    parameters: (
        SimulationParameters | None
    ) = None,
    seed: int | None = None,
    teams: Iterable[str] = EXPECTED_TEAMS,
) -> SimulationResult:
    simulation_count = int(
        simulation_count
    )

    if simulation_count <= 0:
        raise ValueError(
            "simulation_count must be positive"
        )

    team_list = list(teams)

    if parameters is None:
        parameters = SimulationParameters()

    current_ladder = compute_ladder(
        results,
        byes,
        team_list,
    )

    fixtures = future_fixtures(
        results,
        team_list,
    )

    outstanding_byes = (
        future_bye_counts(
            byes,
            team_list,
        )
    )

    priors = build_team_priors(
        current_ladder,
        strength_ratings,
        variability_ratings,
        parameters.strength_adjustment_scale,
    )

    estimated_total_mean, (
        estimated_total_sigma
    ) = estimate_total_points(
        results
    )

    total_sigma = (
        parameters.total_sigma
        if parameters.total_sigma
        is not None
        else estimated_total_sigma
    )

    if total_sigma <= 0:
        raise ValueError(
            "total_sigma must be positive"
        )

    rng = np.random.default_rng(
        seed
    )

    position_lists = {
        team: []
        for team in team_list
    }

    orders = []

    for _ in range(simulation_count):
        ladder = simulate_once(
            current_ladder=(
                current_ladder
            ),
            fixtures=fixtures,
            future_byes=(
                outstanding_byes
            ),
            priors=priors,
            rng=rng,
            parameters=parameters,
            expected_total=(
                estimated_total_mean
            ),
            total_sigma=total_sigma,
        )

        order = tuple(
            ladder["Team"].astype(str)
        )

        orders.append(order)

        for position, team in enumerate(
            order,
            start=1,
        ):
            position_lists[
                team
            ].append(position)

    positions = {
        team: np.asarray(
            values,
            dtype=int,
        )
        for team, values
        in position_lists.items()
    }

    summary_rows = []

    for team in team_list:
        values = positions[team]

        summary_rows.append(
            {
                "Team": team,
                "Top 4 %": float(
                    np.mean(values <= 4)
                    * 100.0
                ),
                "Top 8 %": float(
                    np.mean(values <= 8)
                    * 100.0
                ),
                "Minor Prem. %": float(
                    np.mean(values == 1)
                    * 100.0
                ),
                "Wooden Spoon %": float(
                    np.mean(
                        values
                        == len(team_list)
                    )
                    * 100.0
                ),
                "Median Pos": float(
                    np.median(values)
                ),
                "Mean Pos": float(
                    np.mean(values)
                ),
            }
        )

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(
            by=[
                "Mean Pos",
                "Median Pos",
                "Team",
            ],
            ascending=[
                True,
                True,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return SimulationResult(
        orders=tuple(orders),
        positions=positions,
        summary=summary,
        seed=seed,
        fixture_count=len(fixtures),
        simulation_count=(
            simulation_count
        ),
        estimated_total_mean=(
            estimated_total_mean
        ),
        estimated_total_sigma=float(
            total_sigma
        ),
    )
