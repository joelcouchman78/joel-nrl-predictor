from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from predictor.simulation import SimulationResult


SCENARIO_OUTCOMES = (
    "Home win",
    "Away win",
    "Draw",
)


def position_probability_table(
    result: SimulationResult,
) -> pd.DataFrame:
    teams = result.summary["Team"].astype(str).tolist()
    team_count = len(teams)
    rows: list[dict[str, float | str]] = []

    for team in teams:
        values = result.positions[team]
        row: dict[str, float | str] = {"Team": team}
        for position in range(1, team_count + 1):
            row[str(position)] = float(
                np.mean(values == position) * 100.0
            )
        rows.append(row)

    return pd.DataFrame(rows)


def fixture_probability_table(
    result: SimulationResult,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for index, fixture in enumerate(result.fixtures):
        margins = result.fixture_margins[:, index]
        rows.append(
            {
                "Fixture Index": index,
                "Round": fixture.round,
                "Home": fixture.home,
                "Away": fixture.away,
                "Home Win %": float(np.mean(margins > 0) * 100.0),
                "Draw %": float(np.mean(margins == 0) * 100.0),
                "Away Win %": float(np.mean(margins < 0) * 100.0),
                "Expected Margin": float(np.mean(margins)),
                "Median Margin": float(np.median(margins)),
            }
        )

    return pd.DataFrame(rows)


def team_wins_needed_table(
    result: SimulationResult,
    team: str,
) -> pd.DataFrame:
    _validate_team(result, team)
    future_wins = result.future_wins[team]
    positions = result.positions[team]
    rows: list[dict[str, float | int]] = []

    for win_count in sorted(np.unique(future_wins).tolist()):
        mask = future_wins == win_count
        sample_count = int(mask.sum())
        if sample_count == 0:
            continue

        rows.append(
            {
                "Remaining Wins": int(win_count),
                "Simulation Share %": float(
                    sample_count / result.simulation_count * 100.0
                ),
                "Top 4 %": float(np.mean(positions[mask] <= 4) * 100.0),
                "Top 8 %": float(np.mean(positions[mask] <= 8) * 100.0),
                "Expected Pos": float(np.mean(positions[mask])),
                "Median Pos": float(np.median(positions[mask])),
                "Samples": sample_count,
            }
        )

    return pd.DataFrame(rows)


def team_fixture_leverage_table(
    result: SimulationResult,
    team: str,
) -> pd.DataFrame:
    _validate_team(result, team)
    positions = result.positions[team]
    rows: list[dict[str, float | int | str]] = []

    for index, fixture in enumerate(result.fixtures):
        if team not in {fixture.home, fixture.away}:
            continue

        margins = result.fixture_margins[:, index]
        if team == fixture.home:
            win_mask = margins > 0
            loss_mask = margins < 0
            draw_mask = margins == 0
            opponent = fixture.away
        else:
            win_mask = margins < 0
            loss_mask = margins > 0
            draw_mask = margins == 0
            opponent = fixture.home

        top8_if_win = _conditional_probability(
            positions <= 8, win_mask
        )
        top8_if_loss = _conditional_probability(
            positions <= 8, loss_mask
        )
        top4_if_win = _conditional_probability(
            positions <= 4, win_mask
        )
        top4_if_loss = _conditional_probability(
            positions <= 4, loss_mask
        )

        rows.append(
            {
                "Fixture Index": index,
                "Round": fixture.round,
                "Opponent": opponent,
                "Venue": "Home" if team == fixture.home else "Away",
                "Win %": float(np.mean(win_mask) * 100.0),
                "Draw %": float(np.mean(draw_mask) * 100.0),
                "Loss %": float(np.mean(loss_mask) * 100.0),
                "Top 8 if Win %": top8_if_win,
                "Top 8 if Loss %": top8_if_loss,
                "Top 8 Leverage (pp)": (
                    top8_if_win - top8_if_loss
                    if np.isfinite(top8_if_win)
                    and np.isfinite(top8_if_loss)
                    else np.nan
                ),
                "Top 4 if Win %": top4_if_win,
                "Top 4 if Loss %": top4_if_loss,
                "Top 4 Leverage (pp)": (
                    top4_if_win - top4_if_loss
                    if np.isfinite(top4_if_win)
                    and np.isfinite(top4_if_loss)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def scenario_mask(
    result: SimulationResult,
    selections: Mapping[int, str],
) -> np.ndarray:
    mask = np.ones(result.simulation_count, dtype=bool)

    for fixture_index, outcome in selections.items():
        index = int(fixture_index)
        if not 0 <= index < result.fixture_count:
            raise IndexError(
                f"Fixture index {index} is outside the simulation result"
            )
        if outcome not in SCENARIO_OUTCOMES:
            raise ValueError(
                f"Unknown scenario outcome {outcome!r}; "
                f"expected one of {SCENARIO_OUTCOMES}"
            )

        margins = result.fixture_margins[:, index]
        if outcome == "Home win":
            mask &= margins > 0
        elif outcome == "Away win":
            mask &= margins < 0
        else:
            mask &= margins == 0

    return mask


def scenario_team_summary(
    result: SimulationResult,
    team: str,
    mask: np.ndarray,
) -> dict[str, float | int | str]:
    _validate_team(result, team)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (result.simulation_count,):
        raise ValueError(
            "Scenario mask must have one Boolean value per simulation"
        )

    sample_count = int(mask.sum())
    if sample_count == 0:
        return {
            "Team": team,
            "Samples": 0,
            "Simulation Share %": 0.0,
        }

    positions = result.positions[team][mask]
    points = result.competition_points[team][mask]
    differentials = result.differentials[team][mask]

    return {
        "Team": team,
        "Samples": sample_count,
        "Simulation Share %": float(
            sample_count / result.simulation_count * 100.0
        ),
        "Top 4 %": float(np.mean(positions <= 4) * 100.0),
        "Top 8 %": float(np.mean(positions <= 8) * 100.0),
        "Minor Prem. %": float(np.mean(positions == 1) * 100.0),
        "Wooden Spoon %": float(
            np.mean(positions == len(result.positions)) * 100.0
        ),
        "Expected Pos": float(np.mean(positions)),
        "Median Pos": float(np.median(positions)),
        "Pos P10": _nearest_quantile(positions, 0.10),
        "Pos P90": _nearest_quantile(positions, 0.90),
        "Expected CompPts": float(np.mean(points)),
        "Expected Diff": float(np.mean(differentials)),
    }


def monte_carlo_standard_error(
    probability_percent: float,
    simulation_count: int,
) -> float:
    if simulation_count <= 0:
        raise ValueError("simulation_count must be positive")
    probability = float(probability_percent) / 100.0
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability_percent must be between 0 and 100")
    return float(
        np.sqrt(
            probability
            * (1.0 - probability)
            / int(simulation_count)
        )
        * 100.0
    )


def monte_carlo_error_table(
    result: SimulationResult,
) -> pd.DataFrame:
    rows = []
    for _, row in result.summary.iterrows():
        top4 = float(row["Top 4 %"])
        top8 = float(row["Top 8 %"])
        rows.append(
            {
                "Team": str(row["Team"]),
                "Top 4 %": top4,
                "Top 4 MC SE (pp)": monte_carlo_standard_error(
                    top4, result.simulation_count
                ),
                "Top 8 %": top8,
                "Top 8 MC SE (pp)": monte_carlo_standard_error(
                    top8, result.simulation_count
                ),
            }
        )
    return pd.DataFrame(rows)


def _conditional_probability(
    event: np.ndarray,
    condition: np.ndarray,
) -> float:
    count = int(np.sum(condition))
    if count == 0:
        return float("nan")
    return float(np.mean(event[condition]) * 100.0)


def _validate_team(
    result: SimulationResult,
    team: str,
) -> None:
    if team not in result.positions:
        raise KeyError(f"Unknown team: {team}")


def _nearest_quantile(
    values: np.ndarray,
    probability: float,
) -> float:
    try:
        return float(np.quantile(values, probability, method="nearest"))
    except TypeError:
        return float(
            np.quantile(values, probability, interpolation="nearest")
        )
