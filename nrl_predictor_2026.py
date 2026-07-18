from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import subprocess

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from predictor.ladder import (
    EXPECTED_TEAMS,
    completed_results,
    compute_ladder,
    credited_byes,
    load_byes_csv,
    load_results_csv,
)
from predictor.outputs import (
    SCENARIO_OUTCOMES,
    fixture_probability_table,
    monte_carlo_error_table,
    position_probability_table,
    scenario_mask,
    scenario_team_summary,
    team_fixture_leverage_table,
    team_wins_needed_table,
)
from predictor.simulation import (
    SimulationParameters,
    SimulationResult,
    future_bye_counts,
    future_fixtures,
    run_simulations,
)


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = REPO_ROOT / "data" / "2026" / "nrl_results.csv"
BYES_PATH = REPO_ROOT / "data" / "2026" / "nrl_byes.csv"
META_PATH = REPO_ROOT / "data" / "2026" / "nrl_results.meta.json"
MODEL_SETTINGS_PATH = REPO_ROOT / "data" / "2026" / "app_model_settings.json"

DEFAULT_STRENGTH_RATINGS = {
    "Broncos": 4.0,
    "Bulldogs": 5.5,
    "Cowboys": 5.5,
    "Dolphins": 6.0,
    "Dragons": 6.0,
    "Eels": 5.5,
    "Knights": 4.5,
    "Panthers": 4.5,
    "Rabbitohs": 5.0,
    "Raiders": 5.0,
    "Roosters": 5.0,
    "Sea Eagles": 5.5,
    "Sharks": 5.0,
    "Storm": 5.0,
    "Titans": 5.0,
    "Warriors": 5.0,
    "Wests Tigers": 3.5,
}


@st.cache_data(show_spinner=False)
def load_app_inputs(
    results_path: str,
    byes_path: str,
    metadata_path: str,
    results_modified: str,
    byes_modified: str,
    metadata_modified: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    results = load_results_csv(results_path)
    byes = load_byes_csv(byes_path)
    metadata = json.loads(
        Path(metadata_path).read_text(encoding="utf-8")
    )
    return results, byes, metadata

@st.cache_data(show_spinner=False)
def load_model_settings(settings_path: str) -> dict:
    path = Path(settings_path)
    if not path.exists():
        return {}

    settings = json.loads(path.read_text(encoding="utf-8"))
    if settings.get("rating_mode") != "points":
        return settings

    strengths = settings.get("team_strength_points", {})
    missing = sorted(set(EXPECTED_TEAMS) - set(strengths))
    if missing:
        raise ValueError(
            "Point-mode model settings are missing team strengths for: "
            f"{missing}"
        )

    strength_sds = settings.get("team_strength_sd_points", {})
    missing_sds = sorted(set(EXPECTED_TEAMS) - set(strength_sds))
    if missing_sds:
        raise ValueError(
            "Point-mode model settings are missing strength SDs for: "
            f"{missing_sds}"
        )

    return settings


def model_point_value(settings: dict, key: str, fallback: float) -> float:
    return float(settings.get(key, fallback))



def deployed_commit() -> str:
    try:
        process = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "rev-parse",
                "--short",
                "HEAD",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if process.returncode == 0 and process.stdout.strip():
            return process.stdout.strip()
    except OSError:
        pass
    return "Unavailable"


def file_modified_time(path: Path) -> str:
    try:
        return datetime.fromtimestamp(
            path.stat().st_mtime
        ).astimezone().isoformat(timespec="seconds")
    except OSError:
        return "Unavailable"


def nearest_quantile(values: np.ndarray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="nearest"))
    except TypeError:
        return float(
            np.quantile(values, probability, interpolation="nearest")
        )


def common_combination_table(
    result: SimulationResult,
    size: int,
    ordered: bool,
    limit: int = 10,
) -> pd.DataFrame:
    if ordered:
        combinations = [tuple(order[:size]) for order in result.orders]
    else:
        combinations = [
            tuple(sorted(order[:size])) for order in result.orders
        ]

    counts = Counter(combinations)
    label = f"Ordered Top {size}" if ordered else f"Top {size} Set"
    separator = " > " if ordered else ", "

    return pd.DataFrame(
        [
            {
                label: separator.join(combination),
                "Probability %": (
                    count / result.simulation_count * 100.0
                ),
            }
            for combination, count in counts.most_common(limit)
        ]
    )


def position_distribution_figure(
    result: SimulationResult,
) -> plt.Figure:
    ordered_teams = result.summary["Team"].astype(str).tolist()
    team_count = len(ordered_teams)
    matrix = np.zeros((team_count, team_count), dtype=float)

    for team_index, team in enumerate(ordered_teams):
        counts = Counter(result.positions[team].tolist())
        for position, count in counts.items():
            matrix[team_index, position - 1] = (
                count / result.simulation_count
            )

    figure, axis = plt.subplots(figsize=(13, 8))
    left = np.zeros(team_count, dtype=float)

    for position in range(team_count):
        axis.barh(
            ordered_teams,
            matrix[:, position],
            left=left,
            label=str(position + 1),
        )
        left += matrix[:, position]

    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Probability")
    axis.set_ylabel("Team")
    axis.set_title("Predicted Ladder Position Distribution")
    axis.legend(
        title="Position",
        bbox_to_anchor=(0.5, -0.12),
        loc="upper center",
        ncol=9,
    )
    figure.tight_layout()
    return figure


def position_heatmap_figure(
    result: SimulationResult,
) -> plt.Figure:
    probability_table = position_probability_table(result)
    teams = probability_table["Team"].astype(str).tolist()
    matrix = probability_table.drop(columns="Team").to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(14, 8))
    image = axis.imshow(matrix, aspect="auto")
    axis.set_yticks(np.arange(len(teams)))
    axis.set_yticklabels(teams)
    axis.set_xticks(np.arange(matrix.shape[1]))
    axis.set_xticklabels(
        [str(position) for position in range(1, matrix.shape[1] + 1)]
    )
    axis.set_xlabel("Final ladder position")
    axis.set_ylabel("Team")
    axis.set_title("Probability of Finishing in Every Position")
    figure.colorbar(image, ax=axis, label="Probability (%)")
    figure.tight_layout()
    return figure


def simulation_summary_display(
    result: SimulationResult,
    summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = (
        result.summary.copy()
        if summary is None
        else summary.copy()
    )
    frame["Expected Pos"] = frame["Mean Pos"].round(2)
    frame["80% Pos Range"] = frame.apply(
        lambda row: (
            f'{int(row["Pos P10"])}–{int(row["Pos P90"])}'
        ),
        axis=1,
    )
    frame["Expected Pts"] = frame["Mean CompPts"].round(1)

    percentage_columns = [
        "Top 4 %",
        "Top 8 %",
        "Minor Prem. %",
        "Wooden Spoon %",
    ]
    for column in percentage_columns:
        frame[column] = frame[column].round(1)

    return frame[
        [
            "Team",
            "Expected Pos",
            "80% Pos Range",
            "Expected Pts",
            "Top 4 %",
            "Top 8 %",
            "Minor Prem. %",
            "Wooden Spoon %",
        ]
    ]


def probability_column_config(
    columns: pd.Index | list[str] | None = None,
) -> dict:
    config = {
        "Top 4 %": st.column_config.NumberColumn(format="%.1f"),
        "Top 8 %": st.column_config.NumberColumn(format="%.1f"),
        "Minor Prem. %": st.column_config.NumberColumn(format="%.1f"),
        "Wooden Spoon %": st.column_config.NumberColumn(format="%.1f"),
        "Home Win %": st.column_config.NumberColumn(format="%.1f"),
        "Draw %": st.column_config.NumberColumn(format="%.1f"),
        "Away Win %": st.column_config.NumberColumn(format="%.1f"),
        "Win %": st.column_config.NumberColumn(format="%.1f"),
        "Loss %": st.column_config.NumberColumn(format="%.1f"),
        "Top 8 if Win %": st.column_config.NumberColumn(format="%.1f"),
        "Top 8 if Loss %": st.column_config.NumberColumn(format="%.1f"),
        "Top 8 Leverage (pp)": st.column_config.NumberColumn(
            format="%.1f"
        ),
        "Top 4 if Win %": st.column_config.NumberColumn(format="%.1f"),
        "Top 4 if Loss %": st.column_config.NumberColumn(format="%.1f"),
        "Top 4 Leverage (pp)": st.column_config.NumberColumn(
            format="%.1f"
        ),
        "Simulation Share %": st.column_config.NumberColumn(format="%.1f"),
    }
    if columns is None:
        return config
    column_names = {str(column) for column in columns}
    return {
        key: value
        for key, value in config.items()
        if key in column_names
    }


def display_team_detail(
    result: SimulationResult,
    selected_team: str,
) -> None:
    summary_row = result.summary.loc[
        result.summary["Team"] == selected_team
    ].iloc[0]

    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "Top 8",
        f'{summary_row["Top 8 %"]:.1f}%',
    )
    metric_columns[1].metric(
        "Top 4",
        f'{summary_row["Top 4 %"]:.1f}%',
    )
    metric_columns[2].metric(
        "Expected finish",
        f'{summary_row["Mean Pos"]:.2f}',
    )
    metric_columns[3].metric(
        "80% finishing range",
        (
            f'{int(summary_row["Pos P10"])}–'
            f'{int(summary_row["Pos P90"])}'
        ),
    )
    metric_columns[4].metric(
        "Expected competition points",
        f'{summary_row["Mean CompPts"]:.1f}',
    )

    st.markdown("#### Exact finishing-position probabilities")
    exact = position_probability_table(result)
    exact = exact.loc[exact["Team"] == selected_team].drop(
        columns="Team"
    )
    exact_long = (
        exact.T.reset_index()
        .rename(columns={"index": "Position", exact.index[0]: "Probability %"})
    )
    exact_long["Position"] = exact_long["Position"].astype(int)
    st.bar_chart(
        exact_long.set_index("Position")["Probability %"],
        height=320,
    )
    with st.expander("Show exact percentages", expanded=False):
        st.dataframe(
            exact_long,
            hide_index=True,
            width="stretch",
            column_config={
                "Probability %": st.column_config.NumberColumn(
                    format="%.2f"
                )
            },
        )

    distribution_rows = [
        {
            "Measure": "Competition points",
            "Expected": float(summary_row["Mean CompPts"]),
            "Median": float(summary_row["Median CompPts"]),
            "80% Range": (
                f'{int(summary_row["CompPts P10"])}–'
                f'{int(summary_row["CompPts P90"])}'
            ),
        },
        {
            "Measure": "Points differential",
            "Expected": float(summary_row["Mean Diff"]),
            "Median": float(summary_row["Median Diff"]),
            "80% Range": (
                f'{int(summary_row["Diff P10"])}–'
                f'{int(summary_row["Diff P90"])}'
            ),
        },
        {
            "Measure": "Remaining wins",
            "Expected": float(summary_row["Mean Future Wins"]),
            "Median": float(np.median(result.future_wins[selected_team])),
            "80% Range": (
                f'{int(nearest_quantile(result.future_wins[selected_team], 0.10))}–'
                f'{int(nearest_quantile(result.future_wins[selected_team], 0.90))}'
            ),
        },
    ]
    st.markdown("#### Projected season totals")
    st.dataframe(
        pd.DataFrame(distribution_rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Expected": st.column_config.NumberColumn(format="%.1f"),
            "Median": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    st.markdown("#### Finals chances by number of remaining wins")
    st.caption(
        "Each row conditions on simulations in which the team records "
        "exactly that many wins from its remaining fixtures."
    )
    wins_table = team_wins_needed_table(result, selected_team)
    st.dataframe(
        wins_table,
        hide_index=True,
        width="stretch",
        column_config=probability_column_config(wins_table.columns),
    )

    st.markdown("#### Fixture leverage")
    st.caption(
        "Leverage is the difference between the team's finals chance "
        "after a simulated win and after a simulated loss in that match."
    )
    leverage = team_fixture_leverage_table(result, selected_team)
    if leverage.empty:
        st.info("This team has no remaining fixtures.")
    else:
        leverage_display = leverage.drop(columns=["Fixture Index"])
        st.dataframe(
            leverage_display,
            hide_index=True,
            width="stretch",
            column_config=probability_column_config(
                leverage_display.columns
            ),
        )


def display_scenario_explorer(
    result: SimulationResult,
) -> None:
    if not result.fixtures:
        st.info("There are no remaining fixtures to use in a scenario.")
        return

    teams = result.summary["Team"].astype(str).tolist()
    selected_team = st.selectbox(
        "Team to evaluate",
        teams,
        key="scenario_team",
    )
    scope = st.radio(
        "Fixtures shown",
        ["Selected team's fixtures", "All remaining fixtures"],
        horizontal=True,
        key="scenario_scope",
    )

    fixture_rows = []
    for index, fixture in enumerate(result.fixtures):
        if (
            scope == "Selected team's fixtures"
            and selected_team not in {fixture.home, fixture.away}
        ):
            continue
        fixture_rows.append(
            {
                "Use": False,
                "Fixture Index": index,
                "Round": fixture.round,
                "Match": f"{fixture.home} v {fixture.away}",
                "Outcome": "Home win",
            }
        )

    edited = st.data_editor(
        pd.DataFrame(fixture_rows),
        hide_index=True,
        width="stretch",
        disabled=["Fixture Index", "Round", "Match"],
        column_config={
            "Use": st.column_config.CheckboxColumn(
                "Use",
                help="Include this forced result in the scenario.",
            ),
            "Fixture Index": None,
            "Outcome": st.column_config.SelectboxColumn(
                "Outcome",
                options=list(SCENARIO_OUTCOMES),
                required=True,
            ),
        },
        key="scenario_editor",
    )

    selections = {
        int(row["Fixture Index"]): str(row["Outcome"])
        for _, row in edited.loc[edited["Use"]].iterrows()
    }

    baseline_row = result.summary.loc[
        result.summary["Team"] == selected_team
    ].iloc[0]

    if not selections:
        st.info(
            "Select one or more fixtures with the Use checkbox to "
            "calculate a conditional scenario."
        )
        return

    mask = scenario_mask(result, selections)
    scenario = scenario_team_summary(result, selected_team, mask)
    sample_count = int(scenario["Samples"])

    if sample_count == 0:
        st.warning(
            "None of the stored simulations produced this combination "
            "of results. Remove a condition or rerun with more simulations."
        )
        return

    st.caption(
        f"Scenario matched {sample_count:,} of "
        f"{result.simulation_count:,} simulations "
        f'({scenario["Simulation Share %"]:.2f}%).'
    )
    if sample_count < 200:
        st.warning(
            "This scenario is based on fewer than 200 matching simulations; "
            "the conditional percentages may be noisy."
        )

    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "Top 8",
        f'{scenario["Top 8 %"]:.1f}%',
        delta=(
            f'{scenario["Top 8 %"] - baseline_row["Top 8 %"]:+.1f} pp'
        ),
    )
    metric_columns[1].metric(
        "Top 4",
        f'{scenario["Top 4 %"]:.1f}%',
        delta=(
            f'{scenario["Top 4 %"] - baseline_row["Top 4 %"]:+.1f} pp'
        ),
    )
    metric_columns[2].metric(
        "Expected finish",
        f'{scenario["Expected Pos"]:.2f}',
        delta=(
            f'{scenario["Expected Pos"] - baseline_row["Mean Pos"]:+.2f}'
        ),
        delta_color="inverse",
    )
    metric_columns[3].metric(
        "80% finishing range",
        f'{int(scenario["Pos P10"])}–{int(scenario["Pos P90"])}',
    )
    metric_columns[4].metric(
        "Expected points",
        f'{scenario["Expected CompPts"]:.1f}',
        delta=(
            f'{scenario["Expected CompPts"] - baseline_row["Mean CompPts"]:+.1f}'
        ),
    )


st.set_page_config(
    page_title="Joel's NRL Predictor 2026",
    layout="wide",
)
st.title("Joel's NRL Ladder Predictor (2026)")

try:
    results_df, byes_df, metadata = load_app_inputs(
        str(RESULTS_PATH),
        str(BYES_PATH),
        str(META_PATH),
        file_modified_time(RESULTS_PATH),
        file_modified_time(BYES_PATH),
        file_modified_time(META_PATH),
    )
except Exception as exc:
    st.error(f"The 2026 predictor data could not be loaded: {exc}")
    st.stop()

try:
    model_settings = load_model_settings(str(MODEL_SETTINGS_PATH))
except Exception as exc:
    st.error(f"The 2026 model settings could not be loaded: {exc}")
    st.stop()
using_point_model = model_settings.get("rating_mode") == "points"

current_ladder = compute_ladder(results_df, byes_df)
completed_df = completed_results(results_df)
credited_byes_df = credited_byes(byes_df)
remaining_fixtures = future_fixtures(results_df)
remaining_byes = future_bye_counts(byes_df)

data_cutoff = metadata.get("data_cutoff", "Unknown cutoff")
partial_rounds = [
    int(value) for value in metadata.get("partial_rounds", [])
]
if partial_rounds:
    partial_label = "Round" if len(partial_rounds) == 1 else "Rounds"
    partial_numbers = ", ".join(str(value) for value in partial_rounds)
    data_cutoff = (
        f"{data_cutoff}; {partial_label} {partial_numbers} partial"
    )

st.caption(f"Data cutoff: {data_cutoff}")
st.markdown(
    "Adjust team assumptions, then run a Monte Carlo "
    "simulation of the remaining regular-season fixtures."
)
if using_point_model:
    st.info(
        f"The current data contain {len(completed_df)} completed matches. "
        "The default ratings come from the latest Bayesian team-strength "
        "model and are measured directly in expected margin points. Team "
        "sliders are gut-feel point adjustments with zero impact by default. "
        "Scheduled byes add competition points only."
    )
else:
    st.info(
        f"The current data contain {len(completed_df)} completed matches. "
        "Differential per game supplies the legacy season baseline and "
        "the strength presets apply current-form adjustments. Simulated "
        "scores update PF, PA and differential, while scheduled byes add "
        "competition points only."
    )

metric_columns = st.columns(4)
metric_columns[0].metric("Completed matches", len(completed_df))
metric_columns[1].metric("Remaining fixtures", len(remaining_fixtures))
metric_columns[2].metric("Credited byes", len(credited_byes_df))
metric_columns[3].metric(
    "Future byes",
    sum(remaining_byes.values()),
)

st.subheader("Current Ladder")
current_display = current_ladder.copy()
current_display.insert(0, "Pos", range(1, len(current_display) + 1))
st.dataframe(
    current_display[
        [
            "Pos",
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
    ],
    hide_index=True,
    width="stretch",
)
st.caption(
    "Current interim ordering: competition points, differential, then "
    "points for."
)

st.sidebar.header("Simulation Controls")
simulation_count = st.sidebar.slider(
    "Number of simulations",
    min_value=500,
    max_value=50000,
    value=20000,
    step=500,
    help="Use 20,000 routinely and 50,000 for published probabilities.",
    key="simulation_count",
)
seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=4294967295,
    value=20260609,
    step=1,
    key="seed",
)
team_strength_points: dict[str, float] | None = None
team_strength_sd_points: dict[str, float] | None = None
strength_ratings: dict[str, float] | None = None
variability_ratings: dict[str, int] | None = None
team_adjustments: dict[str, float] = {}

if using_point_model:
    alpha = 1.0
    base_strengths = {
        team: float(value)
        for team, value in model_settings["team_strength_points"].items()
    }
    strength_sds = {
        team: float(value)
        for team, value in model_settings["team_strength_sd_points"].items()
    }

    home_advantage = st.sidebar.slider(
        "Listed-home adjustment (points)",
        min_value=-5.0,
        max_value=5.0,
        value=model_point_value(
            model_settings,
            "home_advantage_points",
            0.0,
        ),
        step=0.5,
        help=(
            "A direct margin-points adjustment for the listed home team. "
            "This is not venue-specific."
        ),
        key="home_advantage_points",
    )
    margin_sigma = st.sidebar.slider(
        "Match randomness (points)",
        min_value=6.0,
        max_value=24.0,
        value=model_point_value(
            model_settings,
            "match_randomness_points",
            16.5,
        ),
        step=0.5,
        help="Independent match-to-match margin noise in points.",
        key="margin_sigma_points",
    )

    st.sidebar.caption(
        "Point mode is active. The fitted model supplies each team's "
        "base rating in points. Team sliders below are gut-feel point "
        "adjustments and default to 0.0."
    )

    st.sidebar.header("Gut-feel Team Adjustments")
    for team in EXPECTED_TEAMS:
        with st.sidebar.expander(team, expanded=False):
            team_adjustments[team] = st.slider(
                f"{team} adjustment (points)",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
                format="%+.1f",
                help=(
                    "Add your own judgement to the fitted model rating. "
                    "Leaving this at 0 uses the model rating unchanged."
                ),
                key=f"point_adjustment_2026_{team}",
            )
            effective = base_strengths[team] + team_adjustments[team]
            st.caption(
                f"Model: {base_strengths[team]:+.1f} pts | "
                f"SD: {strength_sds[team]:.1f} | "
                f"Effective: {effective:+.1f} pts"
            )
            if abs(team_adjustments[team]) >= 4.0:
                st.warning(
                    "This is a large override of the fitted model rating."
                )

    team_strength_points = {
        team: base_strengths[team] + team_adjustments[team]
        for team in EXPECTED_TEAMS
    }
    team_strength_sd_points = {
        team: strength_sds[team]
        for team in EXPECTED_TEAMS
    }
else:
    home_advantage = st.sidebar.slider(
        "Home advantage",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help=(
            "At strength-to-margin scale 10, 0.1 contributes approximately "
            "one margin point to the listed home team."
        ),
        key="home_advantage",
    )
    alpha = st.sidebar.slider(
        "Strength-to-margin scale",
        min_value=5.0,
        max_value=15.0,
        value=10.0,
        step=0.5,
        key="alpha",
    )
    margin_sigma = st.sidebar.slider(
        "Match randomness",
        min_value=6.0,
        max_value=20.0,
        value=16.0,
        step=0.5,
        help=(
            "Independent match-to-match margin noise. Persistent "
            "team-strength uncertainty is handled separately by variability."
        ),
        key="margin_sigma",
    )

    st.sidebar.header("Team Beliefs")
    st.sidebar.caption(
        "Strength presets adjust the differential-per-game baseline. "
        "Variability represents uncertainty about persistent team strength; "
        "zero is the calibrated minimum."
    )
    strength_ratings = {}
    variability_ratings = {}

    for team in EXPECTED_TEAMS:
        with st.sidebar.expander(team, expanded=False):
            strength_ratings[team] = st.slider(
                f"{team} strength",
                min_value=0.0,
                max_value=10.0,
                value=float(DEFAULT_STRENGTH_RATINGS[team]),
                step=0.5,
                format="%.1f",
                key=f"strength_2026_{team}",
            )
            variability_ratings[team] = st.slider(
                f"{team} variability",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                help=(
                    "Persistent season-long strength uncertainty, not "
                    "match-to-match volatility."
                ),
                key=f"variability_2026_{team}",
            )

run_requested = st.sidebar.button(
    "Run Simulation",
    type="primary",
    key="run_simulation",
)

if run_requested:
    parameters = SimulationParameters(
        home_advantage=float(home_advantage),
        alpha=float(alpha),
        margin_sigma=float(margin_sigma),
    )
    with st.spinner("Simulating the remaining season..."):
        simulation_result = run_simulations(
            results=results_df,
            byes=byes_df,
            simulation_count=int(simulation_count),
            strength_ratings=strength_ratings,
            variability_ratings=variability_ratings,
            team_strength_points=team_strength_points,
            team_strength_sd_points=team_strength_sd_points,
            parameters=parameters,
            seed=int(seed),
        )

    st.session_state["simulation_result"] = simulation_result
    st.session_state["simulation_controls"] = {
        "simulation_count": int(simulation_count),
        "seed": int(seed),
        "home_advantage": float(home_advantage),
        "alpha": float(alpha),
        "margin_sigma": float(margin_sigma),
        "rating_mode": "points" if using_point_model else "legacy",
    }

    if using_point_model:
        st.session_state["team_adjustments"] = team_adjustments

simulation_result = st.session_state.get("simulation_result")
simulation_controls = st.session_state.get("simulation_controls")

if simulation_result is None:
    st.info("Set your ratings in the sidebar and select Run Simulation.")
else:
    required_result_fields = (
        "competition_points",
        "future_wins",
        "fixture_margins",
        "top8_cutoff_points",
    )
    if not all(
        hasattr(simulation_result, field)
        for field in required_result_fields
    ):
        st.session_state.pop("simulation_result", None)
        st.warning(
            "The stored simulation predates the expanded outputs. "
            "Run the simulation again."
        )
        st.stop()

    st.divider()
    st.subheader("Final Ladder Projection")
    if simulation_controls:
        st.caption(
            f'Results from {simulation_controls["simulation_count"]:,} '
            f'simulations using seed {simulation_controls["seed"]}.'
        )

    cutoff_values = simulation_result.top8_cutoff_points
    cutoff_columns = st.columns(4)
    cutoff_columns[0].metric(
        "Median 8th-place points",
        f"{np.median(cutoff_values):.0f}",
    )
    cutoff_columns[1].metric(
        "Expected 8th-place points",
        f"{np.mean(cutoff_values):.1f}",
    )
    cutoff_columns[2].metric(
        "80% cutoff range",
        (
            f"{int(nearest_quantile(cutoff_values, 0.10))}–"
            f"{int(nearest_quantile(cutoff_values, 0.90))}"
        ),
    )
    finals_race_count = int(
        (
            (simulation_result.summary["Top 8 %"] >= 10.0)
            & (simulation_result.summary["Top 8 %"] <= 90.0)
        ).sum()
    )
    cutoff_columns[3].metric("Teams in finals race", finals_race_count)

    summary_display = simulation_summary_display(simulation_result)
    st.dataframe(
        summary_display,
        hide_index=True,
        width="stretch",
        column_config={
            **probability_column_config(summary_display.columns),
            "Expected Pos": st.column_config.NumberColumn(format="%.2f"),
            "Expected Pts": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    finals_race = simulation_result.summary.loc[
        simulation_result.summary["Top 8 %"].between(10.0, 90.0)
    ].copy()
    if not finals_race.empty:
        with st.expander("Show the live finals race", expanded=False):
            finals_race_display = simulation_summary_display(
                simulation_result,
                summary=finals_race.reset_index(drop=True),
            )
            st.dataframe(
                finals_race_display,
                hide_index=True,
                width="stretch",
                column_config=probability_column_config(
                    finals_race_display.columns
                ),
            )

    tabs = st.tabs(
        [
            "Position Distribution",
            "Team Detail",
            "Match Forecasts",
            "Scenario Explorer",
            "Top 4 / Top 8 Combinations",
            "Advanced",
        ]
    )

    with tabs[0]:
        st.markdown("### Probability of every finishing position")
        heatmap = position_heatmap_figure(simulation_result)
        st.pyplot(heatmap, width="stretch")
        plt.close(heatmap)

        with st.expander("Show the full numerical matrix", expanded=False):
            probability_table = position_probability_table(simulation_result)
            st.dataframe(
                probability_table,
                hide_index=True,
                width="stretch",
                column_config={
                    str(position): st.column_config.NumberColumn(
                        format="%.1f"
                    )
                    for position in range(1, len(EXPECTED_TEAMS) + 1)
                },
            )

        with st.expander(
            "Show the stacked position-distribution chart",
            expanded=False,
        ):
            figure = position_distribution_figure(simulation_result)
            st.pyplot(figure, width="stretch")
            plt.close(figure)

    with tabs[1]:
        selected_team = st.selectbox(
            "Team",
            simulation_result.summary["Team"].astype(str).tolist(),
            key="team_detail_team",
        )
        display_team_detail(simulation_result, selected_team)

    with tabs[2]:
        st.markdown("### Remaining fixture probabilities")
        st.caption(
            "Expected margin is from the listed home team's perspective; "
            "positive values favour the home team."
        )
        fixtures_table = fixture_probability_table(simulation_result)
        if fixtures_table.empty:
            st.info("There are no remaining fixtures.")
        else:
            fixtures_display = fixtures_table.drop(
                columns=["Fixture Index"]
            )
            st.dataframe(
                fixtures_display,
                hide_index=True,
                width="stretch",
                column_config={
                    **probability_column_config(
                        fixtures_display.columns
                    ),
                    "Expected Margin": st.column_config.NumberColumn(
                        format="%+.1f"
                    ),
                    "Median Margin": st.column_config.NumberColumn(
                        format="%+.1f"
                    ),
                },
            )

    with tabs[3]:
        st.markdown("### Conditional scenario explorer")
        st.caption(
            "This filters the simulations already run. Several forced "
            "results can therefore produce a small matching sample."
        )
        display_scenario_explorer(simulation_result)

    with tabs[4]:
        table_columns = st.columns(2)
        with table_columns[0]:
            st.markdown("#### Most Common Top 8 Sets")
            st.dataframe(
                common_combination_table(
                    simulation_result,
                    size=8,
                    ordered=False,
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "Probability %": st.column_config.NumberColumn(
                        format="%.2f"
                    )
                },
            )
        with table_columns[1]:
            st.markdown("#### Most Common Top 8 Orders")
            st.dataframe(
                common_combination_table(
                    simulation_result,
                    size=8,
                    ordered=True,
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "Probability %": st.column_config.NumberColumn(
                        format="%.2f"
                    )
                },
            )

        table_columns = st.columns(2)
        with table_columns[0]:
            st.markdown("#### Most Common Top 4 Sets")
            st.dataframe(
                common_combination_table(
                    simulation_result,
                    size=4,
                    ordered=False,
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "Probability %": st.column_config.NumberColumn(
                        format="%.2f"
                    )
                },
            )
        with table_columns[1]:
            st.markdown("#### Most Common Top 4 Orders")
            st.dataframe(
                common_combination_table(
                    simulation_result,
                    size=4,
                    ordered=True,
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "Probability %": st.column_config.NumberColumn(
                        format="%.2f"
                    )
                },
            )

    with tabs[5]:
        st.markdown("### Full projection summary")
        full_summary = simulation_result.summary.copy()
        st.dataframe(
            full_summary.round(2),
            hide_index=True,
            width="stretch",
        )

        st.markdown("### Monte Carlo sampling error")
        st.caption(
            "This measures simulation noise only. It does not include "
            "uncertainty from the model specification or strength ratings."
        )
        error_table = monte_carlo_error_table(simulation_result)
        st.dataframe(
            error_table,
            hide_index=True,
            width="stretch",
            column_config={
                "Top 4 %": st.column_config.NumberColumn(format="%.2f"),
                "Top 8 %": st.column_config.NumberColumn(format="%.2f"),
                "Top 4 MC SE (pp)": st.column_config.NumberColumn(
                    format="%.3f"
                ),
                "Top 8 MC SE (pp)": st.column_config.NumberColumn(
                    format="%.3f"
                ),
            },
        )

        st.caption(
            "Estimated completed-match total-points distribution: mean "
            f"{simulation_result.estimated_total_mean:.2f}, standard "
            f"deviation {simulation_result.estimated_total_sigma:.2f}."
        )

with st.expander("Diagnostics and Data Provenance", expanded=False):
    fixture_identity_columns = ["round", "home_team", "away_team"]
    duplicate_count = int(
        results_df.duplicated(
            subset=fixture_identity_columns,
            keep=False,
        ).sum()
    )
    completed_missing_score_count = int(
        completed_df[["home_score", "away_score"]]
        .isna()
        .any(axis=1)
        .sum()
    )
    diagnostics = {
        "deployed_app_commit": deployed_commit(),
        "results_path": str(RESULTS_PATH),
        "results_modified_time": file_modified_time(RESULTS_PATH),
        "byes_path": str(BYES_PATH),
        "metadata_path": str(META_PATH),
        "model_settings_path": str(MODEL_SETTINGS_PATH),
        "rating_mode": (
            "points" if using_point_model else "legacy"
        ),
        "model_run": model_settings.get("model_run"),
        "model_round_range": model_settings.get("round_range"),
        "model_completed_match_count": model_settings.get(
            "completed_match_count"
        ),
        "data_cutoff": data_cutoff,
        "fixture_count": int(len(results_df)),
        "completed_match_count": int(len(completed_df)),
        "remaining_fixture_count": int(len(remaining_fixtures)),
        "team_count": int(
            len(
                set(results_df["home_team"])
                | set(results_df["away_team"])
            )
        ),
        "credited_bye_count": int(len(credited_byes_df)),
        "future_bye_count": int(sum(remaining_byes.values())),
        "duplicate_fixture_rows": duplicate_count,
        "completed_missing_scores": completed_missing_score_count,
        "upstream_scraper_commit": metadata.get("scraper_git_commit"),
        "upstream_database_path": metadata.get("database_path"),
        "fixture_source_path": metadata.get("fixture_path"),
        "export_generated_at": metadata.get(
            "generated_at_australia_sydney"
        ),
    }
    st.json(diagnostics)
