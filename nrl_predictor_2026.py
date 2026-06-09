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

from predictor.simulation import (
    SimulationParameters,
    SimulationResult,
    future_bye_counts,
    future_fixtures,
    run_simulations,
)


REPO_ROOT = Path(__file__).resolve().parent

RESULTS_PATH = (
    REPO_ROOT
    / "data"
    / "2026"
    / "nrl_results.csv"
)

BYES_PATH = (
    REPO_ROOT
    / "data"
    / "2026"
    / "nrl_byes.csv"
)

META_PATH = (
    REPO_ROOT
    / "data"
    / "2026"
    / "nrl_results.meta.json"
)


@st.cache_data(show_spinner=False)
def load_app_inputs(
    results_path: str,
    byes_path: str,
    metadata_path: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    results = load_results_csv(
        results_path
    )

    byes = load_byes_csv(
        byes_path
    )

    metadata = json.loads(
        Path(metadata_path).read_text(
            encoding="utf-8"
        )
    )

    return results, byes, metadata


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

        if process.returncode == 0:
            value = process.stdout.strip()

            if value:
                return value

    except OSError:
        pass

    return "Unavailable"


def file_modified_time(
    path: Path,
) -> str:
    try:
        return datetime.fromtimestamp(
            path.stat().st_mtime
        ).astimezone().isoformat(
            timespec="seconds"
        )

    except OSError:
        return "Unavailable"


def common_combination_table(
    result: SimulationResult,
    size: int,
    ordered: bool,
    limit: int = 10,
) -> pd.DataFrame:
    if ordered:
        combinations = [
            tuple(order[:size])
            for order in result.orders
        ]

    else:
        combinations = [
            tuple(
                sorted(order[:size])
            )
            for order in result.orders
        ]

    counts = Counter(combinations)

    label = (
        f"Ordered Top {size}"
        if ordered
        else f"Top {size} Set"
    )

    separator = (
        " > "
        if ordered
        else ", "
    )

    rows = []

    for combination, count in counts.most_common(
        limit
    ):
        rows.append(
            {
                label: separator.join(
                    combination
                ),
                "Probability %": (
                    count
                    / result.simulation_count
                    * 100.0
                ),
            }
        )

    return pd.DataFrame(rows)


def position_distribution_figure(
    result: SimulationResult,
) -> plt.Figure:
    ordered_teams = (
        result.summary["Team"]
        .astype(str)
        .tolist()
    )

    team_count = len(ordered_teams)

    matrix = np.zeros(
        (
            team_count,
            team_count,
        ),
        dtype=float,
    )

    for team_index, team in enumerate(
        ordered_teams
    ):
        counts = Counter(
            result.positions[
                team
            ].tolist()
        )

        for position, count in counts.items():
            matrix[
                team_index,
                position - 1,
            ] = (
                count
                / result.simulation_count
            )

    figure, axis = plt.subplots(
        figsize=(13, 8)
    )

    left = np.zeros(
        team_count,
        dtype=float,
    )

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
    axis.set_title(
        "Predicted Ladder Position Distribution"
    )

    axis.legend(
        title="Position",
        bbox_to_anchor=(0.5, -0.12),
        loc="upper center",
        ncol=9,
    )

    figure.tight_layout()

    return figure


def simulation_summary_display(
    result: SimulationResult,
) -> pd.DataFrame:
    frame = result.summary.copy()

    percentage_columns = [
        "Top 4 %",
        "Top 8 %",
        "Minor Prem. %",
        "Wooden Spoon %",
    ]

    for column in percentage_columns:
        frame[column] = frame[
            column
        ].round(1)

    frame["Median Pos"] = frame[
        "Median Pos"
    ].round(1)

    frame["Mean Pos"] = frame[
        "Mean Pos"
    ].round(2)

    return frame


st.set_page_config(
    page_title=(
        "Joel's NRL Predictor 2026"
    ),
    layout="wide",
)

st.title(
    "Joel's NRL Ladder Predictor (2026)"
)

try:
    results_df, byes_df, metadata = (
        load_app_inputs(
            str(RESULTS_PATH),
            str(BYES_PATH),
            str(META_PATH),
        )
    )

except Exception as exc:
    st.error(
        "The 2026 predictor data could "
        f"not be loaded: {exc}"
    )

    st.stop()

current_ladder = compute_ladder(
    results_df,
    byes_df,
)

completed_df = completed_results(
    results_df
)

credited_byes_df = credited_byes(
    byes_df
)

remaining_fixtures = future_fixtures(
    results_df
)

remaining_byes = future_bye_counts(
    byes_df
)

data_cutoff = metadata.get(
    "data_cutoff",
    "Unknown cutoff",
)

st.caption(
    f"Data cutoff: {data_cutoff}"
)

st.markdown(
    "Adjust team strength and variability, "
    "then run a Monte Carlo simulation of "
    "the remaining regular-season fixtures."
)

st.warning(
    "This is a provisional baseline model. "
    "It uses current differential per game, "
    "user ratings, home advantage and "
    "match-level randomness. Model calibration "
    "and the exact tertiary NRL ladder "
    "countback remain to be finalised."
)

metric_columns = st.columns(4)

metric_columns[0].metric(
    "Completed matches",
    len(completed_df),
)

metric_columns[1].metric(
    "Remaining fixtures",
    len(remaining_fixtures),
)

metric_columns[2].metric(
    "Credited byes",
    len(credited_byes_df),
)

metric_columns[3].metric(
    "Future byes",
    sum(
        remaining_byes.values()
    ),
)

st.subheader(
    "Current Ladder"
)

current_display = current_ladder.copy()

current_display.insert(
    0,
    "Pos",
    range(
        1,
        len(current_display) + 1,
    ),
)

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
    "Current interim ordering: competition "
    "points, differential, then points for."
)

st.sidebar.header(
    "Simulation Controls"
)

simulation_count = st.sidebar.slider(
    "Number of simulations",
    min_value=500,
    max_value=50000,
    value=2000,
    step=500,
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

home_advantage = st.sidebar.slider(
    "Home advantage",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
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
    value=12.0,
    step=0.5,
    key="margin_sigma",
)

st.sidebar.header(
    "Team Beliefs"
)

st.sidebar.caption(
    "Strength changes expected performance. "
    "Variability changes how widely a team's "
    "sampled performance moves between "
    "simulated seasons. Five is neutral."
)

strength_ratings = {}
variability_ratings = {}

for team in EXPECTED_TEAMS:
    with st.sidebar.expander(
        team,
        expanded=False,
    ):
        strength_ratings[
            team
        ] = st.slider(
            f"{team} strength",
            min_value=0,
            max_value=10,
            value=5,
            step=1,
            key=f"strength_{team}",
        )

        variability_ratings[
            team
        ] = st.slider(
            f"{team} variability",
            min_value=0,
            max_value=10,
            value=5,
            step=1,
            key=f"variability_{team}",
        )

run_requested = st.sidebar.button(
    "Run Simulation",
    type="primary",
    key="run_simulation",
)

if run_requested:
    parameters = SimulationParameters(
        home_advantage=float(
            home_advantage
        ),
        alpha=float(alpha),
        margin_sigma=float(
            margin_sigma
        ),
    )

    with st.spinner(
        "Simulating the remaining season..."
    ):
        simulation_result = (
            run_simulations(
                results=results_df,
                byes=byes_df,
                simulation_count=int(
                    simulation_count
                ),
                strength_ratings=(
                    strength_ratings
                ),
                variability_ratings=(
                    variability_ratings
                ),
                parameters=parameters,
                seed=int(seed),
            )
        )

    st.session_state[
        "simulation_result"
    ] = simulation_result

    st.session_state[
        "simulation_controls"
    ] = {
        "simulation_count": int(
            simulation_count
        ),
        "seed": int(seed),
        "home_advantage": float(
            home_advantage
        ),
        "alpha": float(alpha),
        "margin_sigma": float(
            margin_sigma
        ),
    }

simulation_result = st.session_state.get(
    "simulation_result"
)

simulation_controls = st.session_state.get(
    "simulation_controls"
)

if simulation_result is None:
    st.info(
        "Set your ratings in the sidebar "
        "and select Run Simulation."
    )

else:
    st.divider()

    st.subheader(
        "Final Ladder Probabilities"
    )

    if simulation_controls:
        st.caption(
            "Results from "
            f'{simulation_controls["simulation_count"]:,} '
            "simulations using seed "
            f'{simulation_controls["seed"]}.'
        )

    summary_display = (
        simulation_summary_display(
            simulation_result
        )
    )

    st.dataframe(
        summary_display,
        hide_index=True,
        width="stretch",
        column_config={
            "Top 4 %": (
                st.column_config.NumberColumn(
                    format="%.1f"
                )
            ),
            "Top 8 %": (
                st.column_config.NumberColumn(
                    format="%.1f"
                )
            ),
            "Minor Prem. %": (
                st.column_config.NumberColumn(
                    format="%.1f"
                )
            ),
            "Wooden Spoon %": (
                st.column_config.NumberColumn(
                    format="%.1f"
                )
            ),
            "Median Pos": (
                st.column_config.NumberColumn(
                    format="%.1f"
                )
            ),
            "Mean Pos": (
                st.column_config.NumberColumn(
                    format="%.2f"
                )
            ),
        },
    )

    st.caption(
        "Estimated completed-match total "
        "points distribution: mean "
        f"{simulation_result.estimated_total_mean:.2f}, "
        "standard deviation "
        f"{simulation_result.estimated_total_sigma:.2f}."
    )

    st.subheader(
        "Ladder Position Distribution"
    )

    figure = (
        position_distribution_figure(
            simulation_result
        )
    )

    st.pyplot(
        figure,
        width="stretch",
    )

    plt.close(figure)

    table_columns = st.columns(2)

    with table_columns[0]:
        st.subheader(
            "Most Common Top 8 Sets"
        )

        st.dataframe(
            common_combination_table(
                simulation_result,
                size=8,
                ordered=False,
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Probability %": (
                    st.column_config.NumberColumn(
                        format="%.2f"
                    )
                )
            },
        )

    with table_columns[1]:
        st.subheader(
            "Most Common Top 8 Orders"
        )

        st.dataframe(
            common_combination_table(
                simulation_result,
                size=8,
                ordered=True,
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Probability %": (
                    st.column_config.NumberColumn(
                        format="%.2f"
                    )
                )
            },
        )

    table_columns = st.columns(2)

    with table_columns[0]:
        st.subheader(
            "Most Common Top 4 Sets"
        )

        st.dataframe(
            common_combination_table(
                simulation_result,
                size=4,
                ordered=False,
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Probability %": (
                    st.column_config.NumberColumn(
                        format="%.2f"
                    )
                )
            },
        )

    with table_columns[1]:
        st.subheader(
            "Most Common Top 4 Orders"
        )

        st.dataframe(
            common_combination_table(
                simulation_result,
                size=4,
                ordered=True,
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Probability %": (
                    st.column_config.NumberColumn(
                        format="%.2f"
                    )
                )
            },
        )

with st.expander(
    "Diagnostics and Data Provenance",
    expanded=False,
):
    fixture_identity_columns = [
        "round",
        "home_team",
        "away_team",
    ]

    duplicate_count = int(
        results_df.duplicated(
            subset=fixture_identity_columns,
            keep=False,
        ).sum()
    )

    completed_missing_score_count = int(
        completed_df[
            [
                "home_score",
                "away_score",
            ]
        ].isna().any(axis=1).sum()
    )

    diagnostics = {
        "deployed_app_commit": (
            deployed_commit()
        ),
        "results_path": str(
            RESULTS_PATH
        ),
        "results_modified_time": (
            file_modified_time(
                RESULTS_PATH
            )
        ),
        "byes_path": str(
            BYES_PATH
        ),
        "metadata_path": str(
            META_PATH
        ),
        "data_cutoff": data_cutoff,
        "fixture_count": int(
            len(results_df)
        ),
        "completed_match_count": int(
            len(completed_df)
        ),
        "remaining_fixture_count": int(
            len(remaining_fixtures)
        ),
        "team_count": int(
            len(
                set(
                    results_df[
                        "home_team"
                    ]
                )
                | set(
                    results_df[
                        "away_team"
                    ]
                )
            )
        ),
        "credited_bye_count": int(
            len(credited_byes_df)
        ),
        "future_bye_count": int(
            sum(
                remaining_byes.values()
            )
        ),
        "duplicate_fixture_rows": (
            duplicate_count
        ),
        "completed_missing_scores": (
            completed_missing_score_count
        ),
        "upstream_scraper_commit": (
            metadata.get(
                "scraper_git_commit"
            )
        ),
        "upstream_database_path": (
            metadata.get(
                "database_path"
            )
        ),
        "fixture_source_path": (
            metadata.get(
                "fixture_path"
            )
        ),
        "export_generated_at": (
            metadata.get(
                "generated_at_australia_sydney"
            )
        ),
    }

    st.json(diagnostics)
