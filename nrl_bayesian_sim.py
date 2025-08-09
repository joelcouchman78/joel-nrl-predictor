# File: ~/Projects/joel-nrl-predictor/nrl_bayesian_sim.py

# =========================
# Joel's NRL Ladder Predictor (2025)
# =========================

from pathlib import Path
import os
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------
# Page config & header
# -------------------------
st.set_page_config(page_title="Big Joel's NRL Predictor", layout="wide")
st.title("🏉 Joel's NRL Ladder Predictor (2025)")
st.markdown("Adjust your beliefs about team strength & variability. Click Run Simulation to update predictions.")

# -------------------------
# Team name map (NRL.com -> canonical)
# -------------------------
TEAM_NAME_MAP = {
    "Storm": "Storm",
    "Broncos": "Broncos",
    "Bulldogs": "Bulldogs",
    "Sharks": "Sharks",
    "Panthers": "Panthers",
    "Dolphins": "Dolphins",
    "Warriors": "Warriors",
    "Roosters": "Roosters",
    "Sea Eagles": "Sea Eagles",
    "Wests Tigers": "Wests Tigers",
    "Dragons": "Dragons",
    "Titans": "Titans",
    "Cowboys": "Cowboys",
    "Knights": "Knights",
    "Rabbitohs": "Rabbitohs",
    "Eels": "Eels",
    "Raiders": "Raiders",
}
ALL_TEAMS = list(TEAM_NAME_MAP.values())

# -------------------------
# Data path resolution
# -------------------------
REPO_ROOT = Path(__file__).resolve().parent
REPO_DATA_CSV = REPO_ROOT / "data" / "nrl_results.csv"
DESKTOP_CSV = Path(os.path.expanduser("~/Desktop/nrl_bayesian_app/nrl_results.csv"))

# -------------------------
# Helpers
# -------------------------
def prepare_completed_results(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to 'Full Time' and ensure numeric scores."""
    if df.empty:
        return df
    df = df[df["status"].fillna("").str.lower().eq("full time")].copy()
    if df.empty:
        return df
    df["home_team"] = df["home_team"].map(TEAM_NAME_MAP).fillna(df["home_team"])
    df["away_team"] = df["away_team"].map(TEAM_NAME_MAP).fillna(df["away_team"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df

def compute_ladder_from_results(df: pd.DataFrame, all_teams: list[str]) -> pd.DataFrame:
    """
    Build ladder with GP, W, D, L, PF, PA, Diff, CompPts from completed matches.
    Why: app needs a consistent starting ladder from historical results.
    """
    stats = {
        t: {"GP": 0, "W": 0, "D": 0, "L": 0, "PF": 0, "PA": 0, "Diff": 0, "CompPts": 0}
        for t in all_teams
    }
    for _, r in df.iterrows():
        h, a = r["home_team"], r["away_team"]
        if h not in stats or a not in stats:
            continue
        hs, as_ = int(r["home_score"]), int(r["away_score"])
        stats[h]["GP"] += 1; stats[a]["GP"] += 1
        stats[h]["PF"] += hs; stats[h]["PA"] += as_
        stats[a]["PF"] += as_; stats[a]["PA"] += hs
        if hs > as_:
            stats[h]["W"] += 1; stats[a]["L"] += 1
            stats[h]["CompPts"] += 2
        elif hs < as_:
            stats[a]["W"] += 1; stats[h]["L"] += 1
            stats[a]["CompPts"] += 2
        else:
            stats[h]["D"] += 1; stats[a]["D"] += 1
            stats[h]["CompPts"] += 1; stats[a]["CompPts"] += 1
    for t in all_teams:
        stats[t]["Diff"] = stats[t]["PF"] - stats[t]["PA"]
    ladder_df = (
        pd.DataFrame.from_dict(stats, orient="index")
        .assign(Team=lambda x: x.index)
        [["Team","GP","W","D","L","PF","PA","Diff","CompPts"]]
        .sort_values(["CompPts","Diff","PF"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    return ladder_df

# -------------------------
# Load CSV and prepare results (with diagnostics)
# -------------------------
st.divider()
st.subheader("🔎 Diagnostics")
st.write({
    "app_file": __file__,
    "repo_csv": str(REPO_DATA_CSV),
    "repo_exists": REPO_DATA_CSV.exists(),
    "desktop_csv": str(DESKTOP_CSV),
    "desktop_exists": DESKTOP_CSV.exists(),
})

if REPO_DATA_CSV.exists():
    RESULTS_CSV = REPO_DATA_CSV
elif DESKTOP_CSV.exists():
    RESULTS_CSV = DESKTOP_CSV
else:
    RESULTS_CSV = None

if RESULTS_CSV is not None and RESULTS_CSV.exists():
    raw_df = pd.read_csv(RESULTS_CSV)
    results_df = prepare_completed_results(raw_df)
    st.caption(f"Loaded {len(raw_df)} rows from: {RESULTS_CSV.resolve()} • Completed matches: {len(results_df)}")
else:
    st.error(f"No CSV found. Expected one of: {REPO_DATA_CSV} or {DESKTOP_CSV}.")
    results_df = pd.DataFrame(columns=["home_team","away_team","home_score","away_score","status"])

st.divider()

# -------------------------
# Ladder from completed results (top of app)
# -------------------------
st.subheader("📥 Current Ladder (completed matches only)")
ladder_df = compute_ladder_from_results(results_df, ALL_TEAMS)
st.dataframe(ladder_df.set_index("Team"), use_container_width=True)

# Seeds for the sim:
teams_data = {r.Team: {"CompPts": int(r.CompPts), "Diff": int(r.Diff)} for r in ladder_df.itertuples(index=False)}
teams = list(teams_data.keys())

# -------------------------
# Sidebar: Priors & controls
# -------------------------
st.sidebar.header("🧠 Prior Beliefs")
strength_ratings = {}
variability_ratings = {}

st.sidebar.caption("Give each team a **Strength** and **Variability** (5 = neutral).")
for team in teams:
    with st.sidebar.expander(team, expanded=False):
        strength_ratings[team] = st.slider(f"{team} – Strength", 0, 10, 5, key=f"s_{team}")
        variability_ratings[team] = st.slider(f"{team} – Variability", 0, 10, 5, key=f"v_{team}")

num_sims = st.sidebar.slider("🔁 Number of Simulations", 500, 10000, 2000, step=500)
h = st.sidebar.slider("🏠 Home Advantage (strength units)", 0.0, 1.0, 0.3, 0.05)
alpha = st.sidebar.slider("📈 Strength→Margin scale (α)", 5.0, 15.0, 10.0, 0.5)
sigma = st.sidebar.slider("🎲 Match randomness (σ)", 6.0, 20.0, 12.0, 0.5)

# -------------------------
# Remaining fixtures (placeholder list you curated)
# -------------------------
fixtures = [
    {"home": "Knights", "away": "Panthers"}, {"home": "Raiders", "away": "Sea Eagles"},
    {"home": "Dragons", "away": "Sharks"}, {"home": "Dolphins", "away": "Roosters"},
    {"home": "Bulldogs", "away": "Warriors"}, {"home": "Titans", "away": "Rabbitohs"},
    {"home": "Eels", "away": "Cowboys"}, {"home": "Panthers", "away": "Storm"},
    {"home": "Warriors", "away": "Dragons"}, {"home": "Roosters", "away": "Bulldogs"},
    {"home": "Sharks", "away": "Titans"}, {"home": "Broncos", "away": "Dolphins"},
    {"home": "Rabbitohs", "away": "Eels"}, {"home": "Wests Tigers", "away": "Sea Eagles"},
    {"home": "Cowboys", "away": "Knights"}, {"home": "Rabbitohs", "away": "Dragons"},
    {"home": "Panthers", "away": "Raiders"}, {"home": "Storm", "away": "Bulldogs"},
    {"home": "Sea Eagles", "away": "Dolphins"}, {"home": "Titans", "away": "Warriors"},
    {"home": "Eels", "away": "Roosters"}, {"home": "Knights", "away": "Broncos"},
    {"home": "Wests Tigers", "away": "Cowboys"}, {"home": "Bulldogs", "away": "Panthers"},
    {"home": "Warriors", "away": "Eels"}, {"home": "Storm", "away": "Roosters"},
    {"home": "Raiders", "away": "Wests Tigers"}, {"home": "Dragons", "away": "Sea Eagles"},
    {"home": "Cowboys", "away": "Broncos"}, {"home": "Sharks", "away": "Knights"},
    {"home": "Dolphins", "away": "Titans"}, {"home": "Broncos", "away": "Storm"},
    {"home": "Sea Eagles", "away": "Warriors"}, {"home": "Roosters", "away": "Rabbitohs"},
    {"home": "Dragons", "away": "Panthers"}, {"home": "Titans", "away": "Wests Tigers"},
    {"home": "Bulldogs", "away": "Sharks"}, {"home": "Dolphins", "away": "Raiders"},
    {"home": "Eels", "away": "Knights"},
]

# -------------------------
# Simulation
# -------------------------
if st.button("▶️ Run Simulation"):
    diffs_arr = np.array([info["Diff"] for info in teams_data.values()])
    mean_diff = np.mean(diffs_arr) if len(diffs_arr) else 0.0
    std_diff = np.std(diffs_arr) if np.std(diffs_arr) > 0 else 1.0  # guard

    priors = {}
    for t in teams:
        base = (teams_data[t]["Diff"] - mean_diff) / std_diff
        strength_adj = (strength_ratings[t] - 5) / 5 * 1.5
        std_adj = 0.5 + variability_ratings[t] / 10 * 1.5
        priors[t] = {"mean": base + strength_adj, "std": std_adj}

    ladder_samples = {t: [] for t in teams}
    all_ladders = []
    top8_sets = []

    def sim_once():
        strengths = {t: np.random.normal(priors[t]["mean"], priors[t]["std"]) for t in teams}
        pts = {t: teams_data[t]["CompPts"] for t in teams}
        diffs = {t: teams_data[t]["Diff"] for t in teams}

        for m in fixtures:
            h_team, a_team = m["home"], m["away"]
            if h_team not in strengths or a_team not in strengths:
                continue
            margin = np.random.normal(alpha * (strengths[h_team] + h - strengths[a_team]), sigma)
            if margin > 0:
                pts[h_team] += 2
            elif margin < 0:
                pts[a_team] += 2
            else:
                pts[h_team] += 1; pts[a_team] += 1
            diffs[h_team] += max(0.0, margin)
            diffs[a_team] -= max(0.0, margin)

        return sorted(pts.items(), key=lambda x: (-x[1], -diffs[x[0]]))

    for _ in range(num_sims):
        ladder = sim_once()
        all_ladders.append(ladder)
        top8_sets.append(frozenset([team for team, _ in ladder[:8]]))
        for pos_idx, (t, _) in enumerate(ladder):
            ladder_samples[t].append(pos_idx + 1)

    # -------------------------
    # Final Ladder Probabilities
    # -------------------------
    st.subheader("📊 Final Ladder Probabilities")
    summary_rows = []
    for t in teams:
        pos = np.array(ladder_samples[t])
        row = {
            "Team": t,
            "Top 4 %": (pos <= 4).mean() * 100,
            "Top 8 %": (pos <= 8).mean() * 100,
            "Minor Prem. %": (pos == 1).mean() * 100,
            "Wooden Spoon %": (pos == len(teams)).mean() * 100,
            "Median Pos": int(np.median(pos)),
        }
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows).sort_values("Median Pos")
    st.dataframe(df_summary.set_index("Team").style.format("{:.1f}"))

    # -------------------------
    # Position distribution chart
    # -------------------------
    st.subheader("📈 Ladder Position Distribution")
    fig, ax = plt.subplots(figsize=(12, 8))
    sorted_teams = df_summary["Team"].tolist()
    pos_matrix = np.zeros((len(teams), len(teams)))
    for i, t in enumerate(sorted_teams):
        counts = Counter(ladder_samples[t])
        for pos, count in counts.items():
            pos_matrix[i, pos - 1] = count / num_sims
    bottom = np.zeros(len(teams))
    for p in range(len(teams)):
        ax.barh(sorted_teams, pos_matrix[:, p], left=bottom, label=str(p + 1))
        bottom += pos_matrix[:, p]
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Predicted Ladder Position Distributions")
    st.pyplot(fig)

    # -------------------------
    # Top 8 (unordered)
    # -------------------------
    st.subheader("🎯 Most Common Top 8 Sets (Unordered)")
    top8_counts = Counter(top8_sets)
    top8_table = pd.DataFrame([
        {"Top 8": ", ".join(sorted(combo)), "Probability": f"{(count / num_sims):.2%}"}
        for combo, count in top8_counts.most_common(10)
    ])
    st.dataframe(top8_table)

    # -------------------------
    # Top 8 (ordered)
    # -------------------------
    st.subheader("🏅 Most Common Top 8 Orders")
    top8_ordered = [tuple(t for t, _ in ladder[:8]) for ladder in all_ladders]
    top8_ordered_counts = Counter(top8_ordered)
    top8_ordered_table = pd.DataFrame([
        {"Top 8 Order": " > ".join(combo), "Probability": f"{(count / num_sims):.2%}"}
        for combo, count in top8_ordered_counts.most_common(10)
    ])
    st.dataframe(top8_ordered_table)

    # -------------------------
    # Top 4 (unordered)
    # -------------------------
    st.subheader("🧢 Most Common Top 4 Sets (Unordered)")
    top4_unordered = [frozenset(t for t, _ in ladder[:4]) for ladder in all_ladders]
    top4_unordered_counts = Counter(top4_unordered)
    top4_unordered_table = pd.DataFrame([
        {"Top 4 Teams": ", ".join(sorted(combo)), "Probability": f"{(count / num_sims):.2%}"}
        for combo, count in top4_unordered_counts.most_common(10)
    ])
    st.dataframe(top4_unordered_table)

    # -------------------------
    # Top 4 (ordered)
    # -------------------------
    st.subheader("🥇 Most Common Top 4 Orders")
    top4_ordered = [tuple(t for t, _ in ladder[:4]) for ladder in all_ladders]
    top4_ordered_counts = Counter(top4_ordered)
    top4_ordered_table = pd.DataFrame([
        {"Top 4 Order": " > ".join(combo), "Probability": f"{(count / num_sims):.2%}"}
        for combo, count in top4_ordered_counts.most_common(10)
    ])
    st.dataframe(top4_ordered_table)
