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
    try:
        raw_df = pd.read_csv(RESULTS_CSV)
    except Exception as e:
        st.error(f"Failed to read CSV at {RESULTS_CSV}: {e}")
        raw_df = None
    if isinstance(raw_df, pd.DataFrame):
        try:
            results_df = prepare_completed_results(raw_df)
        except Exception as e:
            st.error(f"Failed to prepare completed results: {e}")
            results_df = pd.DataFrame(columns=["home_team","away_team","home_score","away_score","status"]) 
        st.caption(
            f"Loaded {raw_df.shape[0]} rows from: {RESULTS_CSV.resolve()} • Completed matches: {results_df.shape[0]}"
        )
    else:
        st.error(
            "No readable CSV found. "
            f"Checked: {REPO_DATA_CSV} and {DESKTOP_CSV}. "
            "Commit a CSV to the repo (data/nrl_results.csv) or update the path."
        )
        results_df = pd.DataFrame(columns=["home_team","away_team","home_score","away_score","status"])
else:
    st.error(
        "No readable CSV found. "
        f"Checked: {REPO_DATA_CSV} and {DESKTOP_CSV}. "
        "Commit a CSV to the repo (data/nrl_results.csv) or update the path."
    )
    results_df = pd.DataFrame(columns=["home_team","away_team","home_score","away_score","status"])st.divider()

# -------------------------
# Rest of the app logic follows...
# -------------------------

# -------------------------
# Rest of the app logic follows...
# -------------------------



# -------------------------
# Ladder from completed results (top of app)
# -------------------------

# -------------------------
# Ensure ladder has standard columns and names
# -------------------------
def ensure_ladder_columns(df):
    """
    Normalize/complete ladder columns so we can always display:
    ['Team','Played','Won','Drawn','Lost','CompPts','Diff']
    - Renames common variants (team/points/diff/gp/games/wins/losses/draws).
    - If Team is in the index, moves it to a column.
    - If Played missing but Won/Drawn/Lost exist, computes Played = Won+Drawn+Lost.
    - Fills any still-missing columns with zeros.
    """
    import pandas as _pd
    if df is None or not isinstance(df, _pd.DataFrame):
        return _pd.DataFrame(columns=["Team","Played","Won","Drawn","Lost","CompPts","Diff"])

    # If Team is not a column but the index looks like teams, bring it out.
    if "Team" not in df.columns:
        if df.index.name and df.index.name.lower() == "team":
            df = df.reset_index()
        else:
            # generic: promote index to a column if it's unnamed
            df = df.reset_index()

    # Build a case-insensitive rename map for common variants
    lower_map = {c.lower(): c for c in df.columns}
    rename = {}
    if "team" in lower_map:   rename[lower_map["team"]]   = "Team"
    if "points" in lower_map: rename[lower_map["points"]] = "CompPts"
    if "comppts" in lower_map:rename[lower_map["comppts"]]= "CompPts"
    if "diff" in lower_map:   rename[lower_map["diff"]]   = "Diff"
    if "played" in lower_map: rename[lower_map["played"]] = "Played"
    if "gp" in lower_map:     rename[lower_map["gp"]]     = "Played"
    if "games" in lower_map:  rename[lower_map["games"]]  = "Played"
    if "won" in lower_map:    rename[lower_map["won"]]    = "Won"
    if "wins" in lower_map:   rename[lower_map["wins"]]   = "Won"
    if "drawn" in lower_map:  rename[lower_map["drawn"]]  = "Drawn"
    if "draws" in lower_map:  rename[lower_map["draws"]]  = "Drawn"
    if "lost" in lower_map:   rename[lower_map["lost"]]   = "Lost"
    if "losses" in lower_map: rename[lower_map["losses"]] = "Lost"
    df = df.rename(columns=rename)

    # Add any missing columns (compute Played when possible)
    for col in ["Team","Played","Won","Drawn","Lost","CompPts","Diff"]:
        if col not in df.columns:
            if col == "Played" and all(x in df.columns for x in ["Won","Drawn","Lost"]):
                df["Played"] = df["Won"].fillna(0).astype(int) + df["Drawn"].fillna(0).astype(int) + df["Lost"].fillna(0).astype(int)
            else:
                df[col] = 0

    # Order and return
    return df[["Team","Played","Won","Drawn","Lost","CompPts","Diff"]]


# -----------------------------------------------------------------------------
# Canonicalise completed results (safe on frames without 'status')
# -----------------------------------------------------------------------------
def canonicalize_completed_results(df):
    """
    Return columns: home, away, home_score, away_score.
    If a 'status' column exists, keep only completed rows.
    If no 'status' column exists, assume df already contains completed matches.
    Recognises 'home_team'/'away_team' and common score variants.
    """
    import pandas as _pd

    cols_out = ["home", "away", "home_score", "away_score"]
    if df is None or not isinstance(df, _pd.DataFrame) or df.empty:
        return _pd.DataFrame(columns=cols_out)

    out = df.copy()

    # normalise home/away names
    if "home" not in out.columns and "home_team" in out.columns:
        out = out.rename(columns={"home_team": "home"})
    if "away" not in out.columns and "away_team" in out.columns:
        out = out.rename(columns={"away_team": "away"})

    # status-based filtering ONLY if status exists
    if "status" in out.columns:
        s = out["status"].astype(str).str.strip().str.lower()
        # treat a range of completed markers as completed
        completed_mask = s.str.contains(r"^full\s*time$|^ft$|full\s*time|final|finished|complete", regex=True)
        out = out[completed_mask].copy()

    # choose score column names robustly
    score_candidates = [
        ("home_score", "away_score"),
        ("home_points", "away_points"),
        ("home_pts", "away_pts"),
        ("home", "away"),  # very last resort if scores were already numeric columns (unlikely)
    ]
    use_hs, use_as = None, None
    for hs, a_s in score_candidates:
        if hs in out.columns and a_s in out.columns:
            use_hs, use_as = hs, a_s
            break
    if use_hs is None:
        # try to infer by pattern
        for c in out.columns:
            cl = c.lower()
            if "home" in cl and ("score" in cl or "points" in cl or cl.endswith("_pts")):
                use_hs = c
            if "away" in cl and ("score" in cl or "points" in cl or cl.endswith("_pts")):
                use_as = c
        if use_hs is None or use_as is None:
            # cannot determine, return empty canonical frame
            return _pd.DataFrame(columns=cols_out)

    # coerce scores to int
    out["home_score"] = _pd.to_numeric(out[use_hs], errors="coerce").fillna(0).astype(int)
    out["away_score"] = _pd.to_numeric(out[use_as], errors="coerce").fillna(0).astype(int)

    # map team names to canonical ones if available
    if "TEAM_NAME_MAP" in globals():
        out["home"] = out["home"].map(TEAM_NAME_MAP).fillna(out["home"])
        out["away"] = out["away"].map(TEAM_NAME_MAP).fillna(out["away"])

    # filter to known teams if ALL_TEAMS is present
    if "ALL_TEAMS" in globals():
        out = out[out["home"].isin(ALL_TEAMS) & out["away"].isin(ALL_TEAMS)]

    return out[cols_out]

st.subheader("📥 Current Ladder (completed matches only)")

# -------------------------
# Ladder computation (robust)
# -------------------------
def compute_ladder_from_results(df, all_teams: list[str]) -> pd.DataFrame:
    """
    Build ladder with Team, CompPts, Diff from completed matches.
    Accepts either columns (home_team, away_team, home_score, away_score) or
    canonical (home, away, home_score, away_score).
    """
    import pandas as _pd
    # Empty -> zeroed ladder
    if df is None or not isinstance(df, _pd.DataFrame) or df.empty:
        ladder = _pd.DataFrame({"Team": all_teams, "CompPts": [0]*len(all_teams), "Diff": [0]*len(all_teams)})
        return ladder.sort_values(["CompPts","Diff"], ascending=[False, False]).reset_index(drop=True)

    # Normalise column names
    low_cols = [c.lower() for c in df.columns]
    use_home = "home" if "home" in low_cols else ("home_team" if "home_team" in df.columns else None)
    use_away = "away" if "away" in low_cols else ("away_team" if "away_team" in df.columns else None)
    use_hs   = "home_score"
    use_as   = "away_score"
    if not all(c in df.columns for c in [use_home, use_away, use_hs, use_as] if c):
        raise ValueError("compute_ladder_from_results: missing required columns")

    d = df.copy()
    d[use_hs] = _pd.to_numeric(d[use_hs], errors="coerce").fillna(0).astype(int)
    d[use_as] = _pd.to_numeric(d[use_as], errors="coerce").fillna(0).astype(int)

    pts = {t: 0 for t in all_teams}
    dif = {t: 0 for t in all_teams}

    for _, r in d.iterrows():
        h = r[use_home]; a = r[use_away]
        if h not in pts or a not in pts:
            continue
        hs = int(r[use_hs]); as_ = int(r[use_as])
        margin = hs - as_
        # symmetric differential
        dif[h] += margin
        dif[a] -= margin
        # competition points
        if hs > as_:
            pts[h] += 2
        elif hs < as_:
            pts[a] += 2
        else:
            pts[h] += 1; pts[a] += 1

    ladder = _pd.DataFrame({
        "Team": all_teams,
        "CompPts": [pts[t] for t in all_teams],
        "Diff":    [dif[t] for t in all_teams],
    })
    return ladder.sort_values(["CompPts","Diff"], ascending=[False, False]).reset_index(drop=True)

ladder_df = compute_ladder_from_results(results_df, ALL_TEAMS)
ladder_df = ensure_ladder_columns(ladder_df)
st.dataframe(ladder_df, use_container_width=True)

teams_data = {r.Team: {"CompPts": int(r.CompPts), "Diff": int(r.Diff)} for r in ladder_df.itertuples(index=False)}
teams = list(teams_data.keys())

# -------------------------
# Sidebar: Priors & controls
# -------------------------
st.sidebar.header("🧠 Prior Beliefs")
strength_ratings = {}
variability_ratings = {}

st.sidebar.caption("Give each team a **Strength** (how good you think they are) and **Variability** (how up‑and‑down they are). 5 = neutral.")
for team in teams:
    with st.sidebar.expander(team, expanded=False):
        strength_ratings[team] = st.slider(f"{team} – Strength", 0, 10, 5, key=f"s_{team}")
        variability_ratings[team] = st.slider(f"{team} – Variability", 0, 10, 5, key=f"v_{team}")

num_sims = st.sidebar.slider("🔁 Number of Simulations", 500, 10000, 2000, step=500)
h = st.sidebar.slider("🏠 Home Advantage (strength units)", 0.0, 1.0, 0.3, 0.05)
alpha = st.sidebar.slider("📈 Strength→Margin scale (α)", 5.0, 15.0, 10.0, 0.5)
sigma = st.sidebar.slider("🎲 Match randomness (σ)", 6.0, 20.0, 12.0, 0.5)


# -------------------------
# Build future fixtures from CSV (status != Full Time)
# -------------------------
def build_future_fixtures(raw_df: pd.DataFrame) -> list[dict]:
    if raw_df is None or not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
        return []
    df = raw_df.copy()
    if "status" not in df.columns:
        return []
    s = df["status"].astype(str).str.strip().str.lower()
    df = df[~s.eq("full time")].copy()

    # Derive home/away
    if "TEAM_NAME_MAP" in globals():
        df["home"] = df.get("home", df.get("home_team")).map(TEAM_NAME_MAP).fillna(df.get("home", df.get("home_team")))
        df["away"] = df.get("away", df.get("away_team")).map(TEAM_NAME_MAP).fillna(df.get("away", df.get("away_team")))
    else:
        df["home"] = df.get("home", df.get("home_team"))
        df["away"] = df.get("away", df.get("away_team"))

    # Filter and dedupe
    if "ALL_TEAMS" in globals():
        df = df[df["home"].isin(ALL_TEAMS) & df["away"].isin(ALL_TEAMS)]
    df = df.dropna(subset=["home","away"]).drop_duplicates(subset=["home","away"], keep="first")

    return df[["home","away"]].to_dict(orient="records")

# -------------------------
# Remaining fixtures (placeholder list you curated)
# -------------------------
fixtures = build_future_fixtures(raw_df)
if not fixtures:
    st.error('No future fixtures found in CSV. Ensure upcoming games are present with status != "Full Time".')
# -------------------------
# Simulation
# -------------------------
if st.button("▶️ Run Simulation"):

    # Normalize team-strength base from current points diff
    diffs_arr = np.array([info["Diff"] for info in teams_data.values()])
    mean_diff = np.mean(diffs_arr) if len(diffs_arr) else 0.0
    std_diff = np.std(diffs_arr) if np.std(diffs_arr) > 0 else 1.0

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
            diffs[h_team] += margin
            diffs[a_team] -= margin

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