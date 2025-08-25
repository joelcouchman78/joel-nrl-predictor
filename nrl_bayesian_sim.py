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
# Team name canonicalisation
# -------------------------
import re as _re

def _norm_team(s: str) -> str:
    return _re.sub(r"[^a-z0-9]", "", str(s).lower())

_TEAM_ALIAS_NORM = {
    "broncos":"Broncos","sharks":"Sharks","dragons":"Dragons","seaeagles":"Sea Eagles",
    "warriors":"Warriors","bulldogs":"Bulldogs","rabbitohs":"Rabbitohs","roosters":"Roosters",
    "panthers":"Panthers","eels":"Eels","knights":"Knights","titans":"Titans","storm":"Storm",
    "cowboys":"Cowboys","raiders":"Raiders","dolphins":"Dolphins",
    "weststigers":"Wests Tigers","westtigers":"Wests Tigers","tigers":"Wests Tigers",
    "manlywarringahseaeagles":"Sea Eagles","manlyseaeagles":"Sea Eagles",
    "stgeorgeillawarradragons":"Dragons","cronullasutherlandsharks":"Sharks","cronullasharks":"Sharks",
    "newzealandwarriors":"Warriors","aucklandwarriors":"Warriors",
    "canterburybankstownbulldogs":"Bulldogs","canterburybulldogs":"Bulldogs",
    "southsydneyrabbitohs":"Rabbitohs","sydneyroosters":"Roosters",
    "penrithpanthers":"Panthers","parramattaeels":"Eels","newcastleknights":"Knights",
    "goldcoasttitans":"Titans","melbournestorm":"Storm","northqueenslandcowboys":"Cowboys",
    "canberraraiders":"Raiders","thedolphins":"Dolphins",
    "st.georgeillawarradragons":"Dragons",
}

def canonical_team_name(name: str) -> str | None:
    if name is None:
        return None
    s = str(name).strip()
    if "ALL_TEAMS" in globals() and s in ALL_TEAMS:
        return s
    if "TEAM_NAME_MAP" in globals():
        mapped = TEAM_NAME_MAP.get(s)
        if mapped is not None:
            return mapped
    key = _norm_team(s)
    mapped = _TEAM_ALIAS_NORM.get(key)
    if mapped is not None:
        return mapped
    if "ALL_TEAMS" in globals():
        for t in ALL_TEAMS:
            if _norm_team(t) in key or key in _norm_team(t):
                return t
    return None


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
    raw_df = pd.read_csv(RESULTS_CSV)
    results_df = prepare_completed_results(raw_df)
    st.caption(f"Loaded {len(raw_df)} rows from: {RESULTS_CSV.resolve()} • Completed matches: {len(results_df)}")
else:
    st.error(f"No CSV found. Expected one of: {REPO_DATA_CSV} or {DESKTOP_CSV}.")
    results_df = pd.DataFrame(columns=["home_team","away_team","home_score","away_score","status"])

st.divider()

# -------------------------
# Rest of the app logic follows...
# -------------------------

# -------------------------
# Rest of the app logic follows...
# -------------------------



# -------------------------
# Ladder from completed results (top of app)
# -------------------------
st.subheader("📥 Current Ladder (completed matches only)")

# -------------------------
# Ladder computation (GP/W/D/L, PF/PA, Diff, CompPts)
# -------------------------
def compute_ladder_from_results(df: pd.DataFrame, all_teams: list[str]) -> pd.DataFrame:
    import pandas as _pd
    base = {t: {"GP":0,"W":0,"D":0,"L":0,"PF":0,"PA":0,"Diff":0,"CompPts":0} for t in all_teams}
    if df is None or not isinstance(df, _pd.DataFrame) or df.empty:
        return (_pd.DataFrame.from_dict(base, orient="index")
                .assign(Team=lambda x: x.index)
                [["Team","GP","W","D","L","PF","PA","Diff","CompPts"]]
                .sort_values(["CompPts","Diff","PF"], ascending=[False,False,False])
                .reset_index(drop=True))
    home_col = "home_team" if "home_team" in df.columns else ("home" if "home" in df.columns else None)
    away_col = "away_team" if "away_team" in df.columns else ("away" if "away" in df.columns else None)
    hs_col, as_col = "home_score", "away_score"
    if not all(c in df.columns for c in [home_col, away_col, hs_col, as_col] if c):
        return (_pd.DataFrame.from_dict(base, orient="index")
                .assign(Team=lambda x: x.index)
                [["Team","GP","W","D","L","PF","PA","Diff","CompPts"]]
                .sort_values(["CompPts","Diff","PF"], ascending=[False,False,False])
                .reset_index(drop=True))
    d = df.copy()
    d[hs_col] = _pd.to_numeric(d[hs_col], errors="coerce").fillna(0).astype(int)
    d[as_col] = _pd.to_numeric(d[as_col], errors="coerce").fillna(0).astype(int)

    stats = {t: vals.copy() for t, vals in base.items()}
    for _, r in d.iterrows():
        h = canonical_team_name(r[home_col])
        a = canonical_team_name(r[away_col])
        if h is None or a is None or h not in stats or a not in stats:
            continue
        hs, as_ = int(r[hs_col]), int(r[as_col])
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
    ladder_df = (_pd.DataFrame.from_dict(stats, orient="index")
                 .assign(Team=lambda x: x.index)
                 [["Team","GP","W","D","L","PF","PA","Diff","CompPts"]]
                 .sort_values(["CompPts","Diff","PF"], ascending=[False,False,False])
                 .reset_index(drop=True))
    return ladder_df

ladder_df = compute_ladder_from_results(results_df, ALL_TEAMS)
st.dataframe(ladder_df[["Team","GP","W","D","L","PF","PA","Diff","CompPts"]], use_container_width=True)


# (bye augmentation removed)
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
# Remaining fixtures (CSV only; no static fallback)
# -------------------------
COMPLETED_STATUSES = {"full time", "ft", "final", "finished", "complete"}
def _is_completed(s: str) -> bool:
    return str(s).strip().lower() in COMPLETED_STATUSES
def build_future_fixtures_from_csv(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    d = df.copy()
    if "status" in d.columns:
        d = d[~d["status"].map(_is_completed)]
    hcol = "home_team" if "home_team" in d.columns else ("home" if "home" in d.columns else None)
    acol = "away_team" if "away_team" in d.columns else ("away" if "away" in d.columns else None)
    if not hcol or not acol:
        return []
    out = []
    for _, r in d.iterrows():
        h = canonical_team_name(r[hcol]); a = canonical_team_name(r[acol])
        if h in ALL_TEAMS and a in ALL_TEAMS:
            rnd = r["round"] if "round" in d.columns else None
            out.append({"home": h, "away": a, "round": rnd})
    return out
fixtures = build_future_fixtures_from_csv(raw_df)
if not fixtures:
    st.error("No future fixtures found in CSV — cannot simulate. Check CSV columns & status values.")
    st.stop()
# Remaining fixtures from CSV (+ round), and helpers for byes
# -------------------------
def _status_is_completed(s):
    s = str(s).strip().lower()
    return s in {"full time", "ft", "final", "finished", "complete"}

def build_future_fixtures_from_csv(raw_df: pd.DataFrame):
    """
    Return a list of dicts with keys: {'round','home','away'} for matches not yet completed.
    Team names are canonicalised to match ALL_TEAMS.
    """
    if raw_df is None or raw_df.empty:
        return []
    d = raw_df.copy()
    if "status" in d.columns:
        mask = ~d["status"].map(_status_is_completed)
        d = d.loc[mask]
    hcol = "home_team" if "home_team" in d.columns else ("home" if "home" in d.columns else None)
    acol = "away_team" if "away_team" in d.columns else ("away" if "away" in d.columns else None)
    if hcol is None or acol is None:
        return []
    out = []
    for _, r in d.iterrows():
        rnd = r["round"] if "round" in d.columns else None
        h = canonical_team_name(r[hcol])
        a = canonical_team_name(r[acol])
        if h in ALL_TEAMS and a in ALL_TEAMS:
            out.append({"round": rnd, "home": h, "away": a})
    return out

def present_teams_by_round_full(raw_df: pd.DataFrame):
    """
    For *every* round in the CSV (completed or not), return the set of teams who play in that round.
    This lets us identify byes for both completed rounds (for baseline) and future rounds (in simulation).
    """
    present = {}
    if raw_df is None or raw_df.empty:
        return present
    hcol = "home_team" if "home_team" in raw_df.columns else ("home" if "home" in raw_df.columns else None)
    acol = "away_team" if "away_team" in raw_df.columns else ("away" if "away" in raw_df.columns else None)
    if hcol is None or acol is None or "round" not in raw_df.columns:
        return present
    for rnd, sub in raw_df.groupby("round"):
        homes = sub[hcol].map(canonical_team_name)
        aways = sub[acol].map(canonical_team_name)
        teams = set(h for h in homes.dropna() if h in ALL_TEAMS) | set(a for a in aways.dropna() if a in ALL_TEAMS)
        present[rnd] = teams
    return present

def compute_bye_points_so_far(raw_df: pd.DataFrame, all_teams: list[str]):
    """
    Award 2 comp points per bye for *completed rounds only*.
    A round is completed iff all rows for that round have a completed status.
    """
    bye_pts = {t: 0 for t in all_teams}
    if raw_df is None or raw_df.empty or "round" not in raw_df.columns or "status" not in raw_df.columns:
        return bye_pts
    # completion per round
    s = raw_df["status"].map(_status_is_completed)
    completed_round = {rnd: bool(s.loc[sub.index].all()) for rnd, sub in raw_df.groupby("round")}
    present_map = present_teams_by_round_full(raw_df)
    for rnd, teams_in_round in present_map.items():
        if completed_round.get(rnd, False):
            byes = set(all_teams) - set(teams_in_round)
            for t in byes:
                bye_pts[t] += 2
    return bye_pts

def _round_sort_key(r):
    import re as _re
    s = str(r)
    m = _re.search(r'\d+', s)
    return int(m.group()) if m else 10**9

# Build fixtures and round->present map from the CSV we already loaded
fixtures = build_future_fixtures_from_csv(raw_df)
round_present_map = present_teams_by_round_full(raw_df)

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

    # draw a strength sample per team from priors
    strengths = {}
    for t in teams:
        mu, tau = prior_params[t]["mu"], prior_params[t]["tau"]
        strengths[t] = np.random.normal(mu, tau)

    # start from the real ladder
    pts   = {t: int(teams_data[t]["CompPts"]) for t in teams}
    diffs = {t: float(teams_data[t]["Diff"])   for t in teams}

    # simulate each remaining fixture (from CSV only)
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
# ---- Diagnostics: confirm fixtures come from CSV ----
with st.expander("🧭 Fixture source check", expanded=True):
    try:
        st.write("Future fixtures loaded from CSV:", len(fixtures))
        st.dataframe(pd.DataFrame(fixtures).head(12), use_container_width=True)
    except Exception as _e:
        st.write("No fixtures diagnostic available:", _e)
