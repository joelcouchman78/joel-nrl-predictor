import streamlit as st
import numpy as np
import pandas as pd
from scipy.special import expit
from collections import Counter
import matplotlib.pyplot as plt

# === BASIC SETTINGS ===
st.set_page_config(page_title="NRL Bayesian Ladder Predictor", layout="wide")
st.title("🏉 Joel's NRL Ladder Predictor (2025)")

st.markdown("Adjust your beliefs about team strength & variability. Click **Run Simulation** to update predictions.")

# === TEAM DATA ===
teams_data = {
    'Storm': {'Diff': 248, 'CompPts': 30},
    'Raiders': {'Diff': 152, 'CompPts': 32},
    'Bulldogs': {'Diff': 122, 'CompPts': 28},
    'Sharks': {'Diff': 43,  'CompPts': 24},
    'Panthers': {'Diff': 83,  'CompPts': 23},
    'Broncos': {'Diff': 104, 'CompPts': 22},
    'Dolphins': {'Diff': 171, 'CompPts': 20},
    'Warriors': {'Diff': 26,  'CompPts': 24},
    'Roosters': {'Diff': 20,  'CompPts': 18},
    'Sea Eagles': {'Diff': 6, 'CompPts': 18},
    'Wests Tigers': {'Diff': -121, 'CompPts': 16},
    'Dragons': {'Diff': -58, 'CompPts': 14},
    'Titans': {'Diff': -153, 'CompPts': 10},
    'Cowboys': {'Diff': -177, 'CompPts': 15},
    'Knights': {'Diff': -116, 'CompPts': 12},
    'Rabbitohs': {'Diff': -197, 'CompPts': 12},
    'Eels': {'Diff': -153, 'CompPts': 12},
}
teams = list(teams_data.keys())

# === INPUT SECTION ===
st.sidebar.header("🧠 Prior Beliefs")

strength_ratings = {}
variability_ratings = {}

for team in teams:
    with st.sidebar.expander(team):
        strength = st.slider(f"{team} - Strength", 0, 10, 5, key=f"s_{team}")
        variability = st.slider(f"{team} - Variability", 0, 10, 5, key=f"v_{team}")
        strength_ratings[team] = strength
        variability_ratings[team] = variability

num_sims = st.sidebar.slider("🔁 Number of Simulations", 500, 10000, 2000, step=500)

# === REMAINING FIXTURES ===
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

# === RUN SIMULATION BUTTON ===
if st.button("▶️ Run Simulation"):

    h = 0.3
    alpha = 10.0
    sigma = 12.0

    diffs = np.array([info['Diff'] for info in teams_data.values()])
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    priors = {}
    for team in teams:
        base = (teams_data[team]['Diff'] - mean_diff) / std_diff
        strength_adj = (strength_ratings[team] - 5) / 5 * 1.5
        std_adj = 0.5 + variability_ratings[team] / 10 * 1.5
        priors[team] = {"mean": base + strength_adj, "std": std_adj}

    ladder_samples = {team: [] for team in teams}
    all_ladders = []
    top8_sets = []

    def sim():
        strengths = {t: np.random.normal(priors[t]['mean'], priors[t]['std']) for t in teams}
        pts = {t: teams_data[t]['CompPts'] for t in teams}
        diffs = {t: teams_data[t]['Diff'] for t in teams}

        for match in fixtures:
            h_team = match["home"]
            a_team = match["away"]
            margin = np.random.normal(alpha * (strengths[h_team] + h - strengths[a_team]), sigma)
            if margin > 0:
                pts[h_team] += 2
            else:
                pts[a_team] += 2
            diffs[h_team] += max(0, margin)
            diffs[a_team] -= max(0, margin)

        return sorted(pts.items(), key=lambda x: (-x[1], -diffs[x[0]]))

    # Run simulations and store results
    for _ in range(num_sims):
        ladder = sim()
        all_ladders.append(ladder)
        top8_sets.append(frozenset([team for team, _ in ladder[:8]]))
        for i, (team, _) in enumerate(ladder):
            ladder_samples[team].append(i + 1)

    # === Final Ladder Probabilities ===
    st.subheader("📊 Final Ladder Probabilities")
    result_data = []
    for team in teams:
        pos = np.array(ladder_samples[team])
        row = {
            "Team": team,
            "Top 4 %": (pos <= 4).mean() * 100,
            "Top 8 %": (pos <= 8).mean() * 100,
            "Minor Prem. %": (pos == 1).mean() * 100,
            "Wooden Spoon %": (pos == len(teams)).mean() * 100,
            "Median Pos": int(np.median(pos))
        }
        result_data.append(row)

    df = pd.DataFrame(result_data).sort_values("Median Pos")
    st.dataframe(df.set_index("Team").style.format("{:.1f}"))

    # === Chart ===
    st.subheader("📈 Ladder Position Distribution")
    fig, ax = plt.subplots(figsize=(12, 8))
    sorted_teams = df["Team"].tolist()
    pos_matrix = np.zeros((len(teams), len(teams)))
    for i, team in enumerate(sorted_teams):
        counts = Counter(ladder_samples[team])
        for pos, count in counts.items():
            pos_matrix[i, pos - 1] = count / num_sims
    bottom = np.zeros(len(teams))
    for pos in range(len(teams)):
        ax.barh(sorted_teams, pos_matrix[:, pos], left=bottom, label=str(pos + 1))
        bottom += pos_matrix[:, pos]
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Predicted Ladder Position Distributions")
    st.pyplot(fig)

    # === Top 8 Unordered ===
    st.subheader("🎯 Most Common Top 8 Sets (Unordered)")
    top8_counts = Counter(top8_sets)
    top8_table = pd.DataFrame([
        {"Top 8": ', '.join(sorted(combo)), "Probability": f"{(count/num_sims):.2%}"}
        for combo, count in top8_counts.most_common(10)
    ])
    st.dataframe(top8_table)

    # === Top 8 Ordered ===
    st.subheader("🏅 Most Common Top 8 Orders")
    top8_ordered = [tuple(team for team, _ in ladder[:8]) for ladder in all_ladders]
    top8_ordered_counts = Counter(top8_ordered)
    top8_ordered_table = pd.DataFrame([
        {"Top 8 Order": ' > '.join(combo), "Probability": f"{(count / num_sims):.2%}"}
        for combo, count in top8_ordered_counts.most_common(10)
    ])
    st.dataframe(top8_ordered_table)

    # === Top 4 Unordered ===
    st.subheader("🧢 Most Common Top 4 Sets (Unordered)")
    top4_unordered = [frozenset(team for team, _ in ladder[:4]) for ladder in all_ladders]
    top4_unordered_counts = Counter(top4_unordered)
    top4_unordered_table = pd.DataFrame([
        {"Top 4 Teams": ', '.join(sorted(combo)), "Probability": f"{(count / num_sims):.2%}"}
        for combo, count in top4_unordered_counts.most_common(10)
    ])
    st.dataframe(top4_unordered_table)

    # === Top 4 Ordered ===
    st.subheader("🥇 Most Common Top 4 Orders")
    top4_ordered = [tuple(team for team, _ in ladder[:4]) for ladder in all_ladders]
    top4_ordered_counts = Counter(top4_ordered)
    top4_ordered_table = pd.DataFrame([
        {"Top 4 Order": ' > '.join(combo), "Probability": f"{(count / num_sims):.2%}"}
        for combo, count in top4_ordered_counts.most_common(10)
    ])
    st.dataframe(top4_ordered_table)