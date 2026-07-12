# NRL Bayesian Ladder Predictor

A Streamlit application for simulating the 2026 NRL regular-season ladder. The model starts from completed results and credited byes, simulates every remaining fixture, and updates competition points, points for, points against and differential before applying the ladder ordering.

Users can adjust beliefs about team strength, persistent strength uncertainty, home advantage and match randomness.

## Projection outputs

The application reports:

- expected, median and modal finishing positions;
- an 80% finishing-position range and position variance/standard deviation;
- the probability of every exact ladder position;
- Top 2, Top 4, Top 8, bottom-four, minor-premiership and wooden-spoon probabilities;
- projected final competition points and points differential;
- the simulated eighth-place competition-points cutoff;
- finals probabilities conditional on a team recording a given number of remaining wins;
- match win probabilities and expected margins;
- fixture leverage: how much a win rather than a loss changes a team's Top 4 or Top 8 probability;
- a scenario explorer that filters simulations by one or more forced future results;
- common Top 4 and Top 8 sets and orderings;
- Monte Carlo sampling-error diagnostics and data provenance.

Built with Python, Streamlit, NumPy, pandas and Matplotlib.
