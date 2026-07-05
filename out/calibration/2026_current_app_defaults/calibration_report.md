# Updated NRL Predictor Settings

Generated: 2026-07-05T13:52:43.286592+10:00

Completed matches: **131**
Maximum round represented: **18**

## Recommended global configuration

| Control | Setting |
|---|---:|
| Number of simulations | 20,000 |
| Published probability runs | 50,000 |
| Home advantage | 0.1 |
| Strength-to-margin scale | 10.0 |
| Match randomness | 16.0 |
| Team variability | 0 for every team |
| Strength slider step | 0.5 |


## Match-randomness override

The fitted weighted six-round model produced a raw sigma of approximately 11.0. That value was not used as the app's future-match randomness setting, because the weighted likelihood sigma is not the same quantity as independent future match noise in the app.

The app uses **match randomness = 16.0** with team variability set to zero. Since the app still supplies 5 margin points of persistent uncertainty for each team at variability zero, this gives total margin dispersion:

```text
sqrt(16^2 + 5^2 + 5^2) ≈ 17.49
```

This preserves the updated current-form team sliders while avoiding unrealistically narrow simulations.

## Team strength sliders

**6.0**

- Dolphins
- Dragons

**5.5**

- Sea Eagles
- Cowboys
- Bulldogs
- Eels

**5.0**

- Warriors
- Rabbitohs
- Sharks
- Roosters
- Storm
- Raiders
- Titans

**4.5**

- Panthers
- Knights

**4.0**

- Broncos

**3.5**

- Wests Tigers

## Strength table

| Team         |   Season strength |   Current 6R strength |   Current shift |   Slider continuous |   Slider half-step |
|:-------------|------------------:|----------------------:|----------------:|--------------------:|-------------------:|
| Panthers     |             13.82 |                 12.49 |           -1.34 |                4.55 |               4.50 |
| Dolphins     |              8.44 |                 12.09 |            3.65 |                6.22 |               6.00 |
| Warriors     |              9.02 |                  9.04 |            0.03 |                5.01 |               5.00 |
| Sea Eagles   |              6.87 |                  8.20 |            1.33 |                5.44 |               5.50 |
| Rabbitohs    |              4.79 |                  5.12 |            0.32 |                5.11 |               5.00 |
| Sharks       |              4.09 |                  4.06 |           -0.03 |                4.99 |               5.00 |
| Roosters     |              4.20 |                  3.72 |           -0.47 |                4.84 |               5.00 |
| Cowboys      |             -0.83 |                  0.55 |            1.38 |                5.46 |               5.50 |
| Knights      |              0.79 |                 -0.70 |           -1.49 |                4.50 |               4.50 |
| Storm        |             -0.94 |                 -0.94 |            0.01 |                5.00 |               5.00 |
| Bulldogs     |             -5.19 |                 -3.45 |            1.75 |                5.58 |               5.50 |
| Raiders      |             -5.49 |                 -6.08 |           -0.59 |                4.80 |               5.00 |
| Titans       |             -5.95 |                 -6.30 |           -0.36 |                4.88 |               5.00 |
| Broncos      |             -6.16 |                 -9.02 |           -2.86 |                4.05 |               4.00 |
| Eels         |            -10.06 |                 -9.18 |            0.88 |                5.29 |               5.50 |
| Dragons      |            -11.96 |                 -9.70 |            2.25 |                5.75 |               6.00 |
| Wests Tigers |             -5.43 |                 -9.89 |           -4.46 |                3.51 |               3.50 |

## Model diagnostics

{
  "season": {
    "label": "full-season",
    "divergences": 0,
    "max_rhat": 1.0041696774306252,
    "min_ess_bulk": 5851.531364199827
  },
  "current_6r": {
    "label": "six-round-half-life",
    "divergences": 0,
    "max_rhat": 1.0052248459251005,
    "min_ess_bulk": 5792.276949264185
  }
}
