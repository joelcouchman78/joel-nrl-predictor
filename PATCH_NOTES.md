# Expanded NRL predictor outputs

This package is intended to be extracted over the root of `joel-nrl-predictor`.

## Replaced files

- `nrl_predictor_2026.py`
- `predictor/simulation.py`
- `README.md`

## New files

- `predictor/outputs.py`
- `tests/test_expanded_outputs.py`

## Added outputs

- probability of finishing in every ladder position;
- expected, median and modal finishing positions;
- central 80% finishing-position interval;
- position standard deviation and variance;
- expected final competition points and differential;
- simulated eighth-place competition-points cutoff;
- Top 8 and Top 4 probability conditional on remaining win count;
- remaining-fixture win probabilities and expected margins;
- fixture leverage for Top 4 and Top 8 chances;
- multi-result scenario filtering;
- Monte Carlo standard-error diagnostics;
- a compact default projection table with detailed results behind tabs and expanders.

## Validation performed

```text
python -m compileall -q predictor nrl_predictor_2026.py tests
PYTHONPATH=. pytest -q tests/test_expanded_outputs.py
```

Result:

```text
4 passed
```

The connected GitHub integration returned HTTP 403 for both branch creation and contents writes, so these files were not pushed or deployed automatically.
