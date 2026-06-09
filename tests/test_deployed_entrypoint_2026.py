from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(
    "/Users/joelcouchman/Projects/"
    "joel-nrl-predictor"
)

ENTRYPOINT = (
    ROOT
    / "nrl_bayesian_sim.py"
)


def test_deployed_entrypoint_runs_2026_app() -> None:
    app = AppTest.from_file(
        ENTRYPOINT,
        default_timeout=30,
    )

    app.run()

    assert len(app.exception) == 0
    assert len(app.error) == 0

    assert app.title[0].value == (
        "Joel's NRL Ladder Predictor "
        "(2026)"
    )

    assert any(
        "Rounds 1-14 complete"
        in caption.value
        for caption in app.caption
    )

    app.slider(
        key="simulation_count"
    ).set_value(500)

    app.number_input(
        key="seed"
    ).set_value(20260609)

    app.button(
        key="run_simulation"
    ).click()

    app.run(timeout=120)

    assert len(app.exception) == 0
    assert len(app.error) == 0

    assert any(
        subheader.value
        == "Final Ladder Probabilities"
        for subheader in app.subheader
    )

    assert any(
        "500 simulations"
        in caption.value
        for caption in app.caption
    )
