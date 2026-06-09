from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(
    "/Users/joelcouchman/Projects/"
    "joel-nrl-predictor"
)

APP_PATH = (
    ROOT
    / "nrl_predictor_2026.py"
)


def test_2026_app_initial_render() -> None:
    app = AppTest.from_file(
        APP_PATH,
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
        "Rounds 1-13 complete"
        in caption.value
        for caption in app.caption
    )

    assert len(app.dataframe) >= 1

    run_buttons = [
        button
        for button in app.button
        if button.key
        == "run_simulation"
    ]

    assert len(run_buttons) == 1


def test_2026_app_simulation_path() -> None:
    app = AppTest.from_file(
        APP_PATH,
        default_timeout=30,
    )

    app.run()

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

    assert len(app.dataframe) >= 6
