from pathlib import Path


APP_PATH = Path(__file__).with_name(
    "nrl_predictor_2026.py"
)

source = APP_PATH.read_text(
    encoding="utf-8"
)

exec(
    compile(
        source,
        str(APP_PATH),
        "exec",
    )
)
