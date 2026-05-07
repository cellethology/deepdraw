import pytest

from deepdraw import cli


def test_cli_validation_error_omits_traceback(monkeypatch, capsys):
    def fail_suggest(**kwargs):
        raise ValueError("Measurements are missing labels.")

    monkeypatch.setattr(cli, "suggest_next_batch", fail_suggest)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "suggest",
                "--run-dir",
                "deepdraw_run",
                "--measurements",
                "measurements.csv",
                "--label-column",
                "Expression",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert captured.out == ""
    assert captured.err == "Error: Measurements are missing labels.\n"
    assert "Traceback" not in captured.err


def test_cli_debug_log_level_reraises_validation_error(monkeypatch):
    def fail_suggest(**kwargs):
        raise ValueError("debug me")

    monkeypatch.setattr(cli, "suggest_next_batch", fail_suggest)

    with pytest.raises(ValueError, match="debug me"):
        cli.main(
            [
                "suggest",
                "--log-level",
                "DEBUG",
                "--run-dir",
                "deepdraw_run",
                "--measurements",
                "measurements.csv",
                "--label-column",
                "Expression",
            ]
        )
