import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from detection_regression_infer import cli
from tests.test_ultralytics_adapter import FakeModel


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_model", lambda path: FakeModel())


def test_infer_writes_valid_json(
    fake_model: None, tiny_video: Path, tmp_path: Path, validator: Draft202012Validator
) -> None:
    output = tmp_path / "out" / "before.json"
    code = cli.main(["infer", "--model", "models/before.pt", "--video", str(tiny_video), "--output", str(output)])
    assert code == 0
    result = json.loads(output.read_text())
    validator.validate(result)
    assert result["inference"]["model_id"] == "before"
    assert result["inference"]["image_size"] == 640
    assert result["inference"]["confidence_threshold"] == 0.25


def test_infer_accepts_model_id_and_thresholds(fake_model: None, tiny_video: Path, tmp_path: Path) -> None:
    output = tmp_path / "after.json"
    cli.main([
        "infer", "--model", "x.pt", "--video", str(tiny_video), "--output", str(output),
        "--model-id", "model-v2", "--imgsz", "320", "--conf", "0.4", "--iou", "0.5",
    ])
    inference = json.loads(output.read_text())["inference"]
    assert inference["model_id"] == "model-v2"
    assert (inference["image_size"], inference["confidence_threshold"], inference["iou_threshold"]) == (320, 0.4, 0.5)


def test_infer_reports_missing_video(fake_model: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["infer", "--model", "x.pt", "--video", str(tmp_path / "none.mp4"), "--output", str(tmp_path / "o.json")])
    assert code == 1
    assert "none.mp4" in capsys.readouterr().err
