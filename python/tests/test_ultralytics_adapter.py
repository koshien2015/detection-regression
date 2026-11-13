from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from detection_regression_infer.adapters.ultralytics import infer_video


@dataclass
class FakeBoxes:
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray


@dataclass
class FakeResult:
    boxes: FakeBoxes


class FakeModel:
    """フレーム 1 と 3 にだけ bottle を返す。Ultralytics の Results と同じ属性だけを持つ。"""

    names = {0: "person", 39: "bottle"}

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.count = 0

    def predict(self, frame: np.ndarray, **kwargs: object) -> list[FakeResult]:
        self.calls.append(kwargs)
        index = self.count
        self.count += 1
        if index in (1, 3):
            return [FakeResult(FakeBoxes(np.array([[-2.0, 5.0, 30.0, 60.0]]), np.array([0.5]), np.array([39.0])))]
        return [FakeResult(FakeBoxes(np.zeros((0, 4)), np.zeros(0), np.zeros(0)))]


def test_infer_video_records_every_frame(tiny_video: Path, validator: Draft202012Validator) -> None:
    model = FakeModel()
    result = infer_video(model, tiny_video, model_id="fake", imgsz=320, conf=0.3, iou=0.6)

    validator.validate(result)
    assert [f["frame_index"] for f in result["frames"]] == [0, 1, 2, 3, 4]
    assert [len(f["detections"]) for f in result["frames"]] == [0, 1, 0, 1, 0]
    assert result["source"]["frame_count"] == 5
    assert result["source"]["path"] == "tiny.mp4"
    assert result["frames"][2]["timestamp_ms"] == 200.0


def test_infer_video_clips_boxes_and_passes_settings(tiny_video: Path) -> None:
    model = FakeModel()
    result = infer_video(model, tiny_video, model_id="fake", imgsz=320, conf=0.3, iou=0.6)

    assert result["frames"][1]["detections"][0]["bbox"] == [0.0, 5.0, 30.0, 48.0]
    assert model.calls[0] == {"imgsz": 320, "conf": 0.3, "iou": 0.6, "verbose": False}
    assert result["inference"] == {
        "model_id": "fake",
        "confidence_threshold": 0.3,
        "iou_threshold": 0.6,
        "image_size": 320,
        "classes": {"0": "person", "39": "bottle"},
    }
