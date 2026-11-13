from jsonschema import Draft202012Validator

from detection_regression_infer.convert import build_result, clip_bbox, to_detections

NAMES = {0: "person", 39: "bottle"}


def test_clip_bbox_keeps_inside_box() -> None:
    assert clip_bbox([10.5, 20.0, 30.0, 40.0], 64, 48) == [10.5, 20.0, 30.0, 40.0]


def test_clip_bbox_clamps_to_frame() -> None:
    assert clip_bbox([-3.0, -0.5, 70.0, 49.0], 64, 48) == [0.0, 0.0, 64.0, 48.0]


def test_to_detections_uses_model_class_names() -> None:
    dets = to_detections(
        xyxy=[[1.0, 2.0, 3.0, 4.0]], conf=[0.875], cls=[39.0], names=NAMES, width=64, height=48
    )
    assert dets == [{"class_id": 39, "class_name": "bottle", "confidence": 0.875, "bbox": [1.0, 2.0, 3.0, 4.0]}]


def test_to_detections_empty() -> None:
    assert to_detections(xyxy=[], conf=[], cls=[], names=NAMES, width=64, height=48) == []


def _result(path: str = "tiny.mp4", frames: list | None = None) -> dict:
    return build_result(
        source={"path": path, "sha256": "a" * 64, "fps": 10.0, "width": 64, "height": 48},
        inference={"model_id": "m", "confidence_threshold": 0.25, "iou_threshold": 0.7, "image_size": 640},
        names=NAMES,
        frames=frames if frames is not None else [{"frame_index": 0, "timestamp_ms": 0.0, "detections": []}],
    )


def test_build_result_matches_json_schema(validator: Draft202012Validator) -> None:
    result = _result()
    validator.validate(result)
    assert result["schema_version"] == "1.0"
    assert result["source"]["frame_count"] == 1
    assert result["inference"]["classes"] == {"0": "person", "39": "bottle"}


def test_json_schema_rejects_absolute_source_path(validator: Draft202012Validator) -> None:
    assert not validator.is_valid(_result(path="/Users/someone/tiny.mp4"))
