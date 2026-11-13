import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from detection_regression_infer import cli
from detection_regression_infer.labeling import draft_boxes, import_yolo, prepare_labeling, to_yolo_line, yolo_line_to_bbox
from detection_regression_infer.video import sha256_file

ANNOTATIONS_SCHEMA = Path(__file__).resolve().parents[2] / "schema" / "annotations.schema.json"
CLASSES = {"0": "cap", "1": "person"}


def _det(bbox: list[float], conf: float = 0.5, name: str = "cap") -> dict:
    return {"class_id": 0 if name == "cap" else 1, "class_name": name, "confidence": conf, "bbox": bbox}


def _result(video: Path, dets_by_frame: dict[int, list[dict]], model_id: str = "m") -> dict:
    return {
        "schema_version": "1.0",
        "source": {"path": video.name, "sha256": sha256_file(video), "fps": 10.0, "width": 64, "height": 48, "frame_count": 5},
        "inference": {"model_id": model_id, "confidence_threshold": 0.15, "iou_threshold": 0.7, "image_size": 640, "classes": CLASSES},
        "frames": [{"frame_index": i, "timestamp_ms": i * 100.0, "detections": dets_by_frame.get(i, [])} for i in range(5)],
    }


def test_yolo_line_round_trip() -> None:
    line = to_yolo_line(0, [16.0, 12.0, 32.0, 24.0], 64, 48)
    assert line == "0 0.375000 0.375000 0.250000 0.250000"
    assert yolo_line_to_bbox(line, 64, 48) == (0, pytest.approx([16.0, 12.0, 32.0, 24.0]))


def test_draft_boxes_merges_overlapping_detections_of_both_models() -> None:
    a = [_det([10, 10, 20, 20], 0.3), _det([0, 0, 5, 5], 0.9, "person")]
    b = [_det([11, 10, 21, 20], 0.6), _det([40, 30, 50, 40], 0.2)]
    boxes = draft_boxes([a, b], classes={"cap"})
    # 重なる2つは confidence の高い方に1つにまとめ、person は下書きに入れない
    assert sorted(d["bbox"] for d in boxes) == [[11, 10, 21, 20], [40, 30, 50, 40]]


def test_prepare_labeling_writes_images_drafts_and_manifest(tiny_video: Path, tmp_path: Path) -> None:
    r1 = _result(tiny_video, {1: [_det([10, 10, 20, 20])]})
    r2 = _result(tiny_video, {3: [_det([30, 20, 40, 30])]})
    out = tmp_path / "labeling"
    prepare_labeling(video=tiny_video, results=[r1, r2], frame_indexes=[1, 2, 3], classes={"cap"}, output=out)

    assert sorted(p.name for p in (out / "images").iterdir()) == ["tiny_000001.jpg", "tiny_000002.jpg", "tiny_000003.jpg"]
    drafts = out / "labels_draft"
    assert (drafts / "classes.txt").read_text().splitlines() == ["cap", "person"]
    assert (drafts / "tiny_000001.txt").read_text().startswith("0 ")
    assert not (drafts / "tiny_000002.txt").exists()
    assert json.loads((out / "manifest.json").read_text())["frame_indexes"] == [1, 2, 3]


def test_prepare_labeling_rejects_results_of_other_video(tiny_video: Path, tmp_path: Path) -> None:
    r = _result(tiny_video, {})
    r["source"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="sha256"):
        prepare_labeling(video=tiny_video, results=[r], frame_indexes=[1], classes={"cap"}, output=tmp_path / "o")


def test_import_yolo_builds_annotations_for_every_frame_in_manifest(tiny_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "labeling"
    prepare_labeling(video=tiny_video, results=[_result(tiny_video, {})], frame_indexes=[1, 2], classes={"cap"}, output=out)
    labels = out / "labels"
    labels.mkdir()
    (labels / "classes.txt").write_text("cap\nperson\n")
    (labels / "tiny_000001.txt").write_text("0 0.375 0.375 0.25 0.25\n1 0.5 0.5 0.5 0.5\n")
    # frame 2 はラベルのファイルが無い = 確認したがキャップは写っていない

    annotations = import_yolo(video=tiny_video, labeling_dir=out, labels_dir=labels, classes=["cap"])

    Draft202012Validator(json.loads(ANNOTATIONS_SCHEMA.read_text())).validate(annotations)
    assert annotations["classes"] == ["cap"]
    assert annotations["frames"] == [
        {"frame_index": 1, "objects": [{"class_name": "cap", "bbox": [16.0, 12.0, 32.0, 24.0]}]},
        {"frame_index": 2, "objects": []},
    ]


def test_cli_prepare_and_import(tiny_video: Path, tmp_path: Path) -> None:
    result_path = tmp_path / "r.json"
    result_path.write_text(json.dumps(_result(tiny_video, {1: [_det([10, 10, 20, 20])]})))
    out = tmp_path / "labeling"
    assert cli.main([
        "prepare-labeling", "--video", str(tiny_video), "--result", str(result_path),
        "--frames", "1,2", "--classes", "cap", "--output", str(out),
    ]) == 0
    labels = out / "labels"
    labels.mkdir()
    (labels / "classes.txt").write_text("cap\nperson\n")
    ann = tmp_path / "ann.json"
    assert cli.main([
        "import-yolo", "--video", str(tiny_video), "--labeling-dir", str(out),
        "--labels", str(labels), "--classes", "cap", "--output", str(ann),
    ]) == 0
    assert [f["frame_index"] for f in json.loads(ann.read_text())["frames"]] == [1, 2]


def test_import_labelme_reads_rectangles_of_selected_classes(tiny_video: Path, tmp_path: Path) -> None:
    from detection_regression_infer.labeling import import_labelme

    out = tmp_path / "labeling"
    prepare_labeling(video=tiny_video, results=[_result(tiny_video, {})], frame_indexes=[1, 2], classes={"cap"}, output=out)
    shapes = [
        {"label": "cap", "shape_type": "rectangle", "points": [[32, 24], [16, 12], [16, 24], [32, 12]]},
        {"label": "person", "shape_type": "rectangle", "points": [[0, 0], [10, 10]]},
        {"label": "cap", "shape_type": "point", "points": [[5, 5]]},
    ]
    (out / "images" / "tiny_000001.json").write_text(json.dumps({"shapes": shapes, "imageWidth": 64, "imageHeight": 48}))

    annotations = import_labelme(video=tiny_video, labeling_dir=out, labels_dir=out / "images", classes=["cap"])

    Draft202012Validator(json.loads(ANNOTATIONS_SCHEMA.read_text())).validate(annotations)
    assert annotations["frames"] == [
        {"frame_index": 1, "objects": [{"class_name": "cap", "bbox": [16.0, 12.0, 32.0, 24.0]}]},
        {"frame_index": 2, "objects": []},
    ]


def test_import_labelme_rejects_size_mismatch(tiny_video: Path, tmp_path: Path) -> None:
    from detection_regression_infer.labeling import import_labelme

    out = tmp_path / "labeling"
    prepare_labeling(video=tiny_video, results=[_result(tiny_video, {})], frame_indexes=[1], classes={"cap"}, output=out)
    (out / "images" / "tiny_000001.json").write_text(json.dumps({"shapes": [], "imageWidth": 128, "imageHeight": 48}))
    with pytest.raises(ValueError, match="128"):
        import_labelme(video=tiny_video, labeling_dir=out, labels_dir=out / "images", classes=["cap"])


def test_cli_import_labelme(tiny_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "labeling"
    prepare_labeling(video=tiny_video, results=[_result(tiny_video, {})], frame_indexes=[1], classes={"cap"}, output=out)
    ann = tmp_path / "ann.json"
    code = cli.main([
        "import-labelme", "--video", str(tiny_video), "--labeling-dir", str(out),
        "--labels", str(out / "images"), "--classes", "cap", "--output", str(ann),
    ])
    assert code == 0
    assert json.loads(ann.read_text())["frames"] == [{"frame_index": 1, "objects": []}]
