import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from jsonschema import Draft202012Validator

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "bus" / "make_sample.py"
spec = importlib.util.spec_from_file_location("make_sample", EXAMPLE)
make_sample = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_sample)

ANNOTATIONS_SCHEMA = Path(__file__).resolve().parents[2] / "schema" / "annotations.schema.json"


def test_crop_starts_with_whole_image_and_zooms_in() -> None:
    first = make_sample.crop_at(0.0, 810, 1080)
    last = make_sample.crop_at(1.0, 810, 1080)
    assert (first.x0, first.y0, first.width) == (0.0, 0.0, 810.0)
    assert last.width == pytest.approx(810 * make_sample.END_CROP_RATIO)
    assert last.x0 > 0


def test_transform_box_scales_into_output_frame() -> None:
    crop = make_sample.Crop(x0=100.0, y0=200.0, width=405.0, height=540.0)
    box, visible = make_sample.transform_box([100.0, 200.0, 505.0, 740.0], crop, out_w=810, out_h=1080)
    assert box == pytest.approx([0.0, 0.0, 810.0, 1080.0])
    assert visible == pytest.approx(1.0)


def test_transform_box_reports_visible_ratio_when_cropped() -> None:
    crop = make_sample.Crop(x0=50.0, y0=0.0, width=810.0, height=1080.0)
    box, visible = make_sample.transform_box([0.0, 0.0, 100.0, 100.0], crop, out_w=810, out_h=1080)
    assert box == pytest.approx([0.0, 0.0, 50.0, 100.0])
    assert visible == pytest.approx(0.5)


def test_make_sample_writes_video_and_matching_annotations(tmp_path: Path) -> None:
    image = np.zeros((1080, 810, 3), dtype=np.uint8)
    image[229:739, 0:808] = 200  # 正解の bus の位置だけ明るくする
    image_path = tmp_path / "bus.jpg"
    cv2.imwrite(str(image_path), image)
    video = tmp_path / "bus_pan.mp4"
    annotations = tmp_path / "bus_pan.json"

    make_sample.make_sample(
        image_path=image_path,
        ground_truth=json.loads(EXAMPLE.with_name("ground_truth.json").read_text()),
        video_path=video,
        annotations_path=annotations,
        frames=30,
        annotate_every=10,
    )

    result = json.loads(annotations.read_text())
    Draft202012Validator(json.loads(ANNOTATIONS_SCHEMA.read_text())).validate(result)
    assert [f["frame_index"] for f in result["frames"]] == [0, 10, 20, 29]
    assert result["source"]["path"] == "bus_pan.mp4"
    assert cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT) == 30
    # 先頭フレームは元画像全体なので、正解の5物体がすべて残る
    assert len(result["frames"][0]["objects"]) == 5
