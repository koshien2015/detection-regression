import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "inference-result.schema.json"


@pytest.fixture(scope="session")
def validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    """64x48 で 5 フレームの動画。フレーム i は画素値 i*40 で塗りつぶす。"""
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(5):
        writer.write(np.full((48, 64, 3), i * 40, dtype=np.uint8))
    writer.release()
    return path
