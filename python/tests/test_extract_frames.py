import json
from pathlib import Path

import cv2
import pytest

from detection_regression_infer import cli
from detection_regression_infer.frames import extract_frames, frame_file_name, select_frames
from detection_regression_infer.video import sha256_file


def _diff(video: Path, changed: list[int], total: int = 5) -> dict:
    return {
        "inputs": {"source": {"sha256": sha256_file(video)}},
        "frames": [{"frame_index": i, "has_changes": i in changed} for i in range(total)],
    }


def test_frame_file_name_is_zero_padded() -> None:
    assert frame_file_name(12) == "frame_000012.png"


def test_select_frames_merges_changed_annotated_and_explicit(tiny_video: Path) -> None:
    diff = _diff(tiny_video, changed=[3, 1])
    evaluation = {"frames": [{"frame_index": 4}, {"frame_index": 1}]}
    assert select_frames(diff=diff, evaluation=evaluation, explicit=[0]) == [0, 1, 3, 4]


def test_extract_frames_writes_only_selected_frames(tiny_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "frames"
    written = extract_frames(tiny_video, [1, 3], out)
    assert written == {1: "frame_000001.png", 3: "frame_000003.png"}
    assert sorted(p.name for p in out.iterdir()) == ["frame_000001.png", "frame_000003.png"]
    # 塗りつぶした画素値で、番号どおりのフレームを書き出したかを確かめる（mp4v の圧縮誤差を許す）
    assert abs(float(cv2.imread(str(out / "frame_000003.png")).mean()) - 120) < 3


def test_extract_frames_rejects_index_beyond_video(tiny_video: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="99"):
        extract_frames(tiny_video, [1, 99], tmp_path / "frames")


def test_cli_extract_frames_from_diff(tiny_video: Path, tmp_path: Path) -> None:
    diff_path = tmp_path / "diff.json"
    diff_path.write_text(json.dumps(_diff(tiny_video, changed=[2])))
    out = tmp_path / "report" / "frames"
    code = cli.main(["extract-frames", "--video", str(tiny_video), "--diff", str(diff_path), "--output", str(out), "--frames", "4"])
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == ["frame_000002.png", "frame_000004.png"]


def test_cli_extract_frames_refuses_other_video(tiny_video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    diff = _diff(tiny_video, changed=[2])
    diff["inputs"]["source"]["sha256"] = "f" * 64
    diff_path = tmp_path / "diff.json"
    diff_path.write_text(json.dumps(diff))
    code = cli.main(["extract-frames", "--video", str(tiny_video), "--diff", str(diff_path), "--output", str(tmp_path / "o")])
    assert code == 1
    assert "sha256" in capsys.readouterr().err
