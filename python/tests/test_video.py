import hashlib
from pathlib import Path

from detection_regression_infer.video import iter_frames, read_video_info, sha256_file


def test_iter_frames_reads_every_frame_in_order(tiny_video: Path) -> None:
    frames = list(iter_frames(tiny_video))
    assert [i for i, _ in frames] == [0, 1, 2, 3, 4]
    # 塗りつぶした画素値の順になっていれば、読み飛ばしや順序の入れ替わりが無い
    means = [float(img.mean()) for _, img in frames]
    assert means == sorted(means)


def test_read_video_info(tiny_video: Path) -> None:
    info = read_video_info(tiny_video)
    assert (info.width, info.height) == (64, 48)
    assert info.fps == 10.0


def test_sha256_file(tiny_video: Path) -> None:
    assert sha256_file(tiny_video) == hashlib.sha256(tiny_video.read_bytes()).hexdigest()


def test_iter_frames_raises_for_missing_file(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        list(iter_frames(tmp_path / "missing.mp4"))
