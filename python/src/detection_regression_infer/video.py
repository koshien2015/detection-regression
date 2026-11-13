"""動画の読み込み。フレームは必ず先頭から順に読む（シークはコーデックによって別のフレームを返すため）。"""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int


def _open(path: Path) -> cv2.VideoCapture:
    if not path.is_file():
        raise FileNotFoundError(f"動画が見つからない: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"動画を開けない: {path}")
    return capture


def read_video_info(path: Path) -> VideoInfo:
    capture = _open(path)
    try:
        info = VideoInfo(
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()
    if info.fps <= 0 or info.width <= 0 or info.height <= 0:
        raise ValueError(f"fps・幅・高さを読み取れない: {path}（{info}）")
    return info


def iter_frames(path: Path) -> Iterator[tuple[int, np.ndarray]]:
    """(frame_index, BGR 画像) を1枚ずつ返す。全フレームをメモリに持たない。"""
    capture = _open(path)
    try:
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                return
            yield index, image
            index += 1
    finally:
        capture.release()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
