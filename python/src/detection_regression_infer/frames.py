"""レポート用に、選んだフレームだけを PNG に書き出す。

読み込みは video.iter_frames と同じく先頭から順に行い、推論時と同じ frame_index のフレームを取り出す。
"""

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .video import iter_frames


def frame_file_name(frame_index: int) -> str:
    return f"frame_{frame_index:06d}.png"


def select_frames(
    *,
    diff: Mapping[str, Any] | None = None,
    evaluation: Mapping[str, Any] | None = None,
    explicit: Iterable[int] = (),
) -> list[int]:
    """差分のあるフレーム・正解を付けたフレーム・明示したフレームを合わせ、番号順に返す。"""
    changed = [f["frame_index"] for f in (diff or {}).get("frames", []) if f["has_changes"]]
    annotated = [f["frame_index"] for f in (evaluation or {}).get("frames", [])]
    return sorted({*changed, *annotated, *explicit})


_GAMMA_LUT = np.array([((i / 255.0) ** (1 / 1.5)) * 255 for i in range(256)]).astype("uint8")


def motion_frames(video_path: Path, frame_indexes: Iterable[int]) -> Iterator[tuple[int, np.ndarray]]:
    """前後のフレームとの差分の AND をとり、ガンマ 1.5 で明るくした「動いている成分」だけの画像を返す。

    掲載用の画像で人物を判別しにくくするために使う。フレーム i は i-1・i・i+1 から作り、
    先頭のフレームは前が無いので i-1 の代わりに i を使う（差分は 0 になる）。最後のフレームは作れない。
    """
    wanted = set(frame_indexes)
    last = max(wanted, default=-1)
    prev: np.ndarray | None = None
    current: tuple[int, np.ndarray] | None = None
    for index, image in iter_frames(video_path):
        if current is not None:
            cur_index, cur = current
            if cur_index in wanted:
                before = cur if prev is None else prev
                diff = cv2.bitwise_and(cv2.absdiff(cur, before), cv2.absdiff(image, cur))
                yield cur_index, cv2.LUT(diff, _GAMMA_LUT)
            prev = cur
            if cur_index >= last:
                return
        current = (index, image)


def extract_frames(
    video_path: Path, frame_indexes: Iterable[int], output_dir: Path, *, motion_from: Path | None = None
) -> dict[int, str]:
    """{frame_index: ファイル名} を返す。書き出せない番号が含まれていれば ValueError。

    motion_from を指定すると、video_path の代わりにその動画から差分の成分だけの画像を作って書き出す。
    """
    wanted = set(frame_indexes)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[int, str] = {}
    last = max(wanted, default=-1)
    source = motion_frames(motion_from, wanted) if motion_from is not None else iter_frames(video_path)
    for index, image in source:
        if index > last:
            break
        if index in wanted:
            name = frame_file_name(index)
            if not cv2.imwrite(str(output_dir / name), image):
                raise OSError(f"画像を書き出せない: {output_dir / name}")
            written[index] = name
    missing = sorted(wanted - written.keys())
    if missing:
        raise ValueError(f"書き出せないフレーム番号（動画に無いか、差分を作れない最後のフレーム）: {missing}")
    return dict(sorted(written.items()))
