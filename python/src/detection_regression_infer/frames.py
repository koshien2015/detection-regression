"""レポート用に、選んだフレームだけを PNG に書き出す。

読み込みは video.iter_frames と同じく先頭から順に行い、推論時と同じ frame_index のフレームを取り出す。
"""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import cv2

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
    changed = [f["frame_index"] for f in (diff or {}).get("frames", []) if f.get("has_changes")]
    annotated = [f["frame_index"] for f in (evaluation or {}).get("frames", [])]
    return sorted({*changed, *annotated, *explicit})


def extract_frames(video_path: Path, frame_indexes: Iterable[int], output_dir: Path) -> dict[int, str]:
    """{frame_index: ファイル名} を返す。動画に無い番号が含まれていれば ValueError。"""
    wanted = set(frame_indexes)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[int, str] = {}
    last = max(wanted, default=-1)
    for index, image in iter_frames(video_path):
        if index > last:
            break
        if index in wanted:
            name = frame_file_name(index)
            if not cv2.imwrite(str(output_dir / name), image):
                raise OSError(f"画像を書き出せない: {output_dir / name}")
            written[index] = name
    missing = sorted(wanted - written.keys())
    if missing:
        raise ValueError(f"動画に無いフレーム番号: {missing}")
    return dict(sorted(written.items()))
