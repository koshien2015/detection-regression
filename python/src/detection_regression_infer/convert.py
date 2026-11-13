"""推論結果を共通 JSON スキーマ（schema/inference-result.schema.json）の形に組み立てる。"""

from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = "1.0"


def clip_bbox(xyxy: Sequence[float], width: int, height: int) -> list[float]:
    """BBox をフレームの内側に収める。スキーマはフレームからはみ出す BBox を受け付けないため。"""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    return [
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    ]


def to_detections(
    *,
    xyxy: Sequence[Sequence[float]],
    conf: Sequence[float],
    cls: Sequence[float],
    names: Mapping[int, str],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    return [
        {
            "class_id": int(c),
            "class_name": names[int(c)],
            "confidence": float(p),
            "bbox": clip_bbox(box, width, height),
        }
        for box, p, c in zip(xyxy, conf, cls, strict=True)
    ]


def build_result(
    *,
    source: Mapping[str, Any],
    inference: Mapping[str, Any],
    names: Mapping[int, str],
    frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """frame_count は実際に読めたフレーム数にする。コンテナのメタデータの値は実数とずれることがあるため。"""
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {**source, "frame_count": len(frames)},
        "inference": {**inference, "classes": {str(k): v for k, v in sorted(names.items())}},
        "frames": list(frames),
    }
