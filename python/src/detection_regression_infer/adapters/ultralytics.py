"""Ultralytics YOLO で動画を推論し、共通 JSON スキーマの dict を返す。

ultralytics の import は load_model の中で行う。比較や変換のテストで torch を読み込まないようにするため。
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from ..convert import build_result, to_detections
from ..video import iter_frames, read_video_info, sha256_file


class DetectionModel(Protocol):
    names: Mapping[int, str]

    def predict(self, source: Any, **kwargs: Any) -> list[Any]: ...


def load_model(model_path: Path) -> DetectionModel:
    from ultralytics import YOLO

    return YOLO(str(model_path))


def _to_list(values: Any) -> list[Any]:
    """torch.Tensor でも numpy 配列でも Python のリストにする。"""
    if hasattr(values, "cpu"):
        values = values.cpu().numpy()
    return values.tolist()


def infer_video(
    model: DetectionModel,
    video_path: Path,
    *,
    model_id: str,
    imgsz: int,
    conf: float,
    iou: float,
) -> dict[str, Any]:
    info = read_video_info(video_path)
    names = dict(model.names)
    frames = []
    for index, image in iter_frames(video_path):
        boxes = model.predict(image, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0].boxes
        frames.append(
            {
                "frame_index": index,
                # 再生位置（CAP_PROP_POS_MSEC）はコーデックで揺れるため、番号と fps から計算する
                "timestamp_ms": round(index * 1000.0 / info.fps, 3),
                "detections": to_detections(
                    xyxy=_to_list(boxes.xyxy),
                    conf=_to_list(boxes.conf),
                    cls=_to_list(boxes.cls),
                    names=names,
                    width=info.width,
                    height=info.height,
                ),
            }
        )
    return build_result(
        source={
            "path": video_path.name,
            "sha256": sha256_file(video_path),
            "fps": info.fps,
            "width": info.width,
            "height": info.height,
        },
        inference={"model_id": model_id, "confidence_threshold": conf, "iou_threshold": iou, "image_size": imgsz},
        names=names,
        frames=frames,
    )
