"""ultralytics 同梱の bus.jpg から、ゆっくり拡大しながら右へ動く疑似動画と、その正解アノテーションを作る。

元画像に付けた正解（ground_truth.json）に、動画の各フレームと同じ拡大・移動をかけて写すので、
正解の BBox を手で付け直さずに全フレームの正解が決まる。

使い方（リポジトリのルートで）:
  uv run --project python python examples/bus/make_sample.py \\
    --video samples/bus_pan.mp4 --annotations annotations/bus_pan.json
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detection_regression_infer.video import sha256_file

OUT_W, OUT_H = 600, 800
FPS = 30.0
END_CROP_RATIO = 0.74  # 最後のフレームで元画像の幅の何割を切り出すか
MIN_VISIBLE = 0.5  # 正解の BBox がこの割合以上フレームに残っていれば正解に含める


@dataclass(frozen=True)
class Crop:
    x0: float
    y0: float
    width: float
    height: float


def _ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def crop_at(t: float, img_w: int, img_h: int, end_ratio: float = END_CROP_RATIO) -> Crop:
    """t=0 で画像全体、t=1 で幅 end_ratio まで拡大し、右端へ寄せる切り出し範囲。"""
    e = _ease(t)
    width = img_w * (1 - (1 - end_ratio) * e)
    height = width * img_h / img_w
    return Crop(x0=(img_w - width) * e, y0=(img_h - height) * 0.4, width=width, height=height)


def transform_box(
    bbox: list[float], crop: Crop, out_w: int = OUT_W, out_h: int = OUT_H
) -> tuple[list[float], float]:
    """元画像の BBox を出力フレームの座標に写し、フレーム内に残る面積の割合とともに返す。"""
    s = out_w / crop.width
    x1, y1, x2, y2 = ((bbox[0] - crop.x0) * s, (bbox[1] - crop.y0) * s, (bbox[2] - crop.x0) * s, (bbox[3] - crop.y0) * s)
    cx1, cy1, cx2, cy2 = max(x1, 0.0), max(y1, 0.0), min(x2, float(out_w)), min(y2, float(out_h))
    full = (x2 - x1) * (y2 - y1)
    visible = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    return [cx1, cy1, cx2, cy2], (visible / full if full > 0 else 0.0)


def _render(image: np.ndarray, crop: Crop) -> np.ndarray:
    s = OUT_W / crop.width
    matrix = np.array([[s, 0, -crop.x0 * s], [0, s, -crop.y0 * s]], dtype=np.float64)
    return cv2.warpAffine(image, matrix, (OUT_W, OUT_H), flags=cv2.INTER_AREA)


def make_sample(
    *,
    image_path: Path,
    ground_truth: dict[str, Any],
    video_path: Path,
    annotations_path: Path,
    frames: int = 90,
    annotate_every: int = 15,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"画像を読めない: {image_path}")
    img_h, img_w = image.shape[:2]
    crops = [crop_at(i / (frames - 1), img_w, img_h) for i in range(frames)]

    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (OUT_W, OUT_H))
    for crop in crops:
        writer.write(_render(image, crop))
    writer.release()

    annotated = sorted({*range(0, frames, annotate_every), frames - 1})
    annotation_frames = []
    for index in annotated:
        objects = []
        for obj in ground_truth["objects"]:
            box, visible = transform_box(obj["bbox"], crops[index])
            if visible >= MIN_VISIBLE:
                objects.append({"class_name": obj["class_name"], "bbox": [round(v, 2) for v in box]})
        annotation_frames.append({"frame_index": index, "objects": objects})

    annotations = {
        "schema_version": "1.0",
        "source": {"path": video_path.name, "sha256": sha256_file(video_path), "width": OUT_W, "height": OUT_H},
        "frames": annotation_frames,
    }
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    annotations_path.write_text(json.dumps(annotations, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    import ultralytics

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=90)
    args = parser.parse_args()
    make_sample(
        image_path=Path(os.path.dirname(ultralytics.__file__)) / "assets" / "bus.jpg",
        ground_truth=json.loads((Path(__file__).with_name("ground_truth.json")).read_text()),
        video_path=args.video,
        annotations_path=args.annotations,
        frames=args.frames,
    )
    print(f"{args.video} と {args.annotations} を書き出した")


if __name__ == "__main__":
    main()
