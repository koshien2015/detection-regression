"""正解アノテーションを作るための下書きの書き出し（prepare_labeling）と、YOLO 形式のラベルの取り込み（import_yolo）。

ラベリングツール（X-AnyLabeling など）での流れ:
  1. prepare_labeling で images/ に画像、labels_draft/ に YOLO 形式の下書きと classes.txt を書く
  2. ツールで images/ を開き、labels_draft/ をインポートして直し、YOLO 形式で labels/ にエクスポートする
  3. import_yolo で labels/ を正解アノテーションの JSON にする
     （X-AnyLabeling が images/ に保存した .json をそのまま使うなら import_labelme）

下書きは画像と別のフォルダに書く。X-AnyLabeling は画像フォルダ内の .txt も画像として開こうとするため。
"""

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2

from .video import VideoInfo, iter_frames, read_video_info, sha256_file

SCHEMA_VERSION = "1.0"
DRAFT_MERGE_IOU = 0.5


def to_yolo_line(class_id: int, bbox: Sequence[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    return f"{class_id} {cx:.6f} {cy:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"


def yolo_line_to_bbox(line: str, width: int, height: int) -> tuple[int, list[float]]:
    class_id, cx, cy, bw, bh = line.split()
    cxp, cyp, wp, hp = float(cx) * width, float(cy) * height, float(bw) * width, float(bh) * height
    box = [cxp - wp / 2, cyp - hp / 2, cxp + wp / 2, cyp + hp / 2]
    limits = [width, height, width, height]
    return int(class_id), [min(max(v, 0.0), float(m)) for v, m in zip(box, limits, strict=True)]


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def draft_boxes(detection_lists: Iterable[Sequence[Mapping[str, Any]]], classes: set[str]) -> list[dict[str, Any]]:
    """複数モデルの検出を合わせた下書き。同じクラスで重なる検出は confidence の高い方を残す。

    どちらか一方のモデルにだけ寄った正解にならないよう、両方の検出を候補にする。
    """
    candidates = sorted(
        (d for dets in detection_lists for d in dets if d["class_name"] in classes),
        key=lambda d: (-d["confidence"], d["class_name"], d["bbox"]),
    )
    kept: list[dict[str, Any]] = []
    for d in candidates:
        if all(k["class_name"] != d["class_name"] or _iou(k["bbox"], d["bbox"]) < DRAFT_MERGE_IOU for k in kept):
            kept.append(dict(d))
    return kept


def _class_names(results: Sequence[Mapping[str, Any]]) -> list[str]:
    maps = [r["inference"]["classes"] for r in results]
    if any(m != maps[0] for m in maps):
        raise ValueError("推論結果ごとに class_id と class_name の対応が違う。同じクラス構成のモデルの結果だけを渡す")
    ids = sorted(int(k) for k in maps[0])
    if ids != list(range(len(ids))):
        raise ValueError(f"class_id が 0 からの連番でない: {ids}")
    return [maps[0][str(i)] for i in ids]


def _image_name(video: Path, frame_index: int, suffix: str) -> str:
    return f"{video.stem}_{frame_index:06d}{suffix}"


def prepare_labeling(
    *,
    video: Path,
    results: Sequence[Mapping[str, Any]],
    frame_indexes: Iterable[int],
    classes: set[str],
    output: Path,
) -> None:
    sha = sha256_file(video)
    for r in results:
        if r["source"]["sha256"] != sha:
            raise ValueError(f"推論結果（{r['inference']['model_id']}）の sha256 が動画と一致しない")
    names = _class_names(results)
    wanted = sorted(set(frame_indexes))
    images, drafts = output / "images", output / "labels_draft"
    images.mkdir(parents=True, exist_ok=True)
    drafts.mkdir(parents=True, exist_ok=True)
    (drafts / "classes.txt").write_text("\n".join(names) + "\n")

    info = read_video_info(video)
    by_index = [{f["frame_index"]: f["detections"] for f in r["frames"]} for r in results]
    written = set()
    for index, image in iter_frames(video):
        if index > (wanted[-1] if wanted else -1):
            break
        if index not in wanted:
            continue
        cv2.imwrite(str(images / _image_name(video, index, ".jpg")), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written.add(index)
        boxes = draft_boxes([b.get(index, []) for b in by_index], classes)
        if boxes:
            lines = [to_yolo_line(d["class_id"], d["bbox"], info.width, info.height) for d in boxes]
            (drafts / _image_name(video, index, ".txt")).write_text("\n".join(lines) + "\n")
    missing = sorted(set(wanted) - written)
    if missing:
        raise ValueError(f"動画に無いフレーム番号: {missing}")
    manifest = {"video": video.name, "sha256": sha, "frame_indexes": wanted, "classes": sorted(classes)}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def _build_annotations(
    *,
    video: Path,
    labeling_dir: Path,
    classes: Sequence[str],
    read_objects: Callable[[int, VideoInfo], list[tuple[str, list[float]]]],
) -> dict[str, Any]:
    """manifest.json に載ったフレームを、すべて確認済みとして正解にする。"""
    manifest = json.loads((labeling_dir / "manifest.json").read_text())
    sha = sha256_file(video)
    if manifest["sha256"] != sha:
        raise ValueError("manifest.json の sha256 が動画と一致しない")
    selected = set(classes)
    info = read_video_info(video)
    frames = [
        {
            "frame_index": index,
            "objects": [
                {"class_name": name, "bbox": [round(v, 2) for v in bbox]}
                for name, bbox in read_objects(index, info)
                if name in selected
            ],
        }
        for index in manifest["frame_indexes"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {"path": video.name, "sha256": sha, "width": info.width, "height": info.height},
        "classes": sorted(selected),
        "frames": frames,
    }


def import_yolo(*, video: Path, labeling_dir: Path, labels_dir: Path, classes: Sequence[str]) -> dict[str, Any]:
    """YOLO 形式（.txt）のラベルを取り込む。ラベルのファイルが無いフレームは物体0個。"""
    classes_file = labels_dir / "classes.txt"
    if not classes_file.is_file():
        classes_file = labeling_dir / "labels_draft" / "classes.txt"
    names = [n for n in classes_file.read_text().splitlines() if n.strip()]

    def read_objects(index: int, info: VideoInfo) -> list[tuple[str, list[float]]]:
        label_file = labels_dir / _image_name(video, index, ".txt")
        lines = label_file.read_text().splitlines() if label_file.is_file() else []
        parsed = [yolo_line_to_bbox(line, info.width, info.height) for line in lines if line.strip()]
        return [(names[class_id], bbox) for class_id, bbox in parsed]

    return _build_annotations(video=video, labeling_dir=labeling_dir, classes=classes, read_objects=read_objects)


def import_labelme(*, video: Path, labeling_dir: Path, labels_dir: Path, classes: Sequence[str]) -> dict[str, Any]:
    """labelme 形式（X-AnyLabeling が画像の隣に保存する .json）の矩形を取り込む。.json が無いフレームは物体0個。"""

    def read_objects(index: int, info: VideoInfo) -> list[tuple[str, list[float]]]:
        label_file = labels_dir / _image_name(video, index, ".json")
        if not label_file.is_file():
            return []
        doc = json.loads(label_file.read_text())
        size = (doc.get("imageWidth"), doc.get("imageHeight"))
        if size != (info.width, info.height):
            raise ValueError(f"{label_file.name} の画像サイズ {size[0]}x{size[1]} が動画と違う")
        objects = []
        for shape in doc.get("shapes", []):
            if shape.get("shape_type") != "rectangle":
                continue
            xs = [float(p[0]) for p in shape["points"]]
            ys = [float(p[1]) for p in shape["points"]]
            box = [min(xs), min(ys), max(xs), max(ys)]
            limits = [info.width, info.height, info.width, info.height]
            objects.append((shape["label"], [min(max(v, 0.0), float(m)) for v, m in zip(box, limits, strict=True)]))
        return objects

    return _build_annotations(video=video, labeling_dir=labeling_dir, classes=classes, read_objects=read_objects)
