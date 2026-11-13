"""detection-regression-infer の CLI。推論結果を共通 JSON スキーマで書き出す。"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .adapters.ultralytics import infer_video, load_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="detection-regression-infer")
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("infer", help="動画の全フレームを推論して JSON に書き出す")
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--video", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--model-id", help="結果に記録するモデル名。省略時はモデルのファイル名（拡張子なし）")
    infer.add_argument("--imgsz", type=int, default=640)
    infer.add_argument("--conf", type=float, default=0.25)
    infer.add_argument("--iou", type=float, default=0.7)
    return parser


def _infer(args: argparse.Namespace) -> int:
    if not args.video.is_file():
        print(f"動画が見つからない: {args.video}", file=sys.stderr)
        return 1
    result = infer_video(
        load_model(args.model),
        args.video,
        model_id=args.model_id or args.model.stem,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    detections = sum(len(f["detections"]) for f in result["frames"])
    print(f"{args.output} に書き出した。{result['source']['frame_count']} フレーム・検出 {detections} 件")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "infer":
        return _infer(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
