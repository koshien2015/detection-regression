"""detection-regression-infer の CLI。推論結果を共通 JSON スキーマで書き出す。"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .adapters.ultralytics import infer_video, load_model
from .frames import extract_frames, select_frames
from .video import sha256_file


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

    extract = commands.add_parser("extract-frames", help="差分のあるフレームなどを PNG に書き出す")
    extract.add_argument("--video", type=Path, required=True)
    extract.add_argument("--diff", type=Path, required=True, help="compare が書き出した diff.json")
    extract.add_argument("--evaluation", type=Path, help="evaluate が書き出した evaluation.json（正解を付けたフレームも書き出す）")
    extract.add_argument("--frames", type=_frame_list, default=[], help="追加で書き出すフレーム番号（カンマ区切り）")
    extract.add_argument("--output", type=Path, required=True)
    return parser


def _frame_list(value: str) -> list[int]:
    try:
        return [int(v) for v in value.split(",") if v.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"フレーム番号はカンマ区切りの整数にする: {value}") from error


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


def _extract(args: argparse.Namespace) -> int:
    if not args.video.is_file():
        print(f"動画が見つからない: {args.video}", file=sys.stderr)
        return 1
    diff = json.loads(args.diff.read_text())
    expected = diff["inputs"]["source"]["sha256"]
    actual = sha256_file(args.video)
    if actual != expected:
        print(f"動画の sha256 が diff.json と一致しない（diff: {expected} / 動画: {actual}）", file=sys.stderr)
        return 1
    evaluation = json.loads(args.evaluation.read_text()) if args.evaluation else None
    indexes = select_frames(diff=diff, evaluation=evaluation, explicit=args.frames)
    try:
        written = extract_frames(args.video, indexes, args.output)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"{args.output} に {len(written)} 枚を書き出した")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "infer":
        return _infer(args)
    if args.command == "extract-frames":
        return _extract(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
