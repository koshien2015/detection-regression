"""detection-regression-infer の CLI。推論結果を共通 JSON スキーマで書き出す。"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .adapters.ultralytics import infer_video, load_model
from .frames import extract_frames, select_frames
from .labeling import import_labelme, import_yolo, prepare_labeling
from .video import sha256_file

# ts/src/compare.ts の DIFF_SCHEMA_VERSION と合わせる
SUPPORTED_DIFF_VERSION = "1.0"


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
    extract.add_argument(
        "--motion-from",
        type=Path,
        help="指定した動画（推論に使った動画の元の動画）から、動いている成分だけの画像を作って書き出す。人物を判別しにくくする",
    )

    prep = commands.add_parser("prepare-labeling", help="正解を付けるフレームの画像と、推論結果からの下書きを書き出す")
    prep.add_argument("--video", type=Path, required=True)
    prep.add_argument("--result", type=Path, action="append", required=True, help="下書きに使う推論結果（複数指定可）")
    prep.add_argument("--frames", type=_frame_list, required=True, help="正解を付けるフレーム番号（カンマ区切り）")
    prep.add_argument("--classes", type=_name_list, required=True, help="正解を付けるクラス（カンマ区切り）")
    prep.add_argument("--output", type=Path, required=True)

    for name, help_text, labels_help in [
        ("import-yolo", "YOLO 形式のラベルを正解アノテーションの JSON にする", "YOLO 形式でエクスポートしたラベルのフォルダ"),
        ("import-labelme", "labelme 形式（X-AnyLabeling の保存形式）のラベルを正解アノテーションの JSON にする", "画像の隣に保存された .json のフォルダ（通常は images/）"),
    ]:
        imp = commands.add_parser(name, help=help_text)
        imp.add_argument("--video", type=Path, required=True)
        imp.add_argument("--labeling-dir", type=Path, required=True, help="prepare-labeling の出力先（manifest.json がある）")
        imp.add_argument("--labels", type=Path, required=True, help=labels_help)
        imp.add_argument("--classes", type=_name_list, required=True)
        imp.add_argument("--output", type=Path, required=True)
    return parser


def _name_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


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
    if diff.get("diff_schema_version") != SUPPORTED_DIFF_VERSION:
        print(
            f"diff_schema_version が {diff.get('diff_schema_version')!r}。{SUPPORTED_DIFF_VERSION} のみ扱う",
            file=sys.stderr,
        )
        return 1
    expected = diff["inputs"]["source"]["sha256"]
    actual = sha256_file(args.video)
    if actual != expected:
        print(f"動画の sha256 が diff.json と一致しない（diff: {expected} / 動画: {actual}）", file=sys.stderr)
        return 1
    evaluation = json.loads(args.evaluation.read_text()) if args.evaluation else None
    indexes = select_frames(diff=diff, evaluation=evaluation, explicit=args.frames)
    try:
        written = extract_frames(args.video, indexes, args.output, motion_from=args.motion_from)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"{args.output} に {len(written)} 枚を書き出した")
    return 0


def _prepare(args: argparse.Namespace) -> int:
    results = [json.loads(p.read_text()) for p in args.result]
    try:
        prepare_labeling(
            video=args.video, results=results, frame_indexes=args.frames, classes=set(args.classes), output=args.output
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"{args.output} に {len(set(args.frames))} フレーム分の画像と下書きを書き出した")
    return 0


def _import(args: argparse.Namespace) -> int:
    importer = import_labelme if args.command == "import-labelme" else import_yolo
    try:
        annotations = importer(
            video=args.video, labeling_dir=args.labeling_dir, labels_dir=args.labels, classes=args.classes
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(annotations, ensure_ascii=False, indent=2) + "\n")
    objects = sum(len(f["objects"]) for f in annotations["frames"])
    print(f"{args.output} に書き出した。{len(annotations['frames'])} フレーム・物体 {objects} 個")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare-labeling":
        return _prepare(args)
    if args.command in ("import-yolo", "import-labelme"):
        return _import(args)
    if args.command == "infer":
        return _infer(args)
    if args.command == "extract-frames":
        return _extract(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
