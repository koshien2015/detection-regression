# detection-regression

物体検出モデルを追加学習する前と後で、同じ動画に対する検出結果をフレーム単位で比べるツールです。どのフレームのどの検出が出現・消失・変化したかを JSON と静的な HTML レポートに残します。正解の BBox を付けたフレームでは、学習後に検出できなくなった物体（デグレ。リグレッションとも呼びます）を判定します。

全体の mAP が上がっていても、以前は検出できていた物体を見逃すようになることがあります。ソフトウェアのリグレッションテストと同じように、その変化を再現できる形で記録するのが目的です。

## 必要なもの

- Python 3.11 以上と [uv](https://docs.astral.sh/uv/)（推論とフレーム画像の書き出し）
- Node.js 22 以上と pnpm（比較・デグレ判定・レポート）
- 比べる2つの YOLO モデル（Ultralytics の `.pt`）と、比較に使う動画

比較とデグレ判定のテストは、YOLO のモデルも Python の環境もなしで実行できます。

## インストール

リポジトリのルートで実行します。

```bash
uv sync --project python        # 推論用の Python の依存（ultralytics・opencv）を入れる
pnpm --dir ts install           # 比較・レポート用の Node の依存を入れる
pnpm --dir ts build:viewer      # レポートのビューアをビルドする（ts/viewer/dist/index.html ができる）
```

## 使い方

以下のコマンドはすべてリポジトリのルートで実行します。パスはルートからの相対パスです。

### 1. 学習前と学習後のモデルで推論する

```bash
uv run --project python detection-regression-infer infer \
  --model models/before.pt --video samples/test.mp4 --output results/before.json \
  --imgsz 640 --conf 0.25

uv run --project python detection-regression-infer infer \
  --model models/after.pt --video samples/test.mp4 --output results/after.json \
  --imgsz 640 --conf 0.25
```

動画の全フレームを先頭から順に推論し、検出の無いフレームも含めて JSON に書き出します。`--model-id` を省略すると、モデルのファイル名（拡張子なし）を記録します。推論は1回だけで、以降の比較では YOLO を読み込みません。

### 2. 検出結果を比べる

```bash
pnpm --dir ts -s cli compare \
  --before results/before.json --after results/after.json --output results/diff.json
```

許容幅を変えるときは `--config examples/sample_config.json` のように設定ファイルを渡します。項目は「差分の判定」の表を参照してください。

### 3. 正解を付けたフレームでデグレを判定する（任意）

```bash
pnpm --dir ts -s cli evaluate \
  --before results/before.json --after results/after.json \
  --annotations annotations/test.json --output results/evaluation.json
```

正解アノテーションの書き方は「JSON の形式」を参照してください。全フレームに付ける必要はありません。

### 4. レポートを作る

```bash
uv run --project python detection-regression-infer extract-frames \
  --video samples/test.mp4 --diff results/diff.json --evaluation results/evaluation.json \
  --output results/report/frames

pnpm --dir ts -s cli report \
  --diff results/diff.json --evaluation results/evaluation.json --output results/report
```

`extract-frames` は、差分のあるフレームと正解を付けたフレームだけを PNG に書き出します。ほかのフレームも見たいときは `--frames 120,121` のように番号を足します。人が写った動画のレポートを公開するときは、`--motion-from <元の動画>` を付けると、前後のフレームとの差分（動いている成分）だけの画像を書き出します。輪郭だけが残り、人物を判別しにくくなります。枠の座標は変わりません。`results/report/index.html` をブラウザで開くと、レポートが表示されます。Web サーバーは要りません。

### サンプルで試す

手元に動画が無い場合は、ultralytics に同梱の `bus.jpg` から作る3秒の疑似動画で、推論からデグレ判定まで試せます。`examples/bus/make_sample.py` は、画像をゆっくり拡大・移動させた動画と、元画像の正解（`examples/bus/ground_truth.json`）を各フレームに写した正解アノテーションを書き出します。

```bash
uv run --project python python examples/bus/make_sample.py \
  --video samples/bus_pan.mp4 --annotations annotations/bus_pan.json
```

以降は「使い方」の 1〜4 で、`--video samples/bus_pan.mp4`・`--annotations annotations/bus_pan.json` を指定します。モデルには COCO で学習済みの `yolo26n.pt` と `yolo26s.pt` を使えます（初回の推論で自動的にダウンロードされます）。

### 正解をラベリングツールで付ける

正解アノテーションは、ラベリングツール（X-AnyLabeling など）で付けた YOLO 形式のラベルから作れます。

1. 正解を付けるフレームの画像と下書きを書き出します。下書きには、渡した推論結果の検出を合わせたものが入ります。

   ```bash
   uv run --project python detection-regression-infer prepare-labeling \
     --video samples/test.mp4 --result results/before.json --result results/after.json \
     --frames 10,20,30 --classes bottle --output annotations/labeling/test
   ```

   `annotations/labeling/test/images/` に画像、`labels_draft/` に YOLO 形式の下書きと `classes.txt` ができます。
2. ラベリングツールで `images/` を開き、`labels_draft/` の下書きを YOLO 形式でインポートします。設定ファイルには `labels_draft/classes.txt` を指定します。
3. 下書きを確認して直します。写っているのに枠が無い物体には枠を足し、誤検出の枠は消します。書き出した**すべての**画像を確認してください。
4. YOLO 形式で `annotations/labeling/test/labels/` にエクスポートします。
5. 正解アノテーションに変換します。

   ```bash
   uv run --project python detection-regression-infer import-yolo \
     --video samples/test.mp4 --labeling-dir annotations/labeling/test \
     --labels annotations/labeling/test/labels --classes bottle --output annotations/test.json
   ```

X-AnyLabeling が画像の隣に保存した `.json`（labelme 形式）をそのまま使うときは、手順4のエクスポートを省き、`import-yolo` の代わりに `import-labelme --labels annotations/labeling/test/images` を使います。

`prepare-labeling` で書き出したフレームは、すべて確認済みとして扱います。ラベルのファイルが無いフレームは「物体が写っていない」という正解になります。

## JSON の形式

| ファイル | 作るコマンド | 形式 |
|---|---|---|
| 推論結果（before.json・after.json） | `infer` | `schema/inference-result.schema.json` |
| 正解アノテーション | 手で書く | `schema/annotations.schema.json` |
| 差分（diff.json） | `compare` | `ts/src/compare.ts` の `Diff` 型 |
| デグレ判定（evaluation.json） | `evaluate` | `ts/src/evaluation.ts` の `Evaluation` 型 |

推論結果と正解アノテーションの共通の決まりは次のとおりです。

- BBox は `[x1, y1, x2, y2]` で、元動画の解像度でのピクセル座標です。
- `frame_index` は、動画を先頭から順に読んだときの 0 始まりの番号です。フレームの対応づけはこの番号で行い、`timestamp_ms` は使いません。
- `source.sha256` は動画ファイルの SHA-256 です。別の動画の結果を比べないための識別に使います。
- `source.path` にはファイル名か相対パスを書きます。絶対パスは受け付けません。

正解アノテーションの例です。`objects` が空のフレームは「何も写っていないのが正解」として評価し、`frames` に載せていないフレームは評価から除きます。一部のクラスにだけ正解を付けた場合は、`classes` にそのクラスを書きます。ほかのクラスの検出は FP に数えません。

```json
{
  "schema_version": "1.0",
  "source": { "path": "test.mp4", "sha256": "<動画の sha256>", "width": 1280, "height": 720 },
  "classes": ["bottle"],
  "frames": [
    { "frame_index": 100, "objects": [{ "class_name": "bottle", "bbox": [120, 80, 200, 340] }] },
    { "frame_index": 101, "objects": [] }
  ]
}
```

`<動画の sha256>` は `shasum -a 256 samples/test.mp4` で確かめられます。

## 差分の判定

同じフレームの中で、before と after の検出を一対一に対応づけ、4つに分類します。

| 分類 | 意味 |
|---|---|
| unchanged | 対応づけられ、下の表の許容幅に収まっている |
| modified | 対応づけられたが、いずれかの項目が許容幅を超えた。`changed` に項目を列挙する |
| disappeared | before にあり、after で対応する検出が無い |
| appeared | after にあり、before で対応する検出が無い |

同じクラス（`class_name`）で IoU が `match_iou_threshold` 以上の組を、IoU の大きい順に対応づけます。クラスが変わった検出は disappeared と appeared に数え、同じ位置にある組を `class_change_candidates` に参考として残します。

| 設定の項目 | 既定値 | 意味 |
|---|---|---|
| `match_iou_threshold` | 0.5 | 同じ物体とみなす最小の IoU |
| `iou_tolerance` | 0.8 | before と after の BBox の IoU がこれ未満なら変化 |
| `center_shift_tolerance_ratio` | 0.1 | BBox の中心の移動が before の BBox の対角線の 10% を超えたら変化 |
| `size_change_tolerance_ratio` | 0.15 | 幅か高さの変化が before の 15% を超えたら変化 |
| `confidence_tolerance` | null | null なら confidence の差を変化に数えない。数値（例 0.1）を指定すると、差がそれを超えたら変化 |
| `allow_inference_mismatch` | false | 推論条件（conf・iou・imgsz）が違う結果の比較を、警告付きで許す |

差分があること自体は、改善とも悪化とも判定しません。

## デグレ判定

正解を付けたフレームだけを評価します。confidence の高い検出から順に、同じクラスで IoU が 0.5（`--iou-threshold` で変更可）以上の正解に一対一で割り当て、TP・FP・FN を数えます。

| 遷移 | before | after |
|---|---|---|
| maintained（維持） | PASS | PASS |
| improved（改善） | FAIL | PASS |
| regressed（デグレ） | PASS | FAIL |
| unresolved（未改善） | FAIL | FAIL |

PASS / FAIL は正解の物体ごとに決めます。デグレと改善のうち、判定を分けた IoU が閾値から 0.05 以内のものには「境界」の印を付けます。小さい物体では、正解の枠の付け方や閾値だけで判定が入れ替わるためです。小さい物体の評価では、`--iou-threshold 0.3` も試してください。正解の無い検出（FP）は遷移の表に入れず、モデルごとの件数と precision に反映します。

## 比較を止める条件

次の場合は、黙って比べずにエラーで止めます。

- 動画の sha256・fps・幅・高さ・フレーム数が違う
- 推論条件（confidence 閾値・IoU 閾値・入力サイズ）が違う（`allow_inference_mismatch` で許可した場合を除く）
- 比べる2つの結果でフレームの集合が違う。欠けているフレーム番号を表示します
- `schema_version` が違う

## 制約

- 同じ動画を同じ条件で推論した結果だけを比べます。fps の違う動画どうしの時刻合わせや、別の動画のフレームの対応づけはしません。
- 対応づけは IoU による貪欲法です。物体が密集していると、IoU の合計が最大にならない組を選ぶことがあります。位置が大きく変わった検出は、同じ物体でも disappeared と appeared になります。
- フレームをまたいだ追跡はしません。検出は各フレームで独立に比べます。
- 許容幅の既定値は、COCO の学習済みモデル（yolo26n と yolo26s）と、小さい物体（キャップ）の検出モデルの比較を見て決めた値です。対象や目的に合わせて、設定ファイルで調整してください。
- confidence の差は、既定では変化に数えません。モデルが変わると confidence はほぼ必ず動くため、数えると差分のほとんどが confidence の揺れになるためです。差の値は `metrics.confidence_delta` に記録し、レポートにも表示します。
- フィールドをまたぐ制約（`x1 <= x2`、`frame_index` の重複、BBox が画面の内側か、`class_id` と `class_name` の対応）は、生成した JSON Schema には含まれません。`compare` と `evaluate` が読み込むときに検査します。

## 開発

```bash
pnpm --dir ts test            # 比較・デグレ判定・レポートのテスト（vitest）
pnpm --dir ts typecheck       # 型チェック
pnpm --dir ts gen:schema      # ts/src/schema.ts を変えたら JSON Schema を作り直す
uv run --project python pytest python/tests   # 推論・フレーム書き出しのテスト
```

設計上の判断と、却下した案は `docs/adr/` にあります。
