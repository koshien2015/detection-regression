# detection-regression

物体検出モデルを追加学習する前と後で、同じ動画に対する検出結果をフレーム単位で比べるツールです。どのフレームのどの検出が出現・消失・変化したかを JSON と静的な HTML レポートに残します。正解の BBox を付けたフレームでは、学習後に検出できなくなった物体（回帰）を判定します。

全体の mAP が上がっていても、以前は検出できていた物体を見逃すようになることがあります。ソフトウェアの回帰テストと同じように、その変化を再現できる形で記録するのが目的です。

## 必要なもの

- Python 3.11 以上と [uv](https://docs.astral.sh/uv/)（推論とフレーム画像の書き出し）
- Node.js 22 以上と pnpm（比較・回帰判定・レポート）
- 比べる2つの YOLO モデル（Ultralytics の `.pt`）と、比較に使う動画

比較と回帰判定のテストは、YOLO のモデルも Python の環境もなしで実行できます。

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

### 3. 正解を付けたフレームで回帰を判定する（任意）

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

`extract-frames` は、差分のあるフレームと正解を付けたフレームだけを PNG に書き出します。ほかのフレームも見たいときは `--frames 120,121` のように番号を足します。`results/report/index.html` をブラウザで開くと、レポートが表示されます。Web サーバーは要りません。

## JSON の形式

| ファイル | 作るコマンド | 形式 |
|---|---|---|
| 推論結果（before.json・after.json） | `infer` | `schema/inference-result.schema.json` |
| 正解アノテーション | 手で書く | `schema/annotations.schema.json` |
| 差分（diff.json） | `compare` | `ts/src/compare.ts` の `Diff` 型 |
| 回帰判定（evaluation.json） | `evaluate` | `ts/src/evaluation.ts` の `Evaluation` 型 |

推論結果と正解アノテーションの共通の決まりは次のとおりです。

- BBox は `[x1, y1, x2, y2]` で、元動画の解像度でのピクセル座標です。
- `frame_index` は、動画を先頭から順に読んだときの 0 始まりの番号です。フレームの対応づけはこの番号で行い、`timestamp_ms` は使いません。
- `source.sha256` は動画ファイルの SHA-256 です。別の動画の結果を比べないための識別に使います。
- `source.path` にはファイル名か相対パスを書きます。絶対パスは受け付けません。

正解アノテーションの例です。`objects` が空のフレームは「何も写っていないのが正解」として評価し、`frames` に載せていないフレームは評価から除きます。

```json
{
  "schema_version": "1.0",
  "source": { "path": "test.mp4", "sha256": "<動画の sha256>", "width": 1280, "height": 720 },
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
| `confidence_tolerance` | 0.05 | confidence の差がこれを超えたら変化 |
| `iou_tolerance` | 0.95 | before と after の BBox の IoU がこれ未満なら変化 |
| `center_shift_tolerance_px` | 4 | BBox の中心の移動がこれを超えたら変化 |
| `size_change_tolerance_ratio` | 0.05 | 幅か高さの変化が before の 5% を超えたら変化 |
| `allow_inference_mismatch` | false | 推論条件（conf・iou・imgsz）が違う結果の比較を、警告付きで許す |

差分があること自体は、改善とも悪化とも判定しません。

## 回帰判定

正解を付けたフレームだけを評価します。confidence の高い検出から順に、同じクラスで IoU が 0.5（`--iou-threshold` で変更可）以上の正解に一対一で割り当て、TP・FP・FN を数えます。

| 遷移 | before | after |
|---|---|---|
| maintained（維持） | PASS | PASS |
| improved（改善） | FAIL | PASS |
| regressed（回帰） | PASS | FAIL |
| unresolved（未改善） | FAIL | FAIL |

PASS / FAIL は正解の物体ごとに決めます。正解の無い検出（FP）は遷移の表に入れず、モデルごとの件数と precision に反映します。

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
- 許容幅の既定値は人工データで決めた値です。モデルの入力サイズを変えるだけで、対応づけた検出の大半が modified になることを確かめています。実際の比較では、目的に合わせて設定ファイルで調整してください。
- フィールドをまたぐ制約（`x1 <= x2`、`frame_index` の重複、BBox が画面の内側か、`class_id` と `class_name` の対応）は、生成した JSON Schema には含まれません。`compare` と `evaluate` が読み込むときに検査します。

## 開発

```bash
pnpm --dir ts test            # 比較・回帰判定・レポートのテスト（vitest）
pnpm --dir ts typecheck       # 型チェック
pnpm --dir ts gen:schema      # ts/src/schema.ts を変えたら JSON Schema を作り直す
uv run --project python pytest python/tests   # 推論・フレーム書き出しのテスト
```

設計上の判断と、却下した案は `docs/adr/` にあります。
