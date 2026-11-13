# 0001 言語の分け方：推論は Python、比較以降は TypeScript

## 決定

推論とフレーム画像の抽出は Python、比較・デグレ判定・レポートは TypeScript と React で実装する。両者は共通 JSON スキーマのファイルだけで受け渡す。

## 理由

- 推論には Ultralytics が必要で、Python から離れられない。
- 比較以降は JSON を扱うだけで、言語を選ばない。レポートの UI は React + Vite で作るほうが速い。
- 比較のテストを Python の環境なしで実行できる。

## 却下した案

すべて Python で実装する案。一対一の対応づけに `scipy.optimize.linear_sum_assignment` などの既存ライブラリを使える利点はあるが、レポートの UI を作りにくい。

## 受け入れた代償

- IoU と一対一の対応づけを自前で実装する。正しさは `ts/tests/matching.test.ts` で保証する。
- スキーマを 2 つの言語で扱う。定義元は `ts/src/schema.ts` の zod に一本化し、`pnpm gen:schema` で `schema/*.json` を生成する。生成物と定義のずれは `ts/tests/json-schema.test.ts` で検出する。
- フィールドをまたぐ制約（x1 <= x2、frame_index の重複、BBox が画面内か、class_id と class_name の対応）は zod の superRefine で検査しており、生成した JSON Schema には含まれない。Python 側の出力検証はその分だけ弱い。compare コマンドは読み込んだ JSON を TypeScript 側で必ず全体検証してから比較する。
