# 0002 対応づけと差分の判定のしかた

## クラスの同一性は class_name で判定する

モデルごとに class_id の割り当てが違いうるため、対応づけは class_name で行う。同じ class_id に before と after で別の名前が付いている場合は、比較の出力に警告を残す。

## 一対一の対応づけは IoU の大きい順の貪欲法

同じクラスで IoU が `match_iou_threshold`（既定 0.5）以上の組を、IoU の大きい順に確定させる。IoU が同点の組は confidence の差が小さいほうを優先し、それでも決まらなければ検出の内容（クラス名・BBox・confidence）の並び順で決める。入力配列の順序には依存しない。

却下した案はハンガリアン法（IoU の合計を最大にする割り当て）。貪欲法は、重なり合う検出が密集しているときに全体最適でない組を選ぶことがある。初版の対象（少数の物体）では差が出にくく、結果の説明もしやすいため、貪欲法を選んだ。

## クラスが変わった検出は disappeared と appeared にする

仕様の 2.2 節は「クラスの変化」を差分として挙げ、7.1 節はクラスが違う検出を無理に同一視しないとしている。両方を満たすため、クラスが変わった検出は disappeared と appeared に分類したうえで、同じ位置（IoU が `match_iou_threshold` 以上）にある組を `class_change_candidates` に参考情報として残す。`class_change_candidates` は出現・消失の件数を変えない。

## 「変化」の判定

対応づけた組について、次のいずれかを超えたら modified とし、`changed` に項目を列挙する。

| 項目 | 既定の許容幅 |
|---|---|
| confidence | 差の絶対値 0.05 |
| iou | 0.95 未満 |
| center | 中心の移動 4px |
| size | 幅か高さの変化が before の 5% |

境界の値は浮動小数点の誤差を考慮して 1e-9 の余裕を持たせる。

## 比較を拒否する条件

- 元動画の sha256・fps・幅・高さ・フレーム数のいずれかが違う：`source_mismatch`
- 推論の confidence 閾値・IoU 閾値・入力サイズのいずれかが違う：`inference_mismatch`。`allow_inference_mismatch` を指定したときだけ、警告付きで比較する
- フレームの集合が違う：`frame_set_mismatch`。欠けているフレーム番号を返す
- schema_version が違う：`schema_version_mismatch`。他のフィールドより先に確かめる

ファイル名（source.path）だけが違う場合は、sha256 が一致していれば警告にとどめる。

## フレームの対応づけ

frame_index（先頭から順に読んだ 0 始まりの番号）で対応づける。timestamp_ms は記録するだけで、対応づけには使わない。検出が無いフレームも空配列で必ず記録し、「検出なし」と「フレームの欠損」を区別する。
