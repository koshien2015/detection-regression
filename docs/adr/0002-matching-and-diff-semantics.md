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
| iou | 0.8 未満 |
| center | 中心の移動が before の BBox の対角線の 10% |
| size | 幅か高さの変化が before の 15% |
| confidence | 既定では数えない（`confidence_tolerance: null`） |

当初は confidence 0.05・IoU 0.95・中心 4px・大きさ 5% にしていたが、yolo26n と yolo26s の比較で、対応づけた検出の 6 割近くが modified になった。キャップ検出モデルの学習段階どうしの比較では、unchanged がほぼ 0 件だった。差分が揺れに埋もれて、読み手が本当の変化を探せないため、緩めた。

中心の移動を px でなく対角線に対する割合にしたのは、px では動画の解像度や物体の大きさで意味が変わるため。34×20px のキャップと 800×500px のバスに、同じ 10px を当てるのは不自然になる。

confidence を既定で数えないのは、モデルが変わると confidence はほぼ必ず動くため。差の値は `metrics.confidence_delta` に残すので、confidence の低下を調べたいときは `confidence_tolerance` に数値を指定する。

境界の値は浮動小数点の誤差を考慮して 1e-9 の余裕を持たせる。

## 比較を拒否する条件

- 元動画の sha256・fps・幅・高さ・フレーム数のいずれかが違う：`source_mismatch`
- 推論の confidence 閾値・IoU 閾値・入力サイズのいずれかが違う：`inference_mismatch`。`allow_inference_mismatch` を指定したときだけ、警告付きで比較する
- フレームの集合が違う：`frame_set_mismatch`。欠けているフレーム番号を返す
- schema_version が違う：`schema_version_mismatch`。他のフィールドより先に確かめる

ファイル名（source.path）だけが違う場合は、sha256 が一致していれば警告にとどめる。

## フレームの対応づけ

frame_index（先頭から順に読んだ 0 始まりの番号）で対応づける。timestamp_ms は記録するだけで、対応づけには使わない。検出が無いフレームも空配列で必ず記録し、「検出なし」と「フレームの欠損」を区別する。
