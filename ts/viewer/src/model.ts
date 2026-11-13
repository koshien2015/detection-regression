// ビューアの表示用の純粋関数。React に依存させず、vitest で確かめる。
import type { MatchedEntry } from '../../src/compare.js'
import type { Transition } from '../../src/evaluation.js'
import type { ReportData, ReportFrame } from '../../src/report.js'
import type { BBox, Detection } from '../../src/schema.js'

export type SortKey = 'impact' | 'index'
export type FilterKey = 'all' | 'changed' | 'regressed'

export const sortFrames = (frames: readonly ReportFrame[], key: SortKey): ReportFrame[] =>
  [...frames].sort((a, b) =>
    key === 'index'
      ? a.frame_index - b.frame_index
      : b.regressed_count - a.regressed_count || b.change_count - a.change_count || a.frame_index - b.frame_index,
  )

// 回帰判定があるときは全件から始める。差分が無くても回帰しているフレーム（両方とも見逃し続けている等）を隠さないため
export const defaultFilter = (data: ReportData): FilterKey => (data.evaluation_summary === null ? 'changed' : 'all')

export const filterFrames = (frames: readonly ReportFrame[], key: FilterKey): ReportFrame[] =>
  frames.filter((f) => key === 'all' || (key === 'changed' ? f.diff.has_changes : f.regressed_count > 0))

export type Layer = 'before' | 'after' | 'truth'
export type BoxStatus = 'unchanged' | 'modified' | 'appeared' | 'disappeared' | Transition

export type OverlayBox = { key: string; layer: Layer; status: BoxStatus; bbox: BBox; label: string }

const detectionLabel = (d: Detection): string => `${d.class_name} ${d.confidence.toFixed(2)}`

const box = (layer: Layer, status: BoxStatus, d: Detection, i: number): OverlayBox => ({
  key: `${layer}-${status}-${i}`,
  layer,
  status,
  bbox: d.bbox,
  label: detectionLabel(d),
})

const pairBoxes = (entries: readonly MatchedEntry[], status: 'unchanged' | 'modified'): OverlayBox[] =>
  entries.flatMap((e, i) => [box('before', status, e.before, i), box('after', status, e.after, i)])

export const overlayBoxes = (frame: ReportFrame): OverlayBox[] => [
  ...pairBoxes(frame.diff.unchanged, 'unchanged'),
  ...pairBoxes(frame.diff.modified, 'modified'),
  ...frame.diff.disappeared.map((d, i) => box('before', 'disappeared', d, i)),
  ...frame.diff.appeared.map((d, i) => box('after', 'appeared', d, i)),
  ...(frame.evaluation?.objects ?? []).map(
    (o, i): OverlayBox => ({
      key: `truth-${i}`,
      layer: 'truth',
      status: o.transition,
      bbox: o.ground_truth.bbox,
      label: `正解 ${o.ground_truth.class_name}`,
    }),
  ),
]

export const formatPercent = (v: number | null): string => (v === null ? '—' : `${(v * 100).toFixed(1)}%`)

export const formatSigned = (v: number, digits = 2): string => `${v > 0 ? '+' : ''}${v.toFixed(digits)}`

export const formatTime = (ms: number): string => {
  const totalSeconds = ms / 1000
  const minutes = Math.floor(totalSeconds / 60)
  return `${minutes}:${(totalSeconds - minutes * 60).toFixed(2).padStart(5, '0')}`
}

export const STATUS_LABELS: Record<BoxStatus, string> = {
  unchanged: '維持',
  modified: '変化',
  appeared: '出現',
  disappeared: '消失',
  maintained: '維持',
  improved: '改善',
  regressed: '回帰',
  unresolved: '未改善',
}
