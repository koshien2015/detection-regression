import { compareDetections, greedyMatch, matchDetections, type Pair } from './matching.js'
import {
  parseInferenceResult,
  type Detection,
  type Frame,
  type InferenceResult,
  type ParseError,
} from './schema.js'

export const DIFF_SCHEMA_VERSION = '1.0'

export type CompareOptions = {
  // before と after の検出を同じ物体とみなす最小の IoU
  match_iou_threshold: number
  // 対応づけた検出について、これを超えたら変化とみなす量
  confidence_tolerance: number
  iou_tolerance: number // IoU がこれ未満なら変化
  center_shift_tolerance_px: number
  size_change_tolerance_ratio: number // 幅・高さの変化量 / before の幅・高さ
  allow_inference_mismatch: boolean
}

// undefined の項目は既定値を使う（スプレッドで既定値を undefined に上書きしないため）
export type CompareOptionsInput = { [K in keyof CompareOptions]?: CompareOptions[K] | undefined }

const resolveOptions = (input: CompareOptionsInput): CompareOptions => ({
  ...DEFAULT_COMPARE_OPTIONS,
  ...Object.fromEntries(Object.entries(input).filter(([, v]) => v !== undefined)),
})

export const DEFAULT_COMPARE_OPTIONS: CompareOptions = {
  match_iou_threshold: 0.5,
  confidence_tolerance: 0.05,
  iou_tolerance: 0.95,
  center_shift_tolerance_px: 4,
  size_change_tolerance_ratio: 0.05,
  allow_inference_mismatch: false,
}

// 浮動小数点の誤差で境界の判定が揺れないようにするための許容誤差
const EPSILON = 1e-9

export type ChangeKind = 'confidence' | 'iou' | 'center' | 'size'

export type MatchMetrics = {
  iou: number
  confidence_delta: number
  center_shift_px: number
  width_delta_px: number
  height_delta_px: number
}

export type MatchedEntry = { before: Detection; after: Detection; metrics: MatchMetrics; changed: ChangeKind[] }

export type FrameDiff = {
  frame_index: number
  timestamp_ms: number
  has_changes: boolean
  unchanged: MatchedEntry[]
  modified: MatchedEntry[]
  appeared: Detection[]
  disappeared: Detection[]
  // クラスが違うが位置が重なる組。appeared / disappeared の分類は変えず、参考情報として残す
  class_change_candidates: Pair<Detection, Detection>[]
}

type InputIdentity = Pick<InferenceResult['inference'], 'model_id'> & { inference: InferenceResult['inference'] }

export type Diff = {
  diff_schema_version: string
  options: CompareOptions
  inputs: { source: InferenceResult['source']; before: InputIdentity; after: InputIdentity }
  warnings: string[]
  summary: {
    frames_total: number
    frames_with_changes: number
    unchanged: number
    modified: number
    appeared: number
    disappeared: number
    class_change_candidates: number
  }
  frames: FrameDiff[]
}

export type CompareError =
  | ParseError
  | { kind: 'source_mismatch'; message: string }
  | { kind: 'inference_mismatch'; message: string }
  | {
      kind: 'frame_set_mismatch'
      message: string
      details: { only_in_before: number[]; only_in_after: number[] }
    }

export type CompareOutcome = { ok: true; diff: Diff } | { ok: false; error: CompareError }

const center = ([x1, y1, x2, y2]: readonly number[]): [number, number] => [
  ((x1 ?? 0) + (x2 ?? 0)) / 2,
  ((y1 ?? 0) + (y2 ?? 0)) / 2,
]
const width = (b: readonly number[]): number => (b[2] ?? 0) - (b[0] ?? 0)
const height = (b: readonly number[]): number => (b[3] ?? 0) - (b[1] ?? 0)

const metricsOf = (pair: Pair<Detection, Detection>): MatchMetrics => {
  const [bx, by] = center(pair.before.bbox)
  const [ax, ay] = center(pair.after.bbox)
  return {
    iou: pair.iou,
    confidence_delta: pair.after.confidence - pair.before.confidence,
    center_shift_px: Math.hypot(ax - bx, ay - by),
    width_delta_px: width(pair.after.bbox) - width(pair.before.bbox),
    height_delta_px: height(pair.after.bbox) - height(pair.before.bbox),
  }
}

const exceeds = (value: number, tolerance: number): boolean => value > tolerance + EPSILON

const changesOf = (m: MatchMetrics, before: Detection, o: CompareOptions): ChangeKind[] => {
  const sizeTolW = o.size_change_tolerance_ratio * Math.max(width(before.bbox), 1)
  const sizeTolH = o.size_change_tolerance_ratio * Math.max(height(before.bbox), 1)
  const checks: [ChangeKind, boolean][] = [
    ['confidence', exceeds(Math.abs(m.confidence_delta), o.confidence_tolerance)],
    ['iou', m.iou < o.iou_tolerance - EPSILON],
    ['center', exceeds(m.center_shift_px, o.center_shift_tolerance_px)],
    ['size', exceeds(Math.abs(m.width_delta_px), sizeTolW) || exceeds(Math.abs(m.height_delta_px), sizeTolH)],
  ]
  return checks.filter(([, changed]) => changed).map(([kind]) => kind)
}

const diffFrame = (before: Frame, after: Frame, o: CompareOptions): FrameDiff => {
  const matched = matchDetections(before.detections, after.detections, { iouThreshold: o.match_iou_threshold })
  const entries = matched.pairs.map((pair): MatchedEntry => {
    const metrics = metricsOf(pair)
    return { before: pair.before, after: pair.after, metrics, changed: changesOf(metrics, pair.before, o) }
  })
  const classChanges = greedyMatch(matched.unmatchedBefore, matched.unmatchedAfter, {
    iouThreshold: o.match_iou_threshold,
    compareA: compareDetections,
    compareB: compareDetections,
    classes: 'different',
  }).pairs
  const modified = entries.filter((e) => e.changed.length > 0)
  return {
    frame_index: before.frame_index,
    timestamp_ms: before.timestamp_ms,
    has_changes: modified.length + matched.unmatchedBefore.length + matched.unmatchedAfter.length > 0,
    unchanged: entries.filter((e) => e.changed.length === 0),
    modified,
    appeared: matched.unmatchedAfter,
    disappeared: matched.unmatchedBefore,
    class_change_candidates: classChanges,
  }
}

const SOURCE_KEYS = ['sha256', 'fps', 'width', 'height', 'frame_count'] as const
const INFERENCE_KEYS = ['confidence_threshold', 'iou_threshold', 'image_size'] as const

const differingKeys = <T extends object, K extends keyof T>(a: T, b: T, keys: readonly K[]): string[] =>
  keys.filter((k) => a[k] !== b[k]).map((k) => `${String(k)}: ${String(a[k])} / ${String(b[k])}`)

const classMapWarnings = (before: InferenceResult, after: InferenceResult): string[] =>
  Object.entries(before.inference.classes)
    .filter(([id, name]) => {
      const other = after.inference.classes[id]
      return other !== undefined && other !== name
    })
    .sort(([a], [b]) => Number(a) - Number(b))
    .map(
      ([id, name]) =>
        `class_id ${id} の名前が違う（before: ${name} / after: ${after.inference.classes[id]}）。対応づけは class_name で行う`,
    )

const sortedIndexes = (frames: readonly Frame[]): number[] => frames.map((f) => f.frame_index).sort((a, b) => a - b)

const preview = (xs: number[]): string =>
  xs.length === 0 ? '(なし)' : xs.length > 10 ? `${xs.slice(0, 10).join(', ')} ほか${xs.length - 10}件` : xs.join(', ')

type Checked = { ok: true; warnings: string[] } | { ok: false; error: CompareError }

const checkComparable = (before: InferenceResult, after: InferenceResult, o: CompareOptions): Checked => {
  const sourceDiffs = differingKeys(before.source, after.source, SOURCE_KEYS)
  if (sourceDiffs.length > 0) {
    return { ok: false, error: { kind: 'source_mismatch', message: `元動画が一致しない（before / after）: ${sourceDiffs.join(', ')}` } }
  }
  const inferenceDiffs = differingKeys(before.inference, after.inference, INFERENCE_KEYS)
  if (inferenceDiffs.length > 0 && !o.allow_inference_mismatch) {
    return {
      ok: false,
      error: {
        kind: 'inference_mismatch',
        message: `推論条件が一致しない（before / after）: ${inferenceDiffs.join(', ')}。比較するなら allow_inference_mismatch を指定する`,
      },
    }
  }
  const beforeIdx = new Set(before.frames.map((f) => f.frame_index))
  const afterIdx = new Set(after.frames.map((f) => f.frame_index))
  const onlyBefore = sortedIndexes(before.frames).filter((i) => !afterIdx.has(i))
  const onlyAfter = sortedIndexes(after.frames).filter((i) => !beforeIdx.has(i))
  if (onlyBefore.length + onlyAfter.length > 0) {
    return {
      ok: false,
      error: {
        kind: 'frame_set_mismatch',
        message: `フレームの集合が一致しない。before にのみ存在: ${preview(onlyBefore)} / after にのみ存在: ${preview(onlyAfter)}`,
        details: { only_in_before: onlyBefore, only_in_after: onlyAfter },
      },
    }
  }
  const pathWarning =
    before.source.path === after.source.path
      ? []
      : [`source.path が違う（${before.source.path} / ${after.source.path}）。sha256 は一致している`]
  const inferenceWarnings = inferenceDiffs.map((d) => `推論条件が違う: ${d}`)
  return { ok: true, warnings: [...pathWarning, ...inferenceWarnings, ...classMapWarnings(before, after)] }
}

const count = (frames: FrameDiff[], pick: (f: FrameDiff) => unknown[]): number =>
  frames.reduce((sum, f) => sum + pick(f).length, 0)

export const compareResults = (
  before: InferenceResult,
  after: InferenceResult,
  options: CompareOptionsInput = {},
): CompareOutcome => {
  const o = resolveOptions(options)
  const checked = checkComparable(before, after, o)
  if (!checked.ok) return checked

  const afterByIndex = new Map(after.frames.map((f) => [f.frame_index, f]))
  const frames = [...before.frames]
    .sort((a, b) => a.frame_index - b.frame_index)
    .map((bf) => diffFrame(bf, afterByIndex.get(bf.frame_index) ?? { ...bf, detections: [] }, o))

  return {
    ok: true,
    diff: {
      diff_schema_version: DIFF_SCHEMA_VERSION,
      options: o,
      inputs: {
        source: before.source,
        before: { model_id: before.inference.model_id, inference: before.inference },
        after: { model_id: after.inference.model_id, inference: after.inference },
      },
      warnings: checked.warnings,
      summary: {
        frames_total: frames.length,
        frames_with_changes: frames.filter((f) => f.has_changes).length,
        unchanged: count(frames, (f) => f.unchanged),
        modified: count(frames, (f) => f.modified),
        appeared: count(frames, (f) => f.appeared),
        disappeared: count(frames, (f) => f.disappeared),
        class_change_candidates: count(frames, (f) => f.class_change_candidates),
      },
      frames,
    },
  }
}

const withSide = (side: 'before' | 'after', error: ParseError): CompareError => ({
  ...error,
  message: `${side}: ${error.message}`,
})

export const compareJson = (
  rawBefore: unknown,
  rawAfter: unknown,
  options: CompareOptionsInput = {},
): CompareOutcome => {
  const before = parseInferenceResult(rawBefore)
  if (!before.ok) return { ok: false, error: withSide('before', before.error) }
  const after = parseInferenceResult(rawAfter)
  if (!after.ok) return { ok: false, error: withSide('after', after.error) }
  return compareResults(before.data, after.data, options)
}
