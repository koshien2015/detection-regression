import { compareDetections, compareLabeled, iou } from './matching.js'
import {
  parseAnnotations,
  parseInferenceResult,
  type AnnotatedObject,
  type Annotations,
  type Detection,
  type InferenceResult,
  type ParseError,
} from './schema.js'

export const EVALUATION_SCHEMA_VERSION = '1.0'

export type EvaluateOptions = { iou_threshold: number }
export const DEFAULT_EVALUATE_OPTIONS: EvaluateOptions = { iou_threshold: 0.5 }

export type GroundTruthMatch = {
  groundTruth: AnnotatedObject[] // 内容の並び順にそろえた正解
  matches: (Detection | null)[] // groundTruth と同じ添字。null は見逃し（FN）
  falsePositives: Detection[]
}

// COCO の評価と同じく、confidence の高い検出から順に、まだ割り当てていない同じクラスの正解のうち
// IoU が最大のものへ割り当てる。1つの正解には TP を1つしか割り当てない。
export const matchToGroundTruth = (
  detections: readonly Detection[],
  groundTruth: readonly AnnotatedObject[],
  options: { iouThreshold: number },
): GroundTruthMatch => {
  const gts = [...groundTruth].sort(compareLabeled)
  const dets = [...detections].sort((a, b) => b.confidence - a.confidence || compareDetections(a, b))
  const matches: (Detection | null)[] = gts.map(() => null)
  const falsePositives = dets.filter((d) => {
    const best = gts.reduce<{ index: number; iou: number } | null>((acc, gt, index) => {
      if (matches[index] !== null || gt.class_name !== d.class_name) return acc
      const v = iou(d.bbox, gt.bbox)
      if (v <= 0 || v < options.iouThreshold) return acc
      return acc === null || v > acc.iou ? { index, iou: v } : acc
    }, null)
    if (best === null) return true
    matches[best.index] = d
    return false
  })
  return { groundTruth: gts, matches, falsePositives: [...falsePositives].sort(compareDetections) }
}

export type Status = 'PASS' | 'FAIL'
export type Transition = 'maintained' | 'improved' | 'regressed' | 'unresolved'

const TRANSITIONS: Record<`${Status}->${Status}`, Transition> = {
  'PASS->PASS': 'maintained',
  'FAIL->PASS': 'improved',
  'PASS->FAIL': 'regressed',
  'FAIL->FAIL': 'unresolved',
}

type ModelVerdict = { status: Status; detection: Detection | null }

export type ObjectEvaluation = {
  ground_truth: AnnotatedObject
  before: ModelVerdict
  after: ModelVerdict
  transition: Transition
}

export type FrameEvaluation = {
  frame_index: number
  objects: ObjectEvaluation[]
  false_positives: { before: Detection[]; after: Detection[] }
}

export type ModelScore = { tp: number; fp: number; fn: number; precision: number | null; recall: number | null }

export type Evaluation = {
  evaluation_schema_version: string
  options: EvaluateOptions
  inputs: { source: InferenceResult['source']; before: { model_id: string }; after: { model_id: string } }
  summary: {
    annotated_frames: number
    unannotated_frames: number
    objects: number
  } & Record<Transition, number> & { before: ModelScore; after: ModelScore }
  frames: FrameEvaluation[]
}

export type EvaluateError =
  | ParseError
  | { kind: 'source_mismatch'; message: string }
  | { kind: 'frame_set_mismatch'; message: string; details: { missing_in_results: number[] } }

export type EvaluateOutcome = { ok: true; evaluation: Evaluation } | { ok: false; error: EvaluateError }

const verdict = (detection: Detection | null): ModelVerdict => ({
  status: detection === null ? 'FAIL' : 'PASS',
  detection,
})

const evaluateFrame = (
  frameIndex: number,
  objects: readonly AnnotatedObject[],
  beforeDets: readonly Detection[],
  afterDets: readonly Detection[],
  o: EvaluateOptions,
): FrameEvaluation => {
  const b = matchToGroundTruth(beforeDets, objects, { iouThreshold: o.iou_threshold })
  const a = matchToGroundTruth(afterDets, objects, { iouThreshold: o.iou_threshold })
  return {
    frame_index: frameIndex,
    objects: b.groundTruth.map((gt, i) => {
      const before = verdict(b.matches[i] ?? null)
      const after = verdict(a.matches[i] ?? null)
      return { ground_truth: gt, before, after, transition: TRANSITIONS[`${before.status}->${after.status}`] }
    }),
    false_positives: { before: b.falsePositives, after: a.falsePositives },
  }
}

const score = (frames: FrameEvaluation[], side: 'before' | 'after'): ModelScore => {
  const objects = frames.flatMap((f) => f.objects)
  const tp = objects.filter((x) => x[side].status === 'PASS').length
  const fn = objects.length - tp
  const fp = frames.reduce((sum, f) => sum + f.false_positives[side].length, 0)
  return {
    tp,
    fp,
    fn,
    precision: tp + fp === 0 ? null : tp / (tp + fp),
    recall: tp + fn === 0 ? null : tp / (tp + fn),
  }
}

const checkSources = (before: InferenceResult, after: InferenceResult, ann: Annotations): EvaluateError | null => {
  const differs = (s: InferenceResult['source']) =>
    s.sha256 !== ann.source.sha256 || s.width !== ann.source.width || s.height !== ann.source.height
  const mismatched = [
    ...(differs(before.source) ? ['before'] : []),
    ...(differs(after.source) ? ['after'] : []),
  ]
  if (mismatched.length === 0) return null
  return {
    kind: 'source_mismatch',
    message: `正解アノテーションと元動画（sha256・幅・高さ）が一致しない: ${mismatched.join(', ')}`,
  }
}

export const evaluateResults = (
  before: InferenceResult,
  after: InferenceResult,
  annotations: Annotations,
  options: Partial<EvaluateOptions> = {},
): EvaluateOutcome => {
  const o: EvaluateOptions = { ...DEFAULT_EVALUATE_OPTIONS, ...options }
  const sourceError = checkSources(before, after, annotations)
  if (sourceError !== null) return { ok: false, error: sourceError }

  const beforeByIndex = new Map(before.frames.map((f) => [f.frame_index, f.detections]))
  const afterByIndex = new Map(after.frames.map((f) => [f.frame_index, f.detections]))
  const annotated = [...annotations.frames].sort((x, y) => x.frame_index - y.frame_index)
  const missing = annotated
    .map((f) => f.frame_index)
    .filter((i) => !beforeByIndex.has(i) || !afterByIndex.has(i))
  if (missing.length > 0) {
    return {
      ok: false,
      error: {
        kind: 'frame_set_mismatch',
        message: `正解を付けたフレームが推論結果に無い: ${missing.join(', ')}`,
        details: { missing_in_results: missing },
      },
    }
  }

  const frames = annotated.map((f) =>
    evaluateFrame(f.frame_index, f.objects, beforeByIndex.get(f.frame_index) ?? [], afterByIndex.get(f.frame_index) ?? [], o),
  )
  const objects = frames.flatMap((f) => f.objects)
  const countOf = (t: Transition) => objects.filter((x) => x.transition === t).length

  return {
    ok: true,
    evaluation: {
      evaluation_schema_version: EVALUATION_SCHEMA_VERSION,
      options: o,
      inputs: {
        source: before.source,
        before: { model_id: before.inference.model_id },
        after: { model_id: after.inference.model_id },
      },
      summary: {
        annotated_frames: frames.length,
        unannotated_frames: before.frames.length - frames.length,
        objects: objects.length,
        maintained: countOf('maintained'),
        improved: countOf('improved'),
        regressed: countOf('regressed'),
        unresolved: countOf('unresolved'),
        before: score(frames, 'before'),
        after: score(frames, 'after'),
      },
      frames,
    },
  }
}

const withSide = (side: string, error: ParseError): EvaluateError => ({ ...error, message: `${side}: ${error.message}` })

export const evaluateJson = (
  rawBefore: unknown,
  rawAfter: unknown,
  rawAnnotations: unknown,
  options: Partial<EvaluateOptions> = {},
): EvaluateOutcome => {
  const before = parseInferenceResult(rawBefore)
  if (!before.ok) return { ok: false, error: withSide('before', before.error) }
  const after = parseInferenceResult(rawAfter)
  if (!after.ok) return { ok: false, error: withSide('after', after.error) }
  const annotations = parseAnnotations(rawAnnotations)
  if (!annotations.ok) return { ok: false, error: withSide('annotations', annotations.error) }
  return evaluateResults(before.data, after.data, annotations.data, options)
}
