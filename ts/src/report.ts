import { DIFF_SCHEMA_VERSION, type Diff, type FrameDiff } from './compare.js'
import { EVALUATION_SCHEMA_VERSION, type Evaluation, type FrameEvaluation } from './evaluation.js'

export const REPORT_SCHEMA_VERSION = '1.0'

export type ReportFrame = {
  frame_index: number
  timestamp_ms: number
  image: string | null // index.html からの相対パス。画像を書き出していなければ null
  change_count: number // 出現・消失・変化の件数の合計
  regressed_count: number
  diff: FrameDiff
  evaluation: FrameEvaluation | null
}

export type ReportData = {
  report_schema_version: string
  inputs: Diff['inputs']
  options: Diff['options']
  warnings: string[]
  summary: Diff['summary']
  evaluation_options: Evaluation['options'] | null
  evaluation_summary: Evaluation['summary'] | null
  frames: ReportFrame[]
}

export type ReportOutcome = { ok: true; data: ReportData } | { ok: false; error: string }

const checkInputs = (diff: Diff, evaluation: Evaluation | null): string | null => {
  if (diff.diff_schema_version !== DIFF_SCHEMA_VERSION) {
    return `diff.json の diff_schema_version が ${diff.diff_schema_version}。このツールは ${DIFF_SCHEMA_VERSION} のみ扱う`
  }
  if (evaluation === null) return null
  if (evaluation.evaluation_schema_version !== EVALUATION_SCHEMA_VERSION) {
    return `evaluation.json の evaluation_schema_version が ${evaluation.evaluation_schema_version}。このツールは ${EVALUATION_SCHEMA_VERSION} のみ扱う`
  }
  const same =
    evaluation.inputs.source.sha256 === diff.inputs.source.sha256 &&
    evaluation.inputs.before.model_id === diff.inputs.before.model_id &&
    evaluation.inputs.after.model_id === diff.inputs.after.model_id
  return same ? null : 'diff.json と evaluation.json で、元動画（sha256）か before/after のモデルが違う'
}

const changeCount = (f: FrameDiff): number => f.appeared.length + f.disappeared.length + f.modified.length

// 差分のあるフレームと正解を付けたフレームだけを載せる。全フレームを載せると長い動画で data.js が大きくなるため。
export const buildReportData = (
  diff: Diff,
  evaluation: Evaluation | null,
  images: Readonly<Record<number, string>>,
  framesDir = 'frames',
): ReportOutcome => {
  const error = checkInputs(diff, evaluation)
  if (error !== null) return { ok: false, error }

  const evalByIndex = new Map((evaluation?.frames ?? []).map((f) => [f.frame_index, f]))
  const frames = diff.frames
    .filter((f) => f.has_changes || evalByIndex.has(f.frame_index))
    .map((f): ReportFrame => {
      const ev = evalByIndex.get(f.frame_index) ?? null
      const image = images[f.frame_index]
      return {
        frame_index: f.frame_index,
        timestamp_ms: f.timestamp_ms,
        image: image === undefined ? null : `${framesDir}/${image}`,
        change_count: changeCount(f),
        regressed_count: ev?.objects.filter((o) => o.transition === 'regressed').length ?? 0,
        diff: f,
        evaluation: ev,
      }
    })

  return {
    ok: true,
    data: {
      report_schema_version: REPORT_SCHEMA_VERSION,
      inputs: diff.inputs,
      options: diff.options,
      warnings: diff.warnings,
      summary: diff.summary,
      evaluation_options: evaluation?.options ?? null,
      evaluation_summary: evaluation?.summary ?? null,
      frames,
    },
  }
}

// file:// で開いたページは fetch で JSON を読めないため、データはクラシックスクリプトとして渡す
export const renderDataJs = (data: unknown): string => `window.__REPORT__ = ${JSON.stringify(data)};\n`

const FRAME_FILE = /^frame_(\d+)\.png$/

export const imagesFromFileNames = (names: readonly string[]): Record<number, string> =>
  Object.fromEntries(
    names.flatMap((name) => {
      const m = FRAME_FILE.exec(name)
      return m?.[1] === undefined ? [] : [[Number(m[1]), name]]
    }),
  )
