import { describe, expect, it } from 'vitest'
import { compareJson, compareResults, DEFAULT_COMPARE_OPTIONS, type Diff } from '../src/compare.js'
import { parseInferenceResult, type InferenceResult } from '../src/schema.js'
import { det, frame, makeResult, SHA_B } from './helpers.js'

const parse = (raw: unknown): InferenceResult => {
  const result = parseInferenceResult(raw)
  if (!result.ok) throw new Error(result.error.message)
  return result.data
}

const diffOf = (before: unknown, after: unknown, options = {}): Diff => {
  const outcome = compareResults(parse(before), parse(after), options)
  if (!outcome.ok) throw new Error(outcome.error.message)
  return outcome.diff
}

const single = (beforeDets: ReturnType<typeof det>[], afterDets: ReturnType<typeof det>[]) =>
  diffOf(makeResult({ frames: [frame(0, beforeDets)] }), makeResult({ frames: [frame(0, afterDets)] }))

describe('compareResults: 差分の分類', () => {
  it('同じ JSON 同士なら差分はゼロ', () => {
    const diff = diffOf(makeResult(), makeResult())
    expect(diff.summary).toMatchObject({ frames_total: 3, frames_with_changes: 0, appeared: 0, disappeared: 0, modified: 0 })
    expect(diff.summary.unchanged).toBe(2)
    expect(diff.frames.every((f) => !f.has_changes)).toBe(true)
  })

  it('検出とフレームの順序を入れ替えても結果が同じ', () => {
    const before = makeResult({
      frames: [frame(0, [det([0, 0, 10, 10], 0.9), det([100, 100, 150, 150], 0.6)]), frame(1, [det([5, 5, 30, 30])])],
    })
    const after = makeResult({
      frames: [frame(0, [det([0, 0, 10, 10], 0.5), det([300, 300, 350, 350], 0.7)]), frame(1, [])],
    })
    const shuffledBefore = { ...before, frames: [...before.frames].reverse().map((f) => ({ ...f, detections: [...f.detections].reverse() })) }
    const shuffledAfter = { ...after, frames: [...after.frames].reverse().map((f) => ({ ...f, detections: [...f.detections].reverse() })) }
    expect(diffOf(shuffledBefore, shuffledAfter)).toEqual(diffOf(before, after))
  })

  it('confidence は既定では変化に数えないが、差は記録する', () => {
    const diff = single([det([0, 0, 100, 100], 0.9)], [det([0, 0, 100, 100], 0.6)])
    expect(diff.options.confidence_tolerance).toBeNull()
    expect(diff.frames[0]?.modified).toEqual([])
    expect(diff.frames[0]?.unchanged[0]?.metrics.confidence_delta).toBeCloseTo(-0.3, 12)
  })

  it('confidence_tolerance を指定すれば、それを超えた confidence の変化を modified にする', () => {
    const diff = diffOf(
      makeResult({ frames: [frame(0, [det([0, 0, 100, 100], 0.9)])] }),
      makeResult({ frames: [frame(0, [det([0, 0, 100, 100], 0.6)])] }),
      { confidence_tolerance: 0.05 },
    )
    const entry = diff.frames[0]?.modified[0]
    expect(entry?.changed).toEqual(['confidence'])
    expect(entry?.metrics.confidence_delta).toBeCloseTo(-0.3, 12)
    expect(diff.summary.modified).toBe(1)
  })

  it('confidence の変化が許容幅以内なら unchanged', () => {
    const diff = diffOf(
      makeResult({ frames: [frame(0, [det([0, 0, 100, 100], 0.9)])] }),
      makeResult({ frames: [frame(0, [det([0, 0, 100, 100], 0.86)])] }),
      { confidence_tolerance: 0.05 },
    )
    expect(diff.frames[0]?.unchanged).toHaveLength(1)
    expect(diff.frames[0]?.has_changes).toBe(false)
  })

  it('BBox だけが移動したら modified（iou・center）で、大きさは変わらない', () => {
    const diff = single([det([0, 0, 100, 100])], [det([20, 0, 120, 100])])
    const entry = diff.frames[0]?.modified[0]
    expect(entry?.changed).toEqual(['iou', 'center'])
    expect(entry?.metrics.center_shift_px).toBeCloseTo(20, 12)
    expect(entry?.metrics.center_shift_ratio).toBeCloseTo(20 / Math.hypot(100, 100), 12)
    expect(entry?.metrics.width_delta_px).toBe(0)
  })

  it('BBox の大きさだけが変わったら size を含む', () => {
    const diff = single([det([0, 0, 100, 100])], [det([0, 0, 100, 120])])
    expect(diff.frames[0]?.modified[0]?.changed).toContain('size')
    expect(diff.frames[0]?.modified[0]?.metrics.height_delta_px).toBe(20)
  })

  it('クラスが変化したら disappeared と appeared にし、クラス変化の候補として記録する', () => {
    const before = det([0, 0, 100, 100], 0.9, 'bottle', 39)
    const after = det([0, 0, 100, 100], 0.9, 'person', 0)
    const diff = single([before], [after])
    expect(diff.frames[0]?.disappeared).toEqual([before])
    expect(diff.frames[0]?.appeared).toEqual([after])
    expect(diff.frames[0]?.class_change_candidates).toEqual([{ before, after, iou: 1 }])
    expect(diff.summary).toMatchObject({ appeared: 1, disappeared: 1, modified: 0, class_change_candidates: 1 })
  })

  it('検出が出現・消失したフレームを数える', () => {
    const diff = diffOf(
      makeResult({ frames: [frame(0, [det([0, 0, 10, 10])]), frame(1, []), frame(2, [])] }),
      makeResult({ frames: [frame(0, []), frame(1, [det([0, 0, 10, 10])]), frame(2, [])] }),
    )
    expect(diff.summary).toMatchObject({ frames_with_changes: 2, appeared: 1, disappeared: 1 })
    expect(diff.frames.map((f) => f.has_changes)).toEqual([true, true, false])
  })

  it('両方とも検出が空なら差分なし', () => {
    const diff = single([], [])
    expect(diff.frames[0]).toMatchObject({ has_changes: false, unchanged: [], appeared: [], disappeared: [] })
  })

  it('中心の移動は BBox の対角線に対する割合で判定する（小さい物体ほど少しの移動で変化になる）', () => {
    const small = single([det([0, 0, 20, 20])], [det([3, 0, 23, 20])])
    const large = single([det([0, 0, 400, 400])], [det([3, 0, 403, 400])])
    expect(small.frames[0]?.modified[0]?.changed).toContain('center')
    expect(large.frames[0]?.unchanged).toHaveLength(1)
  })

  it('許容幅を変えると判定が変わり、使った設定を出力に残す', () => {
    const before = makeResult({ frames: [frame(0, [det([0, 0, 100, 100])])] })
    const after = makeResult({ frames: [frame(0, [det([20, 0, 120, 100])])] })
    const diff = diffOf(before, after, { iou_tolerance: 0.5, center_shift_tolerance_ratio: 0.5 })
    expect(diff.summary.modified).toBe(0)
    expect(diff.options).toEqual({ ...DEFAULT_COMPARE_OPTIONS, iou_tolerance: 0.5, center_shift_tolerance_ratio: 0.5 })
  })

  it('undefined の設定項目は既定値を上書きしない', () => {
    const diff = diffOf(makeResult(), makeResult(), { confidence_tolerance: undefined })
    expect(diff.options).toEqual(DEFAULT_COMPARE_OPTIONS)
  })

  it('入力の識別情報を出力に残す', () => {
    const diff = diffOf(makeResult(), makeResult({ inference: { model_id: 'model-v2' } }))
    expect(diff.inputs.before.model_id).toBe('model-v1')
    expect(diff.inputs.after.model_id).toBe('model-v2')
    expect(diff.inputs.source.sha256).toBe('a'.repeat(64))
  })
})

describe('compareResults: 比較の前提が崩れている場合', () => {
  it('動画のハッシュが違えばエラー', () => {
    const outcome = compareResults(parse(makeResult()), parse(makeResult({ source: { sha256: SHA_B } })))
    expect(outcome.ok).toBe(false)
    if (!outcome.ok) expect(outcome.error.kind).toBe('source_mismatch')
  })

  it('フレームの集合が違えばエラーにし、欠けているフレームを示す', () => {
    const outcome = compareResults(
      parse(makeResult({ frames: [frame(0), frame(1), frame(2)] })),
      parse(makeResult({ frames: [frame(0), frame(2)] })),
    )
    expect(outcome).toEqual({
      ok: false,
      error: {
        kind: 'frame_set_mismatch',
        message: expect.stringContaining('1'),
        details: { only_in_before: [1], only_in_after: [] },
      },
    })
  })

  it('推論条件（confidence 閾値など）が違えば既定ではエラー', () => {
    const outcome = compareResults(parse(makeResult()), parse(makeResult({ inference: { confidence_threshold: 0.1 } })))
    expect(outcome.ok).toBe(false)
    if (!outcome.ok) expect(outcome.error.kind).toBe('inference_mismatch')
  })

  it('推論条件の違いを許可すれば警告付きで比較する', () => {
    const diff = diffOf(makeResult(), makeResult({ inference: { image_size: 1280 } }), { allow_inference_mismatch: true })
    expect(diff.warnings.some((w) => w.includes('image_size'))).toBe(true)
  })

  it('同じ class_id に別の class_name が割り当てられていれば警告する', () => {
    const frames = [frame(0, [det([0, 0, 10, 10])])]
    const diff = diffOf(
      makeResult({ frames }),
      makeResult({ frames, inference: { classes: { '0': 'human', '39': 'bottle' } } }),
    )
    expect(diff.warnings.some((w) => w.includes('class_id 0'))).toBe(true)
  })

  it('ファイル名だけが違う場合は警告にとどめる', () => {
    const diff = diffOf(makeResult(), makeResult({ source: { path: 'renamed.mp4' } }))
    expect(diff.warnings.some((w) => w.includes('source.path'))).toBe(true)
  })
})

describe('compareJson: 読み込みから比較まで', () => {
  it('スキーマのバージョンが違えば、どちらの入力かを示してエラー', () => {
    const outcome = compareJson(makeResult(), makeResult({ schema_version: '2.0' }))
    expect(outcome.ok).toBe(false)
    if (!outcome.ok) {
      expect(outcome.error.kind).toBe('schema_version_mismatch')
      expect(outcome.error.message).toContain('after')
    }
  })

  it('正しい入力なら比較結果を返す', () => {
    expect(compareJson(makeResult(), makeResult()).ok).toBe(true)
  })
})

describe('diff.json の契約（Python の extract-frames が読む項目）', () => {
  it('diff_schema_version・inputs.source.sha256・frames[].frame_index・frames[].has_changes を持つ', () => {
    const diff = diffOf(makeResult(), makeResult())
    expect(diff.diff_schema_version).toBe('1.0')
    expect(diff.inputs.source.sha256).toMatch(/^[0-9a-f]{64}$/)
    diff.frames.forEach((f) => {
      expect(typeof f.frame_index).toBe('number')
      expect(typeof f.has_changes).toBe('boolean')
    })
  })
})
