import { describe, expect, it } from 'vitest'
import { evaluateJson, evaluateResults, matchToGroundTruth, type Evaluation } from '../src/evaluation.js'
import { parseAnnotations, parseInferenceResult, type Annotations, type InferenceResult } from '../src/schema.js'
import { det, frame, makeAnnotations, makeResult, SHA_B } from './helpers.js'

const parseResult = (raw: unknown): InferenceResult => {
  const r = parseInferenceResult(raw)
  if (!r.ok) throw new Error(r.error.message)
  return r.data
}

const parseAnn = (raw: unknown): Annotations => {
  const r = parseAnnotations(raw)
  if (!r.ok) throw new Error(r.error.message)
  return r.data
}

const BOTTLE = [120, 80, 200, 340] as [number, number, number, number]
const HIT = det(BOTTLE, 0.9)
const MISS = det([600, 300, 640, 420], 0.9) // 正解と重ならない位置

// frame 0 に bottle が1本写っている、という正解
const oneBottle = makeAnnotations({ frames: [{ frame_index: 0, objects: [{ class_name: 'bottle', bbox: BOTTLE }] }] })

const evaluate = (beforeDets: ReturnType<typeof det>[], afterDets: ReturnType<typeof det>[], annotations = oneBottle): Evaluation => {
  const outcome = evaluateResults(
    parseResult(makeResult({ frames: [frame(0, beforeDets), frame(1), frame(2)] })),
    parseResult(makeResult({ frames: [frame(0, afterDets), frame(1), frame(2)] })),
    parseAnn(annotations),
  )
  if (!outcome.ok) throw new Error(outcome.error.message)
  return outcome.evaluation
}

const transitions = (e: Evaluation) => e.frames.flatMap((f) => f.objects.map((o) => o.transition))

describe('matchToGroundTruth', () => {
  const gt = [{ class_name: 'bottle', bbox: BOTTLE }]

  it('正解と重ならない検出は TP にならない', () => {
    const r = matchToGroundTruth([MISS], gt, { iouThreshold: 0.5 })
    expect(r.matches).toEqual([null])
    expect(r.falsePositives).toEqual([MISS])
  })

  it('クラスが違えば同じ位置でも TP にならない', () => {
    const person = det(BOTTLE, 0.9, 'person', 0)
    expect(matchToGroundTruth([person], gt, { iouThreshold: 0.5 }).matches).toEqual([null])
  })

  it('1つの正解に重なる検出が複数あっても TP は1つだけで、残りは FP', () => {
    const low = det(BOTTLE, 0.4)
    const r = matchToGroundTruth([low, HIT], gt, { iouThreshold: 0.5 })
    expect(r.matches).toEqual([HIT])
    expect(r.falsePositives).toEqual([low])
  })

  it('confidence の高い検出から順に正解へ割り当てる', () => {
    const near = det([121, 80, 201, 340], 0.3) // IoU は HIT より低いが confidence も低い
    const r = matchToGroundTruth([near, det([125, 80, 205, 340], 0.95)], gt, { iouThreshold: 0.5 })
    expect(r.matches[0]?.confidence).toBe(0.95)
  })

  it('入力の順序を入れ替えても結果が同じ', () => {
    const dets = [det(BOTTLE, 0.5), det([122, 80, 202, 340], 0.5), MISS]
    const gts = [...gt, { class_name: 'bottle', bbox: [590, 300, 640, 420] as [number, number, number, number] }]
    const a = matchToGroundTruth(dets, gts, { iouThreshold: 0.5 })
    const b = matchToGroundTruth([...dets].reverse(), [...gts].reverse(), { iouThreshold: 0.5 })
    expect(b.falsePositives).toEqual(a.falsePositives)
    expect(new Set(b.matches)).toEqual(new Set(a.matches))
  })
})

describe('evaluateResults: before/after の遷移', () => {
  it('PASS → PASS は maintained', () => {
    expect(transitions(evaluate([HIT], [HIT]))).toEqual(['maintained'])
  })

  it('PASS → FAIL は regressed', () => {
    const e = evaluate([HIT], [MISS])
    expect(transitions(e)).toEqual(['regressed'])
    expect(e.summary.regressed).toBe(1)
  })

  it('FAIL → PASS は improved', () => {
    expect(transitions(evaluate([], [HIT]))).toEqual(['improved'])
  })

  it('FAIL → FAIL は unresolved', () => {
    expect(transitions(evaluate([MISS], []))).toEqual(['unresolved'])
  })

  it('正解が複数あるフレームは物体ごとに判定する', () => {
    const second = [600, 300, 640, 420] as [number, number, number, number]
    const ann = makeAnnotations({
      frames: [{ frame_index: 0, objects: [{ class_name: 'bottle', bbox: BOTTLE }, { class_name: 'bottle', bbox: second }] }],
    })
    const e = evaluate([HIT, det(second)], [HIT], ann)
    expect(e.frames[0]?.objects.map((o) => [o.ground_truth.bbox, o.transition])).toEqual([
      [BOTTLE, 'maintained'],
      [second, 'regressed'],
    ])
    expect(e.summary).toMatchObject({ objects: 2, maintained: 1, regressed: 1 })
  })

  it('FP は遷移の表に入れず、モデルごとに数える', () => {
    const e = evaluate([HIT], [HIT, MISS])
    expect(transitions(e)).toEqual(['maintained'])
    expect(e.frames[0]?.false_positives.after).toEqual([MISS])
    expect(e.summary.after).toMatchObject({ tp: 1, fp: 1, fn: 0 })
    expect(e.summary.before).toMatchObject({ tp: 1, fp: 0, fn: 0 })
  })

  it('物体が0個と注記されたフレームの検出はすべて FP', () => {
    const ann = makeAnnotations({ frames: [{ frame_index: 0, objects: [] }] })
    const e = evaluate([], [MISS], ann)
    expect(e.frames[0]?.objects).toEqual([])
    expect(e.summary.after).toMatchObject({ tp: 0, fp: 1, fn: 0 })
  })

  it('未アノテーションのフレームは評価に含めず、PASS としても数えない', () => {
    const e = evaluate([HIT], [HIT])
    expect(e.frames.map((f) => f.frame_index)).toEqual([0])
    expect(e.summary).toMatchObject({ annotated_frames: 1, unannotated_frames: 2, objects: 1 })
  })

  it('precision と recall を計算し、分母が0なら null にする', () => {
    const e = evaluate([HIT, MISS], [])
    expect(e.summary.before).toMatchObject({ precision: 0.5, recall: 1 })
    expect(e.summary.after).toMatchObject({ precision: null, recall: 0 })
  })
})

describe('evaluateResults: 前提が崩れている場合', () => {
  it('正解アノテーションの動画が推論結果と違えばエラー', () => {
    const outcome = evaluateResults(
      parseResult(makeResult()),
      parseResult(makeResult()),
      parseAnn(makeAnnotations({ source: { sha256: SHA_B } })),
    )
    expect(outcome.ok).toBe(false)
    if (!outcome.ok) expect(outcome.error.kind).toBe('source_mismatch')
  })

  it('注記したフレームが推論結果に無ければエラー', () => {
    const outcome = evaluateResults(
      parseResult(makeResult({ frames: [frame(0), frame(1)] })),
      parseResult(makeResult({ frames: [frame(0), frame(1)] })),
      parseAnn(makeAnnotations({ frames: [{ frame_index: 2, objects: [] }] })),
    )
    expect(outcome.ok).toBe(false)
    if (!outcome.ok) expect(outcome.error.kind).toBe('frame_set_mismatch')
  })

  it('evaluateJson は読み込みエラーにどの入力かを付ける', () => {
    const outcome = evaluateJson(makeResult(), makeResult(), makeAnnotations({ schema_version: '9' }))
    expect(outcome.ok).toBe(false)
    if (!outcome.ok) expect(outcome.error.message).toContain('annotations')
  })
})
