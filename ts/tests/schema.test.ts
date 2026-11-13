import { describe, expect, it } from 'vitest'
import { parseAnnotations, parseInferenceResult } from '../src/schema.js'
import { det, frame, makeAnnotations, makeResult } from './helpers.js'

const expectInvalid = (result: ReturnType<typeof parseInferenceResult>, pathFragment: string) => {
  expect(result.ok).toBe(false)
  if (result.ok) return
  expect(result.error.kind).toBe('invalid')
  expect(result.error.message).toContain(pathFragment)
}

describe('parseInferenceResult', () => {
  it('正しい推論結果を受け付ける', () => {
    const result = parseInferenceResult(makeResult())
    expect(result.ok).toBe(true)
  })

  it('検出が空のフレームを受け付ける（フレームの欠損とは区別する）', () => {
    const result = parseInferenceResult(makeResult({ frames: [frame(0, [])] }))
    expect(result.ok).toBe(true)
    if (result.ok) expect(result.data.frames[0]?.detections).toEqual([])
  })

  it('schema_version が違うときは専用のエラーを返す', () => {
    const result = parseInferenceResult(makeResult({ schema_version: '2.0' }))
    expect(result).toEqual({
      ok: false,
      error: { kind: 'schema_version_mismatch', expected: '1.0', actual: '2.0', message: expect.any(String) },
    })
  })

  it('schema_version が無いときも専用のエラーを返す', () => {
    const { schema_version: _omit, ...rest } = makeResult()
    const result = parseInferenceResult(rest)
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.kind).toBe('schema_version_mismatch')
  })

  it('オブジェクトでない入力を拒否する', () => {
    const result = parseInferenceResult('not json object')
    expect(result.ok).toBe(false)
  })

  it('frame_index の重複を拒否する', () => {
    expectInvalid(parseInferenceResult(makeResult({ frames: [frame(0), frame(0)] })), 'frame_index')
  })

  it('x1 > x2 の BBox を拒否する', () => {
    expectInvalid(parseInferenceResult(makeResult({ frames: [frame(0, [det([200, 80, 120, 340])])] })), 'bbox')
  })

  it('y1 > y2 の BBox を拒否する', () => {
    expectInvalid(parseInferenceResult(makeResult({ frames: [frame(0, [det([120, 340, 200, 80])])] })), 'bbox')
  })

  it('フレームの外にはみ出す BBox を拒否する', () => {
    expectInvalid(parseInferenceResult(makeResult({ frames: [frame(0, [det([1200, 80, 1281, 340])])] })), 'bbox')
  })

  it('負の座標の BBox を拒否する', () => {
    expectInvalid(parseInferenceResult(makeResult({ frames: [frame(0, [det([-1, 80, 200, 340])])] })), 'bbox')
  })

  it('フレームの端に接する BBox は受け付ける', () => {
    const result = parseInferenceResult(makeResult({ frames: [frame(0, [det([0, 0, 1280, 720])])] }))
    expect(result.ok).toBe(true)
  })

  it('confidence が 1 を超える検出を拒否する', () => {
    expectInvalid(parseInferenceResult(makeResult({ frames: [frame(0, [det([0, 0, 10, 10], 1.2)])] })), 'confidence')
  })

  it('classes の対応と食い違う class_name を拒否する', () => {
    expectInvalid(
      parseInferenceResult(makeResult({ frames: [frame(0, [det([0, 0, 10, 10], 0.9, 'cup', 39)])] })),
      'class_name',
    )
  })

  it('classes に無い class_id を拒否する', () => {
    expectInvalid(
      parseInferenceResult(makeResult({ frames: [frame(0, [det([0, 0, 10, 10], 0.9, 'cup', 41)])] })),
      'class_id',
    )
  })

  it.each(['/Users/someone/sample.mp4', '~/sample.mp4', 'C:\\videos\\sample.mp4'])(
    '絶対パスの source.path を拒否する: %s',
    (path) => {
      expectInvalid(parseInferenceResult(makeResult({ source: { path } })), 'source.path')
    },
  )

  it('sha256 の形式でないハッシュを拒否する', () => {
    expectInvalid(parseInferenceResult(makeResult({ source: { sha256: 'abc' } })), 'source.sha256')
  })

  it('frame_index が frame_count 以上のフレームを拒否する', () => {
    expectInvalid(parseInferenceResult(makeResult({ frames: [frame(3)] })), 'frame_index')
  })

  it('未定義のフィールドを拒否する（綴りの誤りを見逃さないため）', () => {
    const raw = { ...makeResult(), extra: true }
    expectInvalid(parseInferenceResult(raw), 'extra')
  })
})

describe('parseAnnotations', () => {
  it('正しい正解アノテーションを受け付ける', () => {
    expect(parseAnnotations(makeAnnotations()).ok).toBe(true)
  })

  it('物体が0個のフレームを「何も無いことが正解」として保持する', () => {
    const result = parseAnnotations(makeAnnotations())
    expect(result.ok).toBe(true)
    if (result.ok) expect(result.data.frames.find((f) => f.frame_index === 1)?.objects).toEqual([])
  })

  it('frame_index の重複を拒否する', () => {
    const result = parseAnnotations(
      makeAnnotations({ frames: [{ frame_index: 0, objects: [] }, { frame_index: 0, objects: [] }] }),
    )
    expect(result.ok).toBe(false)
  })

  it('フレームの外にはみ出す BBox を拒否する', () => {
    const result = parseAnnotations(
      makeAnnotations({ frames: [{ frame_index: 0, objects: [{ class_name: 'bottle', bbox: [0, 0, 10, 721] }] }] }),
    )
    expect(result.ok).toBe(false)
  })

  it('schema_version が違うときは専用のエラーを返す', () => {
    const result = parseAnnotations(makeAnnotations({ schema_version: '0.9' }))
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.kind).toBe('schema_version_mismatch')
  })
})
