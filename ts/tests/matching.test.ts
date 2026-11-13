import { describe, expect, it } from 'vitest'
import { iou, matchDetections } from '../src/matching.js'
import { det } from './helpers.js'

describe('iou', () => {
  it('同じ BBox は 1', () => {
    expect(iou([0, 0, 10, 10], [0, 0, 10, 10])).toBe(1)
  })

  it('重ならない BBox は 0', () => {
    expect(iou([0, 0, 10, 10], [20, 20, 30, 30])).toBe(0)
  })

  it('辺だけ接する BBox は 0', () => {
    expect(iou([0, 0, 10, 10], [10, 0, 20, 10])).toBe(0)
  })

  it('半分ずれた BBox は 1/3', () => {
    expect(iou([0, 0, 10, 10], [5, 0, 15, 10])).toBeCloseTo(1 / 3, 12)
  })

  it('面積0同士（和集合が0）は NaN でなく 0', () => {
    expect(iou([5, 5, 5, 5], [5, 5, 5, 5])).toBe(0)
  })
})

describe('matchDetections', () => {
  const opts = { iouThreshold: 0.5 }

  it('完全一致した BBox を対応づける', () => {
    const a = det([0, 0, 10, 10])
    const result = matchDetections([a], [a], opts)
    expect(result.pairs).toEqual([{ before: a, after: a, iou: 1 }])
    expect(result.unmatchedBefore).toEqual([])
    expect(result.unmatchedAfter).toEqual([])
  })

  it('少し位置がずれた BBox を対応づける', () => {
    const b = det([0, 0, 100, 100])
    const a = det([5, 0, 105, 100])
    const result = matchDetections([b], [a], opts)
    expect(result.pairs).toHaveLength(1)
    expect(result.pairs[0]?.iou).toBeCloseTo(95 / 105, 12)
  })

  it('IoU が閾値未満なら対応づけない', () => {
    const b = det([0, 0, 10, 10])
    const a = det([5, 0, 15, 10]) // IoU = 1/3
    const result = matchDetections([b], [a], opts)
    expect(result.pairs).toEqual([])
    expect(result.unmatchedBefore).toEqual([b])
    expect(result.unmatchedAfter).toEqual([a])
  })

  it('IoU が閾値ちょうどなら対応づける', () => {
    const b = det([0, 0, 10, 10])
    const a = det([5, 0, 15, 10])
    expect(matchDetections([b], [a], { iouThreshold: 1 / 3 }).pairs).toHaveLength(1)
  })

  it('クラスが違えば同じ位置でも対応づけない', () => {
    const b = det([0, 0, 10, 10], 0.9, 'bottle', 39)
    const a = det([0, 0, 10, 10], 0.9, 'cup', 41)
    const result = matchDetections([b], [a], opts)
    expect(result.pairs).toEqual([])
    expect(result.unmatchedBefore).toEqual([b])
    expect(result.unmatchedAfter).toEqual([a])
  })

  it('class_id が違っても class_name が同じなら対応づける（モデル間で ID の割り当てが違う場合）', () => {
    const b = det([0, 0, 10, 10], 0.9, 'bottle', 39)
    const a = det([0, 0, 10, 10], 0.9, 'bottle', 1)
    expect(matchDetections([b], [a], opts).pairs).toHaveLength(1)
  })

  it('before にしか無い検出は unmatchedBefore になる', () => {
    const b = det([0, 0, 10, 10])
    expect(matchDetections([b], [], opts)).toEqual({ pairs: [], unmatchedBefore: [b], unmatchedAfter: [] })
  })

  it('after にしか無い検出は unmatchedAfter になる', () => {
    const a = det([0, 0, 10, 10])
    expect(matchDetections([], [a], opts)).toEqual({ pairs: [], unmatchedBefore: [], unmatchedAfter: [a] })
  })

  it('同一クラスが複数あるとき、それぞれ近いもの同士を一対一で対応づける', () => {
    const b1 = det([0, 0, 10, 10])
    const b2 = det([100, 100, 110, 110])
    const a1 = det([1, 0, 11, 10])
    const a2 = det([100, 101, 110, 111])
    const result = matchDetections([b1, b2], [a2, a1], opts)
    expect(result.pairs.map((p) => [p.before, p.after])).toEqual([
      [b1, a1],
      [b2, a2],
    ])
  })

  it('同じ BBox に重複した検出があっても一対一を保つ', () => {
    const b = det([0, 0, 10, 10], 0.9)
    const dup = det([0, 0, 10, 10], 0.8)
    const a = det([0, 0, 10, 10], 0.9)
    const result = matchDetections([b, dup], [a], opts)
    expect(result.pairs).toHaveLength(1)
    expect(result.unmatchedBefore).toHaveLength(1)
    expect(result.unmatchedAfter).toEqual([])
  })

  it('IoU が同点のときは confidence が近い検出を対応づける', () => {
    const b = det([0, 0, 10, 10], 0.9)
    const aHigh = det([0, 0, 10, 10], 0.88)
    const aLow = det([0, 0, 10, 10], 0.3)
    const result = matchDetections([b], [aLow, aHigh], opts)
    expect(result.pairs[0]?.after).toEqual(aHigh)
    expect(result.unmatchedAfter).toEqual([aLow])
  })

  it('入力の順序を入れ替えても結果が変わらない', () => {
    const before = [det([0, 0, 10, 10], 0.9), det([2, 0, 12, 10], 0.7), det([50, 50, 60, 60], 0.6, 'person', 0)]
    const after = [det([1, 0, 11, 10], 0.85), det([3, 0, 13, 10], 0.75), det([200, 200, 210, 210], 0.5)]
    const base = matchDetections(before, after, opts)
    const shuffled = matchDetections([...before].reverse(), [after[1]!, after[2]!, after[0]!], opts)
    expect(shuffled).toEqual(base)
  })
})
