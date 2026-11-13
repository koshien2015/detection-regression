import type { BBox, Detection } from './schema.js'

type Box = readonly [number, number, number, number]

const area = ([x1, y1, x2, y2]: Box): number => Math.max(0, x2 - x1) * Math.max(0, y2 - y1)

// 和集合が0（面積0の BBox 同士）のときは 0 とする。NaN を比較に持ち込まないため。
export const iou = (a: Box, b: Box): number => {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]))
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]))
  const inter = ix * iy
  const union = area(a) + area(b) - inter
  return union > 0 ? inter / union : 0
}

type Labeled = { class_name: string; bbox: BBox | Box }

export type Pair<A, B> = { before: A; after: B; iou: number }

export type MatchResult<A, B> = {
  pairs: Pair<A, B>[]
  unmatchedBefore: A[]
  unmatchedAfter: B[]
}

export type Comparator<T> = (x: T, y: T) => number

const compareNumbers = (x: readonly number[], y: readonly number[]): number => {
  for (let i = 0; i < Math.min(x.length, y.length); i += 1) {
    const d = (x[i] ?? 0) - (y[i] ?? 0)
    if (d !== 0) return d
  }
  return x.length - y.length
}

const compareStrings = (x: string, y: string): number => (x < y ? -1 : x > y ? 1 : 0)

// 内容だけで決まる並び順。入力配列の順序に結果が左右されないよう、ソートと同点の解消に使う。
export const compareLabeled: Comparator<Labeled> = (x, y) =>
  compareStrings(x.class_name, y.class_name) || compareNumbers(x.bbox, y.bbox)

export const compareDetections: Comparator<Detection> = (x, y) =>
  compareLabeled(x, y) || y.confidence - x.confidence || x.class_id - y.class_id

type Candidate<A, B> = { a: A; b: B; ai: number; bi: number; iou: number }

// クラスが同じで IoU が閾値以上の組を、IoU の大きい順に一対一で確定させる貪欲法。
// tieBreak は IoU が同点の組の優先順位を決める。最後は内容の並び順で決めるので、結果は決定的になる。
export const greedyMatch = <A extends Labeled, B extends Labeled>(
  befores: readonly A[],
  afters: readonly B[],
  options: {
    iouThreshold: number
    compareA: Comparator<A>
    compareB: Comparator<B>
    tieBreak?: Comparator<Candidate<A, B>>
    // 'same' はクラスが同じ組だけ、'different' はクラスが違う組だけを候補にする（クラス変化の候補探し用）
    classes?: 'same' | 'different'
  },
): MatchResult<A, B> => {
  const wantSame = (options.classes ?? 'same') === 'same'
  const sortedA = [...befores].sort(options.compareA)
  const sortedB = [...afters].sort(options.compareB)
  const candidates = sortedA
    .flatMap((a, ai) =>
      sortedB.map((b, bi) => ({ a, b, ai, bi, iou: iou(a.bbox, b.bbox) })),
    )
    .filter((c) => (c.a.class_name === c.b.class_name) === wantSame)
    .filter((c) => c.iou >= options.iouThreshold && c.iou > 0)
    .sort((x, y) => y.iou - x.iou || (options.tieBreak?.(x, y) ?? 0) || x.ai - y.ai || x.bi - y.bi)

  const usedA = new Set<number>()
  const usedB = new Set<number>()
  const chosen = candidates.filter((c) => {
    if (usedA.has(c.ai) || usedB.has(c.bi)) return false
    usedA.add(c.ai)
    usedB.add(c.bi)
    return true
  })

  return {
    pairs: chosen.sort((x, y) => x.ai - y.ai).map(({ a, b, iou: v }) => ({ before: a, after: b, iou: v })),
    unmatchedBefore: sortedA.filter((_, i) => !usedA.has(i)),
    unmatchedAfter: sortedB.filter((_, i) => !usedB.has(i)),
  }
}

export const matchDetections = (
  before: readonly Detection[],
  after: readonly Detection[],
  options: { iouThreshold: number },
): MatchResult<Detection, Detection> =>
  greedyMatch(before, after, {
    iouThreshold: options.iouThreshold,
    compareA: compareDetections,
    compareB: compareDetections,
    tieBreak: (x, y) => Math.abs(x.a.confidence - x.b.confidence) - Math.abs(y.a.confidence - y.b.confidence),
  })
