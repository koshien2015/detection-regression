import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { compareJson } from '../src/compare.js'
import { evaluateJson } from '../src/evaluation.js'
import { buildReportData, type ReportData } from '../src/report.js'
import { defaultFilter, filterFrames, formatPercent, overlayBoxes, sortFrames } from '../viewer/src/model.js'

const json = (name: string): unknown =>
  JSON.parse(readFileSync(fileURLToPath(new URL(`./fixtures/${name}`, import.meta.url)), 'utf8'))

const report = (): ReportData => {
  const d = compareJson(json('before.json'), json('after.json'))
  const e = evaluateJson(json('before.json'), json('after.json'), json('annotations.json'))
  if (!d.ok || !e.ok) throw new Error('fixture')
  const r = buildReportData(d.diff, e.evaluation, {})
  if (!r.ok) throw new Error(r.error)
  return r.data
}

describe('sortFrames', () => {
  it('impact は回帰の多い順、次に変化の多い順、最後に番号順', () => {
    expect(sortFrames(report().frames, 'impact').map((f) => f.frame_index)).toEqual([1, 2, 3, 0])
  })

  it('index は番号順', () => {
    expect(sortFrames(report().frames, 'index').map((f) => f.frame_index)).toEqual([0, 1, 2, 3])
  })

  it('元の配列を書き換えない', () => {
    const frames = report().frames
    const before = frames.map((f) => f.frame_index)
    sortFrames(frames, 'impact')
    expect(frames.map((f) => f.frame_index)).toEqual(before)
  })
})

describe('filterFrames', () => {
  it('changed は差分のあるフレームだけ、regressed は回帰のあるフレームだけ', () => {
    const frames = report().frames
    expect(filterFrames(frames, 'all')).toHaveLength(4)
    expect(filterFrames(frames, 'changed').map((f) => f.frame_index)).toEqual([1, 2, 3])
    expect(filterFrames(frames, 'regressed').map((f) => f.frame_index)).toEqual([1])
  })
})

describe('overlayBoxes', () => {
  it('変化した検出は before と after の両方の枠を出す', () => {
    const frame2 = report().frames.find((f) => f.frame_index === 2)!
    expect(overlayBoxes(frame2).map((b) => [b.layer, b.status])).toEqual([
      ['before', 'modified'],
      ['after', 'modified'],
    ])
  })

  it('消失は before の枠、出現は after の枠、正解は truth の枠として出す', () => {
    const frames = report().frames
    expect(overlayBoxes(frames.find((f) => f.frame_index === 1)!).map((b) => [b.layer, b.status])).toEqual([
      ['before', 'disappeared'],
      ['truth', 'regressed'],
    ])
    expect(overlayBoxes(frames.find((f) => f.frame_index === 3)!).map((b) => [b.layer, b.status])).toEqual([
      ['after', 'appeared'],
      ['truth', 'improved'],
    ])
  })

  it('ラベルにクラス名と confidence を入れる', () => {
    const frame1 = report().frames.find((f) => f.frame_index === 1)!
    expect(overlayBoxes(frame1)[0]?.label).toBe('bottle 0.40')
  })
})

describe('formatPercent', () => {
  it('null は「—」にする', () => {
    expect(formatPercent(null)).toBe('—')
    expect(formatPercent(0.5)).toBe('50.0%')
  })
})

describe('defaultFilter', () => {
  it('回帰判定があれば all（差分の無いフレームの回帰も見落とさないため）、無ければ changed', () => {
    expect(defaultFilter(report())).toBe('all')
    expect(defaultFilter({ ...report(), evaluation_summary: null })).toBe('changed')
  })
})
