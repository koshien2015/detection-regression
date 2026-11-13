import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { runCli } from '../src/cli.js'
import { compareJson, type Diff } from '../src/compare.js'
import { evaluateJson, type Evaluation } from '../src/evaluation.js'
import { buildReportData, renderDataJs } from '../src/report.js'

const fixture = (name: string) => fileURLToPath(new URL(`./fixtures/${name}`, import.meta.url))
const json = (name: string): unknown => JSON.parse(readFileSync(fixture(name), 'utf8'))

const diff = (): Diff => {
  const o = compareJson(json('before.json'), json('after.json'))
  if (!o.ok) throw new Error(o.error.message)
  return o.diff
}

const evaluation = (): Evaluation => {
  const o = evaluateJson(json('before.json'), json('after.json'), json('annotations.json'))
  if (!o.ok) throw new Error(o.error.message)
  return o.evaluation
}

describe('buildReportData', () => {
  it('差分のあるフレームだけを載せ、画像のあるフレームには画像のパスを付ける', () => {
    const r = buildReportData(diff(), null, { 1: 'frame_000001.png', 2: 'frame_000002.png' })
    if (!r.ok) throw new Error(r.error)
    expect(r.data.frames.map((f) => [f.frame_index, f.image])).toEqual([
      [1, 'frames/frame_000001.png'],
      [2, 'frames/frame_000002.png'],
      [3, null],
    ])
    expect(r.data.evaluation_summary).toBeNull()
  })

  it('正解を付けたフレームは差分が無くても載せ、判定結果を付ける', () => {
    const r = buildReportData(diff(), evaluation(), {})
    if (!r.ok) throw new Error(r.error)
    expect(r.data.frames.map((f) => f.frame_index)).toEqual([0, 1, 2, 3])
    expect(r.data.frames[0]?.evaluation?.objects[0]?.transition).toBe('maintained')
    expect(r.data.frames[2]?.evaluation).toBeNull()
    expect(r.data.evaluation_summary?.regressed).toBe(1)
  })

  it('フレームごとの変化の件数を数える（一覧を差分の大きい順に並べるため）', () => {
    const r = buildReportData(diff(), evaluation(), {})
    if (!r.ok) throw new Error(r.error)
    expect(r.data.frames.map((f) => [f.frame_index, f.change_count, f.regressed_count])).toEqual([
      [0, 0, 0],
      [1, 1, 1],
      [2, 1, 0],
      [3, 1, 0],
    ])
  })

  it('比較と判定で元動画やモデルが違えばエラー', () => {
    const e = evaluation()
    const other = { ...e, inputs: { ...e.inputs, after: { ...e.inputs.after, model_id: 'other' } } }
    const r = buildReportData(diff(), other, {})
    expect(r.ok).toBe(false)
  })

  it('対応する版でない diff.json はエラー', () => {
    const r = buildReportData({ ...diff(), diff_schema_version: '0.1' }, null, {})
    expect(r.ok).toBe(false)
  })
})

describe('renderDataJs', () => {
  it('window.__REPORT__ に代入するスクリプトを作る', () => {
    const script = renderDataJs({ hello: 'world' })
    const sandbox: { __REPORT__?: unknown } = {}
    new Function('window', script)(sandbox)
    expect(sandbox.__REPORT__).toEqual({ hello: 'world' })
  })
})

describe('report コマンド', () => {
  const setup = () => {
    const dir = mkdtempSync(join(tmpdir(), 'dr-'))
    writeFileSync(join(dir, 'diff.json'), JSON.stringify(diff()))
    writeFileSync(join(dir, 'evaluation.json'), JSON.stringify(evaluation()))
    const viewer = join(dir, 'viewer.html')
    writeFileSync(viewer, '<!doctype html><title>viewer</title>')
    const out = join(dir, 'report')
    mkdirSync(join(out, 'frames'), { recursive: true })
    writeFileSync(join(out, 'frames', 'frame_000001.png'), '')
    writeFileSync(join(out, 'frames', 'notes.txt'), '')
    return { dir, viewer, out }
  }

  const run = (args: string[], viewerHtmlPath: string) => {
    const err: string[] = []
    const code = runCli(args, { stdout: () => undefined, stderr: (s) => err.push(s) }, { viewerHtmlPath })
    return { code, err: err.join('') }
  }

  it('ビューアの HTML と data.js を書き出し、frames 内の画像を対応づける', () => {
    const { dir, viewer, out } = setup()
    const result = run(
      ['report', '--diff', join(dir, 'diff.json'), '--evaluation', join(dir, 'evaluation.json'), '--output', out],
      viewer,
    )
    expect(result.code).toBe(0)
    expect(readFileSync(join(out, 'index.html'), 'utf8')).toContain('viewer')
    const sandbox: { __REPORT__?: { frames: { frame_index: number; image: string | null }[] } } = {}
    new Function('window', readFileSync(join(out, 'data.js'), 'utf8'))(sandbox)
    expect(sandbox.__REPORT__?.frames.find((f) => f.frame_index === 1)?.image).toBe('frames/frame_000001.png')
  })

  it('ビューアがビルドされていなければ終了コード1で手順を示す', () => {
    const { dir, out } = setup()
    const result = run(['report', '--diff', join(dir, 'diff.json'), '--output', out], join(dir, 'missing.html'))
    expect(result.code).toBe(1)
    expect(result.err).toContain('build:viewer')
    expect(existsSync(join(out, 'data.js'))).toBe(false)
  })
})
