import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { runCli } from '../src/cli.js'
import type { Diff } from '../src/compare.js'

const fixture = (name: string) => fileURLToPath(new URL(`./fixtures/${name}`, import.meta.url))

const run = (args: string[]) => {
  const out: string[] = []
  const err: string[] = []
  const code = runCli(args, { stdout: (s) => out.push(s), stderr: (s) => err.push(s) })
  return { code, out: out.join(''), err: err.join('') }
}

describe('compare コマンド', () => {
  it('手作業で設計した before/after から、期待どおりの差分を書き出す', () => {
    const output = join(mkdtempSync(join(tmpdir(), 'dr-')), 'diff.json')
    const result = run(['compare', '--before', fixture('before.json'), '--after', fixture('after.json'), '--output', output])
    expect(result.code).toBe(0)

    const diff = JSON.parse(readFileSync(output, 'utf8')) as Diff
    // frame 0: confidence +0.01 で許容幅以内 / frame 1: bottle が消失
    // frame 2: person が右へ20px 移動（IoU 0.82）/ frame 3: bottle が出現
    expect(diff.summary).toEqual({
      frames_total: 4,
      frames_with_changes: 3,
      unchanged: 1,
      modified: 1,
      appeared: 1,
      disappeared: 1,
      class_change_candidates: 0,
    })
    expect(diff.frames.map((f) => f.has_changes)).toEqual([false, true, true, true])
    expect(diff.frames[2]?.modified[0]?.changed).toEqual(['iou', 'center'])
  })

  it('同じファイル同士なら差分ゼロ', () => {
    const output = join(mkdtempSync(join(tmpdir(), 'dr-')), 'diff.json')
    expect(run(['compare', '--before', fixture('before.json'), '--after', fixture('before.json'), '--output', output]).code).toBe(0)
    const diff = JSON.parse(readFileSync(output, 'utf8')) as Diff
    expect(diff.summary.frames_with_changes).toBe(0)
  })

  it('設定ファイルの許容幅を使う', () => {
    const dir = mkdtempSync(join(tmpdir(), 'dr-'))
    const config = join(dir, 'config.json')
    writeFileSync(config, JSON.stringify({ center_shift_tolerance_px: 50, iou_tolerance: 0.5 }))
    const output = join(dir, 'diff.json')
    run(['compare', '--before', fixture('before.json'), '--after', fixture('after.json'), '--output', output, '--config', config])
    const diff = JSON.parse(readFileSync(output, 'utf8')) as Diff
    expect(diff.summary.modified).toBe(0)
  })

  it('比較できない入力なら終了コード1でエラーの内容を表示する', () => {
    const dir = mkdtempSync(join(tmpdir(), 'dr-'))
    const other = join(dir, 'other.json')
    const raw = JSON.parse(readFileSync(fixture('after.json'), 'utf8')) as { source: { sha256: string } }
    writeFileSync(other, JSON.stringify({ ...raw, source: { ...raw.source, sha256: 'f'.repeat(64) } }))
    const result = run(['compare', '--before', fixture('before.json'), '--after', other, '--output', join(dir, 'diff.json')])
    expect(result.code).toBe(1)
    expect(result.err).toContain('source_mismatch')
  })

  it('JSON として読めないファイルなら終了コード1', () => {
    const dir = mkdtempSync(join(tmpdir(), 'dr-'))
    const broken = join(dir, 'broken.json')
    writeFileSync(broken, '{')
    const result = run(['compare', '--before', broken, '--after', fixture('after.json'), '--output', join(dir, 'diff.json')])
    expect(result.code).toBe(1)
    expect(result.err).toContain('broken.json')
  })

  it('必須の引数が無ければ終了コード2で使い方を表示する', () => {
    const result = run(['compare', '--before', fixture('before.json')])
    expect(result.code).toBe(2)
    expect(result.err).toContain('--after')
  })

  it('未知のオプションなら例外を投げずに終了コード2', () => {
    const result = run(['compare', '--before', 'a.json', '--unknown'])
    expect(result.code).toBe(2)
    expect(result.err).toContain('--unknown')
  })

  it('設定ファイルに未知の項目や範囲外の値があれば終了コード1', () => {
    const dir = mkdtempSync(join(tmpdir(), 'dr-'))
    const config = join(dir, 'config.json')
    writeFileSync(config, JSON.stringify({ confidence_tolerance: 2, typo_key: 1 }))
    const result = run([
      'compare', '--before', fixture('before.json'), '--after', fixture('after.json'),
      '--output', join(dir, 'diff.json'), '--config', config,
    ])
    expect(result.code).toBe(1)
    expect(result.err).toContain('confidence_tolerance')
    expect(result.err).toContain('typo_key')
  })

  it('未知のコマンドなら終了コード2', () => {
    expect(run(['unknown']).code).toBe(2)
  })
})
