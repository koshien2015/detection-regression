import { readFileSync, writeFileSync } from 'node:fs'
import { parseArgs } from 'node:util'
import { z } from 'zod'
import { compareJson } from './compare.js'
import { evaluateJson } from './evaluation.js'

export type CliIo = { stdout: (s: string) => void; stderr: (s: string) => void }

const USAGE = `使い方:
  detection-regression compare --before <before.json> --after <after.json> --output <diff.json> [--config <config.json>]
  detection-regression evaluate --before <before.json> --after <after.json> --annotations <annotations.json> --output <evaluation.json> [--iou-threshold 0.5]
`

const ConfigSchema = z
  .strictObject({
    match_iou_threshold: z.number().min(0).max(1),
    confidence_tolerance: z.number().min(0).max(1),
    iou_tolerance: z.number().min(0).max(1),
    center_shift_tolerance_px: z.number().nonnegative(),
    size_change_tolerance_ratio: z.number().nonnegative(),
    allow_inference_mismatch: z.boolean(),
  })
  .partial()

// 終了コード1（入力の内容が正しくない）で止めるためのエラー
class InputError extends Error {}
// 終了コード2（引数の指定が正しくない）で止めるためのエラー
class UsageError extends Error {}

const readJson = (path: string): unknown => {
  try {
    return JSON.parse(readFileSync(path, 'utf8'))
  } catch (error) {
    throw new InputError(`${path} を JSON として読めない: ${(error as Error).message}`)
  }
}

const readConfig = (path: string | undefined): z.infer<typeof ConfigSchema> => {
  if (path === undefined) return {}
  const parsed = ConfigSchema.safeParse(readJson(path))
  if (!parsed.success) {
    throw new InputError(
      `${path} の設定が正しくない:\n${parsed.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`).join('\n')}`,
    )
  }
  return parsed.data
}

const writeJson = (path: string, value: unknown): void => {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`)
}

const parseRequired = <K extends string>(
  args: string[],
  required: readonly K[],
  optional: readonly string[] = [],
): Record<K, string> & Record<string, string | undefined> => {
  const options = Object.fromEntries([...required, ...optional].map((k) => [k, { type: 'string' as const }]))
  const { values } = parseArgs({ args, options, strict: true })
  const missing = required.filter((k) => values[k] === undefined)
  if (missing.length > 0) throw new UsageError(`${missing.map((k) => `--${k}`).join(', ')} を指定する`)
  return values as Record<K, string> & Record<string, string | undefined>
}

const ratio = (value: string | undefined, name: string): number | undefined => {
  if (value === undefined) return undefined
  const n = Number(value)
  if (!Number.isFinite(n) || n < 0 || n > 1) throw new UsageError(`--${name} は 0〜1 の数にする: ${value}`)
  return n
}

const runCompare = (args: string[], io: CliIo): number => {
  const v = parseRequired(args, ['before', 'after', 'output'], ['config'])
  const outcome = compareJson(readJson(v.before), readJson(v.after), readConfig(v.config))
  if (!outcome.ok) {
    io.stderr(`比較できない（${outcome.error.kind}）: ${outcome.error.message}\n`)
    return 1
  }
  writeJson(v.output, outcome.diff)
  const s = outcome.diff.summary
  io.stdout(
    `${v.output} に書き出した。差分のあるフレーム ${s.frames_with_changes}/${s.frames_total}` +
      `（出現 ${s.appeared}・消失 ${s.disappeared}・変化 ${s.modified}・維持 ${s.unchanged}）\n`,
  )
  outcome.diff.warnings.forEach((w) => io.stderr(`警告: ${w}\n`))
  return 0
}

const runEvaluate = (args: string[], io: CliIo): number => {
  const v = parseRequired(args, ['before', 'after', 'annotations', 'output'], ['iou-threshold'])
  const iouThreshold = ratio(v['iou-threshold'], 'iou-threshold')
  const outcome = evaluateJson(
    readJson(v.before),
    readJson(v.after),
    readJson(v.annotations),
    iouThreshold === undefined ? {} : { iou_threshold: iouThreshold },
  )
  if (!outcome.ok) {
    io.stderr(`判定できない（${outcome.error.kind}）: ${outcome.error.message}\n`)
    return 1
  }
  writeJson(v.output, outcome.evaluation)
  const s = outcome.evaluation.summary
  io.stdout(
    `${v.output} に書き出した。正解のあるフレーム ${s.annotated_frames}・物体 ${s.objects}` +
      `（維持 ${s.maintained}・改善 ${s.improved}・回帰 ${s.regressed}・未改善 ${s.unresolved}）\n`,
  )
  return 0
}

const COMMANDS: Record<string, (args: string[], io: CliIo) => number> = {
  compare: runCompare,
  evaluate: runEvaluate,
}

const isParseArgsError = (error: unknown): error is TypeError =>
  error instanceof TypeError && 'code' in error && String(error.code).startsWith('ERR_PARSE_ARGS')

export const runCli = (argv: string[], io: CliIo): number => {
  const [command, ...rest] = argv
  const handler = command === undefined ? undefined : COMMANDS[command]
  if (handler === undefined) {
    io.stderr(`未知のコマンド: ${command ?? '(なし)'}\n${USAGE}`)
    return 2
  }
  try {
    return handler(rest, io)
  } catch (error) {
    if (error instanceof InputError) {
      io.stderr(`${error.message}\n`)
      return 1
    }
    if (error instanceof UsageError || isParseArgsError(error)) {
      io.stderr(`${(error as Error).message}\n${USAGE}`)
      return 2
    }
    throw error
  }
}
