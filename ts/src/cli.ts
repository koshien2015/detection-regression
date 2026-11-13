import { readFileSync, writeFileSync } from 'node:fs'
import { parseArgs } from 'node:util'
import { z } from 'zod'
import { compareJson } from './compare.js'

export type CliIo = { stdout: (s: string) => void; stderr: (s: string) => void }

const USAGE = `使い方:
  detection-regression compare --before <before.json> --after <after.json> --output <diff.json> [--config <config.json>]
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

class CliError extends Error {}

const readJson = (path: string): unknown => {
  try {
    return JSON.parse(readFileSync(path, 'utf8'))
  } catch (error) {
    throw new CliError(`${path} を JSON として読めない: ${(error as Error).message}`)
  }
}

const readConfig = (path: string | undefined): z.infer<typeof ConfigSchema> => {
  if (path === undefined) return {}
  const parsed = ConfigSchema.safeParse(readJson(path))
  if (!parsed.success) {
    throw new CliError(`${path} の設定が正しくない:\n${parsed.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`).join('\n')}`)
  }
  return parsed.data
}

const runCompare = (args: string[], io: CliIo): number => {
  const { values } = parseArgs({
    args,
    options: {
      before: { type: 'string' },
      after: { type: 'string' },
      output: { type: 'string' },
      config: { type: 'string' },
    },
    strict: true,
  })
  const missing = (['before', 'after', 'output'] as const).filter((k) => values[k] === undefined)
  if (missing.length > 0) {
    io.stderr(`${missing.map((k) => `--${k}`).join(', ')} を指定する\n${USAGE}`)
    return 2
  }
  const outcome = compareJson(readJson(values.before!), readJson(values.after!), readConfig(values.config))
  if (!outcome.ok) {
    io.stderr(`比較できない（${outcome.error.kind}）: ${outcome.error.message}\n`)
    return 1
  }
  writeFileSync(values.output!, `${JSON.stringify(outcome.diff, null, 2)}\n`)
  const s = outcome.diff.summary
  io.stdout(
    `${values.output} に書き出した。差分のあるフレーム ${s.frames_with_changes}/${s.frames_total}` +
      `（出現 ${s.appeared}・消失 ${s.disappeared}・変化 ${s.modified}・維持 ${s.unchanged}）\n`,
  )
  outcome.diff.warnings.forEach((w) => io.stderr(`警告: ${w}\n`))
  return 0
}

export const runCli = (argv: string[], io: CliIo): number => {
  const [command, ...rest] = argv
  try {
    if (command === 'compare') return runCompare(rest, io)
    io.stderr(`未知のコマンド: ${command ?? '(なし)'}\n${USAGE}`)
    return 2
  } catch (error) {
    if (error instanceof CliError) {
      io.stderr(`${error.message}\n`)
      return 1
    }
    if (error instanceof TypeError && 'code' in error && String(error.code).startsWith('ERR_PARSE_ARGS')) {
      io.stderr(`${error.message}\n${USAGE}`)
      return 2
    }
    throw error
  }
}
