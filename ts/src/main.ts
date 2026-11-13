import { runCli } from './cli.js'

// pnpm --dir ts cli ... で実行したとき、相対パスをコマンドを打ったディレクトリ基準で解釈する
if (process.env.INIT_CWD !== undefined) process.chdir(process.env.INIT_CWD)

process.exitCode = runCli(process.argv.slice(2), {
  stdout: (s) => process.stdout.write(s),
  stderr: (s) => process.stderr.write(s),
})
