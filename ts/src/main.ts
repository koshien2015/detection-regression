import { runCli } from './cli.js'

process.exitCode = runCli(process.argv.slice(2), {
  stdout: (s) => process.stdout.write(s),
  stderr: (s) => process.stderr.write(s),
})
