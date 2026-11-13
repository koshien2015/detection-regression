import { writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { buildJsonSchemas } from '../src/json-schema.js'

const schemaDir = fileURLToPath(new URL('../../schema/', import.meta.url))

Object.entries(buildJsonSchemas()).forEach(([fileName, schema]) => {
  writeFileSync(`${schemaDir}${fileName}`, `${JSON.stringify(schema, null, 2)}\n`)
  process.stdout.write(`wrote schema/${fileName}\n`)
})
