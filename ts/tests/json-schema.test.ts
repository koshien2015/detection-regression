import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { buildJsonSchemas } from '../src/json-schema.js'

const schemaDir = fileURLToPath(new URL('../../schema/', import.meta.url))

describe('生成済みの JSON Schema', () => {
  it.each(Object.entries(buildJsonSchemas()))(
    '%s が zod の定義と一致している（ずれていたら pnpm gen:schema を実行する）',
    (fileName, schema) => {
      const committed: unknown = JSON.parse(readFileSync(`${schemaDir}${fileName}`, 'utf8'))
      expect(committed).toEqual(schema)
    },
  )

  it('推論結果の JSON Schema が schema_version を固定している', () => {
    const schema = buildJsonSchemas()['inference-result.schema.json'] as {
      properties: { schema_version: { const: string } }
    }
    expect(schema.properties.schema_version.const).toBe('1.0')
  })
})
