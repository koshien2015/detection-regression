import { z } from 'zod'
import { AnnotationsSchema, InferenceResultSchema } from './schema.js'

// zod の定義から JSON Schema を作る。superRefine の制約は含まれない（schema.ts の冒頭を参照）。
export const buildJsonSchemas = (): Record<string, unknown> => ({
  'inference-result.schema.json': z.toJSONSchema(InferenceResultSchema),
  'annotations.schema.json': z.toJSONSchema(AnnotationsSchema),
})
