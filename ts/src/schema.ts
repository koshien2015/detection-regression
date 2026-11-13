import { z } from 'zod'

// 共通JSONスキーマの定義元。schema/*.json はここから生成する（scripts/generate-json-schema.ts）。
// フィールド間の制約（x1 <= x2、frame_index の重複、BBox が画面内か）は superRefine で検査しており、
// 生成した JSON Schema には含まれない。Python 側の検証はその分だけ弱い。

export const SCHEMA_VERSION = '1.0'

// refine でなく regex で書く。JSON Schema の pattern に残り、Python 側の検証にも効くようにするため
const relativePath = z
  .string()
  .regex(
    /^(?!\/|~|[A-Za-z]:[\\/]|\\\\).+$/,
    '絶対パスは記録しない。動画は sha256 で識別するため、ファイル名か相対パスにする',
  )

const sha256 = z.string().regex(/^[0-9a-f]{64}$/, '小文字16進64桁の sha256 にする')

export const BBoxSchema = z
  .tuple([z.number(), z.number(), z.number(), z.number()])
  .describe('[x1, y1, x2, y2]。元動画の解像度でのピクセル座標')

export const DetectionSchema = z.strictObject({
  class_id: z.int().nonnegative(),
  class_name: z.string().min(1),
  confidence: z.number().min(0).max(1),
  bbox: BBoxSchema,
})

export const FrameSchema = z.strictObject({
  frame_index: z.int().nonnegative().describe('先頭から順に読み込んだときの0始まりの番号'),
  timestamp_ms: z.number().nonnegative(),
  detections: z.array(DetectionSchema).describe('検出が無いフレームも空配列で必ず記録する'),
})

const SourceSchema = z.strictObject({
  path: relativePath,
  sha256,
  fps: z.number().positive(),
  width: z.int().positive(),
  height: z.int().positive(),
  frame_count: z.int().nonnegative(),
})

const InferenceSchema = z.strictObject({
  model_id: z.string().min(1),
  confidence_threshold: z.number().min(0).max(1),
  iou_threshold: z.number().min(0).max(1),
  image_size: z.int().positive(),
  classes: z
    .record(z.string().regex(/^\d+$/), z.string().min(1))
    .describe('class_id（文字列）から class_name への対応。モデルごとに異なりうる'),
})

type Issue = { path: (string | number)[]; message: string }

const bboxIssues = (
  bbox: readonly [number, number, number, number],
  size: { width: number; height: number },
  path: (string | number)[],
): Issue[] => {
  const [x1, y1, x2, y2] = bbox
  const issues: Issue[] = []
  if (x1 > x2 || y1 > y2) issues.push({ path, message: 'x1 <= x2 かつ y1 <= y2 にする' })
  if (x1 < 0 || y1 < 0 || x2 > size.width || y2 > size.height) {
    issues.push({ path, message: `フレーム（${size.width}x${size.height}）の内側に収める` })
  }
  return issues
}

const duplicateIndexIssues = (frames: readonly { frame_index: number }[], base: string): Issue[] => {
  const seen = new Set<number>()
  return frames.flatMap((f, i) => {
    if (seen.has(f.frame_index)) {
      return [{ path: [base, i, 'frame_index'], message: `frame_index ${f.frame_index} が重複している` }]
    }
    seen.add(f.frame_index)
    return []
  })
}

const detectionIssues = (
  d: z.infer<typeof DetectionSchema>,
  classes: Record<string, string>,
  source: { width: number; height: number },
  path: (string | number)[],
): Issue[] => {
  const expectedName = classes[String(d.class_id)]
  const classIssue: Issue[] =
    expectedName === undefined
      ? [{ path: [...path, 'class_id'], message: `class_id ${d.class_id} が inference.classes に無い` }]
      : expectedName !== d.class_name
        ? [{ path: [...path, 'class_name'], message: `class_id ${d.class_id} は "${expectedName}" のはず` }]
        : []
  return [...classIssue, ...bboxIssues(d.bbox, source, [...path, 'bbox'])]
}

export const InferenceResultSchema = z
  .strictObject({
    schema_version: z.literal(SCHEMA_VERSION),
    source: SourceSchema,
    inference: InferenceSchema,
    frames: z.array(FrameSchema),
  })
  .superRefine((doc, ctx) => {
    const frameRangeIssues = doc.frames.flatMap((f, i): Issue[] =>
      f.frame_index >= doc.source.frame_count
        ? [{ path: ['frames', i, 'frame_index'], message: 'frame_index が source.frame_count 以上' }]
        : [],
    )
    const detIssues = doc.frames.flatMap((f, i) =>
      f.detections.flatMap((d, j) =>
        detectionIssues(d, doc.inference.classes, doc.source, ['frames', i, 'detections', j]),
      ),
    )
    const all = [...duplicateIndexIssues(doc.frames, 'frames'), ...frameRangeIssues, ...detIssues]
    all.forEach((issue) => ctx.addIssue({ code: 'custom', ...issue }))
  })

export const AnnotatedObjectSchema = z.strictObject({
  class_name: z.string().min(1),
  bbox: BBoxSchema,
})

export const AnnotatedFrameSchema = z.strictObject({
  frame_index: z.int().nonnegative(),
  objects: z
    .array(AnnotatedObjectSchema)
    .describe('空配列は「このフレームには何も写っていないのが正解」を意味する'),
})

export const AnnotationsSchema = z
  .strictObject({
    schema_version: z.literal(SCHEMA_VERSION),
    source: z.strictObject({
      path: relativePath,
      sha256,
      width: z.int().positive(),
      height: z.int().positive(),
    }),
    classes: z
      .array(z.string().min(1))
      .min(1)
      .optional()
      .describe('正解を付けたクラス。指定すると、ほかのクラスの検出は評価しない。省略時はすべてのクラスを評価する'),
    frames: z.array(AnnotatedFrameSchema).describe('ここに無いフレームは未アノテーションとして評価から除く'),
  })
  .superRefine((doc, ctx) => {
    const objIssues = doc.frames.flatMap((f, i) =>
      f.objects.flatMap((o, j) => bboxIssues(o.bbox, doc.source, ['frames', i, 'objects', j, 'bbox'])),
    )
    const allowed = doc.classes === undefined ? null : new Set(doc.classes)
    const classIssues = doc.frames.flatMap((f, i) =>
      f.objects.flatMap((o, j): Issue[] =>
        allowed === null || allowed.has(o.class_name)
          ? []
          : [{ path: ['frames', i, 'objects', j, 'class_name'], message: `${o.class_name} が classes に無い` }],
      ),
    )
    const all = [...duplicateIndexIssues(doc.frames, 'frames'), ...objIssues, ...classIssues]
    all.forEach((issue) => ctx.addIssue({ code: 'custom', ...issue }))
  })

export type InferenceResult = z.infer<typeof InferenceResultSchema>
export type Detection = z.infer<typeof DetectionSchema>
export type Frame = z.infer<typeof FrameSchema>
export type BBox = z.infer<typeof BBoxSchema>
export type Annotations = z.infer<typeof AnnotationsSchema>
export type AnnotatedObject = z.infer<typeof AnnotatedObjectSchema>

export type ParseError =
  | { kind: 'schema_version_mismatch'; expected: string; actual: unknown; message: string }
  | { kind: 'invalid'; message: string }

export type ParseResult<T> = { ok: true; data: T } | { ok: false; error: ParseError }

const formatIssues = (error: z.ZodError): string =>
  error.issues.map((i) => `${i.path.join('.') || '(root)'}: ${i.message}`).join('\n')

// schema_version を先に確かめる。版が違うと他のフィールドの検証エラーが大量に出て、原因が分かりにくくなるため。
const parseVersioned = <T>(schema: z.ZodType<T>, input: unknown): ParseResult<T> => {
  if (typeof input !== 'object' || input === null || Array.isArray(input)) {
    return { ok: false, error: { kind: 'invalid', message: '(root): JSON オブジェクトではない' } }
  }
  const actual = (input as Record<string, unknown>).schema_version
  if (actual !== SCHEMA_VERSION) {
    return {
      ok: false,
      error: {
        kind: 'schema_version_mismatch',
        expected: SCHEMA_VERSION,
        actual,
        message: `schema_version が ${JSON.stringify(actual)}。このツールは ${SCHEMA_VERSION} のみ扱う`,
      },
    }
  }
  const parsed = schema.safeParse(input)
  return parsed.success
    ? { ok: true, data: parsed.data }
    : { ok: false, error: { kind: 'invalid', message: formatIssues(parsed.error) } }
}

export const parseInferenceResult = (input: unknown): ParseResult<InferenceResult> =>
  parseVersioned(InferenceResultSchema, input)

export const parseAnnotations = (input: unknown): ParseResult<Annotations> =>
  parseVersioned(AnnotationsSchema, input)
