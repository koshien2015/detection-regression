// テストで使う推論結果・正解アノテーションの組み立て用ヘルパー。
// 既定値は「正しい」データにしておき、各テストは崩したい箇所だけを上書きする。

export const SHA_A = 'a'.repeat(64)
export const SHA_B = 'b'.repeat(64)

export type RawDetection = {
  class_id: number
  class_name: string
  confidence: number
  bbox: [number, number, number, number]
}

export type RawFrame = {
  frame_index: number
  timestamp_ms: number
  detections: RawDetection[]
}

export const det = (
  bbox: [number, number, number, number],
  confidence = 0.9,
  class_name = 'bottle',
  class_id = 39,
): RawDetection => ({ class_id, class_name, confidence, bbox })

export const frame = (frame_index: number, detections: RawDetection[] = []): RawFrame => ({
  frame_index,
  timestamp_ms: (frame_index * 1000) / 30,
  detections,
})

export const makeResult = (
  overrides: {
    frames?: RawFrame[]
    source?: Record<string, unknown>
    inference?: Record<string, unknown>
    schema_version?: unknown
  } = {},
) => ({
  schema_version: overrides.schema_version ?? '1.0',
  source: {
    path: 'sample.mp4',
    sha256: SHA_A,
    fps: 30,
    width: 1280,
    height: 720,
    frame_count: 3,
    ...overrides.source,
  },
  inference: {
    model_id: 'model-v1',
    confidence_threshold: 0.25,
    iou_threshold: 0.7,
    image_size: 640,
    classes: { '0': 'person', '39': 'bottle' },
    ...overrides.inference,
  },
  frames: overrides.frames ?? [
    frame(0, [det([120, 80, 200, 340])]),
    frame(1, []),
    frame(2, [det([10, 10, 50, 50], 0.5, 'person', 0)]),
  ],
})

export const makeAnnotations = (
  overrides: {
    frames?: { frame_index: number; objects: { class_name: string; bbox: number[] }[] }[]
    source?: Record<string, unknown>
    schema_version?: unknown
  } = {},
) => ({
  schema_version: overrides.schema_version ?? '1.0',
  source: { path: 'sample.mp4', sha256: SHA_A, width: 1280, height: 720, ...overrides.source },
  frames: overrides.frames ?? [
    { frame_index: 0, objects: [{ class_name: 'bottle', bbox: [120, 80, 200, 340] }] },
    { frame_index: 1, objects: [] },
  ],
})
