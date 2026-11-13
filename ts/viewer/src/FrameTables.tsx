import type { MatchedEntry } from '../../src/compare.js'
import type { ReportFrame } from '../../src/report.js'
import type { Detection } from '../../src/schema.js'
import { formatSigned, STATUS_LABELS } from './model.js'

const CHANGE_LABELS = { confidence: 'confidence', iou: 'IoU', center: '位置', size: '大きさ' } as const

const bboxText = (d: { bbox: readonly number[] } | null) => (d === null ? '—' : d.bbox.map((v) => Math.round(v)).join(', '))

const matchedRow = (e: MatchedEntry, status: 'modified' | 'unchanged', i: number) => (
  <tr key={`${status}-${i}`}>
    <td><span className={`badge ${status}`}>{STATUS_LABELS[status]}</span></td>
    <td>{e.after.class_name}</td>
    <td className="num">{e.before.confidence.toFixed(2)}</td>
    <td className="num">{e.after.confidence.toFixed(2)}（{formatSigned(e.metrics.confidence_delta)}）</td>
    <td className="num">{e.metrics.iou.toFixed(3)}</td>
    <td className="num">{e.metrics.center_shift_px.toFixed(1)}px</td>
    <td>{e.changed.map((c) => CHANGE_LABELS[c]).join('・') || '—'}</td>
  </tr>
)

const singleRow = (d: Detection, status: 'appeared' | 'disappeared', i: number) => (
  <tr key={`${status}-${i}`}>
    <td><span className={`badge ${status}`}>{STATUS_LABELS[status]}</span></td>
    <td>{d.class_name}</td>
    <td className="num">{status === 'disappeared' ? d.confidence.toFixed(2) : '—'}</td>
    <td className="num">{status === 'appeared' ? d.confidence.toFixed(2) : '—'}</td>
    <td className="num">—</td>
    <td className="num">—</td>
    <td>{bboxText(d)}</td>
  </tr>
)

export const FrameTables = ({ frame }: { frame: ReportFrame }) => {
  const { diff, evaluation } = frame
  return (
    <div className="tables">
      <h3>検出の差分</h3>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>分類</th><th>クラス</th><th className="num">before</th><th className="num">after</th><th className="num">IoU</th><th className="num">中心の移動</th><th>変化した項目・BBox</th></tr>
          </thead>
          <tbody>
            {diff.disappeared.map((d, i) => singleRow(d, 'disappeared', i))}
            {diff.appeared.map((d, i) => singleRow(d, 'appeared', i))}
            {diff.modified.map((e, i) => matchedRow(e, 'modified', i))}
            {diff.unchanged.map((e, i) => matchedRow(e, 'unchanged', i))}
          </tbody>
        </table>
      </div>
      {diff.class_change_candidates.length > 0 && (
        <p className="muted small">
          クラスが変わった可能性のある組:{' '}
          {diff.class_change_candidates.map((c) => `${c.before.class_name} → ${c.after.class_name}（IoU ${c.iou.toFixed(2)}）`).join('、')}
        </p>
      )}
      {evaluation !== null && (
        <>
          <h3>正解との照合</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>判定</th><th>正解</th><th>before</th><th>after</th></tr>
              </thead>
              <tbody>
                {evaluation.objects.map((o, i) => (
                  <tr key={i}>
                    <td><span className={`badge ${o.transition}`}>{STATUS_LABELS[o.transition]}</span></td>
                    <td>{o.ground_truth.class_name}（{bboxText(o.ground_truth)}）</td>
                    <td>{o.before.status}{o.before.detection && ` ${o.before.detection.confidence.toFixed(2)}`}</td>
                    <td>{o.after.status}{o.after.detection && ` ${o.after.detection.confidence.toFixed(2)}`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted small">
            正解の無い検出（FP）: before {evaluation.false_positives.before.length} 件・after {evaluation.false_positives.after.length} 件
          </p>
        </>
      )}
    </div>
  )
}
