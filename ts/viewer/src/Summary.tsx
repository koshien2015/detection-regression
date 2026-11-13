import type { ReportData } from '../../src/report.js'
import { formatPercent } from './model.js'

const Stat = ({ label, value, tone }: { label: string; value: string | number; tone?: string }) => (
  <div className={`stat ${tone ?? ''}`}>
    <div className="stat-value">{value}</div>
    <div className="stat-label">{label}</div>
  </div>
)

const conditions = (inf: ReportData['inputs']['before']['inference']) =>
  `conf ${inf.confidence_threshold} / iou ${inf.iou_threshold} / imgsz ${inf.image_size}`

export const Summary = ({ data }: { data: ReportData }) => {
  const { source, before, after } = data.inputs
  const s = data.summary
  const e = data.evaluation_summary
  return (
    <header className="summary">
      <div className="title-row">
        <h1>検出結果の比較</h1>
        <p className="models">
          <span className="layer-before">{before.model_id}</span>
          <span aria-hidden="true"> → </span>
          <span className="layer-after">{after.model_id}</span>
        </p>
      </div>
      <dl className="meta">
        <div><dt>動画</dt><dd>{source.path}（{source.width}×{source.height}・{source.fps}fps・{source.frame_count}フレーム）</dd></div>
        <div><dt>sha256</dt><dd><code>{source.sha256.slice(0, 16)}…</code></dd></div>
        <div><dt>推論条件</dt><dd>before: {conditions(before.inference)}／after: {conditions(after.inference)}</dd></div>
        <div>
          <dt>比較の設定</dt>
          <dd>
            対応づけ IoU ≥ {data.options.match_iou_threshold}・confidence ±{data.options.confidence_tolerance}・IoU &lt; {data.options.iou_tolerance}・中心 {data.options.center_shift_tolerance_px}px・大きさ {data.options.size_change_tolerance_ratio * 100}%
          </dd>
        </div>
      </dl>
      {data.warnings.length > 0 && (
        <ul className="warnings">
          {data.warnings.map((w) => <li key={w}>{w}</li>)}
        </ul>
      )}
      <div className="stats">
        <Stat label="差分のあるフレーム" value={`${s.frames_with_changes} / ${s.frames_total}`} />
        <Stat label="出現" value={s.appeared} tone="appeared" />
        <Stat label="消失" value={s.disappeared} tone="disappeared" />
        <Stat label="変化" value={s.modified} tone="modified" />
        <Stat label="維持" value={s.unchanged} />
      </div>
      {e !== null && (
        <div className="stats">
          <Stat label={`正解のあるフレーム（未アノテーション ${e.unannotated_frames}）`} value={`${e.annotated_frames}・物体 ${e.objects}`} />
          <Stat label="回帰" value={e.regressed} tone="regressed" />
          <Stat label="改善" value={e.improved} tone="improved" />
          <Stat label="維持" value={e.maintained} />
          <Stat label="未改善" value={e.unresolved} />
          <Stat label="recall（before → after）" value={`${formatPercent(e.before.recall)} → ${formatPercent(e.after.recall)}`} />
          <Stat label="precision（before → after）" value={`${formatPercent(e.before.precision)} → ${formatPercent(e.after.precision)}`} />
        </div>
      )}
    </header>
  )
}
