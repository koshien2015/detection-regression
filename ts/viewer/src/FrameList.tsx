import type { ReportFrame } from '../../src/report.js'
import { formatTime, type FilterKey, type SortKey } from './model.js'

type Props = {
  frames: ReportFrame[]
  selected: number | null
  onSelect: (index: number) => void
  sort: SortKey
  onSort: (key: SortKey) => void
  filter: FilterKey
  onFilter: (key: FilterKey) => void
  hasEvaluation: boolean
}

const counts = (f: ReportFrame) =>
  [
    f.diff.appeared.length > 0 && <span key="a" className="badge appeared">出現 {f.diff.appeared.length}</span>,
    f.diff.disappeared.length > 0 && <span key="d" className="badge disappeared">消失 {f.diff.disappeared.length}</span>,
    f.diff.modified.length > 0 && <span key="m" className="badge modified">変化 {f.diff.modified.length}</span>,
    f.regressed_count > 0 && <span key="r" className="badge regressed">デグレ {f.regressed_count}</span>,
  ].filter(Boolean)

export const FrameList = (p: Props) => (
  <nav className="frame-list" aria-label="フレーム一覧">
    <div className="list-controls">
      <select name="filter" value={p.filter} onChange={(e) => p.onFilter(e.target.value as FilterKey)} aria-label="絞り込み">
        <option value="changed">差分のあるフレーム</option>
        {p.hasEvaluation && <option value="regressed">デグレのあるフレーム</option>}
        <option value="all">すべて（正解のあるフレームを含む）</option>
      </select>
      <select name="sort" value={p.sort} onChange={(e) => p.onSort(e.target.value as SortKey)} aria-label="並び順">
        <option value="impact">差分の大きい順</option>
        <option value="index">フレーム番号順</option>
      </select>
    </div>
    <p className="muted small">{p.frames.length} フレーム・↑↓キーで移動</p>
    <ol>
      {p.frames.map((f) => (
        <li key={f.frame_index}>
          <button
            type="button"
            className={f.frame_index === p.selected ? 'selected' : ''}
            aria-current={f.frame_index === p.selected}
            onClick={() => p.onSelect(f.frame_index)}
          >
            <span className="frame-no">#{f.frame_index}</span>
            <span className="muted small">{formatTime(f.timestamp_ms)}</span>
            <span className="badges">{counts(f).length > 0 ? counts(f) : <span className="muted small">差分なし</span>}</span>
          </button>
        </li>
      ))}
    </ol>
  </nav>
)
