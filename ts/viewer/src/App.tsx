import { useEffect, useMemo, useState } from 'react'
import type { ReportData } from '../../src/report.js'
import { FrameCanvas } from './FrameCanvas.js'
import { FrameList } from './FrameList.js'
import { FrameTables } from './FrameTables.js'
import { filterFrames, sortFrames, type FilterKey, type Layer, type SortKey } from './model.js'
import { Summary } from './Summary.js'

const LAYERS: { key: Layer; label: string }[] = [
  { key: 'before', label: 'before' },
  { key: 'after', label: 'after' },
  { key: 'truth', label: '正解' },
]

export const App = ({ data }: { data: ReportData }) => {
  const [sort, setSort] = useState<SortKey>('impact')
  const [filter, setFilter] = useState<FilterKey>('changed')
  const [layers, setLayers] = useState<Record<Layer, boolean>>({ before: true, after: true, truth: true })
  const frames = useMemo(() => sortFrames(filterFrames(data.frames, filter), sort), [data.frames, filter, sort])
  const [selected, setSelected] = useState<number | null>(frames[0]?.frame_index ?? null)
  const current = frames.find((f) => f.frame_index === selected) ?? frames[0] ?? null

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return
      const i = frames.findIndex((f) => f.frame_index === current?.frame_index)
      const next = frames[Math.min(frames.length - 1, Math.max(0, i + (e.key === 'ArrowDown' ? 1 : -1)))]
      if (next !== undefined) {
        e.preventDefault()
        setSelected(next.frame_index)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [frames, current])

  return (
    <div className="page">
      <Summary data={data} />
      <div className="workspace">
        <FrameList
          frames={frames}
          selected={current?.frame_index ?? null}
          onSelect={setSelected}
          sort={sort}
          onSort={setSort}
          filter={filter}
          onFilter={setFilter}
          hasEvaluation={data.evaluation_summary !== null}
        />
        <section className="detail">
          {current === null ? (
            <p className="muted">この条件に当てはまるフレームはありません。</p>
          ) : (
            <>
              <div className="detail-head">
                <h2>フレーム {current.frame_index}</h2>
                <div className="toggles" role="group" aria-label="表示する枠">
                  {LAYERS.filter((l) => l.key !== 'truth' || data.evaluation_summary !== null).map((l) => (
                    <label key={l.key} className={`toggle layer-${l.key}`}>
                      <input
                        type="checkbox"
                        name={`layer-${l.key}`}
                        checked={layers[l.key]}
                        onChange={(e) => setLayers({ ...layers, [l.key]: e.target.checked })}
                      />
                      {l.label}
                    </label>
                  ))}
                </div>
              </div>
              <FrameCanvas frame={current} width={data.inputs.source.width} height={data.inputs.source.height} layers={layers} />
              <FrameTables frame={current} />
            </>
          )}
        </section>
      </div>
    </div>
  )
}
