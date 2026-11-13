import type { ReportFrame } from '../../src/report.js'
import { overlayBoxes, type Layer } from './model.js'

type Props = { frame: ReportFrame; width: number; height: number; layers: Record<Layer, boolean> }

// 画像と SVG を同じ縦横比の箱に重ね、SVG の座標系を元動画のピクセル座標にそろえる
export const FrameCanvas = ({ frame, width, height, layers }: Props) => {
  const boxes = overlayBoxes(frame).filter((b) => layers[b.layer])
  const fontSize = Math.max(width, height) * 0.016
  // 縦長の動画でも画面に収まるよう、高さを 70vh までにして幅をそれに合わせる
  const style = { aspectRatio: `${width} / ${height}`, maxWidth: `calc(70vh * ${width / height})` }
  return (
    <figure className="canvas" style={style}>
      {frame.image === null ? (
        <div className="no-image">画像なし（extract-frames で書き出すと表示されます）</div>
      ) : (
        <img src={frame.image} alt={`フレーム ${frame.frame_index}`} />
      )}
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
        {boxes.map((b) => {
          const [x1, y1, x2, y2] = b.bbox
          return (
            <g key={b.key} className={`box layer-${b.layer} status-${b.status}`}>
              <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} vectorEffect="non-scaling-stroke" />
              <text x={x1 + 2} y={b.layer === 'after' ? y2 - 4 : y1 + fontSize} fontSize={fontSize}>
                {b.label}
              </text>
            </g>
          )
        })}
      </svg>
    </figure>
  )
}
