import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import type { ReportData } from '../../src/report.js'
import { App } from './App.js'
import './styles.css'

declare global {
  interface Window {
    __REPORT__?: ReportData
  }
}

const root = document.getElementById('root')
if (root === null) throw new Error('#root が無い')

createRoot(root).render(
  <StrictMode>
    {window.__REPORT__ === undefined ? (
      <main className="empty">
        <h1>データが読み込まれていません</h1>
        <p>
          このページは <code>detection-regression report</code> が書き出した <code>data.js</code> と同じフォルダで開いてください。
        </p>
      </main>
    ) : (
      <App data={window.__REPORT__} />
    )}
  </StrictMode>,
)
