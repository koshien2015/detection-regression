import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { viteSingleFile } from 'vite-plugin-singlefile'

// file:// では外部ファイルの ES モジュールを読み込めないため、JS と CSS を index.html に埋め込む
export default defineConfig({
  root: import.meta.dirname,
  base: './',
  plugins: [react(), viteSingleFile()],
  build: { outDir: 'dist', emptyOutDir: true },
})
