import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({
    plugins: [react()],
    build: {
        chunkSizeWarningLimit: 1400,
        rollupOptions: {
            output: {
                manualChunks: {
                    plotly: ['plotly.js-basic-dist-min', 'react-plotly.js'],
                },
            },
        },
    },
    server: {
        port: 5173,
    },
    preview: {
        port: 4173,
    },
});
