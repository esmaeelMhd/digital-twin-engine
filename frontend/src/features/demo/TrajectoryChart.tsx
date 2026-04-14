import Plotly from 'plotly.js-basic-dist-min';
import createPlotlyComponent from 'react-plotly.js/factory';

import type {
  DemoCompareScenariosResponse,
  DemoDefinition,
  DemoOptimizeControlResponse,
} from '../../api/types';

type TrajectoryChartProps = {
  demo: DemoDefinition;
  comparison: DemoCompareScenariosResponse;
  optimizedResult: DemoOptimizeControlResponse | null;
};

const Plot = createPlotlyComponent(Plotly);

export function TrajectoryChart({
  demo,
  comparison,
  optimizedResult,
}: TrajectoryChartProps) {
  return (
    <div>
      {demo.highlight_states.map((stateName) => {
        const stateIndex = comparison.state_names.indexOf(stateName);
        if (stateIndex < 0) {
          return null;
        }

        const optimizedY = optimizedResult?.predicted_states.map((row) => row[stateIndex]);

        return (
          <Plot
            key={stateName}
            data={[
              {
                x: comparison.times,
                y: comparison.candidate_p95.map((row) => row[stateIndex]),
                mode: 'lines',
                line: { width: 0 },
                showlegend: false,
                hoverinfo: 'skip',
              },
              {
                x: comparison.times,
                y: comparison.candidate_p05.map((row) => row[stateIndex]),
                mode: 'lines',
                fill: 'tonexty',
                fillcolor: 'rgba(26, 74, 62, 0.12)',
                line: { width: 0 },
                name: '90% forecast interval',
                hoverinfo: 'skip',
              },
              {
                x: comparison.times,
                y: comparison.baseline_mean.map((row) => row[stateIndex]),
                mode: 'lines',
                name: 'Baseline',
                line: { color: '#c67a30', dash: 'dash', width: 2 },
              },
              {
                x: comparison.times,
                y: comparison.candidate_mean.map((row) => row[stateIndex]),
                mode: 'lines',
                name: 'Candidate',
                line: { color: '#1a4a3e', width: 3 },
              },
              ...(optimizedY
                ? [
                    {
                      x: comparison.times,
                      y: optimizedY,
                      mode: 'lines',
                      name: 'Recommended',
                      line: { color: '#0f1a14', dash: 'dot', width: 2 },
                    },
                  ]
                : []),
            ]}
            layout={{
              title: {
                text: stateName,
                font: { family: 'Playfair Display, Georgia, serif', size: 20, color: '#0f1a14' },
              },
              height: 300,
              margin: { l: 40, r: 20, t: 50, b: 36 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(255,255,255,0.56)',
              hovermode: 'x unified',
              xaxis: {
                title: 'Time (min)',
                gridcolor: 'rgba(15,26,20,0.08)',
                zeroline: false,
              },
              yaxis: {
                title: 'Value',
                gridcolor: 'rgba(15,26,20,0.08)',
                zeroline: false,
              },
              legend: {
                orientation: 'h',
                y: 1.12,
              },
              font: {
                family: 'Inter, sans-serif',
                color: '#1a2b22',
              },
            }}
            style={{ width: '100%' }}
            config={{ displayModeBar: false, responsive: true }}
          />
        );
      })}
    </div>
  );
}
