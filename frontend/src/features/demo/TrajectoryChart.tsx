import Plotly from 'plotly.js-basic-dist-min';
import createPlotlyComponent from 'react-plotly.js/factory';

import type {
  DemoCompareScenariosRequest,
  DemoCompareScenariosResponse,
  DemoDefinition,
  DemoOptimizeControlResponse,
} from '../../api/types';
import styles from './demo.module.css';

type TrajectoryChartProps = {
  demo: DemoDefinition;
  comparison: DemoCompareScenariosResponse;
  comparisonRequest: DemoCompareScenariosRequest;
  optimizedResult: DemoOptimizeControlResponse | null;
};

const Plot = createPlotlyComponent(Plotly);

function baseLayout(title: string) {
  return {
    title: {
      text: title,
      font: { family: 'Playfair Display, Georgia, serif', size: 20, color: '#0f1a14' },
    },
    height: 300,
    margin: { l: 40, r: 20, t: 50, b: 36 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(255,255,255,0.56)',
    hovermode: 'x unified' as const,
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
      orientation: 'h' as const,
      y: 1.12,
    },
    font: {
      family: 'Inter, sans-serif',
      color: '#1a2b22',
    },
  };
}

export function TrajectoryChart({
  demo,
  comparison,
  comparisonRequest,
  optimizedResult,
}: TrajectoryChartProps) {
  return (
    <div className={styles.chartStack}>
      <div className={styles.chartSection}>
        <div className={styles.chartSectionHeader}>
          <h4 className={styles.chartSectionTitle}>State Forecasts</h4>
          <p className={styles.chartSectionBody}>
            Compare how the baseline, selected candidate, and recommended sequence affect the
            tracked process states.
          </p>
        </div>
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
              layout={baseLayout(stateName)}
              style={{ width: '100%' }}
              config={{ displayModeBar: false, responsive: true }}
            />
          );
        })}
      </div>

      <div className={styles.chartSection}>
        <div className={styles.chartSectionHeader}>
          <h4 className={styles.chartSectionTitle}>Control Schedules</h4>
          <p className={styles.chartSectionBody}>
            These traces show the actual control moves behind the baseline, candidate, and
            recommended trajectories.
          </p>
        </div>
        {demo.system_spec.control_names.map((controlName, controlIndex) => {
          const optimizedY = optimizedResult?.control_sequence.map((row) => row[controlIndex]);

          return (
            <Plot
              key={controlName}
              data={[
                {
                  x: comparison.times,
                  y: comparisonRequest.baseline_controls.map((row) => row[controlIndex]),
                  mode: 'lines',
                  name: 'Baseline',
                  line: { color: '#c67a30', dash: 'dash', width: 2 },
                },
                {
                  x: comparison.times,
                  y: comparisonRequest.candidate_controls.map((row) => row[controlIndex]),
                  mode: 'lines',
                  name: 'Candidate',
                  line: { color: '#1a4a3e', width: 3 },
                },
                ...(optimizedY
                  ? [
                      {
                        x: comparison.times.slice(0, optimizedY.length),
                        y: optimizedY,
                        mode: 'lines',
                        name: 'Recommended',
                        line: { color: '#0f1a14', dash: 'dot', width: 2 },
                      },
                    ]
                  : []),
              ]}
              layout={baseLayout(controlName)}
              style={{ width: '100%' }}
              config={{ displayModeBar: false, responsive: true }}
            />
          );
        })}
      </div>
    </div>
  );
}
