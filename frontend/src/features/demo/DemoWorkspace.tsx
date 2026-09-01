import clsx from 'clsx';
import { Play, Target } from 'lucide-react';
import { lazy, Suspense } from 'react';
import * as Accordion from '@radix-ui/react-accordion';

import { compareScenarios, optimizeControl } from '../../api/client';
import type {
  DemoCompareScenariosRequest,
  DemoCompareScenariosResponse,
  DemoDefinition,
  DemoOptimizeControlRequest,
  DemoOptimizeControlResponse,
} from '../../api/types';
import { MetricCard } from '../../components/MetricCard';
import { formatMetric, titleCase } from '../../lib/format';
import { useDemoScenario } from './useDemoScenario';
import styles from './demo.module.css';

const TrajectoryChart = lazy(async () => {
  const module = await import('./TrajectoryChart');
  return { default: module.TrajectoryChart };
});

type DemoWorkspaceProps = {
  demo: DemoDefinition;
  compareScenario?: (payload: DemoCompareScenariosRequest) => Promise<DemoCompareScenariosResponse>;
  optimizeScenario?: (payload: DemoOptimizeControlRequest) => Promise<DemoOptimizeControlResponse>;
};

export function DemoWorkspace({
  demo,
  compareScenario = compareScenarios,
  optimizeScenario = optimizeControl,
}: DemoWorkspaceProps) {
  const {
    draft,
    comparison,
    comparisonRequest,
    comparisonError,
    comparisonPending,
    optimizedResult,
    optimizationError,
    optimizationPending,
    setSelectedDisturbanceId,
    setSelectedCandidateId,
    setControlAdjustment,
    setDisturbanceAdjustment,
    runScenario,
    optimizeScenario: runOptimizeScenario,
    controlAdjustmentRange,
    disturbanceAdjustmentRange,
  } = useDemoScenario(demo, {
    compareScenario,
    optimizeScenario,
  });
  const editableControlNames = demo.editable_control_names ?? demo.system_spec.control_names;

  const selectedDisturbance = demo.disturbance_presets.find(
    (item) => item.id === draft.selectedDisturbanceId,
  );
  const selectedCandidate = demo.candidate_profiles.find(
    (item) => item.id === draft.selectedCandidateId,
  );

  const constraintRisk = comparison
    ? (comparison.candidate_constraints.above_upper_bound_rate ?? 0) +
      (comparison.candidate_constraints.below_lower_bound_rate ?? 0)
    : null;
  const forecastRuntime =
    comparison?.candidate_source === 'universal_model'
      ? 'Shared Runtime'
      : comparison
        ? titleCase(comparison.candidate_source)
        : null;

  return (
    <div className={styles.workspace}>
      <aside className={clsx(styles.controlPanel, 'glass-panel')}>
        <div>
          <h3 className={styles.panelTitle}>{demo.title}</h3>
          <p className={styles.panelBody}>{demo.description}</p>
          {demo.operator_goal ? (
            <p className={styles.goalText}>Decision focus: {demo.operator_goal}</p>
          ) : null}
          <p className="field-help">
            Baseline stays fixed as the current operating plan. The alternative plan and the
            recommendation are both scored against the same disturbance.
          </p>
        </div>

        <div className={styles.fieldStack}>
          <div>
            <label className="field-label" htmlFor={`${demo.id}-disturbance`}>
              Operating condition
            </label>
            <select
              id={`${demo.id}-disturbance`}
              className="select-shell"
              value={draft.selectedDisturbanceId}
              onChange={(event) => setSelectedDisturbanceId(event.target.value)}
            >
              {demo.disturbance_presets.map((preset) => (
                <option key={preset.id} value={preset.id}>
                  {preset.title}
                </option>
              ))}
            </select>
            {selectedDisturbance ? (
              <p className="field-help">{selectedDisturbance.description}</p>
            ) : null}
          </div>

          <div>
            <label className="field-label" htmlFor={`${demo.id}-candidate`}>
              Alternative operating plan
            </label>
            <select
              id={`${demo.id}-candidate`}
              className="select-shell"
              value={draft.selectedCandidateId}
              onChange={(event) => setSelectedCandidateId(event.target.value)}
            >
              {demo.candidate_profiles.map((preset) => (
                <option key={preset.id} value={preset.id}>
                  {preset.title}
                </option>
              ))}
            </select>
            {selectedCandidate ? <p className="field-help">{selectedCandidate.description}</p> : null}
          </div>
        </div>

        <Accordion.Root type="single" collapsible className={styles.accordion}>
          <Accordion.Item value="fine-trim" className={styles.accordionItem}>
            <Accordion.Trigger className={styles.accordionTrigger}>
              Plan adjustments
            </Accordion.Trigger>
            <Accordion.Content className={styles.accordionContent}>
              <p className="field-help">
                Apply bounded trims on top of the selected operating condition and operating plan
                before sending the comparison to the API.
              </p>
              <div className={styles.sliderGroup}>
                {editableControlNames.map((controlName) => {
                  const range = controlAdjustmentRange(controlName);
                  return (
                    <label key={controlName} className={styles.sliderField}>
                      <span className={styles.sliderHead}>
                        <span>{controlName} trim</span>
                        <strong>{formatMetric(draft.controlAdjustments[controlName])}</strong>
                      </span>
                      <input
                        className="range-shell"
                        type="range"
                        min={-range}
                        max={range}
                        step={range / 50 || 0.01}
                        value={draft.controlAdjustments[controlName] ?? 0}
                        onChange={(event) =>
                          setControlAdjustment(controlName, Number(event.target.value))
                        }
                      />
                    </label>
                  );
                })}
                {demo.system_spec.disturbance_names.map((disturbanceName) => {
                  const range = disturbanceAdjustmentRange(disturbanceName);
                  return (
                    <label key={disturbanceName} className={styles.sliderField}>
                      <span className={styles.sliderHead}>
                        <span>{disturbanceName} trim</span>
                        <strong>{formatMetric(draft.disturbanceAdjustments[disturbanceName])}</strong>
                      </span>
                      <input
                        className="range-shell"
                        type="range"
                        min={-range}
                        max={range}
                        step={range / 50 || 0.01}
                        value={draft.disturbanceAdjustments[disturbanceName] ?? 0}
                        onChange={(event) =>
                          setDisturbanceAdjustment(disturbanceName, Number(event.target.value))
                        }
                      />
                    </label>
                  );
                })}
              </div>
            </Accordion.Content>
          </Accordion.Item>
        </Accordion.Root>

        <div className={styles.targetBlock}>
          <span className="field-label">Success targets</span>
          {demo.highlight_states.map((stateName) => (
            <p key={stateName} className={styles.targetValue}>
              {stateName}: {formatMetric(demo.target_state[stateName])}
            </p>
          ))}
        </div>

        <div className={styles.buttonRow}>
          <button
            className="button-primary"
            type="button"
            disabled={comparisonPending}
            onClick={() => void runScenario()}
          >
            <Play size={16} aria-hidden="true" />
            {comparisonPending ? 'Running…' : demo.run_button_label}
          </button>
          <button
            className="button-secondary"
            type="button"
            disabled={optimizationPending}
            onClick={() => void runOptimizeScenario()}
          >
            <Target size={16} aria-hidden="true" />
            {optimizationPending ? 'Optimising…' : demo.optimize_button_label}
          </button>
        </div>
      </aside>

      <section className={clsx(styles.visualPanel, 'glass-panel')}>
        {comparisonError instanceof Error ? (
          <div className="status-note status-note--warning">{comparisonError.message}</div>
        ) : null}

        {comparison ? (
          <>
            <Suspense fallback={<div className="status-note">Loading chart…</div>}>
              <TrajectoryChart
                demo={demo}
                comparison={comparison}
                comparisonRequest={comparisonRequest}
                optimizedResult={optimizedResult}
              />
            </Suspense>

            <div className="metric-grid">
              <MetricCard
                label="Forecast Runtime"
                value={forecastRuntime}
              />
              {demo.highlight_states.map((stateName) => (
                <MetricCard
                  key={stateName}
                  label={`${stateName} vs Baseline`}
                  value={formatMetric(comparison.summary.candidate_advantage[stateName])}
                />
              ))}
              <MetricCard
                label="Constraint Risk"
                value={formatMetric(constraintRisk)}
                help="Fraction of bounded-state samples outside their allowed region. Lower is better."
              />
            </div>

            <div className="helper-note">
              {comparison.candidate_source === 'universal_model'
                ? 'The shared runtime is generating the forecast bands for the baseline and alternative plans. The recommendation then searches for a nearby stabilization schedule you could review with operations.'
                : 'This run is using simulator ensembles because the shared runtime is unavailable. The comparison flow stays the same, but the uncertainty bands are coming from the physical simulator.'}
            </div>

            {optimizedResult ? (
              <div className={styles.optimizedBlock}>
                <h4 className={styles.optimizedTitle}>Recommended stabilization plan</h4>
                <div className="helper-note">
                  {optimizedResult.source === 'model'
                    ? 'Source: digital-twin model. The winning schedule was re-evaluated with the loaded runtime.'
                    : 'Source: physical simulator. The winning schedule was scored with the process simulator.'}
                </div>
                <div className="metric-grid">
                  {editableControlNames.map((controlName) => {
                    const index = demo.system_spec.control_names.indexOf(controlName);
                    const sequence = optimizedResult.control_sequence.map((row) => row[index]);
                    return (
                      <MetricCard
                        key={controlName}
                        label={`${controlName} Final Setting`}
                        value={formatMetric(sequence[sequence.length - 1])}
                      />
                    );
                  })}
                  <MetricCard
                    label="Residual Target Gap"
                    value={formatMetric(optimizedResult.objective)}
                    help="Lower means the recommended plan finishes closer to the selected success targets."
                  />
                </div>
              </div>
            ) : null}
          </>
        ) : (
          <div className="status-note">
            Run the first plan comparison to populate the forecast and control views.
          </div>
        )}

        {optimizationError instanceof Error ? (
          <div className="status-note status-note--warning">{optimizationError.message}</div>
        ) : null}
      </section>
    </div>
  );
}
