import { lazy, Suspense } from 'react';
import * as Accordion from '@radix-ui/react-accordion';

import type { DemoDefinition } from '../../api/types';
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
};

export function DemoWorkspace({ demo }: DemoWorkspaceProps) {
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
    optimizeScenario,
    controlAdjustmentRange,
    disturbanceAdjustmentRange,
  } = useDemoScenario(demo);

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

  return (
    <div className={styles.workspace}>
      <aside className={`${styles.controlPanel} glass-panel`}>
        <div>
          <h3 className={styles.panelTitle}>{demo.title}</h3>
          <p className={styles.panelBody}>{demo.description}</p>
          {demo.operator_goal ? (
            <p className={styles.goalText}>Operator goal: {demo.operator_goal}</p>
          ) : null}
        </div>

        <div className={styles.fieldStack}>
          <div>
            <label className="field-label" htmlFor={`${demo.id}-disturbance`}>
              Disturbance preset
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
              Candidate operating move
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
              Fine trim
            </Accordion.Trigger>
            <Accordion.Content className={styles.accordionContent}>
              <p className="field-help">
                Apply bounded offsets on top of the preset trajectories before sending the scenario
                to the API.
              </p>
              <div className={styles.sliderGroup}>
                {demo.system_spec.control_names.map((controlName) => {
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
          <span className="field-label">Tracked target state</span>
          {demo.highlight_states.map((stateName) => (
            <p key={stateName} className={styles.targetValue}>
              {stateName}: {formatMetric(demo.target_state[stateName])}
            </p>
          ))}
        </div>

        <div className={styles.buttonRow}>
          <button className="button-primary" type="button" onClick={runScenario}>
            {comparisonPending ? 'Running…' : demo.run_button_label}
          </button>
          <button className="button-secondary" type="button" onClick={optimizeScenario}>
            {optimizationPending ? 'Optimising…' : demo.optimize_button_label}
          </button>
        </div>
      </aside>

      <section className={`${styles.visualPanel} glass-panel`}>
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
                label="Forecast Source"
                value={titleCase(comparison.candidate_source)}
              />
              {demo.highlight_states.map((stateName) => (
                <MetricCard
                  key={stateName}
                  label={`${stateName} Final Delta`}
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
                ? 'Uncertainty bands are coming from the shared checkpoint, while control recommendation still uses a lightweight search over the physical simulator.'
                : 'Release checkpoint is not available for this run, so the uncertainty bands are approximated from simulator ensembles.'}
            </div>

            {optimizedResult ? (
              <div className={styles.optimizedBlock}>
                <h4 className={styles.optimizedTitle}>Recommended sequence</h4>
                <div className="metric-grid">
                  {demo.system_spec.control_names.map((controlName, index) => {
                    const sequence = optimizedResult.control_sequence.map((row) => row[index]);
                    return (
                      <MetricCard
                        key={controlName}
                        label={`${controlName} End`}
                        value={formatMetric(sequence[sequence.length - 1])}
                      />
                    );
                  })}
                  <MetricCard
                    label="Objective"
                    value={formatMetric(optimizedResult.objective)}
                    help="Lower objective means the predicted trajectory ends closer to the target state."
                  />
                </div>
              </div>
            ) : null}
          </>
        ) : (
          <div className="status-note">Waiting for the first scenario result.</div>
        )}

        {optimizationError instanceof Error ? (
          <div className="status-note status-note--warning">{optimizationError.message}</div>
        ) : null}
      </section>
    </div>
  );
}
