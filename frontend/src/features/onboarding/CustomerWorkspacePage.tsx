import { Activity, ArrowLeft, TriangleAlert } from 'lucide-react';
import { Link, useParams } from 'react-router-dom';

import {
  compareOnboardingScenarios,
  optimizeOnboardingControl,
} from '../../api/client';
import { useOnboardingWorkspaceQuery } from '../../api/hooks';
import { MetricCard } from '../../components/MetricCard';
import { formatMetric, titleCase } from '../../lib/format';
import { DemoWorkspace } from '../demo/DemoWorkspace';

import styles from './onboarding.module.css';

export function CustomerWorkspacePage() {
  const { jobId } = useParams();
  const workspaceQuery = useOnboardingWorkspaceQuery(jobId ?? null);

  if (workspaceQuery.isLoading) {
    return (
      <main className={styles.page}>
        <section className={styles.hero}>
          <div className={styles.heroInner}>
            <Link to={jobId ? `/onboard/jobs/${jobId}` : '/onboard'} className={styles.backLink}>
              <ArrowLeft size={16} aria-hidden="true" />
              Back to onboarding job
            </Link>
            <span className="pill">Customer workspace</span>
            <h1 className={styles.heroTitle}>Preparing the adapted planning surface.</h1>
            <p className={styles.heroBody}>
              Loading the customer-specific forecast workspace and soft gate.
            </p>
          </div>
        </section>
        <div className={styles.pageInner}>
          <div className="status-note">Loading customer workspace…</div>
        </div>
      </main>
    );
  }

  if (workspaceQuery.isError || !workspaceQuery.data) {
    return (
      <main className={styles.page}>
        <section className={styles.hero}>
          <div className={styles.heroInner}>
            <Link to={jobId ? `/onboard/jobs/${jobId}` : '/onboard'} className={styles.backLink}>
              <ArrowLeft size={16} aria-hidden="true" />
              Back to onboarding job
            </Link>
            <span className="pill">Customer workspace</span>
            <h1 className={styles.heroTitle}>Customer workspace unavailable.</h1>
            <p className={styles.heroBody}>
              {workspaceQuery.error instanceof Error
                ? workspaceQuery.error.message
                : 'Could not load the customer workspace.'}
            </p>
          </div>
        </section>
      </main>
    );
  }

  const { job, gate, workspace } = workspaceQuery.data;

  return (
    <main className={styles.page}>
      <section className={styles.hero}>
        <div className={styles.heroInner}>
          <Link to={`/onboard/jobs/${job.job_id}`} className={styles.backLink}>
            <ArrowLeft size={16} aria-hidden="true" />
            Back to onboarding job
          </Link>
          <span className="pill">Customer workspace</span>
          <h1 className={styles.heroTitle}>{workspace.title}</h1>
          <p className={styles.heroBody}>
            {workspace.description}
          </p>
        </div>
      </section>

      <div className={styles.pageInner}>
        <section className={`${styles.stepCard} glass-panel`}>
          <div className={styles.stepHeader}>
            <p className="eyebrow">Planning Surface</p>
            <h2 className={styles.stepTitle}>Use the adapted checkpoint, not the public demo runtime.</h2>
            <p className={styles.stepBody}>
              This workspace keeps the familiar compare-and-optimize flow, but the forecasts are
              generated from the calibrated customer job.
            </p>
          </div>

          <div className="metric-grid">
            <MetricCard label="Job status" value={titleCase(job.status)} />
            <MetricCard label="Best validation loss" value={formatMetric(job.metrics.best_val_loss)} />
            <MetricCard label="Forecast RMSE" value={formatMetric(job.metrics.forecast_rmse)} />
            <MetricCard label="Rollout RMSE" value={formatMetric(job.metrics.rollout_rmse)} />
          </div>

          <div className={gate.status === 'warning' ? 'status-note status-note--warning' : 'status-note'}>
            <p>{gate.message}</p>
            {gate.forecast_rmse_ratio != null || gate.rollout_rmse_ratio != null ? (
              <p>
                Forecast ratio: {formatMetric(gate.forecast_rmse_ratio)}. Rollout ratio:{' '}
                {formatMetric(gate.rollout_rmse_ratio)}.
              </p>
            ) : null}
          </div>

          <div className={styles.jobNote}>
            {gate.status === 'warning' ? (
              <TriangleAlert size={16} aria-hidden="true" />
            ) : (
              <Activity size={16} aria-hidden="true" />
            )}
            <span>
              Soft gate only: the workspace stays available even when the fit is weak, but the
              warning banner should change how you interpret the forecast.
            </span>
          </div>
        </section>

        <DemoWorkspace
          demo={workspace}
          compareScenario={(payload) => compareOnboardingScenarios(job.job_id, payload)}
          optimizeScenario={(payload) => optimizeOnboardingControl(job.job_id, payload)}
        />
      </div>
    </main>
  );
}
