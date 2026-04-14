import { MetricCard } from '../../components/MetricCard';
import { Section } from '../../components/Section';
import type { DemoReleaseSnapshot } from '../../api/types';
import { formatMetric, titleCase } from '../../lib/format';

import styles from './marketing.module.css';

type ReleaseOverviewSectionProps = {
  release: DemoReleaseSnapshot;
};

export function ReleaseOverviewSection({ release }: ReleaseOverviewSectionProps) {
  return (
    <Section
      title="Proven Performance"
      subtitle="The shared checkpoint has been evaluated across multiple process families and a customer adaptation pilot."
    >
      <div className={styles.releaseGrid}>
        <MetricCard label="Release Gate" value={titleCase(release.milestone_status)} />
        <MetricCard label="Best Validation Loss" value={formatMetric(release.train_best_val_loss)} />
        <MetricCard
          label={release.eval_metric_name ?? 'Eval Metric'}
          value={formatMetric(release.eval_metric_value)}
        />
        <MetricCard
          label="Customer Adaptation"
          value={formatMetric(release.customer_best_val_loss)}
        />
      </div>

      {Object.keys(release.per_system_total_loss).length > 0 ? (
        <div className="metric-grid">
          {Object.entries(release.per_system_total_loss).map(([system, value]) => (
            <MetricCard key={system} label={titleCase(system)} value={formatMetric(value)} />
          ))}
        </div>
      ) : null}

      <div className={styles.releaseStatus}>
        {release.runtime_loaded ? (
          <div className="status-note">
            Shared runtime is loaded and the interactive demos are using universal-model rollouts.
          </div>
        ) : (
          <div className="status-note status-note--warning">
            Shared runtime is not loaded. The API will fall back to simulator ensembles where needed.
          </div>
        )}
      </div>
    </Section>
  );
}
