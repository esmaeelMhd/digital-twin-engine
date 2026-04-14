import {
  BadgeCheck,
  ChartSpline,
  FlaskConical,
  TrendingUp,
} from 'lucide-react';

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
      title="Proof The Runtime Travels"
      subtitle="The shared checkpoint has already been evaluated across multiple unit families and a customer adaptation pilot, so the story is more than a concept demo."
    >
      <div className={styles.releaseGrid}>
        <MetricCard
          label="Milestone Gate"
          value={titleCase(release.milestone_status)}
          icon={<BadgeCheck size={16} aria-hidden="true" />}
        />
        <MetricCard
          label="Foundation Validation"
          value={formatMetric(release.train_best_val_loss)}
          icon={<TrendingUp size={16} aria-hidden="true" />}
        />
        <MetricCard
          label={release.eval_metric_name ?? 'Cross-System Eval'}
          value={formatMetric(release.eval_metric_value)}
          icon={<ChartSpline size={16} aria-hidden="true" />}
        />
        <MetricCard
          label="Customer Pilot Adaptation"
          value={formatMetric(release.customer_best_val_loss)}
          icon={<FlaskConical size={16} aria-hidden="true" />}
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
            The shared runtime is loaded, and the live comparison below is using universal-model
            rollouts rather than a mocked frontend-only surface.
          </div>
        ) : (
          <div className="status-note status-note--warning">
            Shared runtime is not loaded. The API will fall back to simulator ensembles where
            needed.
          </div>
        )}
      </div>
    </Section>
  );
}
