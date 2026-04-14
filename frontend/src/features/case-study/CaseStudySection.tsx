import {
  Activity,
  BadgeCheck,
  FlaskConical,
  Target,
  TrendingUp,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';

import type { DemoReleaseSnapshot } from '../../api/types';
import { MetricCard } from '../../components/MetricCard';
import { formatMetric, titleCase } from '../../lib/format';

import styles from './case-study.module.css';

type CaseStudySectionProps = {
  release: DemoReleaseSnapshot;
};

export function CaseStudySection({ release }: CaseStudySectionProps) {
  return (
    <div className={styles.caseStudy}>
      <div className={styles.header}>
        <h3 className={styles.title}>Pilot Proof: Customer Adaptation</h3>
        <p className={styles.body}>
          A customer historian export was matched to the closest unit template, adapted against
          real operating history, and validated before any broader deployment work.
        </p>
      </div>

      <div className="metric-grid">
        <MetricCard
          label="Pilot Status"
          value={titleCase(release.customer_status)}
          icon={<BadgeCheck size={16} aria-hidden="true" />}
        />
        <MetricCard
          label="Closest Template"
          value={titleCase(release.customer_best_unit_template)}
          icon={<FlaskConical size={16} aria-hidden="true" />}
        />
        <MetricCard
          label="Adapted Validation Loss"
          value={formatMetric(release.customer_best_val_loss)}
          icon={<Target size={16} aria-hidden="true" />}
        />
        <MetricCard
          label="Forecast RMSE"
          value={formatMetric(release.customer_forecast_rmse)}
          icon={<TrendingUp size={16} aria-hidden="true" />}
        />
        <MetricCard
          label="Rollout RMSE"
          value={formatMetric(release.customer_rollout_rmse)}
          icon={<Activity size={16} aria-hidden="true" />}
        />
      </div>

      {release.customer_report_markdown ? (
        <article className={styles.report}>
          <h4 className={styles.reportTitle}>Pilot validation report</h4>
          <div className={styles.markdown}>
            <ReactMarkdown>{release.customer_report_markdown}</ReactMarkdown>
          </div>
        </article>
      ) : (
        <div className="status-note status-note--warning">
          Customer validation report is not available in the configured release workspace.
        </div>
      )}
    </div>
  );
}
