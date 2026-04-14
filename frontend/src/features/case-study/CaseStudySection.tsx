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
        <h3 className={styles.title}>Case Study: Rapid Plant Adaptation</h3>
        <p className={styles.body}>
          A historian export was matched against the shared checkpoint, adapted, and validated in
          a single working session.
        </p>
      </div>

      <div className="metric-grid">
        <MetricCard
          label="Status"
          value={titleCase(release.customer_status)}
          icon={<BadgeCheck size={16} aria-hidden="true" />}
        />
        <MetricCard
          label="Best Template"
          value={titleCase(release.customer_best_unit_template)}
          icon={<FlaskConical size={16} aria-hidden="true" />}
        />
        <MetricCard
          label="Best Validation Loss"
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
          <h4 className={styles.reportTitle}>Validation report</h4>
          <div className={styles.markdown}>
            <ReactMarkdown>{release.customer_report_markdown}</ReactMarkdown>
          </div>
        </article>
      ) : (
        <div className="status-note status-note--warning">
          Customer validation report was not found in the configured release workspace.
        </div>
      )}
    </div>
  );
}
