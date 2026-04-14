import type { ReactNode } from 'react';

type MetricCardProps = {
  label: string;
  value: ReactNode;
  help?: ReactNode;
};

export function MetricCard({ label, value, help }: MetricCardProps) {
  return (
    <article className="metric-card">
      <span className="metric-card__label">{label}</span>
      <strong className="metric-card__value">{value}</strong>
      {help ? <p className="metric-card__help">{help}</p> : null}
    </article>
  );
}
