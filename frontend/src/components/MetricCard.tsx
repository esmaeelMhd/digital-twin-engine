import type { ReactNode } from 'react';

type MetricCardProps = {
  label: string;
  value: ReactNode;
  help?: ReactNode;
  icon?: ReactNode;
};

export function MetricCard({ label, value, help, icon }: MetricCardProps) {
  return (
    <article className="metric-card">
      <div className="metric-card__head">
        <span className="metric-card__label">{label}</span>
        {icon ? <span className="metric-card__icon">{icon}</span> : null}
      </div>
      <strong className="metric-card__value">{value}</strong>
      {help ? <p className="metric-card__help">{help}</p> : null}
    </article>
  );
}
