import type { PropsWithChildren, ReactNode } from 'react';

type SectionProps = PropsWithChildren<{
  id?: string;
  title?: string;
  subtitle?: string;
  aside?: ReactNode;
}>;

export function Section({ id, title, subtitle, aside, children }: SectionProps) {
  return (
    <section id={id} className="section-shell">
      <div className="section-shell__inner">
        {(title || subtitle || aside) && (
          <div className="section-head">
            <div>
              {title ? <h2 className="section-title">{title}</h2> : null}
              {subtitle ? <p className="section-subtitle">{subtitle}</p> : null}
            </div>
            {aside ? <div>{aside}</div> : null}
          </div>
        )}
        {children}
      </div>
    </section>
  );
}
