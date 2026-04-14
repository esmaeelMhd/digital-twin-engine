import type { DemoFlowsheetItem } from '../../api/types';

import styles from './roadmap.module.css';

type RoadmapSectionProps = {
  flowsheets: DemoFlowsheetItem[];
};

export function RoadmapSection({ flowsheets }: RoadmapSectionProps) {
  return (
    <div className={styles.roadmap}>
      <div className={styles.header}>
        <h3 className={styles.title}>What&apos;s Next: Multi-Unit Modelling</h3>
        <p className={styles.body}>
          V1 is intentionally unit-first. The next surface is plant-section modelling across linked
          unit operations and flowsheet-level interactions.
        </p>
      </div>

      <div className={styles.flowGrid}>
        {flowsheets.map((flowsheet) => (
          <article key={flowsheet.id} className={styles.flowCard}>
            <h4 className={styles.flowTitle}>{flowsheet.title}</h4>
            {flowsheet.description ? <p className={styles.flowBody}>{flowsheet.description}</p> : null}
            <div className={styles.unitRail}>
              {flowsheet.units.map((unit, index) => (
                <div key={`${flowsheet.id}-${unit.name}`} className={styles.unitStep}>
                  <span className={styles.unitNode}>{unit.name}</span>
                  {index < flowsheet.units.length - 1 ? <span className={styles.arrow}>→</span> : null}
                </div>
              ))}
            </div>
            <div className={styles.streamList}>
              {flowsheet.streams.map((stream) => (
                <div key={`${flowsheet.id}-${stream.name}`} className={styles.streamRow}>
                  <strong>{stream.name}</strong>
                  <span>
                    {stream.source} → {stream.target} · {stream.kind}
                  </span>
                </div>
              ))}
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
