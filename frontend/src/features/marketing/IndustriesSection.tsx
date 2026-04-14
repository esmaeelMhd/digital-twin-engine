import { Section } from '../../components/Section';

import { industryCards } from './constants';
import styles from './marketing.module.css';

export function IndustriesSection() {
  return (
    <Section
      title="One Engine, Every Industry"
      subtitle="The same physics-informed architecture adapts to any continuous process family. Bring your historian data and use the same runtime surface across units."
    >
      <div className={styles.cardGrid}>
        {industryCards.map(([icon, title, body]) => (
          <article key={title} className={styles.industryCard}>
            <span className={styles.industryIcon}>{icon}</span>
            <h3 className={styles.cardTitle}>{title}</h3>
            <p className={styles.cardBody}>{body}</p>
          </article>
        ))}
      </div>
    </Section>
  );
}
