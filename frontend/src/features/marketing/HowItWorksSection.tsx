import { Section } from '../../components/Section';

import { howItWorksSteps } from './constants';
import styles from './marketing.module.css';

export function HowItWorksSection() {
  return (
    <Section
      title="How It Works"
      subtitle="From raw historian exports to a live uncertainty-aware twin in three steps."
    >
      <div className={styles.stepsGrid}>
        {howItWorksSteps.map((item) => (
          <article key={item.step} className={styles.stepCard}>
            <span className={styles.stepNumber}>{item.step}</span>
            <h3 className={styles.cardTitle}>{item.title}</h3>
            <p className={styles.cardBody}>{item.body}</p>
          </article>
        ))}
      </div>
    </Section>
  );
}
