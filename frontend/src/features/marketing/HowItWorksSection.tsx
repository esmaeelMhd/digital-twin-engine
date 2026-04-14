import { Section } from '../../components/Section';

import { howItWorksSteps } from './constants';
import styles from './marketing.module.css';

export function HowItWorksSection() {
  return (
    <Section
      title="What A First Pilot Looks Like"
      subtitle="The commercial path is narrow on purpose: one unit, one operating decision, and one review loop with engineering and operations."
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
