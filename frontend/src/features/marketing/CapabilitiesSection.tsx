import { Section } from '../../components/Section';

import { capabilityCards } from './constants';
import styles from './marketing.module.css';

export function CapabilitiesSection() {
  return (
    <Section
      title="What Makes It Different"
      subtitle="This is not a generic black-box model. The process structure still matters, and the training objective knows it."
    >
      <div className={styles.cardGrid}>
        {capabilityCards.map(([icon, title, body]) => (
          <article key={title} className={styles.capabilityCard}>
            <span className={styles.capabilityIcon}>{icon}</span>
            <h3 className={styles.cardTitle}>{title}</h3>
            <p className={styles.cardBody}>{body}</p>
          </article>
        ))}
      </div>
    </Section>
  );
}
