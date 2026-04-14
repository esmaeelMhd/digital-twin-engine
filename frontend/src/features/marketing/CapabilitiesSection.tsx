import { Section } from '../../components/Section';

import { capabilityCards } from './constants';
import styles from './marketing.module.css';

export function CapabilitiesSection() {
  return (
    <Section
      title="Why Pilot Teams Buy In"
      subtitle="A first deployment needs more than a forecast. It has to preserve process context, adapt to plant behavior, and give operations a surface they can review together."
    >
      <div className={styles.cardGrid}>
        {capabilityCards.map(({ icon: Icon, title, body }) => (
          <article key={title} className={styles.capabilityCard}>
            <span className={styles.capabilityIcon}>
              <Icon size={22} aria-hidden="true" />
            </span>
            <h3 className={styles.cardTitle}>{title}</h3>
            <p className={styles.cardBody}>{body}</p>
          </article>
        ))}
      </div>
    </Section>
  );
}
