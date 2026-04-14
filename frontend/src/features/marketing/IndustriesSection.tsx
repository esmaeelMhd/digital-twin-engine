import { Section } from '../../components/Section';

import { industryCards } from './constants';
import styles from './marketing.module.css';

export function IndustriesSection() {
  return (
    <Section
      title="Where Teams Usually Start"
      subtitle="This is built for unit-level operating decisions in real plants. The first win is usually one painful unit, one recurring decision, and one team that needs a better answer than spreadsheets and intuition."
    >
      <div className={styles.cardGrid}>
        {industryCards.map(({ icon: Icon, title, body }) => (
          <article key={title} className={styles.industryCard}>
            <span className={styles.industryIcon}>
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
