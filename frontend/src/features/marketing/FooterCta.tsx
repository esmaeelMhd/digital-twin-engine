import clsx from 'clsx';
import { ArrowRight } from 'lucide-react';

import { Section } from '../../components/Section';

import styles from './marketing.module.css';

export function FooterCta() {
  return (
    <Section>
      <div className={styles.footerCard}>
        <h2 className={styles.footerTitle}>Ready to scope a first pilot?</h2>
        <p className={styles.footerBody}>
          Tell us which unit is painful, what data is available, and which operating decision
          needs a better answer. The first step is usually a bounded pilot, not a platform rollout.
        </p>
        <a
          className={clsx('button-primary', styles.footerButton)}
          href="mailto:s.mohammadi.rl@gmail.com?subject=Digital%20Twin%20Engine%20enquiry"
        >
          Scope a pilot
          <ArrowRight size={16} aria-hidden="true" />
        </a>
      </div>
    </Section>
  );
}
