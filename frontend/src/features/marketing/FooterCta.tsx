import clsx from 'clsx';
import { ArrowRight } from 'lucide-react';

import { Section } from '../../components/Section';

import styles from './marketing.module.css';

export function FooterCta() {
  return (
    <Section>
      <div className={styles.footerCard}>
        <h2 className={styles.footerTitle}>Ready to twin your plant?</h2>
        <p className={styles.footerBody}>
          Tell us about your process, unit operations, and data availability. We can map
          the engine onto your system and show what adaptation would look like in practice.
        </p>
        <a
          className={clsx('button-primary', styles.footerButton)}
          href="mailto:s.mohammadi.rl@gmail.com?subject=Digital%20Twin%20Engine%20enquiry"
        >
          Get in touch
          <ArrowRight size={16} aria-hidden="true" />
        </a>
      </div>
    </Section>
  );
}
