import clsx from 'clsx';
import { ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';

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
        <div className={styles.footerActions}>
          <Link
            className={clsx('button-primary', styles.footerButton)}
            to="/onboard"
          >
            Start onboarding
            <ArrowRight size={16} aria-hidden="true" />
          </Link>
          <a
            className="button-secondary"
            href="mailto:s.mohammadi.rl@gmail.com?subject=Digital%20Twin%20Engine%20enquiry"
          >
            Talk to us
          </a>
        </div>
      </div>
    </Section>
  );
}
