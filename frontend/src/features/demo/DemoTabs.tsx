import * as Tabs from '@radix-ui/react-tabs';
import { useLocation, useNavigate } from 'react-router-dom';

import type { DemoDefinition, DemoFlowsheetItem, DemoReleaseSnapshot } from '../../api/types';
import { Section } from '../../components/Section';
import { CaseStudySection } from '../case-study/CaseStudySection';
import { RoadmapSection } from '../roadmap/RoadmapSection';
import { DemoWorkspace } from './DemoWorkspace';

import styles from './demo.module.css';

type DemoTabsProps = {
  demos: DemoDefinition[];
  flowsheets: DemoFlowsheetItem[];
  release: DemoReleaseSnapshot;
};

export function DemoTabs({ demos, flowsheets, release }: DemoTabsProps) {
  const location = useLocation();
  const navigate = useNavigate();
  const defaultValue = demos[0]?.id ?? 'case-study';
  const tabValues = [...demos.map((demo) => demo.id), 'case-study', 'roadmap'];
  const requestedValue = new URLSearchParams(location.search).get('demo');
  const activeValue =
    requestedValue && tabValues.includes(requestedValue) ? requestedValue : defaultValue;

  const handleValueChange = (nextValue: string) => {
    const nextParams = new URLSearchParams(location.search);
    if (nextValue === defaultValue) {
      nextParams.delete('demo');
    } else {
      nextParams.set('demo', nextValue);
    }
    navigate(
      {
        pathname: location.pathname,
        search: nextParams.toString() ? `?${nextParams.toString()}` : '',
        hash: location.hash,
      },
      { replace: false },
    );
  };

  return (
    <Section
      title="Evaluate One Operating Decision Live"
      subtitle="Each workspace starts from the current operating plan, compares an alternative plan against the same disturbance, and recommends a stabilization plan if neither policy is strong enough."
    >
      <Tabs.Root value={activeValue} onValueChange={handleValueChange} className={styles.tabsRoot}>
        <Tabs.List className={styles.tabList}>
          {demos.map((demo) => (
            <Tabs.Trigger key={demo.id} value={demo.id} className={styles.tabTrigger}>
              {demo.title}
            </Tabs.Trigger>
          ))}
          <Tabs.Trigger value="case-study" className={styles.tabTrigger}>
            Pilot Proof
          </Tabs.Trigger>
          <Tabs.Trigger value="roadmap" className={styles.tabTrigger}>
            Scale-Up Path
          </Tabs.Trigger>
        </Tabs.List>

        {demos.map((demo) => (
          <Tabs.Content key={demo.id} value={demo.id} className={styles.tabContent}>
            <DemoWorkspace demo={demo} />
          </Tabs.Content>
        ))}

        <Tabs.Content value="case-study" className={styles.tabContent}>
          <CaseStudySection release={release} />
        </Tabs.Content>

        <Tabs.Content value="roadmap" className={styles.tabContent}>
          <RoadmapSection flowsheets={flowsheets} />
        </Tabs.Content>
      </Tabs.Root>
    </Section>
  );
}
