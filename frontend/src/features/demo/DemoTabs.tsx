import * as Tabs from '@radix-ui/react-tabs';

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
  const defaultValue = demos[0]?.id ?? 'case-study';

  return (
    <Section
      title="Try It Live"
      subtitle="Each workspace starts from a fixed baseline policy. Select a disturbance regime and an alternate operating move, then compare the resulting forecast and risk profile."
    >
      <Tabs.Root defaultValue={defaultValue} className={styles.tabsRoot}>
        <Tabs.List className={styles.tabList}>
          {demos.map((demo) => (
            <Tabs.Trigger key={demo.id} value={demo.id} className={styles.tabTrigger}>
              {demo.title}
            </Tabs.Trigger>
          ))}
          <Tabs.Trigger value="case-study" className={styles.tabTrigger}>
            Case Study
          </Tabs.Trigger>
          <Tabs.Trigger value="roadmap" className={styles.tabTrigger}>
            Roadmap
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
