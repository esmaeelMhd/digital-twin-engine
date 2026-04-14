import type { LucideIcon } from 'lucide-react';
import {
  Activity,
  Cpu,
  Droplets,
  Factory,
  FlaskConical,
  Orbit,
  Pill,
  ShieldCheck,
  Zap,
} from 'lucide-react';

type MarketingCard = {
  icon: LucideIcon;
  title: string;
  body: string;
};

export const industryCards: MarketingCard[] = [
  {
    icon: FlaskConical,
    title: 'Chemicals and specialty process units',
    body: 'Start with one reactor, separator, or exchanger and show how forecast quality improves operating decisions before rollout.',
  },
  {
    icon: Zap,
    title: 'Energy and thermal assets',
    body: 'Model heat-duty, load-shift, and efficiency tradeoffs where a missed move shows up fast in fuel, power, or emissions.',
  },
  {
    icon: Pill,
    title: 'Pharma and quality-critical production',
    body: 'Surface probabilistic trajectories and deviation risk where operators need more than a point forecast.',
  },
  {
    icon: Droplets,
    title: 'Water, utilities, and environmental systems',
    body: 'Compare dosing, flow, and demand-response plans in a way that remains understandable to plant teams.',
  },
  {
    icon: Factory,
    title: 'Materials, mining, and heavy industry',
    body: 'Bring digital twins to thermal, fluid, and inventory-driven units that need process context, not generic ML dashboards.',
  },
  {
    icon: Cpu,
    title: 'Multi-unit operations on the road to scale-up',
    body: 'Begin with one unit, prove value, then extend toward linked assets and plant-section workflows without changing the core runtime surface.',
  },
];

export const capabilityCards: MarketingCard[] = [
  {
    icon: ShieldCheck,
    title: 'Forecasts grounded in process structure',
    body: 'Mass and energy consistency stay part of the training objective, so the model has to earn trust against plant behavior.',
  },
  {
    icon: Zap,
    title: 'Fast adaptation from real plant history',
    body: 'The foundation model is designed to adapt to plant-specific behavior with a bounded amount of additional data and engineering effort.',
  },
  {
    icon: Activity,
    title: 'Operator-facing risk, not just model output',
    body: 'The runtime returns uncertainty bands, scenario comparisons, and constraint risk so decisions can be reviewed instead of guessed.',
  },
  {
    icon: Orbit,
    title: 'Ready to sit behind live software',
    body: 'The same engine already exposes API-first forecast and recommendation workflows, which makes pilot integration materially easier.',
  },
];

export const howItWorksSteps = [
  {
    step: '01',
    title: 'Pick one unit and one decision problem',
    body:
      'Define the operating question that matters now: stabilize temperature, recover inventory, protect quality, or compare competing operating plans.',
  },
  {
    step: '02',
    title: 'Load historian data and adapt the runtime',
    body:
      'Historian exports are shaped into a training-ready dataset, then matched to the shared plant-model backbone instead of starting from scratch.',
  },
  {
    step: '03',
    title: 'Review plans with engineering and operations',
    body:
      'Use scenario comparison, forecast bands, and recommended stabilization moves to decide whether the pilot should move into a live software surface.',
  },
] as const;
