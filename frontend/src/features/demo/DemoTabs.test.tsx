import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { DemoDefinition, DemoFlowsheetItem, DemoReleaseSnapshot } from '../../api/types';

vi.mock('./DemoWorkspace', () => ({
  DemoWorkspace: ({ demo }: { demo: DemoDefinition }) => <div>{`Workspace ${demo.title}`}</div>,
}));

vi.mock('../case-study/CaseStudySection', () => ({
  CaseStudySection: () => <div>Pilot Proof Content</div>,
}));

vi.mock('../roadmap/RoadmapSection', () => ({
  RoadmapSection: () => <div>Roadmap Content</div>,
}));

import { DemoTabs } from './DemoTabs';

function LocationProbe() {
  const location = useLocation();
  return <div data-testid="location-search">{location.search || '(empty)'}</div>;
}

const demo = {
  id: 'cstr',
  title: 'Reactor Train',
  system: 'cstr',
  kind: 'demo',
  description: 'Reactor decision support',
  operator_goal: 'Protect conversion',
  dt: 1,
  n_steps: 8,
  highlight_states: ['conversion'],
  target_state: { conversion: 0.95 },
  initial_state: { conversion: 0.8 },
  baseline_control_profile: null,
  disturbance_presets: [],
  candidate_profiles: [],
  optimization: {
    n_candidates: 8,
    seed: 7,
  },
  run_button_label: 'Run scenario',
  optimize_button_label: 'Optimize',
  editable_control_names: ['coolant'],
  system_spec: {
    name: 'cstr',
    state_dim: 1,
    control_dim: 1,
    disturbance_dim: 1,
    param_dim: 0,
    state_names: ['conversion'],
    control_names: ['coolant'],
    disturbance_names: ['feed'],
    default_initial_state: [0.8],
    default_nominal_disturbance: [1],
    control_ranges: { coolant: [0, 1] },
    disturbance_ranges: { feed: [0, 2] },
    state_channels: [],
    control_channels: [],
    disturbance_channels: [],
  },
} as DemoDefinition;

const release = {
  release_label: 'test',
  model_available: true,
  config_available: true,
  runtime_samples: 16,
  runtime_loaded: true,
  per_system_total_loss: {},
  customer_report_exists: false,
} as DemoReleaseSnapshot;

const flowsheets = [
  {
    id: 'section-a',
    title: 'Section A',
    description: 'Primary train',
    units: [{ name: 'U-101', family: 'reactor' }],
    streams: [],
  },
] as DemoFlowsheetItem[];

afterEach(() => {
  vi.restoreAllMocks();
});

describe('DemoTabs', () => {
  it('uses the query string to select the initial tab', () => {
    render(
      <MemoryRouter initialEntries={['/?demo=roadmap']}>
        <DemoTabs demos={[demo]} flowsheets={flowsheets} release={release} />
      </MemoryRouter>,
    );

    expect(screen.getByRole('tab', { name: /scale-up path/i })).toHaveAttribute(
      'aria-selected',
      'true',
    );
    expect(screen.getByText('Roadmap Content')).toBeInTheDocument();
  });

  it('updates the location when the active tab changes and clears it for the default tab', async () => {
    render(
      <MemoryRouter initialEntries={['/?demo=roadmap']}>
        <DemoTabs demos={[demo]} flowsheets={flowsheets} release={release} />
        <LocationProbe />
      </MemoryRouter>,
    );

    const pilotProofTab = screen.getByRole('tab', { name: /pilot proof/i });
    fireEvent.mouseDown(pilotProofTab, { button: 0 });
    await waitFor(() => {
      expect(screen.getByTestId('location-search')).toHaveTextContent('?demo=case-study');
    });

    const reactorTab = screen.getByRole('tab', { name: /reactor train/i });
    fireEvent.mouseDown(reactorTab, { button: 0 });
    await waitFor(() => {
      expect(screen.getByRole('tab', { name: /reactor train/i })).toHaveAttribute(
        'aria-selected',
        'true',
      );
    });
    expect(screen.getByText('Workspace Reactor Train')).toBeInTheDocument();
    expect(screen.getByTestId('location-search')).toHaveTextContent('(empty)');
  });
});
