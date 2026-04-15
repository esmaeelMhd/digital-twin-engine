import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import type { JSX } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type {
  DemoCompareScenariosResponse,
  DemoDefinition,
  DemoOptimizeControlResponse,
} from '../../api/types';

vi.mock('react-hot-toast', () => ({
  default: {
    promise: <T,>(promise: Promise<T>) => promise,
  },
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function createQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
}

function createDemo(): DemoDefinition {
  return {
    id: 'reactor-demo',
    title: 'Reactor Train',
    system: 'cstr',
    kind: 'demo',
    description: 'Compare alternative reactor policies.',
    operator_goal: 'Protect conversion',
    dt: 1,
    n_steps: 3,
    highlight_states: ['conversion'],
    target_state: { conversion: 0.95 },
    initial_state: { conversion: 0.8 },
    baseline_control_profile: {
      type: 'constant',
      channels: { coolant_flow: 4 },
    },
    disturbance_presets: [
      {
        id: 'steady',
        title: 'Steady feed',
        description: 'Nominal feed',
        profile: {
          type: 'constant',
          channels: { feed_rate: 6 },
        },
      },
    ],
    candidate_profiles: [
      {
        id: 'nominal',
        title: 'Nominal cooling',
        description: 'Standard cooling profile',
        profile: {
          type: 'constant',
          channels: { coolant_flow: 5 },
        },
      },
    ],
    optimization: {
      n_candidates: 12,
      seed: 17,
    },
    run_button_label: 'Run scenario',
    optimize_button_label: 'Optimize',
    editable_control_names: ['coolant_flow'],
    system_spec: {
      name: 'cstr',
      state_dim: 1,
      control_dim: 1,
      disturbance_dim: 1,
      param_dim: 0,
      state_names: ['conversion'],
      control_names: ['coolant_flow'],
      disturbance_names: ['feed_rate'],
      default_initial_state: [0.8],
      default_nominal_disturbance: [6],
      control_ranges: { coolant_flow: [0, 10] },
      disturbance_ranges: { feed_rate: [0, 10] },
      state_channels: [],
      control_channels: [],
      disturbance_channels: [],
    },
  };
}

const compareResponse: DemoCompareScenariosResponse = {
  system: 'cstr',
  times: [0, 1, 2],
  baseline_source: 'universal_model',
  candidate_source: 'universal_model',
  state_names: ['conversion'],
  baseline_mean: [[0.8], [0.82], [0.84]],
  candidate_mean: [[0.82], [0.86], [0.9]],
  baseline_p05: [[0.78], [0.8], [0.82]],
  baseline_p95: [[0.82], [0.84], [0.86]],
  candidate_p05: [[0.8], [0.84], [0.88]],
  candidate_p95: [[0.84], [0.88], [0.92]],
  summary: {
    final_state_delta_norm: 0.1,
    mean_abs_delta: { conversion: 0.05 },
    candidate_advantage: { conversion: 0.06 },
  },
  baseline_constraints: {
    above_upper_bound_rate: 0,
    below_lower_bound_rate: 0,
  },
  candidate_constraints: {
    above_upper_bound_rate: 0,
    below_lower_bound_rate: 0,
  },
};

const optimizeResponse: DemoOptimizeControlResponse = {
  system: 'cstr',
  control_sequence: [[5], [5], [5]],
  predicted_states: [[0.82], [0.86], [0.9]],
  objective: 0.04,
  tracked_state_names: ['conversion'],
  state_names: ['conversion'],
  constraint_summary: {
    above_upper_bound_rate: 0,
    below_lower_bound_rate: 0,
  },
};

afterEach(() => {
  vi.restoreAllMocks();
  vi.resetModules();
});

describe('DemoWorkspace lazy chart fallback', () => {
  it('shows the Suspense fallback while the chart module is still loading', async () => {
    const chartModule = deferred<{
      TrajectoryChart: () => JSX.Element;
    }>();
    vi.doMock('./TrajectoryChart', () => chartModule.promise);

    const { DemoWorkspace } = await import('./DemoWorkspace');
    const compareScenario = vi.fn().mockResolvedValue(compareResponse);
    const optimizeScenario = vi.fn().mockResolvedValue(optimizeResponse);
    const queryClient = createQueryClient();

    render(
      <QueryClientProvider client={queryClient}>
        <DemoWorkspace
          demo={createDemo()}
          compareScenario={compareScenario}
          optimizeScenario={optimizeScenario}
        />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(compareScenario).toHaveBeenCalled();
    });

    expect(await screen.findByText('Loading chart…')).toBeInTheDocument();

    chartModule.resolve({
      TrajectoryChart: () => <div>Deferred chart resolved</div>,
    });

    expect(await screen.findByText('Deferred chart resolved')).toBeInTheDocument();
  });
});
