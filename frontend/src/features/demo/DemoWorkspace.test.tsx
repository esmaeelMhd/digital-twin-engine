import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
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

vi.mock('./TrajectoryChart', () => ({
  TrajectoryChart: () => <div>Trajectory Ready</div>,
}));

import { DemoWorkspace } from './DemoWorkspace';

function createQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
}

function renderWorkspace(
  demo: DemoDefinition,
  compareScenario: (payload: unknown) => Promise<DemoCompareScenariosResponse>,
  optimizeScenario: (payload: unknown) => Promise<DemoOptimizeControlResponse>,
) {
  const queryClient = createQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <DemoWorkspace
        demo={demo}
        compareScenario={compareScenario as never}
        optimizeScenario={optimizeScenario as never}
      />
    </QueryClientProvider>,
  );
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
      {
        id: 'surge',
        title: 'Feed surge',
        description: 'Low feed edge case',
        profile: {
          type: 'constant',
          channels: { feed_rate: 0.5 },
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
      {
        id: 'aggressive',
        title: 'Aggressive cooling',
        description: 'Push toward the upper bound',
        profile: {
          type: 'constant',
          channels: { coolant_flow: 9.5 },
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
  control_sequence: [[10], [10], [10]],
  predicted_states: [[0.82], [0.88], [0.94]],
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
});

describe('DemoWorkspace', () => {
  it('builds compare and optimize payloads from the selected scenario inputs', async () => {
    const compareScenario = vi.fn().mockResolvedValue(compareResponse);
    const optimizeScenario = vi.fn().mockResolvedValue(optimizeResponse);

    renderWorkspace(createDemo(), compareScenario, optimizeScenario);

    await waitFor(() => {
      expect(compareScenario).toHaveBeenCalledTimes(1);
    });

    fireEvent.change(screen.getByLabelText(/operating condition/i), {
      target: { value: 'surge' },
    });
    fireEvent.change(screen.getByLabelText(/alternative operating plan/i), {
      target: { value: 'aggressive' },
    });

    fireEvent.click(screen.getByRole('button', { name: /plan adjustments/i }));
    fireEvent.change(screen.getByLabelText(/coolant_flow trim/i), {
      target: { value: '1.5' },
    });
    fireEvent.change(screen.getByLabelText(/feed_rate trim/i), {
      target: { value: '-1.5' },
    });

    fireEvent.click(screen.getByRole('button', { name: /run scenario/i }));

    await waitFor(() => {
      expect(compareScenario).toHaveBeenCalledTimes(2);
    });

    const comparePayload = compareScenario.mock.calls[1]?.[0] as {
      baseline_controls: number[][];
      candidate_controls: number[][];
      disturbances: number[][];
      dt: number;
      n_samples: number;
      seed: number;
    };

    expect(comparePayload.baseline_controls).toEqual([[4], [4], [4]]);
    expect(comparePayload.candidate_controls).toEqual([[10], [10], [10]]);
    expect(comparePayload.disturbances).toEqual([[0], [0], [0]]);
    expect(comparePayload.dt).toBe(1);
    expect(comparePayload.n_samples).toBe(20);
    expect(comparePayload.seed).toBe(11);

    fireEvent.click(screen.getByRole('button', { name: /optimize/i }));

    await waitFor(() => {
      expect(optimizeScenario).toHaveBeenCalledTimes(1);
    });

    const optimizePayload = optimizeScenario.mock.calls[0]?.[0] as {
      disturbances: number[][];
      reference_controls: number[][];
      active_control_names: string[];
      target_state: number[];
      tracked_state_names: string[];
      n_candidates: number;
      seed: number;
    };

    expect(optimizePayload.reference_controls).toEqual([[10], [10], [10]]);
    expect(optimizePayload.disturbances).toEqual([[0], [0], [0]]);
    expect(optimizePayload.active_control_names).toEqual(['coolant_flow']);
    expect(optimizePayload.target_state).toEqual([0.95]);
    expect(optimizePayload.tracked_state_names).toEqual(['conversion']);
    expect(optimizePayload.n_candidates).toBe(12);
    expect(optimizePayload.seed).toBe(17);
  });
});
