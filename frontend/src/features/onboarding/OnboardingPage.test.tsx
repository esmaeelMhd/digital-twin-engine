import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { OnboardingTemplate, OnboardingUploadResponse } from '../../api/types';

vi.mock('../../api/hooks', () => ({
  useOnboardingTemplatesQuery: vi.fn(),
  useOnboardingJobQuery: vi.fn(),
}));

vi.mock('../../api/client', () => ({
  uploadOnboardingFile: vi.fn(),
  previewOnboarding: vi.fn(),
  createOnboardingJob: vi.fn(),
  getOnboardingJobReport: vi.fn(),
}));

vi.mock('react-hot-toast', () => ({
  default: {
    promise: <T,>(promise: Promise<T>) => promise,
  },
}));

import * as apiClient from '../../api/client';
import * as apiHooks from '../../api/hooks';
import { OnboardingPage } from './OnboardingPage';

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function createTemplate(id: string, title: string, stateName: string): OnboardingTemplate {
  return {
    id,
    title,
    description: `${title} template`,
    system_spec: {
      name: id,
      state_dim: 1,
      control_dim: 1,
      disturbance_dim: 1,
      param_dim: 0,
      state_names: [stateName],
      control_names: ['coolant_flow'],
      disturbance_names: ['feed_rate'],
      default_initial_state: [1],
      default_nominal_disturbance: [1],
      control_ranges: { coolant_flow: [0, 10] },
      disturbance_ranges: { feed_rate: [0, 10] },
      state_channels: [],
      control_channels: [],
      disturbance_channels: [],
    },
    suggested_objectives: [stateName],
    suggested_controls: ['coolant_flow'],
  };
}

const templates = [
  createTemplate('cstr', 'Reactor Train', 'conversion'),
  createTemplate('hx', 'Heat Exchanger', 'outlet_temp'),
];

const uploadResponse: OnboardingUploadResponse = {
  upload_id: 'upload-1',
  filename: 'historian.csv',
  detected_format: 'csv',
  columns: ['timestamp', 'conversion', 'outlet_temp', 'coolant_flow', 'feed_rate'],
  row_count: 128,
  size_bytes: 2048,
};

function createQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
      mutations: {
        retry: false,
      },
    },
  });
}

function renderPage() {
  const queryClient = createQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/onboard']}>
        <Routes>
          <Route path="/onboard" element={<OnboardingPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.mocked(apiHooks.useOnboardingTemplatesQuery).mockReturnValue({
    data: { templates },
    isLoading: false,
    isError: false,
    error: null,
  } as never);
  vi.mocked(apiHooks.useOnboardingJobQuery).mockReturnValue({
    data: undefined,
    isLoading: false,
    isError: false,
    error: null,
  } as never);
  vi.mocked(apiClient.uploadOnboardingFile).mockResolvedValue(uploadResponse);
  vi.mocked(apiClient.previewOnboarding).mockResolvedValue({
    valid: true,
    preview_id: 'preview-1',
    blocking_errors: [],
    warnings: [],
    ingestion_summary: {
      n_trajectories: 10,
      n_steps_per_trajectory: 20,
      dt: 0.1,
      t_total_seconds: 200,
    },
  } as never);
  vi.mocked(apiClient.createOnboardingJob).mockResolvedValue({ job_id: 'job-1' } as never);
  vi.mocked(apiClient.getOnboardingJobReport).mockResolvedValue({ report_markdown: '' } as never);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('OnboardingPage', () => {
  it('renders an accessible radio group for templates and reports the selected upload', async () => {
    const { container } = renderPage();

    const reactorRadio = screen.getByRole('radio', { name: /reactor train/i });
    const exchangerRadio = screen.getByRole('radio', { name: /heat exchanger/i });
    expect(reactorRadio).toBeChecked();

    fireEvent.click(exchangerRadio);
    expect(exchangerRadio).toBeChecked();

    const fileInput = container.querySelector('input[type="file"]');
    expect(fileInput).not.toBeNull();

    const file = new File(['timestamp,conversion'], 'historian.csv', { type: 'text/csv' });
    fireEvent.change(fileInput as HTMLInputElement, {
      target: { files: [file] },
    });

    await waitFor(() => {
      expect(apiClient.uploadOnboardingFile).toHaveBeenCalledTimes(1);
    });
    expect(vi.mocked(apiClient.uploadOnboardingFile).mock.calls[0]?.[0]).toBe(file);
    expect(await screen.findByText(/selected file: historian\.csv/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /replace upload/i })).toBeInTheDocument();
  });

  it('disables preview while the request is in flight to prevent duplicate submissions', async () => {
    const previewRequest = deferred<{
      valid: boolean;
      preview_id: string;
      blocking_errors: string[];
      warnings: string[];
      ingestion_summary: Record<string, number>;
    }>();
    vi.mocked(apiClient.previewOnboarding).mockReturnValue(previewRequest.promise as never);

    const { container } = renderPage();
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(fileInput, {
      target: {
        files: [new File(['timestamp,conversion'], 'historian.csv', { type: 'text/csv' })],
      },
    });

    await screen.findByText(/selected file: historian\.csv/i);

    const previewButton = screen.getByRole('button', { name: /run preview/i });
    fireEvent.click(previewButton);

    await waitFor(() => {
      expect(apiClient.previewOnboarding).toHaveBeenCalledTimes(1);
    });
    expect(previewButton).toBeDisabled();

    fireEvent.click(previewButton);
    expect(apiClient.previewOnboarding).toHaveBeenCalledTimes(1);

    previewRequest.resolve({
      valid: true,
      preview_id: 'preview-1',
      blocking_errors: [],
      warnings: [],
      ingestion_summary: {
        n_trajectories: 12,
        n_steps_per_trajectory: 24,
        dt: 0.1,
        t_total_seconds: 240,
      },
    });

    await waitFor(() => {
      expect(previewButton).not.toBeDisabled();
    });
  });
});
