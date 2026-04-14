import type {
  DemoCompareScenariosRequest,
  DemoCompareScenariosResponse,
  DemoOptimizeControlRequest,
  DemoOptimizeControlResponse,
  DemoPageResponse,
} from './types';

const apiBaseUrl =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ??
  'http://localhost:8000';

const apiKey = (import.meta.env.VITE_DTE_API_KEY as string | undefined)?.trim();

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(apiKey ? { 'X-API-Key': apiKey } : {}),
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        detail = payload.detail;
      }
    } catch {
      // Fall back to the generic error string above.
    }
    throw new Error(detail);
  }

  return (await response.json()) as T;
}

export function getDemoPage() {
  return request<DemoPageResponse>('/demo/page');
}

export function compareScenarios(payload: DemoCompareScenariosRequest) {
  return request<DemoCompareScenariosResponse>('/demo/compare_scenarios', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function optimizeControl(payload: DemoOptimizeControlRequest) {
  return request<DemoOptimizeControlResponse>('/demo/optimize_control', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}
