import type {
  DemoCompareScenariosRequest,
  DemoCompareScenariosResponse,
  DemoOptimizeControlRequest,
  DemoOptimizeControlResponse,
  DemoPageResponse,
  OnboardingCreateJobRequest,
  OnboardingJobReportResponse,
  OnboardingJobResponse,
  OnboardingPreviewRequest,
  OnboardingPreviewResponse,
  OnboardingTemplateListResponse,
  OnboardingUploadResponse,
  OnboardingWorkspaceResponse,
} from './types';

const apiBaseUrl =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ??
  'http://localhost:8000';

const apiKey = (import.meta.env.VITE_DTE_API_KEY as string | undefined)?.trim();

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const isFormData = typeof FormData !== 'undefined' && init?.body instanceof FormData;
  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...init,
    headers: {
      ...(isFormData ? {} : { 'Content-Type': 'application/json' }),
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

export function getOnboardingTemplates() {
  return request<OnboardingTemplateListResponse>('/onboarding/templates');
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

export function uploadOnboardingFile(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  return request<OnboardingUploadResponse>('/onboarding/uploads', {
    method: 'POST',
    body: formData,
  });
}

export function previewOnboarding(payload: OnboardingPreviewRequest) {
  return request<OnboardingPreviewResponse>('/onboarding/preview', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function createOnboardingJob(payload: OnboardingCreateJobRequest) {
  return request<OnboardingJobResponse>('/onboarding/jobs', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function getOnboardingJob(jobId: string) {
  return request<OnboardingJobResponse>(`/onboarding/jobs/${jobId}`);
}

export function getOnboardingJobReport(jobId: string) {
  return request<OnboardingJobReportResponse>(`/onboarding/jobs/${jobId}/report`);
}

export function getOnboardingWorkspace(jobId: string) {
  return request<OnboardingWorkspaceResponse>(`/onboarding/jobs/${jobId}/workspace`);
}

export function compareOnboardingScenarios(jobId: string, payload: DemoCompareScenariosRequest) {
  return request<DemoCompareScenariosResponse>(`/onboarding/jobs/${jobId}/compare_scenarios`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function optimizeOnboardingControl(jobId: string, payload: DemoOptimizeControlRequest) {
  return request<DemoOptimizeControlResponse>(`/onboarding/jobs/${jobId}/optimize_control`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}
