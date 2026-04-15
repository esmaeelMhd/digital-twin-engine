import { useQuery } from '@tanstack/react-query';

import {
  getDemoPage,
  getOnboardingJob,
  getOnboardingTemplates,
  getOnboardingWorkspace,
} from './client';

export function useDemoPageQuery() {
  return useQuery({
    queryKey: ['demo-page'],
    queryFn: getDemoPage,
  });
}

export function useOnboardingTemplatesQuery() {
  return useQuery({
    queryKey: ['onboarding-templates'],
    queryFn: getOnboardingTemplates,
  });
}

export function useOnboardingJobQuery(jobId: string | null) {
  return useQuery({
    queryKey: ['onboarding-job', jobId],
    queryFn: () => getOnboardingJob(jobId ?? ''),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === 'queued' || status === 'running' ? 1500 : false;
    },
  });
}

export function useOnboardingWorkspaceQuery(jobId: string | null) {
  return useQuery({
    queryKey: ['onboarding-workspace', jobId],
    queryFn: () => getOnboardingWorkspace(jobId ?? ''),
    enabled: Boolean(jobId),
  });
}
