import { useQuery } from '@tanstack/react-query';

import { getDemoPage, getOnboardingJob, getOnboardingTemplates } from './client';

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
