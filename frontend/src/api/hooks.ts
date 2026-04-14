import { useQuery } from '@tanstack/react-query';

import { getDemoPage } from './client';

export function useDemoPageQuery() {
  return useQuery({
    queryKey: ['demo-page'],
    queryFn: getDemoPage,
  });
}
