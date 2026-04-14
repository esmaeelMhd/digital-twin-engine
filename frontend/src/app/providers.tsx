import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { RouterProvider, type RouterProviderProps } from 'react-router-dom';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 30_000,
    },
  },
});

type AppProvidersProps = {
  router: RouterProviderProps['router'];
};

export function AppProviders({ router }: AppProvidersProps) {
  return (
    <QueryClientProvider client={queryClient}>
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            border: '1px solid rgba(15, 26, 20, 0.1)',
            borderRadius: '18px',
            background: 'rgba(255, 255, 255, 0.95)',
            color: '#0f1a14',
            boxShadow: '0 18px 48px rgba(15, 26, 20, 0.12)',
          },
          success: {
            duration: 2600,
          },
        }}
      />
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}
