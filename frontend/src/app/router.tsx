import { createBrowserRouter } from 'react-router-dom';

import { App } from './App';
import { HomePage } from './HomePage';
import { OnboardingPage } from '../features/onboarding/OnboardingPage';

export const router = createBrowserRouter([
  {
    element: <App />,
    children: [
      {
        path: '/',
        element: <HomePage />,
      },
      {
        path: '/onboard',
        element: <OnboardingPage />,
      },
      {
        path: '/onboard/jobs/:jobId',
        element: <OnboardingPage />,
      },
    ],
  },
]);
