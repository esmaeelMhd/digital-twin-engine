import React from 'react';
import ReactDOM from 'react-dom/client';

import { router } from './app/router';
import { AppProviders } from './app/providers';
import './styles/tokens.css';
import './styles/global.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AppProviders router={router} />
  </React.StrictMode>,
);
