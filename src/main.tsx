import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import {installMediaPipeLogFilter} from './lib/eyeTracker.ts';
import App from './App.tsx';
import './index.css';

installMediaPipeLogFilter();

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
