import { render } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { DemoPageResponse } from '../../api/types';
import { HeroSection } from './HeroSection';

function createMatchMedia(matches: boolean) {
  return vi.fn().mockImplementation((query: string) => ({
    matches,
    media: query,
    onchange: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }));
}

const page = {
  product_name: 'Digital Twin Engine',
  headline: 'Decision support for one critical unit',
  summary: 'Plan one bounded pilot first.',
  release: {
    release_label: 'test',
    model_available: true,
    config_available: true,
    runtime_samples: 32,
    runtime_loaded: true,
    per_system_total_loss: {},
    customer_report_exists: false,
  },
  demos: [],
  flowsheets: [],
} as DemoPageResponse;

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('HeroSection', () => {
  it('animates the canvas when reduced motion is not requested', () => {
    const requestAnimationFrameMock = vi.fn(() => 1);
    const cancelAnimationFrameMock = vi.fn();
    const canvasContext = {
      clearRect: vi.fn(),
      setTransform: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      strokeStyle: '',
      fillStyle: '',
      lineWidth: 0,
    };

    vi.stubGlobal('matchMedia', createMatchMedia(false));
    vi.stubGlobal('requestAnimationFrame', requestAnimationFrameMock);
    vi.stubGlobal('cancelAnimationFrame', cancelAnimationFrameMock);
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(
      canvasContext as unknown as CanvasRenderingContext2D,
    );

    const { container } = render(<HeroSection page={page} />);

    expect(requestAnimationFrameMock).toHaveBeenCalled();
    expect(container.querySelector('canvas')).toHaveAttribute('aria-hidden', 'true');
  });

  it('renders a static background when reduced motion is requested', () => {
    const requestAnimationFrameMock = vi.fn(() => 1);
    const canvasContext = {
      clearRect: vi.fn(),
      setTransform: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      strokeStyle: '',
      fillStyle: '',
      lineWidth: 0,
    };

    vi.stubGlobal('matchMedia', createMatchMedia(true));
    vi.stubGlobal('requestAnimationFrame', requestAnimationFrameMock);
    vi.stubGlobal('cancelAnimationFrame', vi.fn());
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(
      canvasContext as unknown as CanvasRenderingContext2D,
    );

    render(<HeroSection page={page} />);

    expect(requestAnimationFrameMock).not.toHaveBeenCalled();
    expect(canvasContext.clearRect).toHaveBeenCalled();
  });
});
