import { useEffect, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import toast from 'react-hot-toast';

import type {
  DemoCompareScenariosRequest,
  DemoCompareScenariosResponse,
  DemoDefinition,
  DemoOptimizeControlRequest,
  DemoOptimizeControlResponse,
} from '../../api/types';
import {
  buildScenarioPayloads,
  cloneDraft,
  defaultDraft,
  type ScenarioDraft,
} from './demoSignals';

type ScenarioApi = {
  compareScenario: (payload: DemoCompareScenariosRequest) => Promise<DemoCompareScenariosResponse>;
  optimizeScenario: (payload: DemoOptimizeControlRequest) => Promise<DemoOptimizeControlResponse>;
};

export function useDemoScenario(demo: DemoDefinition, api: ScenarioApi) {
  const [draft, setDraft] = useState<ScenarioDraft>(() => defaultDraft(demo));
  const [appliedDraft, setAppliedDraft] = useState<ScenarioDraft>(() => defaultDraft(demo));
  const [optimizedResult, setOptimizedResult] = useState<DemoOptimizeControlResponse | null>(null);
  const appliedPayloads = buildScenarioPayloads(demo, appliedDraft);

  const compareMutation = useMutation({
    mutationFn: api.compareScenario,
  });

  const optimizeMutation = useMutation({
    mutationFn: api.optimizeScenario,
    onSuccess: (result) => {
      setOptimizedResult(result);
    },
  });

  useEffect(() => {
    const initialDraft = defaultDraft(demo);
    setDraft(initialDraft);
    setAppliedDraft(initialDraft);
    setOptimizedResult(null);
    compareMutation.mutate(buildScenarioPayloads(demo, initialDraft).compareRequest);
    // compareMutation is stable enough for this effect in practice.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demo]);

  const setSelectedDisturbanceId = (selectedDisturbanceId: string) => {
    setDraft((current) => ({ ...current, selectedDisturbanceId }));
  };

  const setSelectedCandidateId = (selectedCandidateId: string) => {
    setDraft((current) => ({ ...current, selectedCandidateId }));
  };

  const setControlAdjustment = (name: string, value: number) => {
    setDraft((current) => ({
      ...current,
      controlAdjustments: {
        ...current.controlAdjustments,
        [name]: value,
      },
    }));
  };

  const setDisturbanceAdjustment = (name: string, value: number) => {
    setDraft((current) => ({
      ...current,
      disturbanceAdjustments: {
        ...current.disturbanceAdjustments,
        [name]: value,
      },
    }));
  };

  const runScenario = async () => {
    if (compareMutation.isPending) {
      return;
    }
    setOptimizedResult(null);
    const nextDraft = cloneDraft(draft);
    const payloads = buildScenarioPayloads(demo, nextDraft);
    setAppliedDraft(nextDraft);
    await toast.promise(compareMutation.mutateAsync(payloads.compareRequest), {
      loading: `Running ${demo.title} scenario…`,
      success: 'Forecast updated',
      error: (error) =>
        error instanceof Error ? error.message : 'Scenario comparison failed.',
    });
  };

  const optimizeScenario = async () => {
    if (optimizeMutation.isPending) {
      return;
    }
    const payloads = buildScenarioPayloads(demo, draft);
    await toast.promise(optimizeMutation.mutateAsync(payloads.optimizeRequest), {
      loading: `Searching for a stabilizing ${demo.title} sequence…`,
      success: 'Recommended control sequence ready',
      error: (error) =>
        error instanceof Error ? error.message : 'Optimization failed.',
    });
  };

  const controlAdjustmentRange = (ranges: Record<string, [number, number]>, name: string) => {
    const [lower, upper] = ranges[name];
    return 0.15 * (upper - lower);
  };

  return {
    draft,
    comparison: compareMutation.data,
    comparisonRequest: appliedPayloads.compareRequest as DemoCompareScenariosRequest,
    comparisonError: compareMutation.error,
    comparisonPending: compareMutation.isPending,
    optimizedResult,
    optimizationError: optimizeMutation.error,
    optimizationPending: optimizeMutation.isPending,
    setSelectedDisturbanceId,
    setSelectedCandidateId,
    setControlAdjustment,
    setDisturbanceAdjustment,
    runScenario,
    optimizeScenario,
    controlAdjustmentRange: (name: string) =>
      controlAdjustmentRange(demo.system_spec.control_ranges, name),
    disturbanceAdjustmentRange: (name: string) =>
      controlAdjustmentRange(demo.system_spec.disturbance_ranges, name),
  };
}
