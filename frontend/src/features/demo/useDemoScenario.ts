import { useEffect, useState } from 'react';
import { useMutation } from '@tanstack/react-query';

import { compareScenarios, optimizeControl } from '../../api/client';
import type { DemoCompareScenariosRequest, DemoDefinition, NumericDict } from '../../api/types';
import {
  buildScenarioPayloads,
  cloneDraft,
  defaultDraft,
  type ScenarioDraft,
} from './demoSignals';

export function useDemoScenario(demo: DemoDefinition) {
  const [draft, setDraft] = useState<ScenarioDraft>(() => defaultDraft(demo));
  const [appliedDraft, setAppliedDraft] = useState<ScenarioDraft>(() => defaultDraft(demo));
  const [optimizedResult, setOptimizedResult] = useState<Awaited<
    ReturnType<typeof optimizeControl>
  > | null>(null);
  const appliedPayloads = buildScenarioPayloads(demo, appliedDraft);

  const compareMutation = useMutation({
    mutationFn: compareScenarios,
  });

  const optimizeMutation = useMutation({
    mutationFn: optimizeControl,
    onSuccess: (result) => {
      setOptimizedResult(result);
    },
  });

  useEffect(() => {
    const initialDraft = defaultDraft(demo);
    setDraft(initialDraft);
    setAppliedDraft(initialDraft);
    setOptimizedResult(null);
  }, [demo]);

  useEffect(() => {
    compareMutation.mutate(appliedPayloads.compareRequest);
    // compareMutation is stable enough for this effect in practice.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demo, appliedDraft]);

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

  const runScenario = () => {
    setOptimizedResult(null);
    setAppliedDraft(cloneDraft(draft));
  };

  const optimizeScenario = () => {
    const payloads = buildScenarioPayloads(demo, draft);
    optimizeMutation.mutate(payloads.optimizeRequest);
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
