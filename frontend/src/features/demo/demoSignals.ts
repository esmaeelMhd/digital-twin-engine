import type { DemoDefinition, DemoProfile, DemoSystemSpec, NumericDict } from '../../api/types';

export type SignalKind = 'control' | 'disturbance';

export type ScenarioDraft = {
  selectedDisturbanceId: string;
  selectedCandidateId: string;
  controlAdjustments: NumericDict;
  disturbanceAdjustments: NumericDict;
};

function zeroMap(names: string[]) {
  return Object.fromEntries(names.map((name) => [name, 0]));
}

function applyNamedUpdates(
  names: string[],
  baseVector: number[],
  updates?: NumericDict,
) {
  const vector = [...baseVector];
  if (!updates) {
    return vector;
  }

  for (const [name, value] of Object.entries(updates)) {
    const index = names.indexOf(name);
    if (index >= 0) {
      vector[index] = value;
    }
  }
  return vector;
}

function clipByRanges(
  names: string[],
  ranges: Record<string, [number, number]>,
  sequence: number[][],
) {
  return sequence.map((step) =>
    step.map((value, index) => {
      const [lower, upper] = ranges[names[index]];
      return Math.min(Math.max(value, lower), upper);
    }),
  );
}

export function defaultDraft(demo: DemoDefinition): ScenarioDraft {
  return {
    selectedDisturbanceId: demo.disturbance_presets[0]?.id ?? '',
    selectedCandidateId: demo.candidate_profiles[0]?.id ?? '',
    controlAdjustments: zeroMap(demo.system_spec.control_names),
    disturbanceAdjustments: zeroMap(demo.system_spec.disturbance_names),
  };
}

export function cloneDraft(draft: ScenarioDraft): ScenarioDraft {
  return {
    selectedDisturbanceId: draft.selectedDisturbanceId,
    selectedCandidateId: draft.selectedCandidateId,
    controlAdjustments: { ...draft.controlAdjustments },
    disturbanceAdjustments: { ...draft.disturbanceAdjustments },
  };
}

export function namedVectorFromMap(
  names: string[],
  values: NumericDict,
  fallback: number[],
) {
  return names.map((name, index) => values[name] ?? fallback[index] ?? 0);
}

export function buildSignalSequence(
  spec: DemoSystemSpec,
  nSteps: number,
  signalKind: SignalKind,
  profile?: DemoProfile | null,
) {
  const names = signalKind === 'control' ? spec.control_names : spec.disturbance_names;
  const ranges = signalKind === 'control' ? spec.control_ranges : spec.disturbance_ranges;
  const base =
    signalKind === 'control'
      ? names.map((name) => {
          const [lower, upper] = ranges[name];
          return (lower + upper) / 2;
        })
      : [...spec.default_nominal_disturbance];

  if (!profile) {
    return Array.from({ length: nSteps }, () => [...base]);
  }

  const profileType = profile.type?.toLowerCase() ?? 'constant';

  if (profileType === 'constant') {
    const vector = applyNamedUpdates(names, base, profile.channels ?? profile.values);
    return clipByRanges(names, ranges, Array.from({ length: nSteps }, () => [...vector]));
  }

  if (profileType === 'ramp') {
    const start = applyNamedUpdates(names, base, profile.start ?? profile.channels);
    const end = applyNamedUpdates(names, start, profile.end);
    const sequence = Array.from({ length: nSteps }, (_, stepIndex) =>
      names.map((_, channelIndex) => {
        if (nSteps <= 1) {
          return start[channelIndex];
        }
        const ratio = stepIndex / (nSteps - 1);
        return start[channelIndex] + (end[channelIndex] - start[channelIndex]) * ratio;
      }),
    );
    return clipByRanges(names, ranges, sequence);
  }

  if (profileType === 'pulse') {
    const baseline = applyNamedUpdates(names, base, profile.base);
    const pulse = applyNamedUpdates(names, baseline, profile.pulse ?? profile.channels);
    const startStep = Math.max(profile.start_step ?? Math.max(Math.floor(nSteps / 3), 1), 0);
    const duration = Math.max(profile.duration ?? Math.max(Math.floor(nSteps / 5), 1), 1);
    const endStep = Math.min(startStep + duration, nSteps);
    const sequence = Array.from({ length: nSteps }, () => [...baseline]);
    for (let stepIndex = startStep; stepIndex < endStep; stepIndex += 1) {
      sequence[stepIndex] = [...pulse];
    }
    return clipByRanges(names, ranges, sequence);
  }

  return clipByRanges(names, ranges, Array.from({ length: nSteps }, () => [...base]));
}

export function clipAdjustedSequence(
  spec: DemoSystemSpec,
  sequence: number[][],
  adjustments: NumericDict,
  signalKind: SignalKind,
) {
  const names = signalKind === 'control' ? spec.control_names : spec.disturbance_names;
  const ranges = signalKind === 'control' ? spec.control_ranges : spec.disturbance_ranges;
  const adjusted = sequence.map((step) =>
    step.map((value, index) => {
      const name = names[index];
      const [lower, upper] = ranges[name];
      const delta = adjustments[name] ?? 0;
      return Math.min(Math.max(value + delta, lower), upper);
    }),
  );
  return adjusted;
}

export function buildScenarioPayloads(demo: DemoDefinition, draft: ScenarioDraft) {
  const spec = demo.system_spec;
  const initialState = namedVectorFromMap(
    spec.state_names,
    demo.initial_state,
    spec.default_initial_state,
  );
  const targetState = namedVectorFromMap(
    spec.state_names,
    demo.target_state,
    spec.default_initial_state,
  );
  const disturbancePreset = demo.disturbance_presets.find(
    (item) => item.id === draft.selectedDisturbanceId,
  );
  const candidatePreset = demo.candidate_profiles.find(
    (item) => item.id === draft.selectedCandidateId,
  );

  const baselineControls = buildSignalSequence(
    spec,
    demo.n_steps,
    'control',
    demo.baseline_control_profile,
  );
  const disturbances = clipAdjustedSequence(
    spec,
    buildSignalSequence(spec, demo.n_steps, 'disturbance', disturbancePreset?.profile),
    draft.disturbanceAdjustments,
    'disturbance',
  );
  const candidateControls = clipAdjustedSequence(
    spec,
    buildSignalSequence(
      spec,
      demo.n_steps,
      'control',
      candidatePreset?.profile ?? demo.baseline_control_profile,
    ),
    draft.controlAdjustments,
    'control',
  );

  return {
    compareRequest: {
      system: demo.system,
      initial_state: initialState,
      baseline_controls: baselineControls,
      candidate_controls: candidateControls,
      disturbances,
      dt: demo.dt,
      n_samples: 20,
      seed: 11,
    },
    optimizeRequest: {
      system: demo.system,
      initial_state: initialState,
      disturbances,
      reference_controls: candidateControls,
      target_state: targetState,
      tracked_state_names: demo.highlight_states,
      dt: demo.dt,
      n_candidates: demo.optimization.n_candidates,
      seed: demo.optimization.seed,
    },
  };
}
