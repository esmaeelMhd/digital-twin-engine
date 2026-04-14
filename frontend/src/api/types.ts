export type NumericDict = Record<string, number>;

export type DemoProfile = {
  type: string;
  channels?: NumericDict;
  values?: NumericDict;
  start?: NumericDict;
  end?: NumericDict;
  base?: NumericDict;
  pulse?: NumericDict;
  start_step?: number;
  duration?: number;
};

export type DemoPreset = {
  id: string;
  title: string;
  description: string;
  profile?: DemoProfile | null;
};

export type DemoChannelSpec = {
  name: string;
  lower_bound?: number | null;
  upper_bound?: number | null;
  unit?: string | null;
  description?: string | null;
  role?: string | null;
};

export type DemoSystemSpec = {
  name: string;
  state_dim: number;
  control_dim: number;
  disturbance_dim: number;
  param_dim: number;
  state_names: string[];
  control_names: string[];
  disturbance_names: string[];
  default_initial_state: number[];
  default_nominal_disturbance: number[];
  control_ranges: Record<string, [number, number]>;
  disturbance_ranges: Record<string, [number, number]>;
  state_channels: DemoChannelSpec[];
  control_channels: DemoChannelSpec[];
  disturbance_channels: DemoChannelSpec[];
};

export type DemoOptimizationConfig = {
  n_candidates: number;
  seed: number;
};

export type DemoDefinition = {
  id: string;
  title: string;
  system: string;
  kind: string;
  description: string;
  operator_goal?: string | null;
  dt: number;
  n_steps: number;
  highlight_states: string[];
  target_state: NumericDict;
  initial_state: NumericDict;
  baseline_control_profile?: DemoProfile | null;
  disturbance_presets: DemoPreset[];
  candidate_profiles: DemoPreset[];
  optimization: DemoOptimizationConfig;
  run_button_label: string;
  optimize_button_label: string;
  system_spec: DemoSystemSpec;
};

export type DemoReleaseSnapshot = {
  release_label: string;
  model_available: boolean;
  config_available: boolean;
  runtime_samples: number;
  model_path?: string | null;
  config_path?: string | null;
  runtime_loaded: boolean;
  train_best_val_loss?: number | null;
  eval_metric_name?: string | null;
  eval_metric_value?: number | null;
  per_system_total_loss: NumericDict;
  milestone_status?: string | null;
  customer_status?: string | null;
  customer_best_unit_template?: string | null;
  customer_best_val_loss?: number | null;
  customer_forecast_rmse?: number | null;
  customer_rollout_rmse?: number | null;
  customer_report_path?: string | null;
  customer_report_exists: boolean;
  customer_report_markdown?: string | null;
};

export type DemoFlowsheetUnit = {
  name: string;
  family: string;
};

export type DemoFlowsheetStream = {
  name: string;
  source: string;
  target: string;
  kind: string;
};

export type DemoFlowsheetItem = {
  id: string;
  title: string;
  description?: string | null;
  units: DemoFlowsheetUnit[];
  streams: DemoFlowsheetStream[];
};

export type DemoPageResponse = {
  product_name: string;
  headline: string;
  summary: string;
  release: DemoReleaseSnapshot;
  demos: DemoDefinition[];
  flowsheets: DemoFlowsheetItem[];
};

export type DemoCompareScenariosRequest = {
  system: string;
  initial_state: number[];
  baseline_controls: number[][];
  candidate_controls: number[][];
  disturbances: number[][];
  dt: number;
  n_samples: number;
  seed: number;
};

export type DemoCompareScenariosResponse = {
  system: string;
  times: number[];
  baseline_source: string;
  candidate_source: string;
  state_names: string[];
  baseline_mean: number[][];
  candidate_mean: number[][];
  baseline_p05: number[][];
  baseline_p95: number[][];
  candidate_p05: number[][];
  candidate_p95: number[][];
  summary: {
    final_state_delta_norm: number;
    mean_abs_delta: NumericDict;
    candidate_advantage: NumericDict;
  };
  baseline_constraints: NumericDict;
  candidate_constraints: NumericDict;
};

export type DemoOptimizeControlRequest = {
  system: string;
  initial_state: number[];
  disturbances: number[][];
  reference_controls?: number[][];
  target_state: number[];
  tracked_state_names: string[];
  dt: number;
  n_candidates: number;
  seed: number;
};

export type DemoOptimizeControlResponse = {
  system: string;
  control_sequence: number[][];
  predicted_states: number[][];
  objective: number;
  tracked_state_names: string[];
  state_names: string[];
  constraint_summary: NumericDict;
};
