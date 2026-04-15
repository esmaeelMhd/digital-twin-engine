import { useMutation, useQuery } from '@tanstack/react-query';
import {
  Activity,
  ArrowLeft,
  CheckCircle2,
  FlaskConical,
  PlayCircle,
  ShieldCheck,
  UploadCloud,
} from 'lucide-react';
import type { ReactNode } from 'react';
import { useId, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import toast from 'react-hot-toast';
import { Link, useNavigate, useParams } from 'react-router-dom';

import type {
  OnboardingCreateJobRequest,
  OnboardingPreviewRequest,
  OnboardingTemplate,
  OnboardingUploadResponse,
} from '../../api/types';
import {
  createOnboardingJob,
  getOnboardingJobReport,
  previewOnboarding,
  uploadOnboardingFile,
} from '../../api/client';
import { useOnboardingJobQuery, useOnboardingTemplatesQuery } from '../../api/hooks';
import { MetricCard } from '../../components/MetricCard';
import { formatMetric, titleCase } from '../../lib/format';

import styles from './onboarding.module.css';

type ChannelMap = Record<string, string>;

function toggleSelection(name: string, values: string[]) {
  return values.includes(name)
    ? values.filter((item) => item !== name)
    : [...values, name];
}

function defaultChannelMap(names: string[], columns: string[]) {
  return Object.fromEntries(
    names.map((name) => [name, columns.includes(name) ? name : columns[0] ?? '']),
  );
}

function defaultTimestampColumn(columns: string[]) {
  if (columns.includes('timestamp')) {
    return 'timestamp';
  }
  if (columns.includes('time')) {
    return 'time';
  }
  return '';
}

type StepProps = {
  eyebrow: string;
  title: string;
  body: string;
  children: ReactNode;
};

function StepCard({ eyebrow, title, body, children }: StepProps) {
  return (
    <section className={`${styles.stepCard} glass-panel`}>
      <div className={styles.stepHeader}>
        <p className="eyebrow">{eyebrow}</p>
        <h2 className={styles.stepTitle}>{title}</h2>
        <p className={styles.stepBody}>{body}</p>
      </div>
      {children}
    </section>
  );
}

function applyTemplateSelection(
  template: OnboardingTemplate,
  upload: OnboardingUploadResponse | null,
  setSelectedTemplateId: (value: string) => void,
  setObjectiveStateNames: (value: string[]) => void,
  setControlVariableNames: (value: string[]) => void,
  setStateColumnMap: (value: ChannelMap) => void,
  setControlColumnMap: (value: ChannelMap) => void,
  setDisturbanceColumnMap: (value: ChannelMap) => void,
) {
  setSelectedTemplateId(template.id);
  setObjectiveStateNames(template.suggested_objectives);
  setControlVariableNames(template.suggested_controls);
  if (upload) {
    setStateColumnMap(defaultChannelMap(template.system_spec.state_names, upload.columns));
    setControlColumnMap(defaultChannelMap(template.system_spec.control_names, upload.columns));
    setDisturbanceColumnMap(
      defaultChannelMap(template.system_spec.disturbance_names, upload.columns),
    );
  }
}

export function OnboardingPage() {
  const navigate = useNavigate();
  const { jobId: routeJobId } = useParams();
  const templatesQuery = useOnboardingTemplatesQuery();
  const [selectedTemplateId, setSelectedTemplateId] = useState('');
  const [customerName, setCustomerName] = useState('Customer unit pilot');
  const [timestampColumn, setTimestampColumn] = useState('');
  const [dt, setDt] = useState(0.1);
  const [trajectoryDuration, setTrajectoryDuration] = useState(10);
  const [trajectoryStride, setTrajectoryStride] = useState(5);
  const [objectiveStateNames, setObjectiveStateNames] = useState<string[]>([]);
  const [controlVariableNames, setControlVariableNames] = useState<string[]>([]);
  const [stateColumnMap, setStateColumnMap] = useState<ChannelMap>({});
  const [controlColumnMap, setControlColumnMap] = useState<ChannelMap>({});
  const [disturbanceColumnMap, setDisturbanceColumnMap] = useState<ChannelMap>({});
  const [upload, setUpload] = useState<OnboardingUploadResponse | null>(null);
  const [previewId, setPreviewId] = useState<string | null>(null);
  const [previewSummary, setPreviewSummary] = useState<Record<string, unknown> | null>(null);
  const [previewErrors, setPreviewErrors] = useState<string[]>([]);
  const [previewWarnings, setPreviewWarnings] = useState<string[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const uploadInputRef = useRef<HTMLInputElement | null>(null);
  const uploadInputId = useId();
  const uploadStatusId = useId();

  const activeTemplateId = selectedTemplateId || templatesQuery.data?.templates[0]?.id || '';
  const selectedTemplate =
    templatesQuery.data?.templates.find((item) => item.id === activeTemplateId) ?? null;
  const activeJobId = routeJobId ?? jobId;

  const effectiveObjectiveStateNames =
    objectiveStateNames.length > 0
      ? objectiveStateNames
      : selectedTemplate?.suggested_objectives ?? [];
  const effectiveControlVariableNames =
    controlVariableNames.length > 0
      ? controlVariableNames
      : selectedTemplate?.suggested_controls ?? [];
  const effectiveStateColumnMap =
    Object.keys(stateColumnMap).length > 0 || !selectedTemplate || !upload
      ? stateColumnMap
      : defaultChannelMap(selectedTemplate.system_spec.state_names, upload.columns);
  const effectiveControlColumnMap =
    Object.keys(controlColumnMap).length > 0 || !selectedTemplate || !upload
      ? controlColumnMap
      : defaultChannelMap(selectedTemplate.system_spec.control_names, upload.columns);
  const effectiveDisturbanceColumnMap =
    Object.keys(disturbanceColumnMap).length > 0 || !selectedTemplate || !upload
      ? disturbanceColumnMap
      : defaultChannelMap(selectedTemplate.system_spec.disturbance_names, upload.columns);
  const effectiveTimestampColumn =
    timestampColumn || (upload ? defaultTimestampColumn(upload.columns) : '');

  const uploadMutation = useMutation({
    mutationFn: uploadOnboardingFile,
    onSuccess: (result) => {
      setUpload(result);
      setTimestampColumn(defaultTimestampColumn(result.columns));
      if (selectedTemplate) {
        setStateColumnMap(defaultChannelMap(selectedTemplate.system_spec.state_names, result.columns));
        setControlColumnMap(
          defaultChannelMap(selectedTemplate.system_spec.control_names, result.columns),
        );
        setDisturbanceColumnMap(
          defaultChannelMap(selectedTemplate.system_spec.disturbance_names, result.columns),
        );
      }
      setPreviewId(null);
      setPreviewSummary(null);
      setPreviewErrors([]);
      setPreviewWarnings([]);
      setJobId(null);
    },
  });

  const previewMutation = useMutation({
    mutationFn: previewOnboarding,
    onSuccess: (result) => {
      setPreviewId(result.preview_id ?? null);
      setPreviewSummary((result.ingestion_summary as Record<string, unknown> | null) ?? null);
      setPreviewErrors(result.blocking_errors);
      setPreviewWarnings(result.warnings);
      setJobId(null);
    },
  });

  const createJobMutation = useMutation({
    mutationFn: createOnboardingJob,
    onSuccess: (result) => {
      setJobId(result.job_id);
      navigate(`/onboard/jobs/${result.job_id}`);
    },
  });

  const jobQuery = useOnboardingJobQuery(activeJobId);
  const reportQuery = useQuery({
    queryKey: ['onboarding-job-report', activeJobId],
    queryFn: () => getOnboardingJobReport(activeJobId ?? ''),
    enabled: Boolean(activeJobId && jobQuery.data?.status === 'completed'),
  });

  async function handleUpload(file: File) {
    if (uploadMutation.isPending) {
      return;
    }
    await toast.promise(uploadMutation.mutateAsync(file), {
      loading: 'Uploading historian export…',
      success: 'Upload stored and columns detected.',
      error: (error) => (error instanceof Error ? error.message : 'Upload failed.'),
    });
  }

  async function handlePreview() {
    if (!upload || !selectedTemplate || previewMutation.isPending) {
      return;
    }
    const payload: OnboardingPreviewRequest = {
      upload_id: upload.upload_id,
      template_id: selectedTemplate.id,
      customer_name: customerName,
      timestamp_column: effectiveTimestampColumn || null,
      dt,
      trajectory_duration: trajectoryDuration,
      trajectory_stride: trajectoryStride,
      state_column_map: effectiveStateColumnMap,
      control_column_map: effectiveControlColumnMap,
      disturbance_column_map: effectiveDisturbanceColumnMap,
      objective_state_names: effectiveObjectiveStateNames,
      control_variable_names: effectiveControlVariableNames,
    };
    await toast.promise(previewMutation.mutateAsync(payload), {
      loading: 'Running ingestion preview…',
      success: (result) => (result.valid ? 'Preview completed.' : 'Preview found blocking issues.'),
      error: (error) => (error instanceof Error ? error.message : 'Preview failed.'),
    });
  }

  async function handleStartJob() {
    if (!previewId || createJobMutation.isPending) {
      return;
    }
    const payload: OnboardingCreateJobRequest = {
      preview_id: previewId,
      trainable_mode: 'adapters',
      tune_normalization: true,
      time_budget_minutes: 15,
    };
    await toast.promise(createJobMutation.mutateAsync(payload), {
      loading: 'Starting customer adaptation…',
      success: 'Adaptation job started.',
      error: (error) => (error instanceof Error ? error.message : 'Could not start the job.'),
    });
  }

  return (
    <main className={styles.page}>
      <section className={styles.hero}>
        <div className={styles.heroInner}>
          <Link to="/" className={styles.backLink}>
            <ArrowLeft size={16} aria-hidden="true" />
            Back to product overview
          </Link>
          <span className="pill">Customer onboarding MVP</span>
          <h1 className={styles.heroTitle}>Turn one real plant export into a bounded pilot.</h1>
          <p className={styles.heroBody}>
            This flow is intentionally narrow: choose a supported unit template, map real plant
            signals, validate the historian upload, then launch customer adaptation and review the
            fit report.
          </p>
        </div>
      </section>

      <div className={styles.pageInner}>
        <StepCard
          eyebrow="Step 1"
          title="Choose the pilot template"
          body="Keep the first version constrained to supported unit families so the runtime, data mapping, and validation loop stay defensible."
        >
          {templatesQuery.isLoading ? (
            <div className="status-note">Loading supported templates…</div>
          ) : templatesQuery.isError ? (
            <div className="status-note status-note--warning">
              {templatesQuery.error instanceof Error
                ? templatesQuery.error.message
                : 'Could not load templates.'}
            </div>
          ) : (
            <>
              <fieldset className={styles.templateFieldset}>
                <legend className={styles.templateLegend}>Supported unit templates</legend>
                <div className={styles.templateGrid}>
                {templatesQuery.data?.templates.map((template) => (
                  <label
                    key={template.id}
                    className={styles.templateCard}
                    data-selected={template.id === activeTemplateId}
                  >
                    <input
                      className={styles.srOnlyInput}
                      type="radio"
                      name="pilot-template"
                      checked={template.id === activeTemplateId}
                      onChange={() =>
                        applyTemplateSelection(
                          template,
                          upload,
                          setSelectedTemplateId,
                          setObjectiveStateNames,
                          setControlVariableNames,
                          setStateColumnMap,
                          setControlColumnMap,
                          setDisturbanceColumnMap,
                        )
                      }
                    />
                    <span className={styles.templateIcon}>
                      <FlaskConical size={18} aria-hidden="true" />
                    </span>
                    <strong>{template.title}</strong>
                    <span>{template.description}</span>
                  </label>
                ))}
                </div>
              </fieldset>

              <div className={styles.formGrid}>
                <label>
                  <span className="field-label">Customer asset name</span>
                  <input
                    className="input-shell"
                    value={customerName}
                    onChange={(event) => setCustomerName(event.target.value)}
                  />
                </label>
                <label>
                  <span className="field-label">Timestamp column</span>
                  <input
                    className="input-shell"
                    value={effectiveTimestampColumn}
                    onChange={(event) => setTimestampColumn(event.target.value)}
                    placeholder="timestamp or time"
                  />
                </label>
              </div>
            </>
          )}
        </StepCard>

        <StepCard
          eyebrow="Step 2"
          title="Pick the decision objective"
          body="The customer can refine the business goal here, but only within the template’s valid state and control channels."
        >
          {selectedTemplate ? (
            <div className={styles.selectionGrid}>
              <div>
                <span className="field-label">Objective states</span>
                <div className={styles.checkboxGrid}>
                  {selectedTemplate.system_spec.state_names.map((name) => (
                    <label key={name} className={styles.checkPill}>
                      <input
                        type="checkbox"
                        checked={effectiveObjectiveStateNames.includes(name)}
                        onChange={() =>
                          setObjectiveStateNames((current) =>
                            toggleSelection(
                              name,
                              current.length > 0 ? current : effectiveObjectiveStateNames,
                            ),
                          )
                        }
                      />
                      <span>{name}</span>
                    </label>
                  ))}
                </div>
              </div>
              <div>
                <span className="field-label">Manipulated variables</span>
                <div className={styles.checkboxGrid}>
                  {selectedTemplate.system_spec.control_names.map((name) => (
                    <label key={name} className={styles.checkPill}>
                      <input
                        type="checkbox"
                        checked={effectiveControlVariableNames.includes(name)}
                        onChange={() =>
                          setControlVariableNames((current) =>
                            toggleSelection(
                              name,
                              current.length > 0 ? current : effectiveControlVariableNames,
                            ),
                          )
                        }
                      />
                      <span>{name}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="status-note">Select a template first.</div>
          )}
        </StepCard>

        <StepCard
          eyebrow="Step 3"
          title="Upload the historian export"
          body="The upload step stores one CSV or Parquet file, detects the available columns, and prepares the signal-mapping surface."
        >
          <div className={styles.uploadRow}>
            <button
              className={styles.uploadButton}
              type="button"
              disabled={uploadMutation.isPending}
              aria-controls={uploadInputId}
              aria-describedby={uploadStatusId}
              onClick={() => uploadInputRef.current?.click()}
            >
              <UploadCloud size={16} aria-hidden="true" />
              <span>{upload ? 'Replace upload' : 'Upload CSV or Parquet'}</span>
            </button>
            <input
              ref={uploadInputRef}
              id={uploadInputId}
              className={styles.srOnlyInput}
              type="file"
              accept=".csv,.parquet"
              tabIndex={-1}
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) {
                  void handleUpload(file);
                }
                event.target.value = '';
              }}
            />
            <span id={uploadStatusId} className="field-help" aria-live="polite">
              {uploadMutation.isPending
                ? 'Reading upload…'
                : upload
                  ? `Selected file: ${upload.filename}`
                  : 'No file selected yet.'}
            </span>
          </div>

          {upload ? (
            <div className="metric-grid">
              <MetricCard label="File" value={upload.filename} />
              <MetricCard label="Format" value={titleCase(upload.detected_format)} />
              <MetricCard label="Columns" value={String(upload.columns.length)} />
              <MetricCard label="Rows" value={formatMetric(upload.row_count)} />
            </div>
          ) : (
            <div className="status-note">Upload one historian export to continue.</div>
          )}
        </StepCard>

        <StepCard
          eyebrow="Step 4"
          title="Map channels and validate the preview"
          body="The preview step runs ingestion now, not later. If the mapping, sampling, or trajectory extraction are wrong, the user should see it before adaptation starts."
        >
          {upload && selectedTemplate ? (
            <>
              <div className={styles.mappingGrid}>
                <div>
                  <h3 className={styles.mappingTitle}>State channels</h3>
                  {selectedTemplate.system_spec.state_names.map((name) => (
                    <label key={name} className={styles.mappingField}>
                      <span className="field-label">{name}</span>
                      <select
                        className="select-shell"
                        value={effectiveStateColumnMap[name] ?? ''}
                        onChange={(event) =>
                          setStateColumnMap((current) => ({
                            ...(Object.keys(current).length > 0 ? current : effectiveStateColumnMap),
                            [name]: event.target.value,
                          }))
                        }
                      >
                        {upload.columns.map((column) => (
                          <option key={`${name}-${column}`} value={column}>
                            {column}
                          </option>
                        ))}
                      </select>
                    </label>
                  ))}
                </div>
                <div>
                  <h3 className={styles.mappingTitle}>Control channels</h3>
                  {selectedTemplate.system_spec.control_names.map((name) => (
                    <label key={name} className={styles.mappingField}>
                      <span className="field-label">{name}</span>
                      <select
                        className="select-shell"
                        value={effectiveControlColumnMap[name] ?? ''}
                        onChange={(event) =>
                          setControlColumnMap((current) => ({
                            ...(Object.keys(current).length > 0
                              ? current
                              : effectiveControlColumnMap),
                            [name]: event.target.value,
                          }))
                        }
                      >
                        {upload.columns.map((column) => (
                          <option key={`${name}-${column}`} value={column}>
                            {column}
                          </option>
                        ))}
                      </select>
                    </label>
                  ))}
                </div>
                <div>
                  <h3 className={styles.mappingTitle}>Disturbance channels</h3>
                  {selectedTemplate.system_spec.disturbance_names.map((name) => (
                    <label key={name} className={styles.mappingField}>
                      <span className="field-label">{name}</span>
                      <select
                        className="select-shell"
                        value={effectiveDisturbanceColumnMap[name] ?? ''}
                        onChange={(event) =>
                          setDisturbanceColumnMap((current) => ({
                            ...(Object.keys(current).length > 0
                              ? current
                              : effectiveDisturbanceColumnMap),
                            [name]: event.target.value,
                          }))
                        }
                      >
                        {upload.columns.map((column) => (
                          <option key={`${name}-${column}`} value={column}>
                            {column}
                          </option>
                        ))}
                      </select>
                    </label>
                  ))}
                </div>
              </div>

              <div className={styles.formGrid}>
                <label>
                  <span className="field-label">Trajectory duration (s)</span>
                  <input
                    className="input-shell"
                    type="number"
                    min="0.1"
                    step="0.1"
                    value={trajectoryDuration}
                    onChange={(event) => setTrajectoryDuration(Number(event.target.value))}
                  />
                </label>
                <label>
                  <span className="field-label">Trajectory stride (s)</span>
                  <input
                    className="input-shell"
                    type="number"
                    min="0.1"
                    step="0.1"
                    value={trajectoryStride}
                    onChange={(event) => setTrajectoryStride(Number(event.target.value))}
                  />
                </label>
                <label>
                  <span className="field-label">Sampling interval dt (s)</span>
                  <input
                    className="input-shell"
                    type="number"
                    min="0.01"
                    step="0.01"
                    value={dt}
                    onChange={(event) => setDt(Number(event.target.value))}
                  />
                </label>
              </div>

              <div className={styles.actionRow}>
                <button
                  className="button-primary"
                  type="button"
                  disabled={previewMutation.isPending}
                  onClick={() => void handlePreview()}
                >
                  <ShieldCheck size={16} aria-hidden="true" />
                  {previewMutation.isPending ? 'Validating…' : 'Run preview'}
                </button>
              </div>

              {previewErrors.length > 0 ? (
                <div className="status-note status-note--warning">
                  {previewErrors.map((item) => (
                    <p key={item}>{item}</p>
                  ))}
                </div>
              ) : null}

              {previewWarnings.length > 0 ? (
                <div className="status-note">
                  {previewWarnings.map((item) => (
                    <p key={item}>{item}</p>
                  ))}
                </div>
              ) : null}

              {previewSummary ? (
                <div className="metric-grid">
                  <MetricCard
                    label="Trajectories"
                    value={formatMetric(Number(previewSummary['n_trajectories'] ?? 0))}
                  />
                  <MetricCard
                    label="Steps per window"
                    value={formatMetric(Number(previewSummary['n_steps_per_trajectory'] ?? 0))}
                  />
                  <MetricCard
                    label="Preview dt"
                    value={formatMetric(Number(previewSummary['dt'] ?? 0))}
                  />
                  <MetricCard
                    label="Plant seconds"
                    value={formatMetric(Number(previewSummary['t_total_seconds'] ?? 0))}
                  />
                </div>
              ) : null}
            </>
          ) : (
            <div className="status-note">Select a template and upload a file first.</div>
          )}
        </StepCard>

        <StepCard
          eyebrow="Step 5"
          title="Launch adaptation and review the report"
          body="The adaptation job reuses the preview output and runs asynchronously. The report view is the first safe stopping point before any customer-specific optimization is exposed."
        >
          <div className={styles.actionRow}>
            <button
              className="button-primary"
              type="button"
              disabled={!previewId || previewErrors.length > 0 || createJobMutation.isPending}
              onClick={() => void handleStartJob()}
            >
              <PlayCircle size={16} aria-hidden="true" />
              {createJobMutation.isPending ? 'Starting…' : 'Start adaptation'}
            </button>
          </div>

          {jobQuery.data ? (
            <>
              <div className="metric-grid">
                <MetricCard label="Job status" value={titleCase(jobQuery.data.status)} />
                <MetricCard label="Stage" value={titleCase(jobQuery.data.stage)} />
                <MetricCard
                  label="Best validation loss"
                  value={formatMetric(jobQuery.data.metrics.best_val_loss)}
                />
                <MetricCard
                  label="Best template"
                  value={jobQuery.data.metrics.best_unit_template ?? 'n/a'}
                />
              </div>

              <div className={styles.jobNote}>
                <Activity size={16} aria-hidden="true" />
                <span>
                  {jobQuery.data.progress_message ??
                    'Waiting for the job runner to update the adaptation status.'}
                </span>
              </div>

              {jobQuery.data.error ? (
                <div className="status-note status-note--warning">{jobQuery.data.error}</div>
              ) : null}

              {reportQuery.data ? (
                <article className={styles.reportCard}>
                  <div className={styles.reportHeader}>
                    <CheckCircle2 size={18} aria-hidden="true" />
                    <div>
                      <h3 className={styles.reportTitle}>Validation report</h3>
                      <p className={styles.reportBody}>
                        Forecast and rollout errors are shown here before any customer-specific
                        planning surface is unlocked.
                      </p>
                    </div>
                  </div>
                  <div className="metric-grid">
                    <MetricCard
                      label="Forecast RMSE"
                      value={formatMetric(jobQuery.data.metrics.forecast_rmse)}
                    />
                    <MetricCard
                      label="Rollout RMSE"
                      value={formatMetric(jobQuery.data.metrics.rollout_rmse)}
                    />
                  </div>
                  {reportQuery.data.report_markdown ? (
                    <div className={styles.markdown}>
                      <ReactMarkdown>{reportQuery.data.report_markdown}</ReactMarkdown>
                    </div>
                  ) : null}
                  <div className={styles.actionRow}>
                    <Link className="button-primary" to={`/onboard/jobs/${jobQuery.data.job_id}/workspace`}>
                      Open planning workspace
                    </Link>
                  </div>
                </article>
              ) : null}
            </>
          ) : (
            <div className="status-note">
              Start an adaptation job to turn the validated preview into a customer report.
            </div>
          )}
        </StepCard>
      </div>
    </main>
  );
}
