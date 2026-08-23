/**
 * Shared display formatting helpers.
 *
 * Reconstructed from call sites: the original module was never committed
 * because the repository-root .gitignore carried an unanchored `lib/` rule
 * from the standard Python template, which also matched `frontend/src/lib/`.
 */

/**
 * Format a numeric metric for display in the UI.
 *
 * Values span several orders of magnitude across the dashboards (timestep
 * sizes, trajectory counts, risk fractions), so the precision adapts rather
 * than fixing a single decimal count. Non-finite input renders as an em dash
 * instead of "NaN".
 */
export function formatMetric(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '—';
  }

  const magnitude = Math.abs(value);

  if (magnitude === 0) return '0';
  if (Number.isInteger(value) && magnitude < 1e6) return String(value);
  if (magnitude >= 1e6 || magnitude < 1e-3) return value.toExponential(2);
  if (magnitude >= 100) return value.toFixed(1);
  if (magnitude >= 1) return value.toFixed(2);
  return value.toFixed(3);
}

/**
 * Turn an identifier-style string into Title Case for display.
 *
 * Handles the snake_case and kebab-case values that arrive from the API
 * (status, stage, template names), e.g. "in_progress" -> "In Progress".
 */
export function titleCase(value: string | null | undefined): string {
  if (!value) return '';

  return value
    .replace(/[_-]+/g, ' ')
    .trim()
    .split(/\s+/)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
}
