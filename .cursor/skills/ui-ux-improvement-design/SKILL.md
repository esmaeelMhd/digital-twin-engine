---
name: ui-ux-improvement-design
description: Reviews and improves user interfaces for usability, clarity, visual hierarchy, and interaction quality, then translates recommendations into practical frontend changes. Use when users request UI/UX improvements, design polish, better user flows, Streamlit dashboard upgrades, or frontend refactors that affect user experience.
---

# UI/UX Improvement Design

## Purpose

Use this skill to:
- evaluate screens, components, and flows from a user perspective
- identify high-impact usability and design issues
- propose practical implementation changes in code
- communicate recommendations in prioritized findings

## Default Mode

When this skill is active:
1. Inspect the current UI behavior and structure.
2. Evaluate against core UX heuristics and visual design basics.
3. Produce prioritized findings.
4. Provide concrete fixes that can be implemented immediately.

Default style profile: **Marketing Demo** (advisory).
- Optimize for polished storytelling, visual impact, and clear narrative flow.
- Treat this as guidance, not a hard constraint; deviate when user goals require.

## Core Evaluation Lenses

Use these lenses to review UI/UX quality:
- **Clarity**: labels, copy, affordances, feedback, empty states
- **Hierarchy**: spacing, contrast, typography scale, CTA prominence
- **Flow**: task steps, friction points, form burden, navigation predictability
- **Consistency**: reusable patterns, component behavior, naming, spacing system
- **Accessibility**: keyboard flow, focus visibility, semantic structure, readable contrast
- **Responsiveness**: behavior across common viewport sizes and density states

## Marketing Demo Design Heuristics

When the goal is presentation quality:
- lead each page with a strong narrative anchor (headline + value statement + primary CTA)
- prioritize visual rhythm: section pacing, whitespace, and deliberate emphasis
- keep copy punchy and scannable; replace dense text with structured highlights
- use color intentionally to communicate meaning, not decoration
- ensure "wow" does not break usability: preserve legibility and action clarity
- keep transitions/animations subtle and purposeful

## Output Format

Always present findings in priority order:

### Prioritized Findings
- **Critical**: blocks task completion, causes major confusion, or accessibility failure
- **High**: strong usability friction or likely conversion drop
- **Medium**: noticeable polish/clarity issue
- **Low**: minor refinements

For each finding, include:
1. **Issue**: what is wrong and where
2. **Why it matters**: user impact
3. **Recommended fix**: practical change
4. **Implementation hint**: component/page-level guidance
5. **Demo impact**: how the fix improves storytelling or audience confidence

## Implementation Guidance

When turning findings into code changes:
- favor small, testable, reversible edits
- preserve existing design system tokens/components where possible
- avoid introducing one-off styles unless justified
- improve semantics and interaction states with each visual change
- validate empty/loading/error states, not only happy paths

## Stack-Specific Guidance (This Repo)

Primary UI stack in this repository:
- Streamlit apps under `app/`
- Plotly charts for interactive visuals
- Python-first UI logic with cached data/runtime loaders

When reviewing or improving these interfaces:
- prefer Streamlit-native primitives (`st.form`, `st.columns`, `st.tabs`, `st.expander`) before custom HTML/CSS
- keep custom CSS scoped and minimal; avoid brittle selectors that depend on internal Streamlit class names
- ensure long pages preserve flow with clear section titles, progressive disclosure, and concise helper copy
- treat expensive recomputation as UX debt; use cache boundaries intentionally to keep interactions fast
- validate chart readability (titles, axis labels, units, legends, color contrast, hover text signal-to-noise)
- ensure state changes are explicit (loading, auth required, no data, partial data, success, error)

## Streamlit UX Checklist

- Is the first viewport clear about what users can do next?
- Are key controls grouped in sidebars/forms rather than scattered?
- Does each form have clear defaults and submit semantics?
- Are `st.error`, `st.warning`, and `st.info` messages actionable?
- Is rerun behavior predictable and free of accidental state loss?
- Does keyboard navigation remain usable for core tasks?

## Plotly UX Checklist

- Do chart titles answer "what am I looking at?" quickly?
- Are units visible on axes and metrics?
- Are colors meaningful and consistent across charts?
- Is hover information concise and decision-relevant?
- Are multi-series charts visually separable without overload?
- Do subplot layouts remain legible on smaller widths?

## Quick Heuristics Checklist

- Is the primary action obvious within 3 seconds?
- Can a keyboard user complete the core task?
- Are validation and error messages specific and actionable?
- Is spacing and typography consistent within each section?
- Does each screen have a clear state: loading, empty, success, error?
- Does the UI avoid unnecessary cognitive load?

## Response Template

Use this structure when reporting:

```markdown
## UI/UX Findings

- [Severity] [Area/Component]: Issue
  - Why it matters: ...
  - Fix: ...
  - Implementation: ...
  - Demo impact: ...

## Suggested Implementation Order

1. ...
2. ...
3. ...
```

## Optional Scoring Rubric

When useful, score each area from 1-5:
- Narrative clarity
- Visual polish
- Interaction confidence
- Accessibility baseline
- Performance perception

## Profile Switching

Supported profiles:
- `marketing-demo` (default): visual storytelling and polished presentation
- `minimal-technical`: dense, precise, low-ornament interfaces
- `executive-dashboard`: fast decision support, KPI-first hierarchy
- `hybrid-balanced`: practical and clean for everyday product use

Switching rule:
- If the user explicitly requests a profile, use that profile immediately.
- If no profile is specified, keep `marketing-demo` as advisory default.
- If a requested profile conflicts with product constraints, follow constraints and explain trade-offs.

Response note to include when switching:
- `Active profile: <profile-name>`

## Boundaries

- Prioritize user outcomes over stylistic preferences.
- Avoid broad redesign proposals unless requested.
- If context is limited, state assumptions before proposing major changes.
