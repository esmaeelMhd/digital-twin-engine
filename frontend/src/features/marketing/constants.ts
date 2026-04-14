export const industryCards = [
  ['⚗️', 'Chemicals', 'Reactor temperature, selectivity, and yield optimisation'],
  ['⚡', 'Energy', 'Load following, efficiency uplift, and emissions reduction'],
  ['💊', 'Pharma', 'Batch consistency, CQA prediction, and deviation prevention'],
  ['💧', 'Water & Utilities', 'Treatment optimisation, dosing, and demand forecasting'],
  ['🌾', 'Food & Beverage', 'Evaporation, drying, and fermentation control'],
  ['🔩', 'Metals & Materials', 'Furnace dynamics, rolling mill control, and quality windows'],
  ['🏭', 'Manufacturing', 'Thermal, fluid, and mechanical process twins'],
  ['🌲', 'Pulp & Paper', 'Digester, bleaching, and machine-section control'],
  ['⛏️', 'Mining', 'Flotation, leaching, and comminution optimisation'],
  ['🛢️', 'Oil & Gas', 'Separator, compressor, and pipeline network twins'],
] as const;

export const capabilityCards = [
  [
    '🧬',
    'Physics-Informed',
    'Mass, energy, and momentum residuals are embedded directly into the learning objective.',
  ],
  [
    '⚡',
    'Few-Shot Transfer',
    'Pre-trained unit knowledge adapts to plant-specific historian data in minutes instead of months.',
  ],
  [
    '📡',
    'Real-Time API',
    'FastAPI endpoints surface forecasts, risk, and control recommendations to external systems.',
  ],
  [
    '📊',
    'Uncertainty Quantified',
    'Neural SDE diffusion produces confidence bands on every forecast so operators know when to trust it.',
  ],
] as const;

export const howItWorksSteps = [
  {
    step: '01',
    title: 'Connect Your Data',
    body:
      'Ingest historian CSV or Parquet exports through a single CLI path. The engine validates and shapes the data into a training-ready HDF5 dataset.',
  },
  {
    step: '02',
    title: 'Adapt the Foundation Model',
    body:
      'A pre-trained physics-aware neural SDE is matched to your plant dynamics with minimal additional data.',
  },
  {
    step: '03',
    title: 'Forecast and Optimise Live',
    body:
      'The twin runs behind a FastAPI service and returns probabilistic trajectories, constraint risk, and recommended operating moves.',
  },
] as const;
