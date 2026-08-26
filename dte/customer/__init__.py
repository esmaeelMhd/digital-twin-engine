"""Customer onboarding, template matching, and adaptation helpers."""

from dte.customer.adaptation import load_universal_sources, run_customer_adaptation
from dte.customer.onboarding_schema import (
    CustomerMeasurementSpec,
    CustomerOnboardingSpec,
    CustomerSignalSpec,
    CustomerStreamSpec,
    CustomerUnitSpec,
    OperatingRange,
    load_onboarding_spec,
)
from dte.customer.reporting import (
    generate_customer_validation_report,
    render_validation_report_markdown,
)
from dte.evaluation.universal import predict_rollout_samples
from dte.customer.template_matching import TemplateMatch, TemplateMatchResult, match_customer_templates

__all__ = [
    "CustomerMeasurementSpec",
    "CustomerOnboardingSpec",
    "CustomerSignalSpec",
    "CustomerStreamSpec",
    "CustomerUnitSpec",
    "OperatingRange",
    "TemplateMatch",
    "TemplateMatchResult",
    "generate_customer_validation_report",
    "load_onboarding_spec",
    "load_universal_sources",
    "match_customer_templates",
    "predict_rollout_samples",
    "render_validation_report_markdown",
    "run_customer_adaptation",
]
