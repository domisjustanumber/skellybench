"""Model reports: provenance plus apples-to-apples scoring for one model artifact.

Export adapters supply plain dictionaries. The package validates, writes, skips complete reports,
and builds the on-disk overview.
"""

from __future__ import annotations

from .completeness import ModelReportCheck, check_model_report, read_model_report
from .overview import build_model_overview, write_model_overview
from .paths import (
    MODEL_OVERVIEW_STEM,
    MODEL_REPORT_HTML_SUFFIX,
    MODEL_REPORT_JSON_SUFFIX,
    ModelReportPaths,
    model_overview_paths,
    model_report_paths,
)
from .schema import (
    ACCURACY_SCENARIO,
    ACCURACY_STATUSES,
    COMPARISON_STATUSES,
    MODEL_OVERVIEW_KIND,
    MODEL_REPORT_KIND,
    MODEL_REPORT_SCHEMA_VERSION,
    PASS_IDS,
    REPORT_ID_PATTERN,
    THROUGHPUT_SCENARIO,
    ModelReportIdentity,
    ModelReportValidationError,
    build_model_report,
    is_measured_accuracy,
    merge_model_report,
    validate_model_report,
)
from .writer import (
    ModelReportWriteResult,
    ensure_model_report,
    render_model_report_file,
    rewrite_model_report_html,
    write_model_report,
)

__all__ = [
    "ACCURACY_SCENARIO",
    "ACCURACY_STATUSES",
    "COMPARISON_STATUSES",
    "MODEL_OVERVIEW_KIND",
    "MODEL_OVERVIEW_STEM",
    "MODEL_REPORT_HTML_SUFFIX",
    "MODEL_REPORT_JSON_SUFFIX",
    "MODEL_REPORT_KIND",
    "MODEL_REPORT_SCHEMA_VERSION",
    "PASS_IDS",
    "REPORT_ID_PATTERN",
    "THROUGHPUT_SCENARIO",
    "ModelReportCheck",
    "ModelReportIdentity",
    "ModelReportPaths",
    "ModelReportValidationError",
    "ModelReportWriteResult",
    "build_model_overview",
    "build_model_report",
    "check_model_report",
    "ensure_model_report",
    "is_measured_accuracy",
    "merge_model_report",
    "model_overview_paths",
    "model_report_paths",
    "read_model_report",
    "render_model_report_file",
    "rewrite_model_report_html",
    "validate_model_report",
    "write_model_overview",
    "write_model_report",
]
