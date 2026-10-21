"""Stage timings and stage definitions (version 0.4.0)."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from conftest import (
    EVALUATION,
    NULL_STAGES,
    STAGE_DEFINITIONS,
    context,
    dataset,
    execution,
    performance_protocol,
    sha,
    stage_timing,
    subject,
)
from skellybench import SkellybenchValidationError
from skellybench.protocol import PerformanceProtocol, accuracy_protocol
from skellybench.reports import build_benchmark_report
from skellybench.schema import validate_benchmark_report, validate_protocol, validate_timing


def _report(timing: dict[str, Any], protocol: PerformanceProtocol | None = None, mode: str = "device") -> dict[str, Any]:
    ctx = context()
    return {
        "benchmark_type": "performance", "status": "measured", "execution": execution(ctx["host"]),
        "subject": subject("rfdetr-nano_b1_fp32", sha("fp32")), "dataset": dataset(),
        "protocol": (protocol or performance_protocol()).block(batch_size=1),
        "performance": {"records": [{"mode": mode, "timing_scope": "session", "timing": timing}]},
        **ctx,
    }


def _rejects(timing: dict[str, Any], match: str, **kwargs: Any) -> None:
    with pytest.raises(SkellybenchValidationError, match=match):
        build_benchmark_report(**_report(timing, **kwargs))


def test_full_stage_timing_is_accepted() -> None:
    report = build_benchmark_report(**_report(stage_timing(250.0, letterbox=True, decode_provider="CUDAExecutionProvider")))
    timing = report["performance"]["records"][0]["timing"]
    assert set(timing["stages_ms_per_image"]) == {"preprocess", "inference", "postprocess"}
    assert timing["excluded_ms_per_image"] == {"letterbox": 0.5}


def test_stages_are_required_with_exactly_three_keys() -> None:
    timing = stage_timing(100.0)
    del timing["stages_ms_per_image"]
    _rejects(timing, "stages_ms_per_image")
    timing = stage_timing(100.0)
    del timing["stages_ms_per_image"]["postprocess"]
    _rejects(timing, "missing 'postprocess'")
    timing = stage_timing(100.0)
    timing["stages_ms_per_image"]["decode"] = 0.1
    _rejects(timing, "stage 'decode'")


def test_stage_values_are_non_negative_numbers_or_null() -> None:
    for value in (-0.1, "1.0", True, float("nan")):
        timing = stage_timing(100.0)
        timing["stages_ms_per_image"]["inference"] = value
        _rejects(timing, "stages_ms_per_image.inference")
    timing = stage_timing(100.0)
    timing["stages_ms_per_image"]["postprocess"] = None
    build_benchmark_report(**_report(timing))


def test_stages_need_not_add_up_to_ms_per_image() -> None:
    timing = stage_timing(100.0)
    timing["stages_ms_per_image"] = {"preprocess": 50.0, "inference": 50.0, "postprocess": 50.0}
    build_benchmark_report(**_report(timing))


def test_ms_per_image_must_be_non_negative() -> None:
    timing = stage_timing(100.0)
    timing["ms_per_image"] = -1.0
    _rejects(timing, "ms_per_image")


def test_the_old_stage_ms_per_image_key_is_rejected() -> None:
    timing = stage_timing(100.0)
    timing["stage_ms_per_image"] = {"preprocess": 1.0, "inference": 1.0, "decode": 1.0}
    _rejects(timing, "use stages_ms_per_image")


def test_stage_providers_are_checked() -> None:
    timing = stage_timing(100.0, decode_provider="CUDAExecutionProvider")
    build_benchmark_report(**_report(timing))
    broken = copy.deepcopy(timing)
    broken["stage_providers"]["postprocess"] = ""
    _rejects(broken, "stage_providers.postprocess")
    broken = copy.deepcopy(timing)
    broken["stage_providers"]["letterbox"] = "CPU"
    _rejects(broken, "stage 'letterbox'")
    broken = copy.deepcopy(timing)
    del broken["stage_providers"]["inference"]
    _rejects(broken, "missing 'inference'")
    timing["stage_providers"] = None
    build_benchmark_report(**_report(timing))


def test_excluded_times_must_name_excluded_stages() -> None:
    timing = stage_timing(100.0, letterbox=True)
    _rejects(timing, "protocol.excluded_stages",
             protocol=PerformanceProtocol(stage_definitions={"device": {k: v for k, v in STAGE_DEFINITIONS["device"].items()
                                                                        if k != "letterbox"}}))
    timing["excluded_ms_per_image"] = {"letterbox": -1.0}
    _rejects(timing, "excluded_ms_per_image.letterbox")
    timing["excluded_ms_per_image"] = {"resize": 1.0}
    _rejects(timing, "stage 'resize'")


def test_measured_stages_need_definitions_for_their_mode() -> None:
    no_definitions = PerformanceProtocol(excluded_stages=("letterbox",))
    _rejects(stage_timing(100.0), "stage_definitions.device", protocol=no_definitions)
    only_host = performance_protocol(stage_definitions={"host": STAGE_DEFINITIONS["host"]})
    _rejects(stage_timing(100.0), "stage_definitions.device", protocol=only_host)
    build_benchmark_report(**_report(stage_timing(100.0), protocol=only_host, mode="host"))
    unmeasured = {"images_per_second": 100.0, "stages_ms_per_image": dict(NULL_STAGES)}
    build_benchmark_report(**_report(unmeasured, protocol=no_definitions))
    validate_timing(stage_timing(100.0))


def test_stage_definitions_enter_the_protocol_and_its_id() -> None:
    plain = PerformanceProtocol().block(batch_size=1)
    assert plain["stage_definitions"] == {}
    defined = performance_protocol().block(batch_size=1)
    assert defined["stage_definitions"] == STAGE_DEFINITIONS
    assert defined["protocol_id"] != performance_protocol(excluded_stages=("letterbox",), stage_definitions={
        **STAGE_DEFINITIONS, "host": {**STAGE_DEFINITIONS["host"], "inference": "Other text."}}).block(batch_size=1)["protocol_id"]
    assert performance_protocol().block(batch_size=1) == defined
    validate_protocol(defined)
    hash(performance_protocol())


def test_stage_definitions_are_checked() -> None:
    host = STAGE_DEFINITIONS["host"]
    cases = [
        ({"gpu": host}, "must be one of"),
        ({"host": {k: v for k, v in host.items() if k != "inference"}}, "missing 'inference'"),
        ({"host": {**host, "decode": "x"}}, "stage 'decode'"),
        ({"host": {**host, "preprocess": ""}}, "stage_definitions.host.preprocess"),
        ({"host": "text"}, "must be an object"),
    ]
    for definitions, match in cases:
        with pytest.raises(ValueError, match=match):
            PerformanceProtocol(excluded_stages=("letterbox",), stage_definitions=definitions)
    with pytest.raises(ValueError, match="stage 'letterbox'"):
        PerformanceProtocol(stage_definitions={"host": host})
    block = performance_protocol().block(batch_size=1)
    block["stage_definitions"]["device"]["inference"] = 3
    with pytest.raises(SkellybenchValidationError, match="stage_definitions.device.inference"):
        validate_protocol(block)


def test_accuracy_protocol_takes_optional_stage_definitions() -> None:
    plain = accuracy_protocol(batch_size=1, images=2, evaluation=EVALUATION)
    assert "stage_definitions" not in plain
    host = {k: v for k, v in STAGE_DEFINITIONS["host"].items() if k != "letterbox"}
    defined = accuracy_protocol(batch_size=1, images=2, evaluation=EVALUATION, stage_definitions={"host": host})
    assert defined["stage_definitions"] == {"host": host}
    assert defined["protocol_id"] != plain["protocol_id"]
    with pytest.raises(ValueError, match="letterbox"):
        accuracy_protocol(batch_size=1, images=2, evaluation=EVALUATION, stage_definitions={"host": STAGE_DEFINITIONS["host"]})


def test_accuracy_report_with_measured_stages_needs_definitions(family) -> None:
    from conftest import write_accuracy
    from skellybench._internal.io import read_json

    report = read_json(write_accuracy(family, context(), subject("a", sha("a")), execution(context()["host"]), 0.5))
    validate_benchmark_report(report)
    report["performance"]["records"][0]["timing"]["stages_ms_per_image"] = {
        "preprocess": 10.0, "inference": 10.0, "postprocess": 4.0}
    with pytest.raises(SkellybenchValidationError, match="stage_definitions.host"):
        validate_benchmark_report(report)
