from acl import ActionOutcome, SagaActionResult
from saga import SagaEngine, SagaStep


def _run_case(name: str, http_status: int, error_code: str, outcome: ActionOutcome, semantic_status: str) -> None:
    engine = SagaEngine(kind="transactional", persist_path=None)

    def forward(_: dict) -> dict:
        return {"ok": True}

    def compensate(_: dict) -> SagaActionResult:
        return SagaActionResult(
            outcome=outcome,
            semantic_status=semantic_status,
            raw_metadata={"http_status": http_status, "error_code": error_code, "duffel_request_id": f"req-{name}"},
        )

    step = SagaStep(name=f"step-{name}", forward=forward, compensate=compensate)
    ok, _ = engine.run([step], context={})
    assert ok is True
    engine.compensate_all()
    print(
        f"[{name}] http_status={http_status}, error_code={error_code}, "
        f"saga_outcome={step.outcome}, semantic_status={step.semantic_status}, saga_step_status={step.status.value}"
    )


if __name__ == "__main__":
    _run_case("already-cancelled", 422, "already_cancelled", ActionOutcome.SUCCESS, "ALREADY_CANCELLED_SUCCESS")
    _run_case("server-error", 500, "server_error", ActionOutcome.RETRYABLE, "CANCEL_CONFIRM_RETRYABLE")
    _run_case("validation-non-idempotent", 422, "validation_required", ActionOutcome.FAILED, "CANCEL_CREATE_VALIDATION_FAILED")
