import hashlib
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .datasets import Example
from .detectors import detect
from .llm import DetectorFailure, DetectorUnavailable
from .schemas import Method


def metrics(records: list[dict]) -> dict:
    evaluated = [r for r in records if r.get("prediction") is not None and r.get("status") == "ok"]
    report = {
        "total": len(records),
        "evaluated": len(evaluated),
        "coverage": len(evaluated) / len(records) if records else 0,
        "abstained": sum(r.get("status") == "abstained" for r in records),
        "errors": sum(r.get("status") == "error" for r in records),
        "precision": None,
        "recall": None,
        "f1": None,
        "accuracy": None,
        "auc_roc": None,
        "cohen_kappa": None,
        "brier_score": None,
        "confusion_matrix": None,
        "positive_class": "hallucination=1",
    }
    if not evaluated:
        return report
    labels = [r["label"] for r in evaluated]
    predictions = [int(r["prediction"]) for r in evaluated]
    scores = [r["score"] for r in evaluated]
    p, r, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    report.update(
        precision=float(p),
        recall=float(r),
        f1=float(f1),
        accuracy=float(accuracy_score(labels, predictions)),
        confusion_matrix=confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    )
    if len(set(labels)) == 2:
        report["auc_roc"] = float(roc_auc_score(labels, scores))
        report["cohen_kappa"] = float(cohen_kappa_score(labels, predictions))
    if all(r.get("calibrated", False) for r in evaluated):
        report["brier_score"] = sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(labels)
    return report


def calibrated_score(score: float, calibration: dict) -> float:
    z = calibration["coefficient"] * score + calibration["intercept"]
    return 1 / (1 + math.exp(-max(-700, min(700, z))))


def validate_calibration(calibration: dict, examples: list[Example], method: Method):
    if calibration.get("method") != method.value or calibration.get("split") != "validation":
        raise ValueError("Calibration must match the method and come from validation only")
    for key in ("coefficient", "intercept", "threshold"):
        if not isinstance(calibration.get(key), (int, float)) or not math.isfinite(calibration[key]):
            raise ValueError("Invalid calibration coefficients")
    if not 0 <= calibration["threshold"] <= 1 or not calibration.get("group_ids"):
        raise ValueError("Calibration needs a valid threshold and source group IDs")
    used = set(calibration.get("group_ids", []))
    if used & {e.group_id for e in examples}:
        raise ValueError("Calibration/test group leakage detected")


def evaluate(
    examples: list[Example],
    methods: list[Method],
    source: Path,
    output: Path,
    threshold: float = 0.5,
    calibration: dict | None = None,
) -> dict:
    if not examples:
        raise ValueError("No examples selected; an empty run is not a benchmark")
    if output.exists():
        raise ValueError("Output directory already exists; choose a fresh path to preserve previous runs")
    if calibration and len(methods) != 1:
        raise ValueError("Apply one calibration file to its matching method only")
    if calibration:
        validate_calibration(calibration, examples, methods[0])
        threshold = calibration["threshold"]
    output.mkdir(parents=True)
    started = time.perf_counter()
    all_records, reports = [], {}
    for method in methods:
        records = []
        for example in examples:
            record = {
                "id": example.id,
                "group_id": example.group_id,
                "dataset": example.dataset,
                "split": example.split,
                "method": method.value,
                "label": example.label,
                "prediction": None,
                "score": None,
                "calibrated": calibration is not None,
            }
            try:
                result = detect(example.request(method=method, threshold=threshold))
                record.update(
                    status="abstained" if result.is_hallucination is None else "ok",
                    prediction=result.is_hallucination,
                    score=result.score,
                    result=result.model_dump(),
                )
                if calibration and record["status"] == "ok":
                    record["score"] = calibrated_score(result.score, calibration)
                    record["prediction"] = record["score"] >= threshold
            except (DetectorUnavailable, DetectorFailure, ValueError) as exc:
                record.update(status="error", error=str(exc))
            records.append(record)
        reports[method.value] = metrics(records)
        reports[method.value]["by_dataset"] = {
            name: metrics([r for r in records if r["dataset"] == name])
            for name in sorted({r["dataset"] for r in records})
        }
        all_records.extend(records)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "splits": dict(Counter(e.split for e in examples)),
        "threshold": threshold,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "metrics": reports,
        "configuration": {
            k: os.getenv(k)
            for k in [
                "JUDGE_MODEL",
                "NLI_MODEL",
                "NLI_REVISION",
                "HHEM_MODEL",
                "HHEM_REVISION",
                "MODEL_DEVICE",
            ]
        },
        "calibration": calibration,
        "limitations": [
            "Metrics exclude errors/abstentions; always report coverage.",
            "Cohen kappa here compares detector labels with gold, not human-human agreement.",
            "Demo data are authored fixtures, not an official benchmark.",
            "Scores are uncalibrated unless a held-out calibration file was explicitly applied.",
        ],
    }
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_records), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return summary


def calibrate(records: list[dict], method: str, target: float = 0.8) -> dict:
    records = [r for r in records if r.get("method") == method and r.get("status") == "ok"]
    if not records or any(r["split"] != "validation" for r in records):
        raise ValueError("Calibration requires successful validation predictions only, never train/test")
    if len({r["label"] for r in records}) != 2:
        raise ValueError("Calibration needs both positive and negative labels")
    model = LogisticRegression(random_state=42).fit(
        [[r["score"]] for r in records], [r["label"] for r in records]
    )
    calibration = {
        "method": method,
        "split": "validation",
        "coefficient": float(model.coef_[0][0]),
        "intercept": float(model.intercept_[0]),
        "n": len(records),
        "group_ids": sorted({r["group_id"] for r in records}),
        "target_precision_recall": target,
        "note": "Platt scaling and threshold fitted on validation; validation metrics are optimistic. Report test separately.",
    }
    scores = [calibrated_score(r["score"], calibration) for r in records]
    candidates = sorted({0.0, 0.5, 1.0, *scores})
    choices = []
    for threshold in candidates:
        rows = [dict(r, score=s, prediction=s >= threshold) for r, s in zip(records, scores)]
        report = metrics(rows)
        meets = report["precision"] >= target and report["recall"] >= target
        choices.append((meets, report["f1"], threshold, report))
    meets, _, threshold, report = max(choices, key=lambda c: (c[0], c[1], -abs(c[2] - 0.5)))
    calibration.update(threshold=threshold, target_met_on_validation=meets, validation_metrics=report)
    return calibration


def human_agreement(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 2 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Need at least 2 distinct annotation IDs")
    if any(type(r[k]) is not int or r[k] not in (0, 1) for r in rows for k in ("annotator_a", "annotator_b")):
        raise ValueError("Human annotation labels must be 0/1 integers")
    a, b = [r["annotator_a"] for r in rows], [r["annotator_b"] for r in rows]
    kappa = float(cohen_kappa_score(a, b)) if len(set(a + b)) == 2 else None
    return {"n": len(rows), "agreement": sum(x == y for x, y in zip(a, b)) / len(rows), "cohen_kappa": kappa}
