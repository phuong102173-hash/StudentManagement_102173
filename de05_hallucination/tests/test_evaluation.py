import json

import pytest

from hallucination_lab.datasets import load_examples, prepare_halueval, prepare_truthfulqa, split_for
from hallucination_lab.evaluation import calibrate, evaluate, human_agreement, metrics, validate_calibration
from hallucination_lab.schemas import Method


def row(label, score, split="validation", group="g"):
    return {
        "id": f"{group}:{label}",
        "group_id": group,
        "label": label,
        "score": score,
        "prediction": score >= 0.5,
        "status": "ok",
        "method": "evidence_baseline",
        "split": split,
    }


def test_metrics_known_confusion_and_auc():
    rows = [row(1, 0.9), row(1, 0.8), row(0, 0.7), row(0, 0.1)]
    report = metrics(rows)
    assert report["precision"] == pytest.approx(2 / 3)
    assert report["recall"] == 1
    assert report["f1"] == pytest.approx(0.8)
    assert report["auc_roc"] == 1
    assert report["confusion_matrix"] == [[1, 1], [0, 2]]


def test_abstentions_and_errors_are_not_counted_as_correct():
    report = metrics([row(1, 0.9), {"status": "abstained", "prediction": None}, {"status": "error"}])
    assert report["coverage"] == pytest.approx(1 / 3)
    assert report["errors"] == 1 and report["abstained"] == 1
    assert report["auc_roc"] is None
    assert metrics([])["f1"] is None


def test_halueval_pairs_keep_group_and_split(tmp_path):
    path = tmp_path / "qa.json"
    path.write_text(
        json.dumps(
            [
                {
                    "knowledge": "source",
                    "question": "q",
                    "right_answer": "right",
                    "hallucinated_answer": "wrong",
                }
            ]
        )
    )
    examples = prepare_halueval(path, "qa", 42)
    assert [e.label for e in examples] == [0, 1]
    assert len({e.split for e in examples}) == 1
    assert len({e.group_id for e in examples}) == 1
    assert examples[0].context == "source"
    assert split_for("x", 42) == split_for("x", 42)


def test_truthfulqa_never_leaks_reference_answer_as_context(tmp_path):
    path = tmp_path / "truth.csv"
    path.write_text("Question,Best Answer,Incorrect Answers\nQuestion?,Right,Wrong;Another wrong\n")
    examples = prepare_truthfulqa(path, 42)
    assert all(e.context == "" for e in examples)
    assert [e.label for e in examples] == [0, 1]


def test_calibration_rejects_test_data_and_group_leakage():
    with pytest.raises(ValueError, match="validation"):
        calibrate([row(0, 0.1, split="test"), row(1, 0.9, split="test")], "evidence_baseline")
    records = [row(0, 0.1, group="a"), row(1, 0.9, group="b"), row(0, 0.2, group="c"), row(1, 0.8, group="d")]
    calibration = calibrate(records, "evidence_baseline")
    assert 0 <= calibration["threshold"] <= 1
    assert calibration["target_met_on_validation"] is True

    class TestExample:
        group_id = "a"

    with pytest.raises(ValueError, match="leakage"):
        validate_calibration(calibration, [TestExample()], Method.BASELINE)


def test_demo_evaluation_writes_actual_predictions(tmp_path):
    from pathlib import Path

    source = Path(__file__).parents[1] / "data" / "demo.jsonl"
    examples = [e for e in load_examples(source) if e.split == "test"]
    out = tmp_path / "run"
    summary = evaluate(examples, [Method.BASELINE], source, out)
    records = (out / "predictions.jsonl").read_text().splitlines()
    assert len(records) == len(examples) > 0
    assert summary["metrics"]["evidence_baseline"]["evaluated"] == len(examples)
    assert json.loads((out / "summary.json").read_text())["source_sha256"]
    with pytest.raises(ValueError, match="already exists"):
        evaluate(examples, [Method.BASELINE], source, out)


def test_human_kappa(tmp_path):
    path = tmp_path / "annotations.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"id": str(i), "annotator_a": a, "annotator_b": b})
            for i, (a, b) in enumerate([(0, 0), (0, 1), (1, 1), (1, 1)])
        )
    )
    report = human_agreement(path)
    assert report["agreement"] == 0.75
    assert report["cohen_kappa"] == pytest.approx(0.5)
