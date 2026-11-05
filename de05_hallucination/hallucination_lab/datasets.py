import csv
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .schemas import DetectRequest


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    split: Literal["train", "validation", "test"]
    label: Literal[0, 1]  # 1 = hallucination, never the reverse
    claim: str = Field(min_length=1)
    context: str = ""
    question: str = ""
    samples: list[str] = Field(default_factory=list)

    def request(self, **kwargs) -> DetectRequest:
        return DetectRequest(
            claim=self.claim, context=self.context, question=self.question, samples=self.samples, **kwargs
        )


def split_for(group_id: str, seed: int = 42) -> str:
    # All variants from one original example remain in the same split.
    bucket = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def read_json(path: Path):
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(data, list):
        raise ValueError("Expected JSON array or JSONL records")
    return data


def load_examples(path: Path) -> list[Example]:
    examples = [Example.model_validate(row) for row in read_json(path)]
    if not examples:
        raise ValueError("Dataset is empty")
    ids = [e.id for e in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate example IDs")
    groups = {}
    for example in examples:
        if example.group_id in groups and groups[example.group_id] != example.split:
            raise ValueError(f"Split leakage for group {example.group_id}")
        groups[example.group_id] = example.split
    return examples


def prepare_halueval(path: Path, task: str, seed: int) -> list[Example]:
    keys = {
        "qa": ("knowledge", "question", "right_answer", "hallucinated_answer"),
        "dialogue": ("knowledge", "dialogue_history", "right_response", "hallucinated_response"),
        "summarization": ("document", None, "right_summary", "hallucinated_summary"),
    }
    output = []
    for i, row in enumerate(read_json(path)):
        group = f"halueval:{task}:{i}"
        if task == "general":
            raw_label = row["hallucination_label"].lower().strip()
            if raw_label not in {"yes", "no"}:
                raise ValueError(f"Unexpected label: {raw_label}")
            output.append(
                Example(
                    id=group,
                    group_id=group,
                    dataset="HaluEval/general",
                    split=split_for(group, seed),
                    label=int(raw_label == "yes"),
                    claim=row["chatgpt_response"],
                    question=row["user_query"],
                    # General queries have no source document. Do not synthesize one from their label.
                )
            )
            continue
        context_key, question_key, right_key, wrong_key = keys[task]
        for label, answer_key in [(0, right_key), (1, wrong_key)]:
            output.append(
                Example(
                    id=f"{group}:{label}",
                    group_id=group,
                    dataset=f"HaluEval/{task}",
                    split=split_for(group, seed),
                    label=label,
                    claim=row[answer_key],
                    context=row[context_key],
                    question=row.get(question_key, "") if question_key else "",
                )
            )
    return output


def prepare_truthfulqa(path: Path, seed: int, contexts: Path | None = None) -> list[Example]:
    # Reference answers are candidate outputs, never evidence. Contexts must be independently supplied.
    evidence = json.loads(contexts.read_text(encoding="utf-8")) if contexts else {}
    if not isinstance(evidence, dict) or any(not isinstance(v, str) for v in evidence.values()):
        raise ValueError("Contexts must be a JSON object: question -> independent source text")
    output = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            group = f"truthfulqa:{i}"
            candidates = [(0, row["Best Answer"])]
            incorrect = row.get("Best Incorrect Answer", "").strip()
            if not incorrect:
                incorrect = row["Incorrect Answers"].split(";")[0].strip()
            candidates.append((1, incorrect))
            for label, answer in candidates:
                output.append(
                    Example(
                        id=f"{group}:{label}",
                        group_id=group,
                        dataset="TruthfulQA/candidate-detection",
                        split=split_for(group, seed),
                        label=label,
                        claim=answer,
                        question=row["Question"],
                        context=evidence.get(row["Question"], ""),
                    )
                )
    return output


def save_examples(examples: list[Example], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(e.model_dump_json() + "\n" for e in examples), encoding="utf-8")
