import os
import threading
from functools import lru_cache
from importlib.util import find_spec

from .llm import DetectorFailure, DetectorUnavailable

_lock = threading.RLock()


def enabled() -> bool:
    return os.getenv("ENABLE_LOCAL_MODELS") == "1" and find_spec("transformers") is not None


@lru_cache(maxsize=2)
def load_nli():
    if not enabled():
        raise DetectorUnavailable("NLI cần cài .[models] và đặt ENABLE_LOCAL_MODELS=1.")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_id = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
    revision = os.getenv("NLI_REVISION", "main")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=False,
            )
            .to(os.getenv("MODEL_DEVICE", "cpu"))
            .eval()
        )
        labels = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
        if not {"entailment", "neutral", "contradiction"}.issubset(labels):
            raise ValueError("NLI checkpoint must provide semantic id2label labels")
        return tokenizer, model, labels
    except Exception as exc:
        raise DetectorFailure(
            "Không tải được NLI model. Kiểm tra checkpoint, cache và quyền tải Hugging Face."
        ) from exc


def nli_pairs(pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
    if not enabled():
        raise DetectorUnavailable("NLI cần cài .[models] và đặt ENABLE_LOCAL_MODELS=1.")
    import torch

    with _lock:
        tokenizer, model, labels = load_nli()
        results = []
        for offset in range(0, len(pairs), 8):
            batch = pairs[offset : offset + 8]
            # Bounded retrieval chunks are scored separately, never silently drop the hypothesis.
            inputs = tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                padding=True,
                truncation="only_first",
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                probs = model(**inputs).logits.softmax(dim=-1).cpu().tolist()
            results.extend({label: row[index] for label, index in labels.items()} for row in probs)
        return results


@lru_cache(maxsize=1)
def load_hhem():
    if not enabled() or find_spec("sentence_transformers") is None:
        raise DetectorUnavailable("HHEM cần cài .[models] và đặt ENABLE_LOCAL_MODELS=1.")
    from sentence_transformers import CrossEncoder

    try:
        return CrossEncoder(
            os.getenv("HHEM_MODEL", "vectara/hallucination_evaluation_model"),
            revision=os.getenv("HHEM_REVISION", "hhem-1.0-open"),
            device=os.getenv("MODEL_DEVICE", "cpu"),
            trust_remote_code=False,
            max_length=512,
        )
    except Exception as exc:
        raise DetectorFailure(
            "Không tải được HHEM v1. Kiểm tra revision hhem-1.0-open và quyền tải model."
        ) from exc


def hhem_pairs(pairs: list[tuple[str, str]]) -> list[float]:
    with _lock:
        scores = load_hhem().predict(pairs, batch_size=8, show_progress_bar=False)
    result = [float(s) for s in scores]
    if any(not 0 <= s <= 1 for s in result):
        raise DetectorFailure("HHEM checkpoint không trả về điểm factual consistency trong [0,1].")
    return result
