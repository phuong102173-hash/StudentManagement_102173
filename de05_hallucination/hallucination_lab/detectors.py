import math
import re
import time

from . import llm, models
from .llm import DetectorFailure, DetectorUnavailable
from .schemas import DetectionResult, DetectRequest, Evidence, Finding, Method
from .text import has_negation, normalize, retrieve, sentences, tokens

RESEARCH_METHODS = [Method.FACTSCORE, Method.SELFCHECK, Method.HHEM, Method.JUDGE, Method.NLI]


def result(
    req: DetectRequest,
    findings: list[Finding],
    explanation: str,
    warnings: list[str] | None = None,
    score: float | None = None,
) -> DetectionResult:
    known = [f.score for f in findings if f.score is not None]
    # A single bad factual claim can invalidate the output; do not average it away.
    if score is None and known:
        score = max(known)
    if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
        raise DetectorFailure("Detector trả điểm không hợp lệ.")
    evidence = [e for f in findings for e in f.evidence]
    abstaining = score is None or (any(f.score is None for f in findings) and score < req.threshold)
    return DetectionResult(
        method=req.method,
        score=score,
        is_hallucination=None if abstaining else score >= req.threshold,
        confidence=None if abstaining else score if score >= req.threshold else 1 - score,
        threshold=req.threshold,
        explanation=explanation,
        findings=findings,
        evidence=evidence,
        warnings=warnings or [],
        confidence_kind="heuristic" if req.method == Method.BASELINE else "uncalibrated_score",
    )


def unknown(req: DetectRequest, message: str) -> DetectionResult:
    return result(
        req,
        [],
        message,
        ["Không đủ thông tin để đưa ra kết luận; null không có nghĩa là không có hallucination."],
    )


def baseline(req: DetectRequest) -> DetectionResult:
    if not tokens(req.context):
        return unknown(req, "Hãy cung cấp ngữ cảnh hoặc tài liệu nguồn để đối chiếu.")
    findings = []
    for claim in sentences(req.claim):
        chunks = retrieve(claim, req.context)
        source = chunks[0] if chunks else ""
        c_tokens, s_tokens = tokens(claim), tokens(source)
        overlap = len(c_tokens & s_tokens) / max(len(c_tokens), 1)
        same = normalize(claim).strip(".!? ") == normalize(source).strip(".!? ")
        numbers_c, numbers_s = (
            set(re.findall(r"\d+(?:[.,]\d+)?", claim)),
            set(re.findall(r"\d+(?:[.,]\d+)?", source)),
        )
        number_conflict = bool(numbers_c and numbers_s and not numbers_c <= numbers_s)
        negation_conflict = has_negation(claim) != has_negation(source)
        if same:
            score, verdict, reason = 0.03, "supported", "Câu khớp trực tiếp với câu trong tài liệu nguồn."
        elif overlap >= 0.55 and (number_conflict or negation_conflict):
            score, verdict = 0.9, "contradicted"
            reason = "Có chênh lệch số liệu hoặc dấu hiệu phủ định trong câu có nhiều từ trùng khớp."
        elif overlap >= 0.85:
            score, verdict, reason = (
                0.25,
                "supported",
                "Độ trùng từ vựng cao; cần kiểm tra thêm quan hệ ngữ nghĩa.",
            )
        else:
            score, verdict, reason = (
                0.65,
                "unsupported",
                "Chưa tìm được câu hỗ trợ đủ gần về từ vựng trong nguồn.",
            )
        findings.append(
            Finding(
                claim=claim,
                score=score,
                verdict=verdict,
                explanation=reason,
                error_type="factual_error"
                if verdict == "contradicted"
                else "unsupported"
                if verdict == "unsupported"
                else "none",
                evidence=[Evidence(quote=source, source="context", relation="related")] if source else [],
            )
        )
    return result(
        req,
        findings,
        "Đối chiếu từ vựng, số liệu và phủ định với ngữ cảnh được cung cấp.",
        [
            "Baseline minh họa, không phải mô hình AI đã huấn luyện; dễ sai với diễn đạt lại, đổi chủ thể và nhiều mệnh đề.",
            "Điểm và confidence là heuristic, chưa được hiệu chỉnh. Unsupported không đồng nghĩa thông tin sai ngoài đời.",
        ],
    )


def from_judge(claim: str, context: str) -> Finding:
    output = llm.judge(claim, context)
    relation = (
        "support"
        if output.verdict == "supported"
        else "contradiction"
        if output.verdict == "contradicted"
        else "related"
    )
    return Finding(
        claim=claim,
        score=output.hallucination_score,
        verdict=output.verdict,
        explanation=output.explanation,
        error_type=output.error_type,
        evidence=[Evidence(quote=q, source="context", relation=relation) for q in output.evidence_quotes],
    )


def judge_detector(req: DetectRequest) -> DetectionResult:
    if not tokens(req.context):
        return unknown(req, "LLM judge cần context để đánh giá độ trung thực có căn cứ.")
    return result(
        req,
        [from_judge(req.claim, req.context)],
        "LLM đánh giá có schema và kiểm tra trích dẫn.",
        [
            "Judge có thể thiên lệch hoặc bị ảnh hưởng bởi prompt injection; bằng chứng trích dẫn cần được đọc lại.",
        ],
    )


def factscore(req: DetectRequest) -> DetectionResult:
    if not tokens(req.context):
        return unknown(req, "FActScore cần tài liệu nguồn để kiểm chứng từng atomic fact.")
    facts = llm.atomic_facts(req.claim)
    findings = [from_judge(fact, "\n".join(retrieve(fact, req.context, k=5))) for fact in facts]
    if any(e.quote not in req.context for f in findings for e in f.evidence):
        raise DetectorFailure("Trích dẫn phải là đoạn liên tục trong tài liệu nguồn gốc.")
    known = [f for f in findings if f.score is not None]
    score = sum(f.verdict != "supported" for f in known) / len(known) if known else None
    return result(
        req,
        findings,
        "Tách atomic facts → truy hồi câu liên quan → kiểm chứng từng fact.",
        [
            "Bản FActScore-inspired dùng nguồn do người dùng cung cấp; không tái tạo Wikipedia retrieval của repo gốc.",
            "score = 1 − tỷ lệ atomic facts được hỗ trợ; không phải xác suất đã hiệu chỉnh.",
        ],
        score=score,
    )


def nli(req: DetectRequest) -> DetectionResult:
    if not tokens(req.context):
        return unknown(req, "NLI cần context (premise) để kiểm tra claim (hypothesis).")
    findings = []
    for claim in sentences(req.claim):
        chunks = retrieve(claim, req.context)
        scores = models.nli_pairs([(chunk, claim) for chunk in chunks])
        best = max(range(len(scores)), key=lambda i: scores[i]["entailment"])
        chosen = scores[best]
        score = 1 - chosen["entailment"]
        verdict = (
            "supported"
            if score < 0.5
            else "contradicted"
            if chosen["contradiction"] >= 0.5
            else "unsupported"
        )
        findings.append(
            Finding(
                claim=claim,
                score=score,
                verdict=verdict,
                explanation=f"NLI: entailment={chosen['entailment']:.3f}, contradiction={chosen['contradiction']:.3f}.",
                error_type="none"
                if verdict == "supported"
                else "factual_error"
                if verdict == "contradicted"
                else "unsupported",
                evidence=[
                    Evidence(
                        quote=chunks[best],
                        source="context",
                        relation="support" if verdict == "supported" else "related",
                    )
                ],
            )
        )
    return result(
        req,
        findings,
        "DeBERTa NLI: score = 1 − P(entailment), lấy câu nguồn hỗ trợ tốt nhất.",
        [
            "Checkpoint mặc định dành cho tiếng Anh. Truy hồi top-4 và giới hạn 512 tokens có thể bỏ sót bằng chứng.",
        ],
    )


def hhem(req: DetectRequest) -> DetectionResult:
    if not tokens(req.context):
        return unknown(req, "HHEM cần tài liệu nguồn để chấm factual consistency.")
    findings = []
    for claim in sentences(req.claim):
        chunks = retrieve(claim, req.context)
        scores = models.hhem_pairs([(chunk, claim) for chunk in chunks])
        best = max(range(len(scores)), key=lambda i: scores[i])
        score = 1 - scores[best]
        findings.append(
            Finding(
                claim=claim,
                score=score,
                verdict="unsupported" if score >= 0.5 else "supported",
                explanation=f"HHEM factual-consistency score: {scores[best]:.3f}.",
                error_type="unsupported" if score >= 0.5 else "none",
                evidence=[Evidence(quote=chunks[best], source="context", relation="related")],
            )
        )
    return result(
        req,
        findings,
        "HHEM-1.0 CrossEncoder đối chiếu từng câu với nguồn liên quan.",
        [
            "Sử dụng checkpoint HHEM v1 mở, không phải HHEM thương mại; chủ yếu dành cho tiếng Anh.",
            "Truy hồi top-4 và giới hạn 512 tokens có thể bỏ sót bằng chứng.",
        ],
    )


def selfcheck(req: DetectRequest) -> DetectionResult:
    samples = req.samples
    if not samples:
        if not req.question.strip():
            return unknown(
                req, "SelfCheckGPT cần ít nhất 2 câu trả lời lấy mẫu độc lập hoặc câu hỏi để LLM sinh mẫu."
            )
        samples = llm.sample_answers(req.question)
    if len(samples) < 2 or len({normalize(s).strip() for s in samples}) < 2:
        return unknown(req, "Cần ít nhất 2 mẫu khác nhau từ cùng model và cùng câu hỏi.")
    findings = []
    for claim in sentences(req.claim):
        judgments = [from_judge(claim, sample) for sample in samples]
        # Prompting variant: supported=0, not supported=1, unknown=0.5.
        scores = [0 if j.verdict == "supported" else 0.5 if j.verdict == "unknown" else 1 for j in judgments]
        score = sum(scores) / len(scores)
        findings.append(
            Finding(
                claim=claim,
                score=score,
                verdict="unsupported" if score >= 0.5 else "supported",
                explanation=f"SelfCheckGPT-Prompt: trung bình bất đồng trên {len(samples)} mẫu độc lập.",
                error_type="unsupported" if score >= 0.5 else "none",
                evidence=[
                    Evidence(quote=e.quote, source=f"sample_{i + 1}", relation="sample")
                    for i, j in enumerate(judgments)
                    for e in j.evidence
                ],
            )
        )
    return result(
        req,
        findings,
        "SelfCheckGPT biến thể LLM-Prompting, không phải phép đo trùng từ.",
        [
            "Nhất quán giữa các mẫu không chứng minh sự thật: cùng model có thể lặp lại cùng một lỗi.",
            "Mẫu phải đến từ cùng generator; hệ thống không thể xác minh nguồn gốc các mẫu nhập tay.",
        ],
    )


def availability() -> list[dict]:
    llm_ready, local_ready = llm.configured(), models.enabled()
    descriptions = {
        Method.BASELINE: (True, "Baseline minh họa", "Chạy ngay, không cần API key"),
        Method.FACTSCORE: (llm_ready, "FActScore · atomic facts", "Cần endpoint LLM"),
        Method.SELFCHECK: (llm_ready, "SelfCheckGPT · prompting", "Cần endpoint LLM và mẫu độc lập"),
        Method.HHEM: (local_ready, "HHEM · factual consistency", "Cần local models; lần đầu tải checkpoint"),
        Method.JUDGE: (llm_ready, "LLM-as-Judge · Qwen", "Cần endpoint LLM"),
        Method.NLI: (local_ready, "NLI · DeBERTa", "Cần local models; lần đầu tải checkpoint"),
        Method.ENSEMBLE: (
            llm_ready and local_ready,
            "Ensemble · 5 phương pháp",
            "Cần cả 5 detector; không bỏ qua detector lỗi",
        ),
    }
    return [
        {
            "id": m.value,
            "configured": v[0],
            "name": v[1],
            "requirement": v[2],
            "note": "configured chỉ phản ánh cấu hình, chưa xác nhận model tải hoặc suy luận thành công.",
        }
        for m, v in descriptions.items()
    ]


def detect(req: DetectRequest) -> DetectionResult:
    start = time.perf_counter()
    if len(sentences(req.claim)) > 30:
        raise DetectorFailure("Tối đa 30 câu mỗi lần kiểm tra. Hãy chia nhỏ văn bản.")
    functions = {
        Method.BASELINE: baseline,
        Method.FACTSCORE: factscore,
        Method.SELFCHECK: selfcheck,
        Method.HHEM: hhem,
        Method.JUDGE: judge_detector,
        Method.NLI: nli,
    }
    try:
        if req.method == Method.ENSEMBLE:
            components = [detect(req.model_copy(update={"method": m})) for m in RESEARCH_METHODS]
            if any(c.is_hallucination is None for c in components):
                output = unknown(req, "Một detector chưa đủ bằng chứng; ensemble không đưa ra kết luận.")
            else:
                score = sum(c.score for c in components) / len(components)
                output = result(
                    req, [], "Trung bình đều điểm của 5 detector; chưa được calibration.", score=score
                )
            output.components = components
            output.evidence = [e for c in components for e in c.evidence]
            output.warnings.append("Các detector có tương quan; điểm ensemble chưa phải xác suất đúng.")
        else:
            output = functions[req.method](req)
    except (DetectorFailure, DetectorUnavailable):
        raise
    except Exception as exc:
        raise DetectorFailure(
            "Detector không hoàn tất suy luận. Kiểm tra cài đặt model và kích thước đầu vào."
        ) from exc
    output.latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return output
