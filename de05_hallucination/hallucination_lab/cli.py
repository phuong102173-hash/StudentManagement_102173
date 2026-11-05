import argparse
import json
from pathlib import Path

from .datasets import load_examples, prepare_halueval, prepare_truthfulqa, read_json, save_examples
from .evaluation import calibrate, evaluate, human_agreement
from .schemas import Method


def main():
    parser = argparse.ArgumentParser(description="Đề 5 — chuẩn bị dữ liệu, đánh giá, calibration")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--dataset", choices=["halueval", "truthfulqa"], required=True)
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--task", choices=["qa", "dialogue", "summarization", "general"], default="qa")
    prepare.add_argument(
        "--contexts", type=Path, help="TruthfulQA: JSON map question -> independently sourced evidence"
    )
    prepare.add_argument("--seed", type=int, default=42)
    run = commands.add_parser("evaluate")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--methods", nargs="+", choices=[m.value for m in Method], default=[Method.BASELINE.value]
    )
    run.add_argument("--split", choices=["train", "validation", "test"], default="test")
    run.add_argument("--limit", type=int, help="First N records in the selected fixed split")
    run.add_argument("--threshold", type=float, default=0.5)
    run.add_argument("--calibration", type=Path)
    cal = commands.add_parser("calibrate")
    cal.add_argument("--predictions", type=Path, required=True)
    cal.add_argument("--method", choices=[m.value for m in Method], required=True)
    cal.add_argument("--output", type=Path, required=True)
    cal.add_argument("--target", type=float, default=0.8)
    human = commands.add_parser("agreement")
    human.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            if args.output.exists():
                raise ValueError("Output already exists; choose a fresh path")
            examples = (
                prepare_halueval(args.input, args.task, args.seed)
                if args.dataset == "halueval"
                else prepare_truthfulqa(args.input, args.seed, args.contexts)
            )
            save_examples(examples, args.output)
            print(f"Saved {len(examples)} candidates. Grouped 70/15/15 hash split; hallucination=1.")
        elif args.command == "evaluate":
            if not 0 <= args.threshold <= 1 or args.limit is not None and args.limit < 1:
                raise ValueError("Threshold must be in [0,1]; limit must be positive")
            examples = [e for e in load_examples(args.data) if e.split == args.split]
            if args.limit:
                examples = examples[: args.limit]
            calibration = json.loads(args.calibration.read_text()) if args.calibration else None
            summary = evaluate(
                examples,
                [Method(m) for m in args.methods],
                args.data,
                args.output,
                args.threshold,
                calibration,
            )
            print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
            if any(r["errors"] or r["abstained"] for r in summary["metrics"].values()):
                raise SystemExit(2)  # partial runs are saved, but never reported as full success
        elif args.command == "calibrate":
            if not 0 < args.target <= 1:
                raise ValueError("Target must be in (0,1]")
            calibration = calibrate(read_json(args.predictions), args.method, args.target)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(calibration, stream, indent=2, ensure_ascii=False, allow_nan=False)
            print(json.dumps(calibration, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(human_agreement(args.annotations), indent=2))
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
