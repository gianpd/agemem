"""
HotpotQA evaluation following AgeMem paper (arXiv:2601.01885v1) methodology.

Key features:
- Uses HuggingFace datasets for HotpotQA
- Multiple evaluation modes:
  - "ltm": Context stored in LTM (oracle, gold paragraphs)
  - "corpus": Context ingested to corpus, agent uses corpus tools
- J-score: LLM-as-a-Judge evaluation (0-1 scale)
- Multi-hop reasoning evaluation

Target metrics from paper:
- AgeMem-noRL: 54.49 J-score
- AgeMem (RL):  55.49 J-score

Accountability additions vs. original:
- Every J-score carries a ParseStatus so 0.0 (genuine) != 0.0 (parse failure)
- EvalResult.error is typed; a hard_fail flag controls whether errors abort the run
- All config (models, dirs, mode) is logged at run start via RunConfig
- Corpus documents are tracked in an explicit registry — no content-string scanning
- Summary distinguishes parse failures from true zeros
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & lightweight value types
# ---------------------------------------------------------------------------

class ParseStatus(str, Enum):
    """Distinguishes a genuine zero score from a parse failure."""
    OK = "ok"
    PARSE_FAILURE = "parse_failure"
    EXCEPTION = "exception"


class EvalMode(str, Enum):
    LTM = "ltm"
    CORPUS = "corpus"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HotpotSample:
    """A single HotpotQA sample."""
    id: str
    question: str
    answer: str
    supporting_facts: list[tuple[str, int]]   # (title, sentence_idx)
    context: list[tuple[str, list[str]]]       # (title, sentences)
    question_type: str
    level: str

    @property
    def gold_titles(self) -> set[str]:
        return {title for title, _ in self.supporting_facts}

    def gold_context_text(self) -> str:
        """
        Return only the gold paragraphs as a formatted string.
        Matches the Stage-1 context used in the AgeMem paper.
        """
        chunks = [
            f"Title: {title}\nParagraph: {' '.join(sentences)}"
            for title, sentences in self.context
            if title in self.gold_titles
        ]
        return "\n\n".join(chunks)


@dataclass
class JudgeScore:
    """Wraps a raw J-score with provenance."""
    value: float                    # 0.0–1.0; may be 0.0 due to parse failure
    status: ParseStatus
    raw_response: str               # full judge output, always preserved


@dataclass
class EvalResult:
    """Result for a single HotpotQA sample."""
    sample_id: str
    question: str
    expected_answer: str
    predicted_answer: str
    judge: JudgeScore
    latency_ms: float
    error: Optional[str] = None     # set only on exception; does NOT swallow score

    # Convenience accessor so callers don't reach into judge
    @property
    def j_score(self) -> float:
        return self.judge.value

    @property
    def scored(self) -> bool:
        """True when the score is trustworthy (not a parse/exception fallback)."""
        return self.judge.status == ParseStatus.OK

    def to_dict(self) -> dict:
        d = asdict(self)
        # Flatten judge into top-level for readability in JSON
        d["j_score"] = self.j_score
        d["parse_status"] = self.judge.status.value
        d["judge_raw_response"] = self.judge.raw_response
        del d["judge"]
        return d


@dataclass
class RunConfig:
    """
    All runtime parameters in one place.
    Logged at the start of every run so results are fully reproducible.
    """
    mode: EvalMode
    split: str
    setting: str
    judge_model: str
    persist_dir: Path
    corpus_dir: Optional[Path]
    limit: int
    hard_fail: bool = False         # if True, raise on sample exception instead of recording
    skip_ingest: bool = False       # if True, use existing corpus docs without re-ingesting

    def log(self) -> None:
        logger.info("=== RunConfig ===")
        for k, v in asdict(self).items():
            logger.info("  %s = %s", k, v)
        logger.info("=================")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mode"] = self.mode.value
        # Path objects are not JSON-serialisable — convert to strings
        d["persist_dir"] = str(self.persist_dir)
        d["corpus_dir"] = str(self.corpus_dir) if self.corpus_dir else None
        return d


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_sample(item: dict) -> HotpotSample:
    return HotpotSample(
        id=item.get("_id") or item.get("id"),
        question=item["question"],
        answer=item["answer"],
        supporting_facts=list(zip(
            item["supporting_facts"]["title"],
            item["supporting_facts"]["sent_id"],
        )),
        context=list(zip(
            item["context"]["title"],
            item["context"]["sentences"],
        )),
        question_type=item.get("type", "unknown"),
        level=item.get("level", "unknown"),
    )


def load_hotpotqa_hf(
    split: str = "validation",
    setting: str = "distractor",
    limit: int = 0,
) -> list[HotpotSample]:
    """Load HotpotQA from HuggingFace."""
    from datasets import load_dataset

    logger.info("Loading HotpotQA split=%s setting=%s limit=%s", split, setting, limit or "all")
    dataset = load_dataset("hotpot_qa", setting, split=split)
    samples = [_parse_sample(item) for item in (dataset if not limit else list(dataset)[:limit])]
    logger.info("Loaded %d samples", len(samples))
    return samples


def load_hotpotqa_json(path: Path, limit: int = 0) -> list[HotpotSample]:
    """Load HotpotQA from a local JSON file."""
    logger.info("Loading HotpotQA from %s (limit=%s)", path, limit or "all")
    raw = json.loads(path.read_text())
    if limit:
        raw = raw[:limit]
    samples = [_parse_sample(item) for item in raw]
    logger.info("Loaded %d samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# J-score parsing  (isolated + testable)
# ---------------------------------------------------------------------------

def parse_j_score(response: str) -> JudgeScore:
    """
    Parse a J-score from a judge response.

    Returns a JudgeScore with status OK on success, PARSE_FAILURE otherwise.
    Callers can distinguish a genuine 0.0 from a failed parse.
    """
    text = response.strip()

    # 1. Explicit marker wins
    m = re.search(r"FINAL_SCORE:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if m:
        v = float(m.group(1))
        if 0.0 <= v <= 1.0:
            return JudgeScore(value=v, status=ParseStatus.OK, raw_response=response)

    # 2. Last valid [0, 1] number in the text (handles chain-of-thought responses)
    candidates = [float(n) for n in re.findall(r"\b([0-9]*\.?[0-9]+)\b", text)]
    valid = [v for v in candidates if 0.0 <= v <= 1.0]
    if valid:
        return JudgeScore(value=valid[-1], status=ParseStatus.OK, raw_response=response)

    # 3. Binary fallback
    lower = text.lower()
    if "yes" in lower:
        return JudgeScore(value=1.0, status=ParseStatus.OK, raw_response=response)
    if "no" in lower:
        return JudgeScore(value=0.0, status=ParseStatus.OK, raw_response=response)

    # 4. Could not parse — return 0.0 but flag it
    logger.warning("Could not parse J-score from response: %r", text[:200])
    return JudgeScore(value=0.0, status=ParseStatus.PARSE_FAILURE, raw_response=response)


# ---------------------------------------------------------------------------
# Corpus registry  (replaces fragile content-string scanning)
# ---------------------------------------------------------------------------

@dataclass
class CorpusRegistry:
    """
    Tracks which files were ingested for the current sample so they can be
    removed precisely without scanning all corpus files.
    """
    _paths: list[Path] = field(default_factory=list)

    def register(self, path: Path) -> None:
        self._paths.append(path)

    def clear(self) -> None:
        removed, failed = 0, 0
        for p in self._paths:
            try:
                p.unlink(missing_ok=True)
                removed += 1
            except OSError as e:
                logger.warning("Could not remove corpus file %s: %s", p, e)
                failed += 1
        logger.debug("Corpus cleared: %d removed, %d failed", removed, failed)
        self._paths.clear()


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class HotpotEvaluator:
    """
    HotpotQA evaluator following the AgeMem paper methodology.

    Design principles:
    - RunConfig holds all parameters (no hidden env-var surprises at call time)
    - Every score carries ParseStatus (no silent zero-on-failure)
    - Corpus cleanup uses an explicit registry, not content scanning
    - run() delegates to _evaluate_sample(); summary is a pure function of results
    """

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.orchestrator = None
        self.judge = None
        self._corpus_registry = CorpusRegistry()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialise orchestrator and judge. Called once before run()."""
        from core.factory import OrchestratorFactory
        from evaluation.llm_judge import LLMJudge

        self.config.log()

        logger.info("Building orchestrator (persist_dir=%s)", self.config.persist_dir)
        self.orchestrator = OrchestratorFactory.build(
            persist_dir=self.config.persist_dir,
            include_web_tools=False,
        )

        api_base = os.getenv("JUDGE_BASE_URL", "http://localhost:8080/v1")
        api_key = (
            "EMPTY"
            if "localhost" in api_base
            else (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))
        )
        logger.info("Initialising LLMJudge (model=%s, api_base=%s)", self.config.judge_model, api_base)
        self.judge = LLMJudge(
            api_base=api_base,
            api_key=api_key,
            model=self.config.judge_model,
            temperature=0.0,
            max_tokens=512,
        )

    # ------------------------------------------------------------------
    # Context storage helpers
    # ------------------------------------------------------------------

    def _store_ltm(self, sample: HotpotSample) -> None:
        from core.types import TriggerKind

        text = sample.gold_context_text()
        if not text.strip():
            logger.warning("Sample %s has empty gold context", sample.id)
            return

        result = self.orchestrator._ltm.add(
            content=text,
            learning_score=0.5,
            tags=["hotpot_stage1", "gold_context"],
            trigger=TriggerKind.SYSTEM_RULE,
        )
        if not result.success:
            logger.warning("LTM storage failed for sample %s", sample.id)

    def _store_corpus(self, sample: HotpotSample) -> None:
        from ingest.ingest import ingest

        if not self.config.corpus_dir:
            raise ValueError("corpus_dir must be set when mode=corpus")

        # Skip ingestion if flag is set (assumes docs already ingested)
        if self.config.skip_ingest:
            logger.debug("Skipping ingestion for sample %s (--skip-ingest)", sample.id)
            return

        for title, sentences in sample.context:
            if title not in sample.gold_titles:
                continue

            safe_title = title.replace("/", "_")
            doc_path = self.config.corpus_dir / f"{safe_title}.md"
            doc_path.write_text(
                f"---\ntitle: {title}\nsource: hotpotqa\nquestion_id: {sample.id}\n---\n\n"
                f"# {title}\n\n{' '.join(sentences)}\n"
            )
            try:
                ingest(str(doc_path))
                self._corpus_registry.register(doc_path)
                logger.debug("Ingested: %s", title)
            except Exception:
                logger.exception("Ingest failed for title=%s sample=%s", title, sample.id)

    def _store_context(self, sample: HotpotSample) -> None:
        if self.config.mode == EvalMode.CORPUS:
            self._store_corpus(sample)
        else:
            self._store_ltm(sample)

    # ------------------------------------------------------------------
    # Per-sample evaluation
    # ------------------------------------------------------------------

    def _evaluate_sample(self, sample: HotpotSample) -> EvalResult:
        start = time.perf_counter()
        predicted = ""
        judge_score = JudgeScore(value=0.0, status=ParseStatus.EXCEPTION, raw_response="")
        error: Optional[str] = None

        try:
            self._store_context(sample)
            predicted = self.orchestrator.chat(sample.question)

            raw = self.judge.evaluate(
                question=sample.question,
                expected_answer=sample.answer,
                model_response=predicted,
                behavior_type="HOTPOT_J",
            ).raw_response
            judge_score = parse_j_score(raw)

        except Exception as exc:
            error = str(exc)
            logger.exception("Exception evaluating sample %s", sample.id)
            if self.config.hard_fail:
                raise

        latency_ms = (time.perf_counter() - start) * 1000
        return EvalResult(
            sample_id=sample.id,
            question=sample.question,
            expected_answer=sample.answer,
            predicted_answer=predicted,
            judge=judge_score,
            latency_ms=latency_ms,
            error=error,
        )

    def _reset_state(self) -> None:
        """Reset per-sample state between evaluations."""
        self.orchestrator.reset_stm()
        self.orchestrator.clear_ltm()
        if self.config.mode == EvalMode.CORPUS:
            self._corpus_registry.clear()

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self, samples: list[HotpotSample], output_path: Optional[Path] = None) -> list[EvalResult]:
        self.setup()
        results: list[EvalResult] = []

        for i, sample in enumerate(samples, start=1):
            logger.info(
                "[%d/%d] id=%s type=%s level=%s",
                i, len(samples), sample.id, sample.question_type, sample.level,
            )
            logger.debug("Q: %s", sample.question)
            logger.debug("Gold answer: %s", sample.answer)

            result = self._evaluate_sample(sample)
            results.append(result)

            logger.info(
                "  j_score=%.2f status=%s latency=%.0fms error=%s",
                result.j_score, result.judge.status.value, result.latency_ms, result.error,
            )
            self._reset_state()

        _print_summary(results, self.config)

        if output_path:
            _save_results(results, self.config, output_path)

        return results


# ---------------------------------------------------------------------------
# Summary & persistence  (pure functions, easy to test independently)
# ---------------------------------------------------------------------------

def _print_summary(results: list[EvalResult], config: RunConfig) -> None:
    scored = [r for r in results if r.scored]
    parse_failures = [r for r in results if r.judge.status == ParseStatus.PARSE_FAILURE]
    exceptions = [r for r in results if r.judge.status == ParseStatus.EXCEPTION]

    j_vals = [r.j_score for r in scored] or [0.0]
    mean = sum(j_vals) / len(j_vals)
    variance = sum((v - mean) ** 2 for v in j_vals) / len(j_vals)
    std = variance ** 0.5

    print(f"\n{'='*60}")
    print("HOTPOTQA EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Mode            : {config.mode.value}")
    print(f"Judge model     : {config.judge_model}")
    print(f"Total samples   : {len(results)}")
    print(f"Scored (OK)     : {len(scored)}")
    print(f"Parse failures  : {len(parse_failures)}  <- scores are 0.0 but NOT genuine zeros")
    print(f"Exceptions      : {len(exceptions)}")
    print(f"Mean J-score    : {mean:.4f}  (over scored samples only)")
    print(f"Std dev         : {std:.4f}")
    print(f"Min / Max       : {min(j_vals):.2f} / {max(j_vals):.2f}")
    print(f"\nPaper targets   : AgeMem-noRL=54.49  AgeMem-RL=55.49")
    print(f"{'='*60}\n")


def _save_results(
    results: list[EvalResult],
    config: RunConfig,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.to_dict(),
        "results": [r.to_dict() for r in results],
        "summary": {
            "total": len(results),
            "scored": sum(1 for r in results if r.scored),
            "parse_failures": sum(1 for r in results if r.judge.status == ParseStatus.PARSE_FAILURE),
            "exceptions": sum(1 for r in results if r.judge.status == ParseStatus.EXCEPTION),
            "mean_j_score": (
                sum(r.j_score for r in results if r.scored) / max(1, sum(1 for r in results if r.scored))
            ),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info("Results saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_config(args: argparse.Namespace) -> RunConfig:
    persist_dir = args.persist_dir or Path(tempfile.mkdtemp(prefix="hotpot_eval_"))
    mode = EvalMode(args.mode)

    corpus_dir: Optional[Path] = None
    if mode == EvalMode.CORPUS:
        corpus_dir = args.corpus_dir or Path(tempfile.mkdtemp(prefix="hotpot_corpus_"))

    judge_model = args.judge_model or os.getenv("JUDGE_BASE_MODEL", "Qwen3.5-9B-UD-Q4_K_XL.gguf")

    return RunConfig(
        mode=mode,
        split=args.split,
        setting=args.setting,
        judge_model=judge_model,
        persist_dir=persist_dir,
        corpus_dir=corpus_dir,
        limit=args.limit,
        hard_fail=args.hard_fail,
        skip_ingest=getattr(args, 'skip_ingest', False),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="HotpotQA evaluation (AgeMem paper style)")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--setting", default="distractor", choices=["distractor", "fullwiki"])
    parser.add_argument("--mode", default="ltm", choices=["ltm", "corpus"])
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None, help="Local JSON override")
    parser.add_argument(
        "--output", type=Path,
        default=Path("evaluation/results/hotpot_results.json"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--persist-dir", type=Path, default=None)
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument(
        "--hard-fail", action="store_true",
        help="Abort run on first sample exception instead of recording and continuing",
    )
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Skip ingestion in corpus mode (use existing corpus documents)",
    )
    args = parser.parse_args()

    config = _build_config(args)
    samples = (
        load_hotpotqa_json(args.data, config.limit)
        if args.data
        else load_hotpotqa_hf(config.split, config.setting, config.limit)
    )

    evaluator = HotpotEvaluator(config)
    evaluator.run(samples, args.output)


if __name__ == "__main__":
    main()