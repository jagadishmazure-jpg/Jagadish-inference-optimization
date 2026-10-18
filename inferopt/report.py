"""Comparison tables and constraint-aware recommendations."""

from dataclasses import dataclass

from .configs import InferenceConfig

QUALITY_NOTES = {
    "fp16": "baseline quality — the reference everything else is judged against.",
    "int8": "near-baseline quality on most tasks; re-run evals to confirm.",
    "int4": "lossy — meaningful quality risk; do not ship without evals.",
}


@dataclass(frozen=True)
class ComparisonRow:
    config: InferenceConfig
    mean_ttft_ms: float
    mean_tpot_ms: float
    mean_tokens_per_sec: float
    total_cost_usd: float
    quality_note: str


@dataclass(frozen=True)
class ComparisonTable:
    rows: tuple

    def markdown(self) -> str:
        lines = [
            "| config | TTFT (ms) | TPOT (ms) | tok/s | cost (USD) | quality |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for r in self.rows:
            lines.append(
                f"| {r.config.describe()} | {r.mean_ttft_ms:.1f} | "
                f"{r.mean_tpot_ms:.1f} | {r.mean_tokens_per_sec:.1f} | "
                f"{r.total_cost_usd:.4f} | {r.quality_note} |"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class Recommendation:
    config: InferenceConfig
    reason: str
    binding_constraint: str | None  # None => cheapest overall won outright
    quality_check: str = ""          # what quality evidence the winner has
    excluded: tuple = ()             # configs the quality gate removed, with why


# Placeholder eval scores for the offline demo ONLY. They are assumptions, not
# measurements: replace them with your own eval results (same tasks, each
# quantization) before trusting a recommendation.
SIMULATED_QUALITY = {"fp16": 1.00, "int8": 0.99, "int4": 0.93}


def compare(results: list) -> ComparisonTable:
    """Aggregate grid results into one row per config."""
    by_config: dict = {}
    for res in results:
        by_config.setdefault(res.config, []).append(res.metrics)

    rows = []
    for config, metrics_list in by_config.items():
        n = len(metrics_list)
        rows.append(
            ComparisonRow(
                config=config,
                mean_ttft_ms=sum(m.ttft_ms for m in metrics_list) / n,
                mean_tpot_ms=sum(m.tpot_ms for m in metrics_list) / n,
                mean_tokens_per_sec=sum(m.tokens_per_sec for m in metrics_list) / n,
                total_cost_usd=sum(m.est_cost_usd for m in metrics_list),
                quality_note=QUALITY_NOTES[config.quantization],
            )
        )
    rows.sort(key=lambda r: r.total_cost_usd)
    return ComparisonTable(rows=tuple(rows))


def recommend(
    results: list,
    max_ttft_ms: float | None = None,
    max_cost_usd: float | None = None,
    quality: dict | None = None,
    min_relative_quality: float = 0.97,
    gated: tuple = ("int4",),
) -> Recommendation:
    """Pick the cheapest config that satisfies the constraints AND the quality gate.

    Quality gate: a config whose quantization is in ``gated`` (int4 by default)
    is only eligible with eval evidence. ``quality`` maps quantization to an
    eval score on your tasks; the config must score at least
    ``min_relative_quality`` of the fp16 score. With no score it is excluded,
    so a lossy config is never recommended on speed and cost alone.

    The binding constraint is the reason the outright-cheapest config lost.
    Raises ValueError when no config satisfies the constraints.
    """
    table = compare(results)
    if not table.rows:
        raise ValueError("no results to recommend from")
    quality = quality or {}
    base = quality.get("fp16")

    def quality_problem(row: ComparisonRow) -> str | None:
        q = row.config.quantization
        if q not in gated:
            return None
        if q not in quality or not base:
            return f"{row.config.describe()}: no {q} eval score (needs fp16 and {q})"
        rel = quality[q] / base
        if rel < min_relative_quality:
            return (f"{row.config.describe()}: {q} scores {rel:.2f} of fp16, "
                    f"below the {min_relative_quality:.2f} floor")
        return None

    def satisfies(row: ComparisonRow) -> bool:
        if max_ttft_ms is not None and row.mean_ttft_ms > max_ttft_ms:
            return False
        if max_cost_usd is not None and row.total_cost_usd > max_cost_usd:
            return False
        return True

    excluded = tuple(p for p in (quality_problem(r) for r in table.rows) if p)
    eligible = [r for r in table.rows if satisfies(r) and quality_problem(r) is None]
    if not eligible:
        raise ValueError(
            "no config satisfies the constraints and the quality gate "
            f"(max_ttft_ms={max_ttft_ms}, max_cost_usd={max_cost_usd}; "
            f"excluded: {list(excluded)})"
        )

    winner = eligible[0]  # rows are pre-sorted cheapest-first
    cheapest = table.rows[0]

    binding = None
    if cheapest.config != winner.config:
        violated = []
        if max_ttft_ms is not None and cheapest.mean_ttft_ms > max_ttft_ms:
            violated.append(
                f"max_ttft_ms={max_ttft_ms} "
                f"(cheapest config needs {cheapest.mean_ttft_ms:.1f} ms)"
            )
        if max_cost_usd is not None and cheapest.total_cost_usd > max_cost_usd:
            violated.append(
                f"max_cost_usd={max_cost_usd} "
                f"(cheapest config costs {cheapest.total_cost_usd:.4f} USD)"
            )
        problem = quality_problem(cheapest)
        if problem:
            violated.append(f"quality gate ({problem})")
        binding = "; ".join(violated) if violated else "unknown"

    wq = winner.config.quantization
    if wq in quality and base:
        check = f"{wq} scores {quality[wq] / base:.2f} of fp16 on the supplied evals"
    elif wq == "fp16":
        check = "fp16 is the quality baseline"
    else:
        check = f"{wq} is not gated, but has no eval score: confirm on your evals"

    reason = (
        f"{winner.config.describe()} is the cheapest config satisfying "
        f"max_ttft_ms={max_ttft_ms}, max_cost_usd={max_cost_usd} and the quality gate "
        f"at {winner.total_cost_usd:.4f} USD and {winner.mean_ttft_ms:.1f} ms TTFT."
    )
    return Recommendation(config=winner.config, reason=reason, binding_constraint=binding,
                          quality_check=check, excluded=excluded)
