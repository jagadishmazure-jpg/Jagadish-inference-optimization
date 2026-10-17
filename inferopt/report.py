"""Comparison tables and constraint-aware recommendations."""

from dataclasses import dataclass, field

from .benchmark import BenchmarkResult
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
) -> Recommendation:
    """Pick the cheapest config that satisfies the constraints.

    The binding constraint is the reason the outright-cheapest config lost:
    if the cheapest config already satisfies everything, nothing binds.
    Raises ValueError when no config satisfies the constraints.
    """
    table = compare(results)
    if not table.rows:
        raise ValueError("no results to recommend from")

    def satisfies(row: ComparisonRow) -> bool:
        if max_ttft_ms is not None and row.mean_ttft_ms > max_ttft_ms:
            return False
        if max_cost_usd is not None and row.total_cost_usd > max_cost_usd:
            return False
        return True

    eligible = [r for r in table.rows if satisfies(r)]
    if not eligible:
        raise ValueError(
            "no config satisfies the constraints "
            f"(max_ttft_ms={max_ttft_ms}, max_cost_usd={max_cost_usd})"
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
        binding = "; ".join(violated) if violated else "unknown"

    reason = (
        f"{winner.config.describe()} is the cheapest config satisfying "
        f"max_ttft_ms={max_ttft_ms}, max_cost_usd={max_cost_usd} "
        f"at {winner.total_cost_usd:.4f} USD and {winner.mean_ttft_ms:.1f} ms TTFT."
    )
    return Recommendation(
        config=winner.config, reason=reason, binding_constraint=binding
    )
