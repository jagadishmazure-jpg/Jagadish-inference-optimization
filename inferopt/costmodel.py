"""Simulated inference backend.

DOCUMENTED SIMULATION — not real measurements. Every number below is a
plausible stand-in chosen so the *relative* effects of each optimization
are correct and testable:

  int4 is faster/cheaper than int8, which is faster/cheaper than fp16
  batching raises throughput sub-linearly
  kv_cache cuts prefill cost for repeated prompt prefixes
  prompt_cache turns exact repeats into near-free hits

Replace SimulatedBackend with OllamaLiveRunner (live.py) for real data.
"""

from dataclasses import dataclass

from .configs import InferenceConfig
from .workloads import Workload

# --- Simulated constants -------------------------------------------------
# Prefill (time-to-first-token) cost per 1,000 input tokens, in ms.
TTFT_BASE_MS_PER_1K = {"fp16": 180.0, "int8": 120.0, "int4": 85.0}
# Decode cost per output token (batch_size=1), in ms.
TPOT_BASE_MS = {"fp16": 45.0, "int8": 28.0, "int4": 18.0}
# Simulated price per million tokens (input and output priced equally here;
# production pricing usually splits them — see README).
PRICE_PER_MTOK_USD = {"fp16": 3.00, "int8": 1.50, "int4": 0.60}
# Fraction of prefill skipped when the kv cache already holds the prefix.
KV_CACHE_PREFIX_DISCOUNT = 0.60
# TTFT of an exact prompt-cache hit (no prefill at all).
PROMPT_CACHE_HIT_TTFT_MS = 5.0
# Throughput scales as batch_size ** BATCH_THROUGHPUT_EXPONENT (< 1:
# sub-linear, i.e. batching helps but with diminishing returns).
BATCH_THROUGHPUT_EXPONENT = 0.7
# Batching is not free for latency: scheduling/queueing overhead grows
# with batch size. Documented so "batch everything" is not a free lunch.
BATCH_TTFT_OVERHEAD_PER_SLOT = 0.05


@dataclass(frozen=True)
class RunMetrics:
    """Outcome of running one workload under one config."""

    ttft_ms: float  # mean time to first token
    tpot_ms: float  # mean time per output token
    tokens_per_sec: float  # effective decode throughput incl. batching
    est_cost_usd: float  # simulated spend for the workload
    num_prompts: int


class SimulatedBackend:
    """Deterministic latency/throughput/cost model.

    Keeps per-instance cache state (seen prefixes, prompt cache), so create
    a fresh instance per benchmark run for comparable numbers.
    """

    def __init__(self) -> None:
        self._seen_prefixes: set = set()
        self._prompt_cache: set = set()

    @staticmethod
    def _prompt_key(config: InferenceConfig, input_tokens: int,
                    output_tokens: int, prefix_id: str) -> tuple:
        return (config.describe(), input_tokens, output_tokens, prefix_id)

    def run(self, config: InferenceConfig, workload: Workload) -> RunMetrics:
        q = config.quantization
        ttft_sum = 0.0
        total_in = 0
        total_out = 0

        for prompt in workload.prompts:
            in_tok = prompt.input_tokens
            # max_tokens caps generation: the knob has a real effect.
            out_tok = min(prompt.output_tokens, config.max_tokens)
            total_in += in_tok
            total_out += out_tok

            key = self._prompt_key(config, in_tok, prompt.output_tokens,
                                   prompt.prefix_id)
            if config.prompt_cache == "on" and key in self._prompt_cache:
                # Exact repeat: prefill skipped entirely.
                ttft = PROMPT_CACHE_HIT_TTFT_MS
            else:
                prefill = (in_tok / 1000.0) * TTFT_BASE_MS_PER_1K[q]
                # Batch scheduling overhead: bigger batches wait longer.
                prefill *= 1.0 + BATCH_TTFT_OVERHEAD_PER_SLOT * (config.batch_size - 1)
                if (
                    config.kv_cache == "on"
                    and prompt.prefix_id
                    and prompt.prefix_id in self._seen_prefixes
                ):
                    # Repeated prefix: kv cache skips most of the prefill.
                    prefill *= 1.0 - KV_CACHE_PREFIX_DISCOUNT
                ttft = prefill
                if prompt.prefix_id:
                    self._seen_prefixes.add(prompt.prefix_id)
                if config.prompt_cache == "on":
                    self._prompt_cache.add(key)

            ttft_sum += ttft

        n = len(workload.prompts)
        avg_ttft = ttft_sum / n if n else 0.0
        tpot = TPOT_BASE_MS[q]
        tokens_per_sec = (1000.0 / tpot) * (config.batch_size ** BATCH_THROUGHPUT_EXPONENT)
        est_cost = (total_in + total_out) / 1_000_000 * PRICE_PER_MTOK_USD[q]

        return RunMetrics(
            ttft_ms=avg_ttft,
            tpot_ms=tpot,
            tokens_per_sec=tokens_per_sec,
            est_cost_usd=est_cost,
            num_prompts=n,
        )
