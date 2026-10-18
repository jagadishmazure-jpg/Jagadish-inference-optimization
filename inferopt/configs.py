"""Inference configuration: the knobs this lab benchmarks."""

from dataclasses import dataclass

QUANTIZATIONS = ("fp16", "int8", "int4")
CACHE_STATES = ("on", "off")


@dataclass(frozen=True)
class InferenceConfig:
    """One point in the optimization space.

    quantization: weight precision. int4 < int8 < fp16 in latency/cost,
        at the price of potential quality loss (validate on evals).
    batch_size: requests decoded together. Improves throughput
        sub-linearly; per-request latency may rise slightly.
    kv_cache: reuse key/value cache for repeated prompt prefixes.
        Helps workloads that share a system prompt.
    prompt_cache: exact-prompt cache. A repeat of an identical prompt
        skips prefill entirely (cache hit).
    max_tokens: generation cap per prompt. Caps cost and tail latency.
    """

    quantization: str = "fp16"
    batch_size: int = 1
    kv_cache: str = "on"
    prompt_cache: str = "off"
    max_tokens: int = 512

    def __post_init__(self) -> None:
        if self.quantization not in QUANTIZATIONS:
            raise ValueError(
                f"quantization must be one of {QUANTIZATIONS}, got {self.quantization!r}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.kv_cache not in CACHE_STATES:
            raise ValueError(f"kv_cache must be 'on'/'off', got {self.kv_cache!r}")
        if self.prompt_cache not in CACHE_STATES:
            raise ValueError(
                f"prompt_cache must be 'on'/'off', got {self.prompt_cache!r}"
            )
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")

    def warnings(self) -> list:
        """Config smells worth a second look. Warnings, not errors."""
        out = []
        if self.quantization == "int4" and self.kv_cache == "off":
            out.append(
                "int4 with kv_cache off: you pay the quality cost of aggressive "
                "quantization without the latency win of prefix reuse."
            )
        if self.quantization == "int4":
            out.append(
                "int4 is lossy: re-run your quality evals before trusting it "
                "in production."
            )
        if self.batch_size > 8:
            out.append(
                f"batch_size={self.batch_size}: throughput keeps climbing but "
                "per-request tail latency degrades — measure p99, not the mean."
            )
        return out

    def describe(self) -> str:
        """One-line summary, stable for use as a table key."""
        return (
            f"{self.quantization}/b{self.batch_size}"
            f"/kv:{self.kv_cache}/pc:{self.prompt_cache}"
            f"/mt:{self.max_tokens}"
        )
