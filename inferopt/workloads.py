"""Synthetic workloads. Token counts are illustrative, not measured."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    input_tokens: int
    output_tokens: int
    prefix_id: str = ""  # shared id => repeated prompt prefix


@dataclass(frozen=True)
class Workload:
    name: str
    prompts: tuple

    def total_tokens(self) -> int:
        return sum(p.input_tokens + p.output_tokens for p in self.prompts)


def short_chat() -> Workload:
    """20 quick chat turns: small in, small out, no shared prefix."""
    return Workload(
        name="short-chat",
        prompts=tuple(
            Prompt(input_tokens=30, output_tokens=50, prefix_id=f"chat-{i}")
            for i in range(20)
        ),
    )


def long_doc() -> Workload:
    """5 document Q&A turns: big in (2k tokens), moderate out."""
    return Workload(
        name="long-doc",
        prompts=tuple(
            Prompt(input_tokens=2000, output_tokens=200, prefix_id=f"doc-{i}")
            for i in range(5)
        ),
    )


def repeated_prefix() -> Workload:
    """10 turns sharing one 1500-token system prefix.

    The workload kv_cache was made for: first prompt pays full prefill,
    the other nine should be heavily discounted.
    """
    return Workload(
        name="repeated-prefix",
        prompts=tuple(
            Prompt(input_tokens=1550, output_tokens=100, prefix_id="system-v1")
            for _ in range(10)
        ),
    )


ALL_WORKLOADS = (short_chat, long_doc, repeated_prefix)
