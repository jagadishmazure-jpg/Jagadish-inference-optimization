"""inferopt — inference optimization benchmark lab.

Measure what inference optimizations actually buy you (TTFT, TPOT,
tokens/sec, cost) with a documented simulated backend, and re-run the
same workloads against a live Ollama server when one is available.
"""

from .configs import InferenceConfig
from .costmodel import RunMetrics, SimulatedBackend
from .workloads import Workload, long_doc, repeated_prefix, short_chat
from .benchmark import BenchmarkResult, BenchmarkRunner
from .report import ComparisonRow, ComparisonTable, Recommendation, compare, recommend
from .live import LiveUnavailable, OllamaLiveRunner

__all__ = [
    "InferenceConfig",
    "RunMetrics",
    "SimulatedBackend",
    "Workload",
    "short_chat",
    "long_doc",
    "repeated_prefix",
    "BenchmarkResult",
    "BenchmarkRunner",
    "ComparisonRow",
    "ComparisonTable",
    "Recommendation",
    "compare",
    "recommend",
    "LiveUnavailable",
    "OllamaLiveRunner",
]
