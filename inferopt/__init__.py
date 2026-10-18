"""inferopt — inference optimization benchmark lab.

Measure what inference optimizations actually buy you (TTFT, TPOT,
tokens/sec, cost) with a documented simulated backend, and re-run the
same workloads against a live Ollama server when one is available.
"""

from .benchmark import BenchmarkResult, BenchmarkRunner
from .configs import InferenceConfig
from .costmodel import RunMetrics, SimulatedBackend
from .live import LiveUnavailable, OllamaLiveRunner
from .report import SIMULATED_QUALITY, ComparisonRow, ComparisonTable, Recommendation, compare, recommend
from .workloads import Workload, long_doc, repeated_prefix, short_chat

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
    "SIMULATED_QUALITY",
    "LiveUnavailable",
    "OllamaLiveRunner",
]
