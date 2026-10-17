"""Grid benchmark runner: configs x workloads, deterministic via seed."""

import random
from dataclasses import dataclass, replace

from .configs import InferenceConfig
from .costmodel import RunMetrics, SimulatedBackend
from .workloads import Workload


@dataclass(frozen=True)
class BenchmarkResult:
    config: InferenceConfig
    workload_name: str
    metrics: RunMetrics


class BenchmarkRunner:
    """Runs every config against every workload.

    A fresh SimulatedBackend per run keeps cache state from leaking
    across grid cells. Small seeded gaussian noise simulates measurement
    jitter; the same seed always reproduces the same table.
    """

    def __init__(self, seed: int = 7, noise_pct: float = 0.02) -> None:
        self.seed = seed
        self.noise_pct = noise_pct

    def run_grid(
        self,
        configs: list,
        workloads: list,
    ) -> list:
        rng = random.Random(self.seed)
        results = []
        for config in configs:
            for workload in workloads:
                backend = SimulatedBackend()
                metrics = backend.run(config, workload)
                metrics = self._add_noise(rng, metrics)
                results.append(
                    BenchmarkResult(
                        config=config,
                        workload_name=workload.name,
                        metrics=metrics,
                    )
                )
        return results

    def _add_noise(self, rng: random.Random, metrics: RunMetrics) -> RunMetrics:
        if self.noise_pct <= 0:
            return metrics

        def jitter(x: float) -> float:
            return x * (1.0 + rng.gauss(0.0, self.noise_pct))

        return replace(
            metrics,
            ttft_ms=jitter(metrics.ttft_ms),
            tpot_ms=jitter(metrics.tpot_ms),
            tokens_per_sec=jitter(metrics.tokens_per_sec),
            est_cost_usd=jitter(metrics.est_cost_usd),
        )
