"""Benchmark grid -> comparison table -> quality-gated recommendation.

ALL latency, throughput and cost numbers below are from SimulatedBackend's
documented cost model, not real measurements. The quality scores are
placeholder assumptions (SIMULATED_QUALITY) standing in for your own evals.
"""
from inferopt import (
    SIMULATED_QUALITY,
    BenchmarkRunner,
    InferenceConfig,
    compare,
    long_doc,
    recommend,
    repeated_prefix,
    short_chat,
)

configs = [
    InferenceConfig(),                                   # fp16, KV cache on (normal server default)
    InferenceConfig(quantization="int8", batch_size=4),
    InferenceConfig(quantization="int4", kv_cache="on"),
]
results = BenchmarkRunner(seed=7).run_grid(configs, [short_chat(), long_doc(), repeated_prefix()])
print("SIMULATED results (cost model, not a GPU):")
print(compare(results).markdown())

print("\n1) No eval evidence supplied:")
rec = recommend(results, max_ttft_ms=500.0, max_cost_usd=0.05)
print("  pick:", rec.config.describe())
print("  binding:", rec.binding_constraint)
print("  quality check:", rec.quality_check)

print(f"\n2) Placeholder eval scores {SIMULATED_QUALITY}, floor 0.97 of fp16:")
rec = recommend(results, max_ttft_ms=500.0, max_cost_usd=0.05, quality=SIMULATED_QUALITY)
print("  pick:", rec.config.describe())
print("  excluded:", list(rec.excluded))

print("\n3) Evals show int4 within 2% of fp16:")
rec = recommend(results, max_ttft_ms=500.0, max_cost_usd=0.05,
                quality={"fp16": 0.90, "int8": 0.89, "int4": 0.885})
print("  pick:", rec.config.describe(), "|", rec.quality_check)
