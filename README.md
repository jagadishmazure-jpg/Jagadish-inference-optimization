# Jagadish-inference-optimization

A benchmark lab that **measures** what inference optimizations actually buy you —
TTFT, TPOT, tokens/sec, and cost — instead of assuming.

## Why this exists

Inference is where AI budgets die: every token has a price, and every
millisecond of latency is user patience spent. Teams routinely pick
quantization levels, batch sizes, and caching strategies on vibes. This repo
turns each knob into a row in a comparison table with measured numbers, so
"should we quantize to int4?" becomes a data question, not an opinion.

> **Simulation note:** the default backend (`SimulatedBackend`) uses a
> documented deterministic cost model — plausible relative effects, not real
> measurements. It exists so the benchmark logic, tables, and
> recommendations are fully testable offline. Swap in `OllamaLiveRunner`
> for real numbers (see "Live runs" below).

## The knobs

| Knob | Values | What it does |
| --- | --- | --- |
| `quantization` | fp16 / int8 / int4 | Weight precision. Lower = faster + cheaper, but lossy — int4 needs evals before production. |
| `batch_size` | ≥ 1 | Requests decoded together. Throughput scales sub-linearly; per-request TTFT pays scheduling overhead. |
| `kv_cache` | on / off | Reuses the key/value cache for repeated prompt prefixes (shared system prompts). |
| `prompt_cache` | on / off | Exact prompt repeats skip prefill entirely. |
| `max_tokens` | ≥ 1 | Generation cap. Caps cost and tail latency per request. |

## Quickstart

```python
from inferopt import (
    BenchmarkRunner, InferenceConfig, compare, recommend,
    short_chat, long_doc, repeated_prefix,
)

configs = [
    InferenceConfig(),                                            # fp16 baseline
    InferenceConfig(quantization="int8", batch_size=4),           # balanced
    InferenceConfig(quantization="int4", kv_cache="on"),          # aggressive
]
workloads = [short_chat(), long_doc(), repeated_prefix()]

results = BenchmarkRunner(seed=7).run_grid(configs, workloads)
print(compare(results).markdown())

rec = recommend(results, max_ttft_ms=500.0, max_cost_usd=0.05)
print(rec.reason)
print("binding constraint:", rec.binding_constraint)
```

## How to read the comparison table

- **TTFT (ms)** — mean time to first token. Dominated by prefill: long
  inputs and no cache hurt here. This is what users *feel*.
- **TPOT (ms)** — mean time per output token. Dominated by decode;
  quantization moves this most.
- **tok/s** — effective throughput including batching gains.
- **cost (USD)** — simulated spend for the workload set. Priced per
  million tokens; int4 ≈ 5× cheaper than fp16 in the model.
- **quality** — the trade-off note. int4's speed is real; so is its
  quality risk. Never ship a quantization change without re-running
  your quality evals.

`recommend()` picks the cheapest config satisfying your constraints and
tells you which constraint actually decided — the *binding* constraint.
If the cheapest config wins outright, it says so.

## Live runs (Ollama)

```bash
ollama serve
ollama pull llama3.1:8b
```

```python
from inferopt import OllamaLiveRunner, InferenceConfig, short_chat

runner = OllamaLiveRunner(model="llama3.1:8b")
if runner.available():
    metrics = runner.run(InferenceConfig(), short_chat())
    print(metrics)
```

The live runner exposes the same `run(config, workload)` interface as the
simulated backend, so tables built from real measurements drop straight
into `compare()` / `recommend()`. Quantization mapping is approximate —
pick the Ollama model tag whose quantization matches what you want to
measure (e.g. a `Q4_K_M` tag for int4).

## Production path

- **vLLM / TensorRT-LLM** replace the toy backends: continuous batching,
  paged KV cache, and real quantization kernels.
- Split input/output pricing (production bills them differently).
- Measure p99 TTFT/TPOT, not just means — tail latency is the SLA.
- Gate every quantization change on quality evals (see the
  `Jagadish-llm-evals-observability` companion project).

## Synthetic data

All workloads are synthetic token counts. No real prompts, no PII, no
model weights ship with this repo.
