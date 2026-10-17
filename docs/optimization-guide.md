# Optimization guide: what each technique trades off

Every optimization buys speed or money with *something*. This guide states
the price in plain terms.

## Quantization (fp16 → int8 → int4)

**What:** store model weights in fewer bits.
**Buys:** lower latency (less memory bandwidth per token) and lower cost
(smaller GPUs, cheaper per-token pricing).
**Costs:** quality. int8 is near-lossless on most tasks; int4 is visibly
lossy on hard reasoning. The cost is not uniform — code and math degrade
first, chit-chat last.
**Rule:** never ship a quantization change without re-running quality
evals. The benchmark table tells you what you gain; only evals tell you
what you paid.

## Batching

**What:** decode several requests together.
**Buys:** throughput — tokens/sec climbs sub-linearly with batch size.
**Costs:** per-request latency. Requests wait for the batch to fill and
share scheduling overhead, so TTFT rises. Tail latency (p99) degrades
before the mean does.
**Rule:** batch for throughput-bound workloads (offline jobs, bulk
pipelines); keep batches small for interactive chat.

## KV cache

**What:** reuse computed key/value tensors for prompt prefixes the model
has already seen.
**Buys:** prefill time on repeated prefixes — shared system prompts,
few-shot examples, RAG templates.
**Costs:** GPU memory (the cache lives somewhere) and complexity on
prefix changes. Zero benefit when every prompt is unique.
**Rule:** biggest win for agent loops and chat apps with a fixed system
prompt; measure on the `repeated-prefix` workload.

## Prompt cache (exact-match)

**What:** skip prefill entirely when the identical prompt repeats.
**Buys:** near-zero TTFT on cache hits.
**Costs:** only helps exact repeats — same tokens, same order. Useless
for open-ended chat; excellent for templated pipelines and eval harnesses
that replay prompts.
**Rule:** check your hit rate first. Below ~20% hits, the bookkeeping is
not worth it.

## max_tokens caps

**What:** hard cap on generated tokens per request.
**Buys:** bounded cost and bounded tail latency. A runaway generation
can't burn a dollar or hang a request.
**Costs:** truncated answers when set too low. Silent quality loss —
the model doesn't know it was cut off.
**Rule:** set from measured output-length distributions (p99 + margin),
not from guesses.

## Putting it together

Optimizations compose, but their interactions are not always friendly:
batching fights TTFT, int4 fights quality, caching fights memory. The
lab's job is to make the trade-offs visible in one table so the choice
is explicit — and `recommend()` exists to turn "we have a 500 ms TTFT
budget" into "here is the cheapest config that meets it, and here is
the constraint that decided."
