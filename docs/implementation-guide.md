# Implementation guide: from offline build to Azure

> Status: **planned**. Nothing here has been deployed. The offline build in this
> repository is what exists today; this guide is how it would be taken to a
> real Azure environment.

## Azure service mapping

| Offline component | Azure service (planned) | Notes |
|---|---|---|
| Serving | vLLM on Azure Container Apps serverless GPU or AKS GPU node pools |  |
| Hosted models | Azure AI Foundry deployments (provisioned vs standard) | Compare cost per 1M tokens at expected load |
| Quality evals | Azure AI Foundry evaluations on a fixed set per quantisation | Feeds the quality gate |
| Metrics | Azure Monitor: TTFT, TPOT, tokens per second, GPU utilisation |  |

## Steps

1. Run the live runner (or vLLM's benchmark) on the target GPU and replace the simulated table.
2. Run the task's eval set for fp16, int8 and int4 and pass the scores to `recommend()`.
3. Set latency and cost limits from the product requirement, not defaults.

## Guardrails to keep when going live

- int4 is never recommended without eval scores for fp16 and int4 that pass the floor.
- Every recommendation states the binding constraint and what was excluded and why.
- Simulated numbers are labelled as simulated in code, output and docs.

## What to measure before calling it production-ready

- TTFT p95 and TPOT p95 at the expected concurrency.
- Cost per 1M output tokens at real utilisation.
- Eval score of each quantisation relative to fp16.
