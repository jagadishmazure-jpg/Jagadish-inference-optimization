"""Live runner against a local Ollama server.

Same interface as SimulatedBackend.run(config, workload) so benchmark
tables can be produced from real measurements. Requires Ollama running
on localhost:11434 (`ollama serve`) with the model pulled
(`ollama pull llama3.1:8b`).

Quantization mapping is approximate: Ollama serves quantized GGUF
builds; pick the model tag whose quantization matches the config you
want to measure (e.g. a Q4_K_M tag for int4). The runner does not
pretend to control server-side quantization — it documents what ran.
"""

import json
import urllib.error
import urllib.request

from .configs import InferenceConfig
from .costmodel import PRICE_PER_MTOK_USD, RunMetrics
from .workloads import Workload


class LiveUnavailable(Exception):
    """Raised when no Ollama server is reachable."""


class OllamaLiveRunner:
    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        timeout_s: float = 10.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s

    def available(self) -> bool:
        """Probe the server. Never raises."""
        try:
            with urllib.request.urlopen(
                f"{self.host}/api/tags", timeout=self.timeout_s
            ) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    def _generate(self, prompt_text: str, num_predict: int) -> dict:
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt_text,
                "stream": False,
                "options": {"num_predict": num_predict},
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise LiveUnavailable(
                f"Ollama not reachable at {self.host}: {exc}"
            ) from exc

    def run(self, config: InferenceConfig, workload: Workload) -> RunMetrics:
        if not self.available():
            raise LiveUnavailable(f"Ollama not reachable at {self.host}")

        ttft_sum = 0.0
        tpot_sum = 0.0
        total_in = 0
        total_out = 0
        n = 0

        for i, prompt in enumerate(workload.prompts):
            # Synthetic prompt text sized ~ input_tokens (approx 2 tokens/word).
            text = f"Benchmark prompt {i}: " + "lorem " * max(
                1, prompt.input_tokens // 2
            )
            num_predict = min(prompt.output_tokens, config.max_tokens)
            data = self._generate(text, num_predict)

            # Ollama reports nanosecond timings.
            prompt_eval_ns = data.get("prompt_eval_duration", 0) or 0
            eval_ns = data.get("eval_duration", 0) or 0
            eval_count = data.get("eval_count", 0) or 0

            ttft_sum += (prompt_eval_ns / 1e6) or 0.0
            tpot_sum += (eval_ns / 1e6) / eval_count if eval_count else 0.0
            total_in += prompt.input_tokens
            total_out += num_predict
            n += 1

        price = PRICE_PER_MTOK_USD.get(config.quantization, 0.0)
        return RunMetrics(
            ttft_ms=ttft_sum / n if n else 0.0,
            tpot_ms=tpot_sum / n if n else 0.0,
            tokens_per_sec=(1000.0 / (tpot_sum / n)) if n and tpot_sum else 0.0,
            est_cost_usd=(total_in + total_out) / 1_000_000 * price,
            num_prompts=n,
        )
