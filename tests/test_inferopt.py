"""Tests for inferopt. Deterministic, stdlib only, no network."""

import urllib.error
from unittest import mock

import pytest

from inferopt import (
    BenchmarkResult,
    BenchmarkRunner,
    InferenceConfig,
    LiveUnavailable,
    OllamaLiveRunner,
    RunMetrics,
    SimulatedBackend,
    compare,
    long_doc,
    recommend,
    repeated_prefix,
    short_chat,
)


# --- configs -------------------------------------------------------------

def test_config_defaults_valid():
    cfg = InferenceConfig()
    assert cfg.quantization == "fp16"
    assert "fp16" in cfg.describe()


def test_config_rejects_bad_quantization():
    with pytest.raises(ValueError):
        InferenceConfig(quantization="fp32")


def test_config_rejects_bad_batch_size():
    with pytest.raises(ValueError):
        InferenceConfig(batch_size=0)


def test_config_rejects_bad_max_tokens():
    with pytest.raises(ValueError):
        InferenceConfig(max_tokens=0)


def test_int4_without_kv_cache_warns():
    cfg = InferenceConfig(quantization="int4", kv_cache="off")
    assert any("kv_cache" in w for w in cfg.warnings())


def test_int4_always_warns_about_quality():
    cfg = InferenceConfig(quantization="int4")
    assert any("eval" in w for w in cfg.warnings())


def test_fp16_has_no_warnings():
    assert InferenceConfig().warnings() == []


# --- workloads -----------------------------------------------------------

def test_short_chat_shape():
    wl = short_chat()
    assert len(wl.prompts) == 20
    assert all(p.output_tokens == 50 for p in wl.prompts)


def test_long_doc_shape():
    wl = long_doc()
    assert len(wl.prompts) == 5
    assert all(p.input_tokens == 2000 for p in wl.prompts)


def test_repeated_prefix_shares_prefix_id():
    wl = repeated_prefix()
    assert len(wl.prompts) == 10
    assert len({p.prefix_id for p in wl.prompts}) == 1


# --- costmodel -----------------------------------------------------------

def test_ttft_ordering_by_quantization():
    wl = short_chat()
    ttfts = {}
    for q in ("fp16", "int8", "int4"):
        m = SimulatedBackend().run(InferenceConfig(quantization=q), wl)
        ttfts[q] = m.ttft_ms
    assert ttfts["int4"] < ttfts["int8"] < ttfts["fp16"]


def test_cost_ordering_by_quantization():
    wl = short_chat()
    costs = {}
    for q in ("fp16", "int8", "int4"):
        m = SimulatedBackend().run(InferenceConfig(quantization=q), wl)
        costs[q] = m.est_cost_usd
    assert costs["int4"] < costs["int8"] < costs["fp16"]


def test_batching_improves_throughput():
    wl = short_chat()
    tps1 = SimulatedBackend().run(InferenceConfig(batch_size=1), wl).tokens_per_sec
    tps4 = SimulatedBackend().run(InferenceConfig(batch_size=4), wl).tokens_per_sec
    assert tps4 > tps1


def test_kv_cache_helps_repeated_prefix_workload():
    wl = repeated_prefix()
    off = SimulatedBackend().run(InferenceConfig(kv_cache="off"), wl).ttft_ms
    on = SimulatedBackend().run(InferenceConfig(kv_cache="on"), wl).ttft_ms
    assert on < off


def test_kv_cache_no_benefit_without_repeated_prefix():
    wl = short_chat()  # every prompt has a distinct prefix_id
    off = SimulatedBackend().run(InferenceConfig(kv_cache="off"), wl).ttft_ms
    on = SimulatedBackend().run(InferenceConfig(kv_cache="on"), wl).ttft_ms
    assert on == pytest.approx(off)


def test_prompt_cache_second_run_is_hit():
    wl = short_chat()
    backend = SimulatedBackend()
    cfg = InferenceConfig(prompt_cache="on")
    first = backend.run(cfg, wl).ttft_ms
    second = backend.run(cfg, wl).ttft_ms
    assert second < first


def test_max_tokens_caps_cost():
    wl = short_chat()  # 50 output tokens per prompt
    cheap = SimulatedBackend().run(InferenceConfig(max_tokens=10), wl).est_cost_usd
    full = SimulatedBackend().run(InferenceConfig(max_tokens=512), wl).est_cost_usd
    assert cheap < full


def test_metrics_carry_prompt_count():
    m = SimulatedBackend().run(InferenceConfig(), long_doc())
    assert m.num_prompts == 5


# --- benchmark -----------------------------------------------------------

def _grid(seed=7):
    configs = [InferenceConfig(), InferenceConfig(quantization="int4")]
    workloads = [short_chat(), long_doc()]
    return BenchmarkRunner(seed=seed).run_grid(configs, workloads)


def test_grid_completes_all_cells():
    assert len(_grid()) == 2 * 2


def test_grid_is_deterministic_for_same_seed():
    a = _grid(seed=7)
    b = _grid(seed=7)
    for ra, rb in zip(a, b):
        assert ra.metrics.ttft_ms == rb.metrics.ttft_ms
        assert ra.metrics.est_cost_usd == rb.metrics.est_cost_usd


# --- report --------------------------------------------------------------

def test_compare_one_row_per_config():
    table = compare(_grid())
    assert len(table.rows) == 2
    assert table.rows[0].total_cost_usd <= table.rows[1].total_cost_usd


def test_compare_marks_int4_quality_risk():
    table = compare(_grid())
    int4_row = next(r for r in table.rows if r.config.quantization == "int4")
    assert "eval" in int4_row.quality_note


def test_compare_markdown_has_header():
    md = compare(_grid()).markdown()
    assert md.startswith("| config |")


def test_recommend_unconstrained_picks_cheapest():
    rec = recommend(_grid())
    assert rec.config.quantization == "int4"
    assert rec.binding_constraint is None


def test_recommend_respects_ttft_constraint_and_cites_it():
    # Hand-built grid: cheapest config is slow, pricier config is fast.
    # (In the simulated model cheap and fast align, so the grid alone
    # can never exercise the binding-constraint path.)
    cheap = InferenceConfig(quantization="int4")
    fast = InferenceConfig(quantization="fp16")
    results = [
        BenchmarkResult(cheap, "w", RunMetrics(500.0, 18.0, 55.0, 1.0, 10)),
        BenchmarkResult(fast, "w", RunMetrics(100.0, 45.0, 22.0, 5.0, 10)),
    ]
    rec = recommend(results, max_ttft_ms=200.0)
    assert rec.config.quantization == "fp16"
    assert rec.binding_constraint is not None
    assert "max_ttft_ms" in rec.binding_constraint
    assert "500.0" in rec.binding_constraint


def test_recommend_impossible_constraints_raise():
    with pytest.raises(ValueError):
        recommend(_grid(), max_ttft_ms=0.0001)


# --- live -----------------------------------------------------------------

def _refused(*args, **kwargs):
    raise urllib.error.URLError("connection refused")


def test_live_available_false_when_refused():
    runner = OllamaLiveRunner()
    with mock.patch("urllib.request.urlopen", side_effect=_refused):
        assert runner.available() is False


def test_live_available_true_on_200():
    runner = OllamaLiveRunner()
    fake = mock.MagicMock()
    fake.status = 200
    fake.__enter__.return_value = fake
    with mock.patch("urllib.request.urlopen", return_value=fake):
        assert runner.available() is True


def test_live_run_raises_when_unavailable():
    runner = OllamaLiveRunner()
    with mock.patch("urllib.request.urlopen", side_effect=_refused):
        with pytest.raises(LiveUnavailable):
            runner.run(InferenceConfig(), short_chat())


def test_live_run_parses_ollama_timings():
    runner = OllamaLiveRunner()
    tag_resp = mock.MagicMock()
    tag_resp.status = 200
    tag_resp.__enter__.return_value = tag_resp
    gen_resp = mock.MagicMock()
    gen_resp.read.return_value = (
        b'{"prompt_eval_duration": 200000000, "eval_duration": 900000000,'
        b' "eval_count": 20}'
    )
    gen_resp.__enter__.return_value = gen_resp

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        return tag_resp if url.endswith("/api/tags") else gen_resp

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        m = runner.run(InferenceConfig(), short_chat())
    assert m.ttft_ms == pytest.approx(200.0)
    assert m.tpot_ms == pytest.approx(45.0)  # 900ms / 20 tokens
    assert m.num_prompts == 20
