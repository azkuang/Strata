"""Fast (non-GPU) tests for the M5 benchmark prompt pool."""

import pytest

from strata.benchmark_prompts import BASE_PROMPTS, make_prompt_pool


def test_make_prompt_pool_returns_exact_count():
    assert len(make_prompt_pool(1)) == 1
    assert len(make_prompt_pool(8)) == 8
    assert len(make_prompt_pool(32)) == 32


def test_make_prompt_pool_cycles_deterministically_through_base_prompts():
    pool = make_prompt_pool(len(BASE_PROMPTS) + 3)
    assert pool[: len(BASE_PROMPTS)] == BASE_PROMPTS
    assert pool[len(BASE_PROMPTS)] == BASE_PROMPTS[0]
    assert pool[len(BASE_PROMPTS) + 1] == BASE_PROMPTS[1]


def test_make_prompt_pool_raises_for_non_positive_n():
    with pytest.raises(ValueError):
        make_prompt_pool(0)
    with pytest.raises(ValueError):
        make_prompt_pool(-1)
