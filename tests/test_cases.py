"""Тест-кейси з zoriana_bot/cases.py: сценарії лаби 2 і додаткові проблемні ситуації."""

import pytest

from zoriana_bot.cases import CASES, run_case


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_case(case):
    result = run_case(case)
    assert not result["problems"], result["problems"]
