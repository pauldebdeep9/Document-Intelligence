from __future__ import annotations

import pytest

from isc.common.confidence import Confidence, Signal, Thresholds


def test_independent_is_pessimistic():
    c = Confidence.independent(Confidence.of(Signal.OCR, 0.9), Confidence.of(Signal.MODEL, 0.9))
    assert c.score == pytest.approx(0.81)


def test_weakest_link_takes_minimum():
    c = Confidence.weakest_link(Confidence.of(Signal.OCR, 0.9), Confidence.of(Signal.MODEL, 0.4))
    assert c.score == pytest.approx(0.4)


def test_corroborate_can_only_increase():
    a, b = Confidence.of(Signal.LEXICAL, 0.7), Confidence.of(Signal.MASTER_DATA, 0.8)
    assert Confidence.corroborate(a, b).score > max(a.score, b.score)


def test_factors_survive_combination():
    c = Confidence.independent(Confidence.of(Signal.OCR, 0.9), Confidence.of(Signal.MODEL, 0.8))
    assert {f.signal for f in c.factors} == {Signal.OCR, Signal.MODEL}


def test_weakest_identifies_the_culprit():
    c = Confidence.independent(
        Confidence.of(Signal.OCR, 0.3, "blurry scan"), Confidence.of(Signal.MODEL, 0.95)
    )
    weakest = c.weakest()
    assert weakest is not None and weakest.signal is Signal.OCR


@pytest.mark.parametrize(
    "score,expected",
    [(0.95, "accept"), (0.7, "review"), (0.4, "low_confidence"), (0.1, "reject")],
)
def test_routing(score, expected):
    assert Thresholds().route(Confidence(score)) == expected


def test_out_of_range_rejected():
    with pytest.raises(ValueError):
        Confidence(1.5)
