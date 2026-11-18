'''
@author: Michael

Tests for the high precision Decimal functions (src/phase/decimal_additional_functions/decimal_additional_functions.py).
'''

import math
from decimal import Decimal, getcontext

import pytest

from phase.decimal_additional_functions import cos, exp, pi, precision, sin
from phase.decimal_additional_functions.decimal_additional_functions import moneyfmt

PI_50 = Decimal("3.14159265358979323846264338327950288419716939937510")
E_50 = Decimal("2.71828182845904523536028747135266249775724709369995")
TOLERANCE = Decimal("1e-42")


def test_precision_is_set_globally():
    assert precision == 45
    assert getcontext().prec == precision


def test_pi():
    assert abs(pi() - PI_50) < TOLERANCE
    assert getcontext().prec == precision  # The temporary extra digits are removed again.


def test_exp():
    assert abs(exp(Decimal(1)) - E_50) < TOLERANCE
    assert exp(Decimal(0)) == 1
    assert abs(exp(Decimal("0.3")) * exp(Decimal("1.2")) - exp(Decimal("1.5"))) < TOLERANCE


@pytest.mark.parametrize("x, sine, cosine", [
    (PI_50 / 6, Decimal("0.5"), Decimal(3).sqrt() / 2),
    (PI_50 / 4, Decimal(2).sqrt() / 2, Decimal(2).sqrt() / 2),
    (PI_50 / 2, Decimal(1), Decimal(0)),
    (PI_50, Decimal(0), Decimal(-1)),
])
def test_sin_and_cos_at_special_angles(x, sine, cosine):
    assert abs(sin(x) - sine) < TOLERANCE
    assert abs(cos(x) - cosine) < TOLERANCE


@pytest.mark.parametrize("x", [Decimal("-2.5"), Decimal("0.1"), Decimal("1"), Decimal("5.9")])
def test_sin_and_cos_match_math(x):
    assert float(sin(x)) == pytest.approx(math.sin(float(x)), abs = 1e-15)
    assert float(cos(x)) == pytest.approx(math.cos(float(x)), abs = 1e-15)
    assert abs(sin(x) ** 2 + cos(x) ** 2 - 1) < TOLERANCE
    assert sin(-x) == -sin(x)
    assert cos(-x) == cos(x)


def test_moneyfmt():
    d = Decimal("-1234567.8901")

    assert moneyfmt(d, curr = "$") == "-$1,234,567.89"
    assert moneyfmt(d, places = 0, sep = ".", dp = "", neg = "", trailneg = "-") == "1.234.568-"
    assert moneyfmt(d, curr = "$", neg = "(", trailneg = ")") == "($1,234,567.89)"
    assert moneyfmt(Decimal(123456789), sep = " ") == "123 456 789.00"
    assert moneyfmt(Decimal("-0.02"), neg = "<", trailneg = ">") == "<0.02>"


# END OF FILE
