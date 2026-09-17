"""Exact arithmetic for short Weierstrass curves over Q."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import log


@dataclass(frozen=True)
class Point:
    x: Fraction | None = None
    y: Fraction | None = None

    @property
    def is_infinity(self) -> bool:
        return self.x is None and self.y is None

    def __str__(self) -> str:
        if self.is_infinity:
            return "O"
        return f"({self.x}, {self.y})"


INFINITY = Point()


@dataclass(frozen=True)
class EllipticCurve:
    """The curve y^2 = x^3 + a*x + b over the rationals."""

    a: Fraction
    b: Fraction

    def __post_init__(self) -> None:
        if 4 * self.a**3 + 27 * self.b**2 == 0:
            raise ValueError("singular curve: discriminant is zero")

    @property
    def discriminant(self) -> Fraction:
        return -16 * (4 * self.a**3 + 27 * self.b**2)

    def is_on_curve(self, point: Point) -> bool:
        if point.is_infinity:
            return True
        assert point.x is not None and point.y is not None
        return point.y**2 == point.x**3 + self.a * point.x + self.b

    def require_point(self, point: Point) -> None:
        if not self.is_on_curve(point):
            raise ValueError(f"point {point} is not on the curve")

    def negate(self, point: Point) -> Point:
        self.require_point(point)
        if point.is_infinity:
            return point
        assert point.x is not None and point.y is not None
        return Point(point.x, -point.y)

    def add(self, left: Point, right: Point) -> Point:
        self.require_point(left)
        self.require_point(right)
        if left.is_infinity:
            return right
        if right.is_infinity:
            return left
        assert left.x is not None and left.y is not None
        assert right.x is not None and right.y is not None
        if left.x == right.x and left.y == -right.y:
            return INFINITY
        if left == right:
            if left.y == 0:
                return INFINITY
            slope = (3 * left.x**2 + self.a) / (2 * left.y)
        else:
            if left.x == right.x:
                return INFINITY
            slope = (right.y - left.y) / (right.x - left.x)
        x3 = slope**2 - left.x - right.x
        y3 = slope * (left.x - x3) - left.y
        result = Point(x3, y3)
        self.require_point(result)
        return result

    def multiply(self, scalar: int, point: Point) -> Point:
        self.require_point(point)
        if scalar < 0:
            return self.multiply(-scalar, self.negate(point))
        result = INFINITY
        addend = point
        value = scalar
        while value:
            if value & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            value >>= 1
        return result


def naive_x_height(point: Point) -> float:
    """Logarithmic naive height h(x)=log(max(|num(x)|, den(x)))."""
    if point.is_infinity:
        return 0.0
    assert point.x is not None
    return log(max(abs(point.x.numerator), point.x.denominator))
