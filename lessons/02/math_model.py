"""Small, checked affine calculations for the RUNTIME lessons."""
import math


def vector(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError('Expected a nonempty vector')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError('Vector values must be finite numbers')
    return list(values)


def matrix(rows):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError('Expected a nonempty matrix')
    result = [vector(row) for row in rows]
    if len({len(row) for row in result}) != 1:
        raise ValueError('Matrix must be rectangular')
    return result


def dot(a, b):
    a, b = vector(a), vector(b)
    if len(a) != len(b):
        raise ValueError('Dot-product dimensions differ')
    return math.fsum(x*y for x, y in zip(a, b))


def linear(x, weights, bias=None):
    """A row vector times W[input_dimension][output_dimension], plus bias."""
    x, weights = vector(x), matrix(weights)
    if len(x) != len(weights):
        raise ValueError('Linear input dimension differs from weight rows')
    n = len(weights[0])
    bias = [0.0]*n if bias is None else vector(bias)
    if len(bias) != n:
        raise ValueError('Bias dimension differs from output dimension')
    return [dot(x, [row[j] for row in weights]) + bias[j] for j in range(n)]

