"""Exact algebra checks used by reproduce.py."""

from __future__ import annotations

import sympy as sp


def run_symbolic_checks() -> dict[str, str]:
    """Verify the Squier identities and commutator invariants exactly."""
    s = sp.symbols("s", nonzero=True)
    beta_one = sp.Matrix([[-s**2, s], [0, 1]])
    beta_two = sp.Matrix([[1, 0], [s, -s**2]])
    form = sp.Matrix([[s + s**-1, -1], [-1, s + s**-1]])

    def star(matrix: sp.Matrix) -> sp.Matrix:
        return matrix.T.xreplace({s: s**-1})

    zero = sp.zeros(2)
    assert sp.simplify(star(beta_one) * form * beta_one - form) == zero
    assert sp.simplify(star(beta_two) * form * beta_two - form) == zero
    assert sp.simplify(
        beta_one * beta_two * beta_one - beta_two * beta_one * beta_two
    ) == zero

    order_difference = sp.simplify(beta_one * beta_two - beta_two * beta_one)
    expected_difference = sp.Matrix(
        [[s**2, -s * (1 + s**2)], [s * (1 + s**2), -s**2]]
    )
    assert sp.simplify(order_difference - expected_difference) == zero

    group_commutator = sp.simplify(
        beta_two.inv() * beta_one.inv() * beta_two * beta_one
    )
    expected_trace = 1 - s**2 - s**-2
    assert sp.simplify(sp.det(group_commutator) - 1) == 0
    assert sp.simplify(sp.trace(group_commutator) - expected_trace) == 0

    determinant_form = sp.factor(form.det())
    assert sp.simplify(determinant_form - (s**4 + s**2 + 1) / s**2) == 0

    return {
        "beta_i_star_J_beta_i": "J for i=1,2",
        "braid_relation": "beta_1 beta_2 beta_1 = beta_2 beta_1 beta_2",
        "det_J_s": str(determinant_form),
        "det_group_commutator": "1",
        "trace_group_commutator": str(expected_trace),
        "order_difference": str(order_difference),
    }
