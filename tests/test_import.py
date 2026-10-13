"""
Purpose:
    Verify import behavior without requiring network access.

Input:
    Small temporary repositories, records, models, or fixtures created by the tests.

Output:
    Passing assertions when the implementation follows the Review-1 contract.

Pipeline stage:
    Regression testing for the GraphFix implementation.
"""

import graphfix


def test_package_import() -> None:
    """The GraphFix package is importable and exposes its version."""
    assert graphfix.__version__ == "0.1.0"
