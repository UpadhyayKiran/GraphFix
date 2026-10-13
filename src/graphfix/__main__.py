"""
Purpose:
    Expose the public main names used by other GraphFix modules.

Input:
    Imports from the implementation modules in this package.

Output:
    A small, stable package-level import surface.

Pipeline stage:
    Package wiring for the GraphFix localization pipeline.
"""

from graphfix.cli import main

raise SystemExit(main())
