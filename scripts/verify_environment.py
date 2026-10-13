"""
Purpose:
    Check the supported Python range, GraphFix source and installation, and Git executable.

Input:
    The current Python environment and system Git installation.

Output:
    A short environment status printed to the console.

Pipeline stage:
    Environment verification before setup or a demo.

When to run:
    Run this script when you need this operational check or stage.

Run example:
    python scripts/verify_environment.py
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
SOURCE_PACKAGE = SOURCE_ROOT / "graphfix" / "__init__.py"
REQUIRED_PYTHON = ">=3.11,<3.12"


def supported_python() -> bool:
    """Return whether the running interpreter is Python 3.11.x."""
    return sys.version_info[:2] == (3, 11)


def source_tree_import() -> ModuleType:
    """Load only the source-tree package initializer without changing ``sys.path``."""
    spec = importlib.util.spec_from_file_location("_graphfix_source_check", SOURCE_PACKAGE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {SOURCE_PACKAGE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def installed_package_version() -> str:
    """Import the installed package normally and return its distribution version."""
    distribution_version = version("graphfix-local")
    module = importlib.import_module("graphfix")
    module_version = getattr(module, "__version__", None)
    if module_version != distribution_version:
        raise ImportError(
            "installed module version does not match the graphfix-local distribution"
        )
    return distribution_version


def git_version() -> str:
    """Return the installed Git version, or a clear unavailable status."""
    try:
        result = subprocess.run(
            ["git", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return f"unavailable ({error})"
    return result.stdout.strip()


def main() -> int:
    """Print separate setup checks and return a failing status if any required check fails."""
    valid = True
    print(f"Python version: {sys.version.split()[0]}")
    if supported_python():
        print(f"Python requirement ({REQUIRED_PYTHON}): OK")
    else:
        print(f"Unsupported Python version. GraphFix requires Python {REQUIRED_PYTHON}.")
        valid = False

    try:
        source_module = source_tree_import()
    except (ImportError, OSError) as error:
        print(f"GraphFix source-tree import check: failed ({error})")
        valid = False
    else:
        print(
            "GraphFix source-tree import check: "
            f"OK (version {source_module.__version__})"
        )

    try:
        package_version = installed_package_version()
    except (PackageNotFoundError, ImportError) as error:
        print(f"GraphFix installed-package import check: failed ({error})")
        valid = False
    else:
        print(f"GraphFix installed-package import check: OK (version {package_version})")

    installed_git = git_version()
    print(f"Git version: {installed_git}")
    if installed_git.startswith("unavailable"):
        valid = False

    print(f"Environment verification: {'OK' if valid else 'FAILED'}")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
