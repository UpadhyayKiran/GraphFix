"""Export an existing GraphFix RepositoryGraph to optional Neo4j persistence."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from graphfix.graph import build_repository_graph
from graphfix.graph.neo4j_export import Neo4jGraphExporter, load_neo4j_driver
from graphfix.parsing import PythonRepositoryParser
from graphfix.repositories import RepositoryManager, RepositoryRequest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = ROOT / "data" / "cache" / "repositories"
DEFAULT_REPOSITORY = "vllm-project/vllm"
DEFAULT_INSTANCE = "vllm-project__vllm-9390"
DEFAULT_COMMIT = "83450458339b07765b0e72a822e5fe93eeaf5258"


@dataclass(frozen=True)
class Neo4jSettings:
    """Credentials read only when the optional export command runs."""

    uri: str
    user: str
    password: str
    database: str

    @classmethod
    def from_environment(cls) -> Neo4jSettings:
        """Read required Neo4j connection values without embedding credentials."""
        required = ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            names = ", ".join(missing)
            raise RuntimeError(f"Missing required Neo4j environment variable(s): {names}")
        return cls(
            uri=os.environ["NEO4J_URI"],
            user=os.environ["NEO4J_USER"],
            password=os.environ["NEO4J_PASSWORD"],
            database=os.environ.get("NEO4J_DATABASE", "neo4j"),
        )


def parse_args() -> argparse.Namespace:
    """Parse repository identity and safe cache location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--instance-id", default=DEFAULT_INSTANCE)
    parser.add_argument("--commit", default=DEFAULT_COMMIT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--batch-size", type=int, default=1_000)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main() -> int:
    """Reconstruct, parse, build once in Python, then export the resulting graph."""
    args = parse_args()
    try:
        settings = Neo4jSettings.from_environment()
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    request = RepositoryRequest(
        instance_id=args.instance_id,
        repo=args.repository,
        base_commit=args.commit,
    )
    checkout = RepositoryManager(args.cache_root).reconstruct(request)
    if not checkout.success or checkout.checkout_path is None:
        detail = checkout.failure.message if checkout.failure else "unknown reconstruction error"
        raise RuntimeError(f"Repository reconstruction failed: {detail}")
    parsed = PythonRepositoryParser().parse(Path(checkout.checkout_path))
    graph = build_repository_graph(parsed, repository_name=args.repository)

    driver = load_neo4j_driver(settings.uri, settings.user, settings.password)
    try:
        exporter = Neo4jGraphExporter(
            driver,
            database=settings.database,
            batch_size=args.batch_size,
        )
        summary = exporter.export(graph)
    finally:
        driver.close()

    print(f"Repository: {args.repository}")
    print(f"Commit: {checkout.resolved_commit}")
    print(f"Python RepositoryGraph nodes: {summary.python_nodes}")
    print(f"Python RepositoryGraph edges: {summary.python_edges}")
    print(f"Neo4j exported nodes: {summary.neo4j_nodes}")
    print(f"Neo4j exported relationships: {summary.neo4j_relationships}")
    print(f"Parity check: {'PASS' if summary.parity_ok else 'FAIL'}")
    return 0 if summary.parity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
