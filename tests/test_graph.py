"""
Purpose:
    Verify graph behavior without requiring network access.

Input:
    Small temporary repositories, records, models, or fixtures created by the tests.

Output:
    Passing assertions when the implementation follows the Review-1 contract.

Pipeline stage:
    Regression testing for the GraphFix implementation.
"""

from pathlib import Path

from graphfix.graph import build_repository_graph
from graphfix.parsing import PythonRepositoryParser


def test_builds_containment_definition_import_call_and_inheritance_edges(tmp_path: Path) -> None:
    """Verify builds containment definition import call and inheritance edges."""
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "base.py").write_text("class Base:\n    pass\n", encoding="utf-8")
    (package / "service.py").write_text(
        "from pkg.base import Base\n"
        "def helper():\n    return 1\n"
        "class Service(Base):\n"
        "    def run(self):\n        return helper()\n",
        encoding="utf-8",
    )
    parsed = PythonRepositoryParser().parse(tmp_path)

    graph = build_repository_graph(parsed, repository_name="owner/repo")
    stats = graph.stats

    assert stats.nodes_by_type == {
        "class": 2,
        "directory": 1,
        "file": 2,
        "function": 1,
        "method": 1,
        "repository": 1,
    }
    assert stats.edges_by_type["CONTAINS"] >= 6
    assert stats.edges_by_type["DEFINED_IN"] == 4
    assert stats.edges_by_type["IMPORTS"] == 1
    assert stats.edges_by_type["INHERITS"] == 1
    assert stats.edges_by_type["CALLS"] == 1
    method = next(node for node in graph.nodes if node.name == "run")
    assert [node.name for node in graph.callees(method.node_id)] == ["helper"]
    assert [node.name for node in graph.callers(graph.callees(method.node_id)[0].node_id)] == [
        "run"
    ]
    service_file = next(node for node in graph.nodes if node.name == "service.py")
    service_class = next(node for node in graph.nodes if node.name == "Service")
    assert [node.name for node in graph.imports(service_file.node_id)] == ["base.py"]
    assert [node.name for node in graph.bases(service_class.node_id)] == ["Base"]
    assert method in graph.neighbors(service_file.node_id, "DEFINED_IN")
