# Optional Neo4j Graph Inspection

## Purpose and architecture

The heterogeneous `RepositoryGraph` is constructed programmatically in Python from the
AST-derived `ParsedRepository`. Neo4j is an optional persistence and interactive inspection
layer for the same generated graph; it is not part of the localization/evaluation critical
path.

```text
Python source -> AST -> ParsedRepository -> Python RepositoryGraph (source of truth)
                                             |-> existing GraphFix pipeline
                                             `-> optional Neo4j export -> Browser/Cypher
```

Normal GraphFix execution, the offline Streamlit demonstration, and all frozen experiments do
not import or require the Neo4j driver.

## Setup

Run a local Neo4j server or Neo4j Desktop database, then install only the optional exporter
dependency:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[neo4j]"
```

Set connection details for the current PowerShell session. Choose a dedicated empty database
for the clearest parity inspection when your Neo4j edition supports multiple databases.

```powershell
$env:NEO4J_URI = "bolt://localhost:7687"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASSWORD = "<your-password>"
$env:NEO4J_DATABASE = "neo4j"
```

Never commit the password. `NEO4J_DATABASE` is optional and defaults to `neo4j`; the other three
variables are required only for this export command.

## Export the representative graph

```powershell
.\.venv\Scripts\python.exe scripts\export_repository_graph_to_neo4j.py
```

The defaults select `vllm-project/vllm`, instance `vllm-project__vllm-9390`, and historical
commit `83450458339b07765b0e72a822e5fe93eeaf5258`. The script reuses GraphFix repository
reconstruction, parsing, and `build_repository_graph()`, then exports that already-built graph
in batches. Nodes and relationships use `MERGE`, so repeating the same export is idempotent.

## Review Call 3 Cypher views

Use the qualified name to avoid ambiguity when locating `benchmark()`:

```cypher
MATCH (f:GraphFixNode:Function)
WHERE f.qualified_name ENDS WITH '.benchmark'
MATCH (f)-[r]-(neighbor:GraphFixNode)
RETURN f, r, neighbor
LIMIT 50;
```

Inspect statically resolved outgoing calls:

```cypher
MATCH (f:GraphFixNode:Function)-[r:CALLS]->(callee:GraphFixNode)
WHERE f.qualified_name ENDS WITH '.benchmark'
RETURN f, r, callee
LIMIT 50;
```

Show the exact symbol-to-file definition relationship:

```cypher
MATCH (f:GraphFixNode:Function)-[r:DEFINED_IN]->(file:GraphFixNode:File)
WHERE f.qualified_name ENDS WITH '.benchmark'
RETURN f, r, file;
```

Inspect symbols contained by the representative file:

```cypher
MATCH (file:GraphFixNode:File)-[r:CONTAINS]->(symbol:GraphFixNode)
WHERE file.path = 'benchmarks/benchmark_serving.py'
RETURN file, r, symbol
LIMIT 100;
```

Summarize the heterogeneous schema:

```cypher
MATCH (source:GraphFixNode)-[r]->(target:GraphFixNode)
RETURN source.node_type, type(r), target.node_type, count(*) AS relationships
ORDER BY relationships DESC;
```

## Scope and limitations

- The exporter preserves GraphFix `node_id` as the unique Neo4j identity and exports only
  metadata present on `GraphNode`.
- Neo4j does not generate or alter the repository graph and is never queried by localization.
- A running Neo4j server and valid local credentials are required for live parity and
  idempotency checks.
- Because GraphFix node IDs are graph-local (including the literal repository ID), use a
  dedicated database when inspecting a different repository graph rather than silently mixing
  multiple exports.
