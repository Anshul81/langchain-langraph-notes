# Neo4j Interview Preparation — Complete Guide

> 🎯 **One-stop reference** — Read this before your interview to refresh fundamentals, project-specific knowledge, and practice with real interview questions.

---

## Table of Contents
1. [Neo4j Fundamentals Refresher](#1-neo4j-fundamentals-refresher)
2. [Why Neo4j? Why Not PostgreSQL?](#2-why-neo4j-why-not-postgresql)
3. [Project Architecture Questions](#3-project-architecture-questions)
4. [Neomodel OGM Questions](#4-neomodel-ogm-questions)
5. [Cypher Query Questions](#5-cypher-query-questions)
6. [Transaction Management](#6-transaction-management)
7. [Performance & Optimization](#7-performance--optimization)
8. [Challenges Faced & Solutions](#8-challenges-faced--solutions)
9. [Graph Data Modeling](#9-graph-data-modeling)
10. [Integration & Architecture](#10-integration--architecture)
11. [Advanced Neo4j Concepts](#11-advanced-neo4j-concepts)
12. [Scenario-Based Questions](#12-scenario-based-questions)

---

## 1. Neo4j Fundamentals Refresher

### Q1. What is Neo4j?
**A:** Neo4j is a **native graph database** that stores data as nodes, relationships, and properties. Unlike relational databases that use tables and JOINs, Neo4j stores relationships as first-class citizens directly in the storage engine. This makes traversing relationships O(1) per hop rather than requiring expensive JOIN operations.

### Q2. What are the building blocks of a Neo4j graph?
**A:**
| Component | Description | Example from our project |
|---|---|---|
| **Node** | An entity/record | `Snippet`, `Dbcall`, `Parent`, `Module` |
| **Label** | A tag/category on a node | `:Snippet`, `:Dbcall`, `:DecisionNode` |
| **Relationship** | A directed connection between two nodes | `CONTAINS_DB_CALLS`, `IS_A_CHILD_OF_PARENT` |
| **Property** | Key-value pair on nodes or relationships | `name: "PROCESS-DATA"`, `operation_type: "READ"` |

### Q3. What is Cypher?
**A:** Cypher is Neo4j's declarative query language, inspired by ASCII art patterns:
- `()` represents nodes
- `-->` represents relationships
- `[]` represents relationship types
- Example: `(s:Snippet)-[:CONTAINS_DB_CALLS]->(d:Dbcall)` means "find a Snippet node connected to a Dbcall via CONTAINS_DB_CALLS relationship"

### Q4. Explain MATCH vs MERGE vs CREATE
**A:**
| Command | Behavior | When to use |
|---|---|---|
| `MATCH` | Finds existing patterns; fails silently if not found | Reading data |
| `CREATE` | Always creates new nodes/relationships (may create duplicates) | When you're sure it doesn't exist |
| `MERGE` | Finds or creates — acts as an upsert | When you want to avoid duplicates |

In our project we use `MERGE` for Parent and Module nodes to avoid duplicates when multiple snippets belong to the same class:
```cypher
MERGE (p:Parent {project_id: s.project_id, run_id: s.run_id, name: parent_name})
```

### Q5. What is the difference between `OPTIONAL MATCH` and `MATCH`?
**A:** 
- `MATCH` filters out rows where the pattern doesn't exist (like INNER JOIN)
- `OPTIONAL MATCH` keeps rows with `null` where pattern doesn't exist (like LEFT OUTER JOIN)

In our project we use `OPTIONAL MATCH` extensively for DB catalog queries because a DatabaseEntity may or may not have associated Dbcall nodes:
```cypher
OPTIONAL MATCH (dbe)<-[:USES_ENTITY]-(dbc:Dbcall)
```

### Q6. What are indexes in Neo4j?
**A:** Neo4j supports several index types:
- **B-tree indexes** — For exact lookups and range queries on properties
- **Full-text indexes** — For text search using Lucene (we use these for searching DB entities, parent names, etc.)
- **Composite indexes** — On multiple properties
- **Unique constraints** — Enforce uniqueness (we use `unique_index=True` on `key` properties)

In our project, we use full-text indexes heavily:
```cypher
CALL db.index.fulltext.queryNodes("databaseEntityFullTextIndex", $search_string + "*")
```

### Q7. What is the difference between a property and a label?
**A:**
- **Label** = A type/category tag on a node (like a table name). Nodes can have multiple labels. Labels are indexed automatically.
- **Property** = A key-value pair stored on a node or relationship (like a column value). Properties can be indexed manually.

### Q8. What data types does Neo4j support?
**A:** `Boolean`, `Integer`, `Float`, `String`, `List` (of primitives), `Date`, `DateTime`, `Duration`, `Point`, `null`. Neo4j does **not** natively support nested objects/maps as properties (unlike MongoDB). We work around this by serializing complex objects as JSON strings or using `JSONProperty` via neomodel.

### Q9. What is a traversal in Neo4j?
**A:** A traversal is the process of walking through the graph following relationships. Neo4j performs **index-free adjacency** — each node directly stores pointers to its neighbors, making traversals O(1) per relationship hop. This is the key advantage over relational databases where each "hop" requires a JOIN.

### Q10. Explain the difference between directed and undirected relationships
**A:** In Neo4j, **all relationships are stored as directed** (they have a start and end node). However, during querying you can ignore direction using `(a)-[r]-(b)` instead of `(a)-[r]->(b)`. In our project, relationships like `CONTAINS_DB_CALLS` are always directional: `(Snippet)-[:CONTAINS_DB_CALLS]->(Dbcall)`.

---

## 2. Why Neo4j? Why Not PostgreSQL?

### Q11. Why did you choose Neo4j over PostgreSQL for this project?
**A:** Our project (RapidX CodeScout) builds a **knowledge graph of source code** — programs contain functions, functions call other functions, functions contain DB calls, DB calls access tables, tables have fields, programs belong to modules, etc. This is a **deeply interconnected, hierarchical graph** with relationships that are:

1. **Multi-level**: Module → Parent → Snippet → Dbcall → Variable (5 levels deep)
2. **Many-to-many**: A snippet can call multiple other snippets, and be called by multiple parents
3. **Traversal-heavy**: Our queries traverse 4-5 relationship hops (e.g., "Find all database tables accessed by programs in this module")

In PostgreSQL, this would require:
```sql
-- PostgreSQL: 5 JOINs to get from Module to Variable
SELECT v.* FROM modules m
JOIN parents p ON p.module_id = m.id
JOIN snippets s ON s.parent_id = p.id
JOIN db_calls d ON d.snippet_id = s.id
JOIN variables v ON v.db_call_id = d.id
WHERE m.name = 'ModuleX';
```

In Neo4j:
```cypher
-- Neo4j: Natural graph traversal
MATCH (m:Module {name: 'ModuleX'})<-[:IS_A_PART_OF_MODULE]-(p:Parent)
      <-[:IS_A_CHILD_OF_PARENT]-(s:Snippet)-[:CONTAINS_DB_CALLS]->(d:Dbcall)
      -[:CONTAINS_VARIABLE]->(v:Variable)
RETURN v;
```

### Q12. What specific problems does Neo4j solve that PostgreSQL cannot?
**A:**

| Problem | PostgreSQL | Neo4j |
|---|---|---|
| **Call chain analysis** (A calls B calls C calls D) | Recursive CTEs — slow, hard to write | Variable-length path: `(a)-[:CALLS*1..10]->(b)` |
| **Impact analysis** ("What is affected if I change this table?") | Multiple JOINs across many tables | Single traversal: reverse-walk from DatabaseEntity to Snippet to Parent |
| **Program flow visualization** | Need to reconstruct graph in application layer | Graph is the native storage — query returns the structure directly |
| **Schema flexibility** | ALTER TABLE for new properties; migrations | Just add a property — no schema migration needed |
| **Relationship properties** | Requires junction tables | Properties directly on relationships: `[:CALLS {execution_order: 3}]` |

### Q13. What are the disadvantages of Neo4j?
**A:**
- **Not ideal for heavy aggregations** (SUM, AVG over millions of rows) — PostgreSQL with columnar storage is better
- **No native support for complex OLAP queries**
- **Storage cost** — Relationships are stored explicitly, so more disk usage
- **Less mature ecosystem** compared to PostgreSQL (fewer ORMs, tools, hosted options)
- **Licensing** — Enterprise features (clustering, backup) require a commercial license
- **Learning curve** — Developers need to learn Cypher and graph thinking

### Q14. Do you use both Neo4j and PostgreSQL?
**A:** Yes! We use a **polyglot persistence** approach:
- **PostgreSQL** — For transactional metadata: `code_analysis_run_file` (file parsing status), `project` details, `run` tracking, user authentication. These are structured, tabular data.
- **Neo4j** — For the code knowledge graph: Snippets, DbCalls, DecisionNodes, Callees, Parents, Modules, and all their relationships. This is deeply connected, traversal-heavy data.

This is a common architectural pattern — use the right database for the right workload.

---

## 3. Project Architecture Questions

### Q15. Describe the graph model in your project
**A:** Our graph represents a complete code comprehension knowledge graph:

```
Module ← Snippet → Parent
           ↓
    ┌──────┼──────────────┐
    ↓      ↓              ↓
  Dbcall  DecisionNode  Callee  ServiceCall  InputEntity ...
    ↓
  Variable
```

**Node types (labels):**
- `Snippet` — A function/paragraph/method
- `Dbcall` — A database call (EXEC SQL, SQL query)
- `DecisionNode` — Business rules (IF/EVALUATE blocks)
- `Callee` — Function call targets
- `ServiceCall` — External service calls (EXEC CICS)
- `Parent` — Class/program containing snippets
- `Module` — Folder/module grouping
- `DatabaseEntity` — A database table/view
- `DatabaseField` — A column in a table
- `Variable` — A data variable used in code
- `InputEntity` — Screen/form inputs
- `InputInterface` — UI screen definitions
- `ReportInterface` — Report definitions
- `ExecutionFlow` — Grouped execution sequences
- `TransactionCommit/Rollback` — Transaction boundaries

### Q16. What relationships exist in your graph?
**A:** Key relationships:
| Relationship | From | To | Meaning |
|---|---|---|---|
| `CONTAINS_DB_CALLS` | Snippet | Dbcall | Snippet contains a database call |
| `CONTAINS_DMN` | Snippet | DecisionNode | Snippet contains business rules |
| `CONTAINS_SERVICE_CALLS` | Snippet | ServiceCall | Snippet calls an external service |
| `CONTAINS_VARIABLE` | Snippet/Dbcall | Variable | Contains a data variable |
| `HAS_CALLEE` | Snippet | Callee | Snippet calls another function |
| `IS_A_CHILD_OF_PARENT` | Snippet | Parent | Function belongs to a class/program |
| `IS_A_PART_OF_MODULE` | Snippet/Parent | Module | Belongs to a module |
| `CALLS` | Snippet | Snippet | Direct call relationship (linked later) |
| `USES_ENTITY` | Dbcall | DatabaseEntity | A db call accesses this table |
| `HAS_FIELD` | DatabaseEntity | DatabaseField | Table has this column |
| `PARTICIPATES_IN_FLOW` | Snippet | ExecutionFlow | Part of an execution flow |
| `MAPPED_TO` | DatabaseEntity | FunctionalModule | Mapped to functional area |

### Q17. How many nodes does a typical parsed file generate?
**A:** For a typical COBOL program (~1000 lines):
- 1 Parent node
- 10-30 Snippet nodes (paragraphs)
- 5-20 Dbcall nodes (SQL statements)
- 10-40 DecisionNode nodes (IF/EVALUATE)
- 5-15 Callee nodes (PERFORM calls)
- 20-50 Variable nodes
- A few ServiceCall, InputEntity nodes

So roughly **50-150 nodes per file** with **80-250 relationships**. For a project with 500 files, that's ~50,000-75,000 nodes.

### Q18. How is data scoped in your graph?
**A:** We use `project_id`, `run_id`, and `run_file_id` as **scoping properties** on every node. This enables:
- **Multi-tenancy**: Different projects don't interfere
- **Versioning**: Each parsing run gets a `run_id`, so we can compare versions
- **File-level operations**: We can delete and re-parse a single file using `run_file_id`

```cypher
-- Delete all nodes for a specific file
MATCH (n) WHERE n.run_file_id = $run_file_id DETACH DELETE n;
```

---

## 4. Neomodel OGM Questions

### Q19. What is neomodel and why do you use it?
**A:** Neomodel is a Python **Object Graph Mapper (OGM)** for Neo4j — similar to how SQLAlchemy is an ORM for SQL databases. It maps Python classes to Neo4j node labels and properties. We use it because:
- Type safety via Python property definitions
- Built-in relationship management
- Transaction support (`db.begin()`, `db.commit()`, `db.rollback()`)
- Cleaner code than raw Cypher strings

### Q20. How do you define a node model in neomodel?
**A:** By extending `StructuredNode`:
```python
from neomodel import StructuredNode, StringProperty, IntegerProperty, ArrayProperty, RelationshipTo
from uuid import uuid4

class Dbcall(StructuredNode):
    key = StringProperty(unique_index=True, default=uuid4)
    project_id = IntegerProperty()
    name = StringProperty(default="Database call")
    table_names = ArrayProperty(StringProperty(), default=[])
    snippet = StringProperty()
    operation_type = StringProperty(default="READ")
    
    # Relationships
    contains_variable = RelationshipTo("Variable", "CONTAINS_VARIABLE")
```

### Q21. How are relationships defined in neomodel?
**A:** Using `RelationshipTo`, `RelationshipFrom`, or `Relationship`:
```python
class Snippet(StructuredNode):
    # Outgoing relationships
    contains_dbcalls = RelationshipTo("Dbcall", "CONTAINS_DB_CALLS")
    contains_dmn = RelationshipTo("DecisionNode", "CONTAINS_DMN", model=RelationShipOrder)
    has_callee = RelationshipTo("Callee", "HAS_CALLEE")
    
    # Connecting nodes
    snippet_node.contains_dbcalls.connect(db_node)
```

The `model=RelationShipOrder` parameter allows **relationship properties** (e.g., ordering):
```python
class RelationShipOrder(StructuredRel):
    order = IntegerProperty(default=1)
```

### Q22. What is the `derive_from` pattern in your codebase?
**A:** It's a **factory method** that converts a Pydantic schema object into a neomodel StructuredNode. This separates the validation layer from the persistence layer:

```python
# Snippet.derive_from() converts FunctionSnippetSchema → Snippet (neomodel node)
@staticmethod
def derive_from(snippet_data: FunctionSnippetSchema):
    snippet = Snippet(
        program_id=snippet_data.program_id,
        project_id=snippet_data.project_id,
        # ... map all fields
    )
    return snippet
```

Similarly, `to_db_call()` converts `DbCallDataEntity` → `Dbcall`:
```python
def to_db_call(self, snippet: FunctionSnippetSchema, db_call: DbCallDataEntity):
    return Dbcall(
        name=snippet.function_name,       # From parent context
        snippet=db_call.snippet,           # From the specific call
        table_names=db_call.table_names,
        # ...
    )
```

### Q23. How do you handle arrays and JSON in neomodel?
**A:** 
- `ArrayProperty(StringProperty())` — For lists of strings (e.g., `table_names`, `aliases`)
- `JSONProperty()` — For JSON objects stored as Neo4j maps
- `ArrayProperty(JSONProperty())` — For lists of JSON objects (e.g., `variables`)

```python
class Dbcall(StructuredNode):
    table_names = ArrayProperty(StringProperty(), default=[])  # ["TABLE1", "TABLE2"]
    variables = ArrayProperty(JSONProperty(), default=[])       # [{term: "VAR1", type: "STR"}]
```

### Q24. How does neomodel connect to Neo4j?
**A:** We use the Neo4j Python driver with neomodel:
```python
from neo4j import GraphDatabase

class Neo4j:
    def connect(self):
        AUTH = (os.environ.get('NEO4j_USERNAME'), os.environ.get('NEO4j_PASSWORD'))
        self.__driver = GraphDatabase.driver(
            os.environ.get('NEO4j_URI'), 
            auth=AUTH,
            connection_acquisition_timeout=2,
            connection_timeout=1,
            max_connection_lifetime=240
        )
        return self.__driver

# Then set the connection for neomodel
db.set_connection(driver=neo4j_driver.driver)
```

---

## 5. Cypher Query Questions

### Q25. Write a Cypher query to find all database tables accessed by a specific program
**A:**
```cypher
MATCH (p:Parent {name: $program_name, project_id: $project_id})
      <-[:IS_A_CHILD_OF_PARENT]-(s:Snippet)
      -[:CONTAINS_DB_CALLS]->(d:Dbcall)
RETURN DISTINCT d.table_names AS tables, d.operation_type AS operation
```

### Q26. Explain the UNWIND clause
**A:** `UNWIND` expands a list into individual rows — like `unnest()` in PostgreSQL. We use it for batch node creation:
```cypher
UNWIND $functions AS function
MERGE (c:Callee {function_name: function[0], function_class: function[1]})
```

### Q27. What is a CALL subquery?
**A:** `CALL {}` is a scoped subquery in Cypher that allows you to run a separate query block with its own scope. We use it extensively in our DB catalog queries to collect aggregations independently:
```cypher
CALL {
    WITH dbe
    OPTIONAL MATCH (dbe)<-[:USES_ENTITY]-(dbc:Dbcall)
    RETURN 
        COLLECT(DISTINCT dbc.query) AS queries,
        COLLECT(DISTINCT dbc.operation_type) AS accessType
}
```

### Q28. What is APOC and how do you use it?
**A:** APOC (Awesome Procedures on Cypher) is a library of utility procedures for Neo4j. We use:
- `apoc.coll.toSet()` — Deduplicate a list
- `apoc.coll.flatten()` — Flatten nested lists
- `apoc.text.join()` — Join strings
- `apoc.convert.toJson()` — Convert to JSON

```cypher
apoc.coll.toSet(apoc.coll.flatten(COLLECT(dbc.aliases))) AS flattenedAliases
```

### Q29. How do you do full-text search in Neo4j?
**A:** We create full-text indexes and query them:
```cypher
-- Create index (one-time)
CALL db.index.fulltext.createNodeIndex("databaseEntityFullTextIndex", ["DatabaseEntity"], ["name"])

-- Query
CALL db.index.fulltext.queryNodes("databaseEntityFullTextIndex", $search_string + "*") 
YIELD node AS dbe, score
WHERE dbe.run_id = $run_id AND dbe.project_id = $project_id
RETURN dbe.name, score
```

We use wildcard search with `*` for prefix matching — this is critical for our DB catalog search feature.

### Q30. What does DETACH DELETE do?
**A:** `DETACH DELETE` removes a node **and all its relationships**. Without `DETACH`, Neo4j would throw an error if the node has relationships. We use this for cleanup:
```cypher
MATCH (n) WHERE n.run_file_id = $run_file_id DETACH DELETE n;
```

### Q31. How do you do pagination in Cypher?
**A:** Using `SKIP` and `LIMIT`:
```cypher
RETURN dbe.name AS name ORDER BY name ASC SKIP $offset LIMIT $limit
```

### Q32. Explain parameterized queries and why they matter
**A:** We always use `$parameter` syntax instead of string interpolation:
```cypher
-- GOOD: Parameterized
MATCH (s:Snippet {project_id: $project_id, run_id: $run_id})

-- BAD: String interpolation (SQL injection risk)
MATCH (s:Snippet {project_id: {project_id}})
```
Benefits: Prevents injection attacks, enables query plan caching.

---

## 6. Transaction Management

### Q33. How do you manage transactions in your project?
**A:** We wrap the entire file-level parsing in a single Neo4j transaction:
```python
db.set_connection(driver=neo4j_driver.driver)
db.begin()             # Start transaction
try:
    # Create all nodes for this file
    snippet_node = neo_service.create_function_snippet_node(snippet)
    neo_service.create_call_nodes(snippet, snippet_node)
    # ... more node creation
    db.commit()        # Commit all at once
except Exception:
    db.rollback()      # Rollback everything if anything fails
```

This gives us **file-level atomicity** — either all nodes for a file are created, or none are.

### Q34. Why is file-level atomicity important?
**A:** Without it, a partial failure would leave **orphan nodes** — Dbcall nodes without their parent Snippet, or DecisionNodes disconnected from their function. This would corrupt the graph and break downstream queries. By wrapping in a transaction, we ensure consistency.

### Q35. How do you handle concurrent parsing?
**A:** We use Azure Service Bus messages — each file is a separate message processed by a separate pod. To prevent duplicate processing:
```python
if file_data.parse_status in ["InProgress", "Completed"]:
    return  # Skip — already being processed
di[CodeAnalysisRunFileRepository].update_parse_status_of_file(run_file_id, "InProgress")
```

This is a **pessimistic locking** strategy using PostgreSQL status flags as a coordination mechanism.

### Q36. What are ACID properties and does Neo4j support them?
**A:** ACID = Atomicity, Consistency, Isolation, Durability. Yes, Neo4j is **fully ACID compliant**:
- **Atomicity**: Transactions succeed or fail entirely (we use `db.begin()/commit()/rollback()`)
- **Consistency**: Schema constraints are enforced (we use `unique_index=True` on key properties)
- **Isolation**: Read-committed isolation level by default
- **Durability**: WAL (Write-Ahead Log) ensures data survives crashes

---

## 7. Performance & Optimization

### Q37. How do you optimize Neo4j performance in your project?
**A:**
1. **Indexes** on frequently queried properties: `project_id`, `run_id`, `run_file_id`, `name`, `key`
2. **Unique constraints** on `key` (UUID) to enable fast lookups
3. **Batch operations** using `UNWIND` for callee creation instead of individual queries
4. **Full-text indexes** for search features instead of `CONTAINS` or regex
5. **Connection pooling** via the Neo4j driver with configured timeouts
6. **Transaction batching** — one transaction per file instead of per node

### Q38. What is index-free adjacency?
**A:** This is Neo4j's core architectural advantage. Each node stores a direct physical pointer to its neighbor nodes. When traversing a relationship, Neo4j follows the pointer directly — it doesn't need to scan an index or do a table lookup. This makes each relationship traversal O(1), regardless of the total graph size.

Compare with PostgreSQL: Each JOIN requires an index lookup + row fetch, making deep JOINs exponentially slower.

### Q39. How do you handle large graphs?
**A:** Our strategies:
- **Scoping queries** with `project_id` and `run_id` to limit the search space
- **SKIP/LIMIT** for pagination in API responses
- **OPTIONAL MATCH** instead of `MATCH` to avoid query failure on missing paths
- **APOC procedures** for complex aggregations instead of multiple round-trips
- **Lazy loading** — not loading all relationships upfront

### Q40. How do you handle connection timeouts?
**A:** We configure the driver with explicit timeouts:
```python
GraphDatabase.driver(uri, auth=AUTH,
    connection_acquisition_timeout=2,    # Wait max 2s to get a connection from pool
    connection_timeout=1,                # Connection establishment timeout
    liveness_check_timeout=251,          # Verify connection is alive
    max_connection_lifetime=240          # Recycle connections every 4 minutes
)
```

### Q41. What is the `analysis_status` field and why does it exist?
**A:** We use `analysis_status` (values: `NOT_STARTED`, `IN_PROGRESS`, `COMPLETED`) on nodes like `Snippet`, `Dbcall`, `Parent`, `DatabaseEntity` to track which nodes have been processed by our AI summarization pipeline (snippet-analyser-service). This enables:
- **Incremental processing** — Only summarize nodes that haven't been processed yet
- **Retry logic** — Re-process failed nodes
- **Progress tracking** — How much of the project is analyzed?

---

## 8. Challenges Faced & Solutions

### Q42. What challenges did you face using Neo4j in production?
**A:**

#### Challenge 1: Duplicate Node Creation
**Problem:** When Azure Service Bus retried messages (due to lock renewal failure), the same file was parsed twice, creating duplicate nodes.
**Solution:** Added status-checking before parsing:
```python
if file_data.parse_status in ["InProgress", "Completed"]:
    return  # Skip duplicates
```
And `DETACH DELETE` before re-parsing to clean up existing nodes.

#### Challenge 2: Transaction Size Limits
**Problem:** Very large COBOL programs (5000+ lines) generated hundreds of nodes in a single transaction, causing memory pressure.
**Solution:** One transaction per file (not per project). Also, we moved callees from `MERGE`-based batch queries to individual `node.save()` + `connect()` calls which are more memory-efficient.

#### Challenge 3: Connection Pool Exhaustion
**Problem:** Under heavy load with 10+ pods parsing simultaneously, connections were exhausted.
**Solution:** Configured `max_connection_lifetime=240` and `connection_acquisition_timeout=2` to recycle connections aggressively and fail fast rather than queue indefinitely.

#### Challenge 4: Schema Evolution
**Problem:** As we added new node types (InputInterface, ReportInterface, DatabaseEntity), existing data didn't have the new properties.
**Solution:** Neo4j's schema-less nature actually helped — new properties default to `null`. We used `COALESCE()` in queries to handle missing properties gracefully.

#### Challenge 5: Complex Cypher Queries for UI
**Problem:** The DB Catalog feature needed aggregated data from 4-5 levels of traversal with pagination, sorting, and filtering.
**Solution:** Used `CALL {}` subqueries to isolate aggregations, `APOC` for deduplication, and dynamic query building in Python:
```python
def build_base_query(filters=None, sort_by=None):
    facet_filters = []
    if filters and "facets" in filters:
        for k, filter in filters["facets"].items():
            facet_filters.append(f" dbe.{k} IN {filter}")
    # ... dynamic query construction
```

#### Challenge 6: Neomodel vs Raw Cypher Trade-offs
**Problem:** Neomodel's OGM is great for CRUD, but complex queries (with UNWIND, CALL subqueries, APOC) can't be expressed through the OGM.
**Solution:** We use a **hybrid approach**:
- `neomodel` for node creation and simple operations (`node.save()`, `.connect()`)
- `db.cypher_query()` for complex analytical queries (density calculations, linking, interdependency)

### Q43. How did you handle data migration when the schema changed?
**A:** Neo4j is **schema-flexible**, so we didn't need traditional migrations. Instead:
- New properties just appear on new nodes (existing nodes have `null`)
- For mandatory updates, we run Cypher scripts: `MATCH (n:Dbcall) WHERE n.tech IS NULL SET n.tech = 'NA'`
- For new node types, we deployed the updated parser — old data stays intact

### Q44. How did you debug Neo4j query performance issues?
**A:** 
1. **EXPLAIN/PROFILE** in Neo4j Browser to see query plans
2. **db.stats()** to check index usage
3. Added logging around `db.cypher_query()` calls with timing
4. Used Neo4j Browser's query monitoring for long-running queries
5. Ensured indexes exist for all properties used in `WHERE` clauses

---

## 9. Graph Data Modeling

### Q45. Why are DbCalls stored as separate nodes instead of properties on Snippet?
**A:** Because:
1. **Multiple DbCalls per Snippet** — A function may have 5-10 SQL statements. Storing as an array loses individual identity.
2. **Independent querying** — We need to query "all DbCalls that access TABLE_X" without loading every Snippet.
3. **Variable attachment** — Each DbCall has its own set of Variables connected via `CONTAINS_VARIABLE`.
4. **AI enrichment** — Each DbCall gets its own summary, tags, and analysis status independently.
5. **Relationship semantics** — The `CONTAINS_DB_CALLS` relationship explicitly models the containment, which is queryable.

### Q46. Why do you store `callees` both as a JSON string AND as separate Callee nodes?
**A:** Historical reasons + optimization:
- The `callees` JSON string on `Snippet` is a **denormalized snapshot** for quick access without traversal
- The `Callee` nodes with `HAS_CALLEE` relationships are the **normalized form** used for graph queries like "find all callers of function X"
- We initially used only the JSON approach, then added proper nodes when we needed graph traversal for the function linker service

### Q47. How do you model the hierarchy: Module → Parent → Snippet?
**A:** Using relationships, not nested properties:
```
Module <-[:IS_A_PART_OF_MODULE]- Parent <-[:IS_A_CHILD_OF_PARENT]- Snippet
```
This allows querying at any level: "All snippets in module X" or "All modules containing programs that access table Y". The Parent node is created via a `MERGE` Cypher query after all snippets are saved, ensuring deduplication.

### Q48. Why is `execution_order` stored on both the node and the relationship?
**A:** Different meanings:
- On the **node** (e.g., `Dbcall.execution_order`) — The order within the function/paragraph
- On the **relationship** (e.g., `CONTAINS_DMN` with `model=RelationShipOrder`) — The order among all decision nodes of that snippet

This enables sorting results by execution order for program flow reconstruction.

---

## 10. Integration & Architecture

### Q49. Describe the microservice architecture around Neo4j
**A:**
```
Azure Service Bus → Parser Service → Neo4j ← Mirrormotive (API Gateway)
                         ↓                         ↑
                    PostgreSQL              Snippet Analyser Service
                    (run metadata)         (AI enrichment → Neo4j)
```

- **Parser Service** — Receives file parsing messages, parses code, writes to Neo4j
- **Function Linker Service** — Creates `CALLS` relationships between Snippets based on Callee nodes
- **Snippet Analyser Service** — AI-enriches nodes (summaries, tags) by reading from Neo4j and writing back
- **Mirrormotive (API Gateway)** — Serves UI queries by reading from Neo4j
- **Hierarchy Service** — Builds execution flows and functional groupings

### Q50. How does the Parser Service communicate with Neo4j?
**A:** Through two mechanisms:
1. **Neomodel OGM** — For node CRUD (`node.save()`, `.connect()`)
2. **Raw Cypher** via `db.cypher_query()` — For complex operations (MERGE Parent/Module, DETACH DELETE, density calculations)

Both use the same Neo4j Python driver under the hood.

### Q51. How does data flow from parsing to the UI?
**A:**
1. File uploaded → PostgreSQL stores file metadata → Message to Azure Service Bus
2. Parser Service picks up message → Parses file → Writes nodes to Neo4j → Updates PostgreSQL status
3. Function Linker links Snippets via `CALLS` relationships
4. Snippet Analyser enriches nodes with AI summaries
5. Frontend calls API Gateway → Cypher queries Neo4j → Returns structured JSON

---

## 11. Advanced Neo4j Concepts

### Q52. What is the Bolt protocol?
**A:** Bolt is Neo4j's binary protocol for client-server communication. It's more efficient than HTTP because it uses binary serialization and persistent connections. Our Python driver uses Bolt by default (`bolt://` or `neo4j://` URI).

### Q53. What is causal consistency in Neo4j?
**A:** In a Neo4j cluster, a write on the leader may not be immediately visible on followers. Causal consistency ensures that if a client writes data and then reads, it sees its own write. The driver handles this via bookmarks. We don't use clustering in our current setup, but it's important for scaling.

### Q54. Explain write-ahead logging (WAL) in Neo4j
**A:** Before modifying the graph store, Neo4j writes the change to a transaction log file. If the system crashes, it replays the log to recover. This ensures **durability** (the "D" in ACID).

### Q55. What are Neo4j constraints and how do you use them?
**A:** We use:
- **Unique node property constraint**: `key = StringProperty(unique_index=True, default=uuid4)` — ensures no two Snippets have the same key
- This also creates an index automatically for fast lookups

### Q56. What is the difference between Neo4j Community and Enterprise?
**A:**
| Feature | Community | Enterprise |
|---|---|---|
| Clustering | ❌ | ✅ (Causal clustering) |
| Online backup | ❌ | ✅ |
| Role-based access | ❌ | ✅ |
| Unlimited databases | ❌ | ✅ |
| Security (encryption) | Basic | Advanced |
| License | GPLv3 | Commercial |

### Q57. What is the difference between `db.cypher_query()` and `node.save()`?
**A:**
- `node.save()` — Neomodel OGM method that generates a `CREATE` or `MERGE` Cypher query under the hood. Good for simple node operations.
- `db.cypher_query()` — Execute raw Cypher directly. Needed for complex queries that the OGM can't express (JOINs, aggregations, APOC calls).

---

## 12. Scenario-Based Questions

### Q58. If a file is re-parsed, how do you update the graph?
**A:** We use a **delete-and-recreate** strategy:
1. `MATCH (n) WHERE n.run_file_id = $run_file_id DETACH DELETE n` — Remove all nodes for that file
2. Parse the file again and create new nodes
3. All within a single transaction

This is simpler than diffing and updating individual properties.

### Q59. How would you find which programs are affected if DATABASE_TABLE_X is modified?
**A:**
```cypher
MATCH (dbe:DatabaseEntity {name: "TABLE_X"})
      <-[:USES_ENTITY]-(dbc:Dbcall)
      <-[:CONTAINS_DB_CALLS]-(s:Snippet)
      -[:IS_A_CHILD_OF_PARENT]->(p:Parent)
RETURN DISTINCT p.name AS program, s.function_name AS function, 
       dbc.operation_type AS operation
```

This is exactly the kind of **impact analysis** query that makes Neo4j shine — in PostgreSQL, this would be 4 JOINs.

### Q60. How would you find the complete call chain from a root function?
**A:**
```cypher
MATCH path = (s:Snippet {name: $root_name})-[:CALLS*1..10]->(target:Snippet)
RETURN path
```

The `*1..10` is a **variable-length relationship** — Neo4j traverses 1 to 10 hops. This is practically impossible to do efficiently in SQL without recursive CTEs.

### Q61. How would you calculate the "complexity" of a program?
**A:** We compute `rule_density_score` and `db_density_score`:
```cypher
MATCH (s:Snippet {project_id: $project_id, run_id: $run_id})
OPTIONAL MATCH (s)-[:CONTAINS_DMN]->(d:DecisionNode)
WITH s,
  COALESCE(SUM(size(split(d.snippet, '\n'))), 0) AS total_rule_lines,
  size(split(s.snippet, '\n')) AS total_snippet_node_lines
SET s.rule_density_score = round(total_rule_lines*1.0 / total_snippet_node_lines*100) / 100.0
```

This traverses from Snippet to its DecisionNodes, counts lines, and computes a density percentage.

### Q62. Design question: How would you add a new node type (e.g., "APIEndpoint") to the graph?
**A:**
1. **Pydantic schema**: Create `APIEndpointSchema(BaseModel)` in `data_entity_schema.py`
2. **Add to FunctionSnippetSchema**: `api_endpoints: Optional[List[APIEndpointSchema]]`
3. **Neomodel node**: Create `APIEndpoint(StructuredNode)` in `models/neo4j/api_endpoint.py`
4. **Add relationship on Snippet**: `contains_api_endpoints = RelationshipTo("APIEndpoint", "CONTAINS_API_ENDPOINTS")`
5. **Conversion method**: Add `to_api_endpoint()` on Snippet model
6. **NeoModelService**: Add creation logic in `create_call_nodes()`
7. **Visitor**: Add extraction logic in the parser visitor
8. **Data model constructor**: Add assembly function

No database migration needed — Neo4j handles the new label and properties automatically!

### Q63. If you had to migrate from Neo4j to another database, what would you choose?
**A:** If we absolutely had to:
- **Amazon Neptune** — Managed graph database, supports both Gremlin and SPARQL
- **ArangoDB** — Multi-model database supporting graphs + documents
- **PostgreSQL with Apache AGE extension** — Adds graph query capabilities to PostgreSQL

But honestly, the migration would be painful because our entire query layer is built around Cypher and traversal patterns.

### Q64. How do you test Neo4j-related code?
**A:** 
- **Unit tests** with test visitor inputs — verify that the correct number of DbCalls, DecisionNodes, etc., are extracted
- **Integration tests** — Parse real files and verify the CodeContext output
- **Cypher query tests** — Run queries against a test Neo4j instance

Example from the codebase:
```python
def verify_db_calls_count(self, code_context, program_id, expected):
    snippets = code_context.function_snippets
    total_db_calls = sum(len(s.db_calls or []) for s in snippets)
    assert total_db_calls == expected["db_calls"]
```

---

## Quick-Reference Cypher Cheat Sheet

```cypher
-- Create
CREATE (n:Label {prop: value})

-- Find
MATCH (n:Label {prop: value}) RETURN n

-- Update
MATCH (n:Label {prop: value}) SET n.newProp = newValue

-- Delete
MATCH (n:Label {prop: value}) DETACH DELETE n

-- Find or create
MERGE (n:Label {prop: value})

-- Relationship
MATCH (a:A {name: 'x'}), (b:B {name: 'y'})
CREATE (a)-[:REL_TYPE {prop: value}]->(b)

-- Variable-length path
MATCH path = (a)-[:CALLS*1..5]->(b) RETURN path

-- Aggregation
MATCH (s:Snippet)-[:CONTAINS_DB_CALLS]->(d:Dbcall)
RETURN s.name, COUNT(d) AS dbCallCount ORDER BY dbCallCount DESC

-- Full-text search
CALL db.index.fulltext.queryNodes("indexName", "search*") YIELD node, score

-- List operations
UNWIND $list AS item CREATE (n:Node {prop: item})
WITH COLLECT(DISTINCT x) AS uniqueList
RETURN apoc.coll.toSet(list) AS deduped

-- Conditional
MATCH (s:Snippet)
RETURN CASE WHEN s.tech = 'cobol' THEN s.program_id ELSE s.class_name END AS name

-- Exists check
MATCH (n:Snippet) WHERE EXISTS((n)-[:CONTAINS_DB_CALLS]->()) RETURN n

-- Count
MATCH (n:Snippet {project_id: 123}) RETURN COUNT(n)
```

---

## Key Terminology Quick Glossary

| Term | Meaning |
|---|---|
| **Node** | A data entity in the graph (like a row) |
| **Label** | A type tag on a node (like a table name) |
| **Relationship** | A connection between two nodes |
| **Property** | A key-value attribute on a node or relationship |
| **Cypher** | Neo4j's query language |
| **Bolt** | Neo4j's binary communication protocol |
| **APOC** | Library of utility functions for Neo4j |
| **OGM** | Object Graph Mapper (neomodel = Python OGM for Neo4j) |
| **Index-free adjacency** | Each node stores direct pointers to neighbors |
| **Traversal** | Walking the graph following relationships |
| **MERGE** | Find-or-create (upsert) operation |
| **DETACH DELETE** | Delete a node and all its relationships |
| **WAL** | Write-Ahead Log for crash recovery |
| **Causal consistency** | Read-your-own-writes guarantee in clusters |
| **StructuredNode** | Neomodel's base class for Neo4j node models |
| **StructuredRel** | Neomodel's base class for relationship models |

---

> [!TIP]
> **Final Interview Tips:**
> 1. Always relate answers back to your **project** — interviewers love hearing real-world examples
> 2. Know the **graph model** cold — draw it if asked
> 3. Be ready to write Cypher on a whiteboard
> 4. Mention the **polyglot persistence** pattern (PostgreSQL + Neo4j)
> 5. Talk about **trade-offs** you've made — shows maturity
> 6. If asked about something you don't know, say "I haven't encountered that in production, but from what I understand..." and reason through it
