// LDBC SNB Interactive — IC13: Single shortest path between two persons.
// RFC-023 requires an explicit upper bound (`*..15`) so the BFS has a
// finite ceiling. Neo4j's open-ended `*` is rejected.
MATCH (a:Person {_id: $person1Id}), (b:Person {_id: $person2Id})
MATCH path = shortestPath((a)-[:KNOWS*..15]-(b))
RETURN length(path) AS shortestPathLength
