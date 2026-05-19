// LDBC SNB Interactive — IC13: Single shortest path between two persons.
// OUT-OF-SCOPE v0 (RFC-004): requires `shortestPath`. Aterriza con RFC-009.
MATCH (a:Person {_id: $person1Id}), (b:Person {_id: $person2Id}),
      path = shortestPath((a)-[:KNOWS*]-(b))
RETURN length(path) AS shortestPathLength
