// LDBC SNB Interactive — IC14: Weighted shortest paths between two persons.
// OUT-OF-SCOPE v0 (RFC-004): requires `allShortestPaths`. Aterriza con RFC-009.
MATCH path = allShortestPaths((a:Person {id: $person1Id})-[:KNOWS*]-(b:Person {id: $person2Id}))
WITH nodes(path) AS pathPersons
RETURN pathPersons
