// LDBC SNB Interactive — IC14: Weighted shortest paths between two persons.
// RFC-023 requires an explicit upper bound on the path length so the
// BFS has a ceiling; weighting via edge properties is a follow-up.
MATCH (a:Person {_id: $person1Id}), (b:Person {_id: $person2Id})
MATCH path = allShortestPaths((a)-[:KNOWS*..15]-(b))
WITH nodes(path) AS pathPersons
RETURN pathPersons
