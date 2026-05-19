// LDBC SNB Interactive — IC1: Friends with a given name (transitive *1..3).
// Canonical LDBC uses `shortestPath((p)-[:KNOWS*1..3]-(friend))` which emits
// a single row per (p, friend) pair. v0 lowers `shortestPath` to a plain
// variable-length expand and dedupes via `WITH DISTINCT friend` to match
// the same row-count contract.
MATCH (p:Person {_id: $personId}), (friend:Person {firstName: $firstName})
WHERE NOT p._id = friend._id
MATCH (p)-[:KNOWS*1..3]-(friend)
WITH DISTINCT friend
RETURN friend._id AS friendId,
       friend.firstName AS firstName,
       friend.lastName AS lastName
ORDER BY friend.lastName ASC, friend._id ASC
LIMIT 20
