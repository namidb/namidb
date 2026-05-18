// LDBC SNB Interactive Update — IU8: add a friendship between two
// Persons. Stored as a single directed KNOWS edge; bidirectional
// traversal is provided by `-[:KNOWS]-` at query time.
MATCH (a:Person {id: $person1Id}), (b:Person {id: $person2Id})
CREATE (a)-[k:KNOWS {creationDate: $creationDate}]->(b)
RETURN a.id AS aId, b.id AS bId
