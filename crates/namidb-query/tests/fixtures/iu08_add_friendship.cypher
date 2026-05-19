// LDBC SNB Interactive Update — IU8: add a friendship between two
// Persons. Stored as a single directed KNOWS edge; bidirectional
// traversal is provided by `-[:KNOWS]-` at query time.
MATCH (a:Person {_id: $person1Id}), (b:Person {_id: $person2Id})
CREATE (a)-[k:KNOWS {creationDate: $creationDate}]->(b)
RETURN a._id AS aId, b._id AS bId
