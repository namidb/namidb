// LDBC SNB Interactive Update — IU6 simplified: add a Message authored
// by a Person. The official IU6 also wires Post→Forum + Post→Tag*; we
// keep just Post + HAS_CREATOR.
MATCH (author:Person {_id: $authorId})
CREATE (m:Message {_id: $messageId, content: $content, creationDate: $creationDate})
CREATE (m)-[:HAS_CREATOR]->(author)
RETURN m._id AS messageId
