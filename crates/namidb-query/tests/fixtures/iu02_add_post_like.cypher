// LDBC SNB Interactive Update — IU2: a Person likes a Message.
MATCH (p:Person {_id: $personId}), (m:Message {_id: $messageId})
CREATE (p)-[l:LIKES {creationDate: $creationDate}]->(m)
RETURN p._id AS personId, m._id AS messageId
