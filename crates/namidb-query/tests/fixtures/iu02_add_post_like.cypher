// LDBC SNB Interactive Update — IU2: a Person likes a Message.
MATCH (p:Person {id: $personId}), (m:Message {id: $messageId})
CREATE (p)-[l:LIKES {creationDate: $creationDate}]->(m)
RETURN p.id AS personId, m.id AS messageId
