// LDBC SNB Interactive — IC2: Recent messages by friends.
MATCH (p:Person {id: $personId})-[:KNOWS]-(friend:Person)<-[:HAS_CREATOR]-(message:Message)
WHERE message.creationDate <= $maxDate
RETURN friend.id AS personId,
       friend.firstName AS personFirstName,
       friend.lastName AS personLastName,
       message.id AS messageId,
       message.creationDate AS messageCreationDate
ORDER BY messageCreationDate DESC, messageId ASC
LIMIT 20
