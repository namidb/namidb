// LDBC SNB Interactive — IC2: Recent messages by friends.
MATCH (p:Person {_id: $personId})-[:KNOWS]-(friend:Person)<-[:HAS_CREATOR]-(message:Message)
WHERE message.creationDate <= $maxDate
RETURN friend._id AS personId,
       friend.firstName AS personFirstName,
       friend.lastName AS personLastName,
       message._id AS messageId,
       message.creationDate AS messageCreationDate
ORDER BY messageCreationDate DESC, messageId ASC
LIMIT 20
