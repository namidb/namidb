// LDBC SNB Interactive — IC9: Recent messages by friends-of-friends.
MATCH (p:Person {id: $personId})-[:KNOWS*1..2]-(friend:Person)<-[:HAS_CREATOR]-(message:Message)
WHERE NOT friend.id = p.id
  AND message.creationDate < $maxDate
RETURN friend.id AS personId,
       friend.firstName AS personFirstName,
       friend.lastName AS personLastName,
       message.id AS messageId,
       message.creationDate AS messageCreationDate
ORDER BY messageCreationDate DESC, messageId ASC
LIMIT 20
