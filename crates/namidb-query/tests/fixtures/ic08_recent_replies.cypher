// LDBC SNB Interactive — IC8: Recent replies to a person's messages.
MATCH (p:Person {id: $personId})<-[:HAS_CREATOR]-(message:Message)<-[:REPLY_OF]-(reply:Comment)-[:HAS_CREATOR]->(replier:Person)
RETURN replier.id AS personId,
       replier.firstName AS personFirstName,
       replier.lastName AS personLastName,
       reply.creationDate AS commentCreationDate,
       reply.id AS commentId,
       reply.content AS commentContent
ORDER BY commentCreationDate DESC, commentId ASC
LIMIT 20
