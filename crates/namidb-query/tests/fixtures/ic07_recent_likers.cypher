// LDBC SNB Interactive — IC7: Recent likers of a person's messages.
MATCH (p:Person {_id: $personId})<-[:HAS_CREATOR]-(message:Message)<-[like:LIKES]-(liker:Person)
WITH liker, message, like.creationDate AS likeTime
ORDER BY likeTime DESC, message._id ASC
WITH liker, head(collect(message)) AS topMessage, head(collect(likeTime)) AS topLikeTime
RETURN liker._id AS personId,
       liker.firstName AS personFirstName,
       liker.lastName AS personLastName,
       topLikeTime AS likeCreationDate,
       topMessage._id AS messageId
ORDER BY likeCreationDate DESC, personId ASC
LIMIT 20
