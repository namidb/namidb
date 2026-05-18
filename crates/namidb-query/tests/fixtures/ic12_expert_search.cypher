// LDBC SNB Interactive — IC12: Expert search by tag class.
MATCH (p:Person {id: $personId})-[:KNOWS]-(friend:Person)<-[:HAS_CREATOR]-(comment:Comment)-[:REPLY_OF]->(post:Post)-[:HAS_TAG]->(tag:Tag)-[:HAS_TYPE]->(tagClass:TagClass)-[:IS_SUBCLASS_OF*0..5]->(parent:TagClass {name: $tagClassName})
WITH friend, collect(DISTINCT tag.name) AS tagNames, count(DISTINCT comment) AS replyCount
RETURN friend.id AS personId,
       friend.firstName AS personFirstName,
       friend.lastName AS personLastName,
       tagNames,
       replyCount
ORDER BY replyCount DESC, personId ASC
LIMIT 20
