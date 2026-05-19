// LDBC SNB Interactive — IC6: Tag co-occurrence.
MATCH (p:Person {_id: $personId})-[:KNOWS*1..2]-(friend:Person)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(knownTag:Tag {name: $tagName}),
      (post)-[:HAS_TAG]->(otherTag:Tag)
WHERE NOT otherTag = knownTag
  AND NOT friend._id = p._id
WITH otherTag, count(post) AS postCount
RETURN otherTag.name AS tagName, postCount
ORDER BY postCount DESC, tagName ASC
LIMIT 10
