// LDBC SNB Interactive — IC4: New topics on friend posts (tags that
// appeared during the window but not before). Uses OPTIONAL MATCH + IS NULL
// to express the "not previously seen" filter — equivalent to the LDBC
// canonical form which originally relied on `WHERE NOT EXISTS { ... }`
// (Cypher-25 subquery, out-of-scope in v0).
MATCH (p:Person {id: $personId})-[:KNOWS]-(friend:Person)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag)
WHERE post.creationDate >= $startDate
  AND post.creationDate < $endDate
WITH friend, tag, count(post) AS postCount
OPTIONAL MATCH (friend)<-[:HAS_CREATOR]-(oldPost:Post)-[:HAS_TAG]->(tag)
WHERE oldPost.creationDate < $startDate
WITH tag, postCount, oldPost
WHERE oldPost IS NULL
RETURN tag.name AS tagName, postCount
ORDER BY postCount DESC, tagName ASC
LIMIT 10
