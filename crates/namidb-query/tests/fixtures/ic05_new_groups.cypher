// LDBC SNB Interactive — IC5: New groups (forum memberships) that friends
// joined after a given date.
MATCH (p:Person {_id: $personId})-[:KNOWS*1..2]-(friend:Person)
WHERE NOT friend._id = p._id
WITH DISTINCT friend
MATCH (friend)<-[membership:HAS_MEMBER]-(forum:Forum)
WHERE membership.joinDate >= $minDate
OPTIONAL MATCH (friend)<-[:HAS_CREATOR]-(post:Post)<-[:CONTAINER_OF]-(forum)
WITH forum, count(post) AS postCount
RETURN forum.title AS forumName, postCount
ORDER BY postCount DESC, forum._id ASC
LIMIT 20
