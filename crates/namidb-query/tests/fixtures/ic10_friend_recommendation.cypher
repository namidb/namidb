// LDBC SNB Interactive — IC10: Friend recommendation by shared interests.
MATCH (p:Person {_id: $personId})-[:KNOWS*2..2]-(friend:Person)-[:IS_LOCATED_IN]->(city:City)
WHERE NOT friend._id = p._id
  AND NOT (p)-[:KNOWS]-(friend)
WITH p, friend, city
MATCH (p)-[:HAS_INTEREST]->(tag:Tag)<-[:HAS_INTEREST]-(friend)
WITH friend, city, count(DISTINCT tag) AS commonInterestCount
RETURN friend._id AS personId,
       friend.firstName AS personFirstName,
       friend.lastName AS personLastName,
       commonInterestCount,
       city.name AS personCityName
ORDER BY commonInterestCount DESC, personId ASC
LIMIT 10
