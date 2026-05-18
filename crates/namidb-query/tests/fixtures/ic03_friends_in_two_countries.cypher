// LDBC SNB Interactive — IC3: Friends located in two countries.
MATCH (p:Person {id: $personId})-[:KNOWS*1..2]-(friend:Person)-[:IS_LOCATED_IN]->(city:City)-[:IS_PART_OF]->(country:Country)
WHERE NOT friend.id = p.id
  AND country.name IN [$countryAName, $countryBName]
WITH friend, country
MATCH (friend)<-[:HAS_CREATOR]-(message:Message)-[:IS_LOCATED_IN]->(country)
WHERE message.creationDate >= $startDate
  AND message.creationDate < $endDate
WITH friend, country.name AS countryName, count(message) AS messageCount
WITH friend, collect({name: countryName, count: messageCount}) AS countries
WHERE size(countries) = 2
RETURN friend.id AS personId,
       friend.firstName AS personFirstName,
       friend.lastName AS personLastName,
       countries
ORDER BY personId ASC
LIMIT 20
