// LDBC SNB Interactive — IC11: Job referral search.
MATCH (p:Person {id: $personId})-[:KNOWS*1..2]-(friend:Person)-[work:WORK_AT]->(company:Company)-[:IS_LOCATED_IN]->(country:Country {name: $countryName})
WHERE work.workFrom < $workFromYear
  AND NOT friend.id = p.id
RETURN friend.id AS personId,
       friend.firstName AS personFirstName,
       friend.lastName AS personLastName,
       company.name AS organizationName,
       work.workFrom AS organizationWorkFromYear
ORDER BY organizationWorkFromYear ASC, personId ASC, organizationName DESC
LIMIT 10
