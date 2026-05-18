// LDBC SNB Interactive Update — IU1 simplified: insert a Person and
// connect to one existing friend. The official IU1 inserts a Person plus
// edges to a list of friends + studies-at university + works-at company
// + lives-in city. We keep just the Person + a single KNOWS for v0.
CREATE (p:Person {id: $personId, firstName: $firstName, lastName: $lastName, age: $age})
WITH p
MATCH (f:Person {id: $friendId})
CREATE (p)-[r:KNOWS]->(f)
RETURN p.id AS createdId
