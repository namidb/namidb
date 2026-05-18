//! Four LDBC SNB Interactive Complex Read queries NamiDB supports
//! end-to-end today. Each query is parameterised by an integer; the
//! benchmark picks `param_count` distinct values and runs each multiple
//! times.
//!
//! The queries omit features NamiDB doesn't yet parse (recursive
//! variable-length paths beyond `*1..2`, `STDEV`, etc.); they remain
//! recognisably the LDBC shape so a Kuzu run over the same dataset
//! produces comparable rows.

/// One of the four supported LDBC SNB Complex Read queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Query {
 /// IC2 — recent messages by friends of a given Person.
 Ic02,
 /// IC7 — recent likers of any of a Person's messages.
 Ic07,
 /// IC8 — recent replies to any of a Person's messages.
 Ic08,
 /// IC9 — recent messages by friends-of-friends.
 Ic09,
}

impl Query {
 pub fn name(self) -> &'static str {
 match self {
 Query::Ic02 => "ic02",
 Query::Ic07 => "ic07",
 Query::Ic08 => "ic08",
 Query::Ic09 => "ic09",
 }
 }

 /// Render the query as Cypher text with `$personId` substituted by
 /// the supplied id literal. The text is consumed verbatim by
 /// `namidb_query::parse`, so anything that pretends to be a
 /// parameter must already be inlined.
 pub fn cypher(self, person_id: &str) -> String {
 match self {
 Query::Ic02 => format!(
 "MATCH (p:Person {{id: '{pid}'}})-[:KNOWS]->(friend:Person)<-[:HAS_CREATOR]-(message:Post) \
 RETURN friend.firstName AS personFirstName, friend.lastName AS personLastName, \
 message.content AS messageContent, message.creationDate AS messageCreationDate \
 ORDER BY messageCreationDate DESC LIMIT 20",
 pid = person_id,
 ),
 Query::Ic07 => format!(
 "MATCH (p:Person {{id: '{pid}'}})<-[:HAS_CREATOR]-(message:Post)<-[liker:LIKES]-(fan:Person) \
 RETURN fan.firstName AS personFirstName, fan.lastName AS personLastName, \
 liker.creationDate AS likeCreationDate, message.content AS messageContent \
 ORDER BY likeCreationDate DESC LIMIT 20",
 pid = person_id,
 ),
 Query::Ic08 => format!(
 "MATCH (p:Person {{id: '{pid}'}})<-[:HAS_CREATOR]-(post:Post)<-[:REPLY_OF]-(reply:Comment) \
 RETURN reply.content AS replyContent, reply.creationDate AS replyDate, \
 post.content AS postContent \
 ORDER BY replyDate DESC LIMIT 20",
 pid = person_id,
 ),
 Query::Ic09 => format!(
 "MATCH (p:Person {{id: '{pid}'}})-[:KNOWS]->(friend:Person)-[:KNOWS]->(fof:Person) \
 <-[:HAS_CREATOR]-(msg:Post) \
 RETURN fof.firstName AS personFirstName, fof.lastName AS personLastName, \
 msg.content AS messageContent, msg.creationDate AS messageCreationDate \
 ORDER BY messageCreationDate DESC LIMIT 20",
 pid = person_id,
 ),
 }
 }
}

#[cfg(test)]
mod tests {
 use super::*;

 #[test]
 fn ic02_cypher_substitutes_param() {
 let q = Query::Ic02.cypher("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
 assert!(q.contains("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"));
 assert!(q.contains("MATCH"));
 assert!(q.contains("LIMIT 20"));
 }
}
