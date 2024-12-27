use crate::embeddings::{
    EmbeddingConfig, EmbeddingPreload, EmbeddingService, OramaFastembedModel, OramaModel,
};
use chrono::{DateTime, Utc};
use petgraph::data::Build;
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

pub type Metadata = HashMap<String, serde_json::Value>;
pub type Timestamp = DateTime<Utc>;
pub type SemanticSearchResult = Vec<(String, f32, Metadata)>;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Message {
    content: String,
    timestamp: Timestamp,
    embedding: Vec<f32>,
    meta: Metadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Entity {
    name: String,
    entity_type: String,
    first_mentioned: Timestamp,
    meta: Metadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum NodeType {
    Session { user_id: String, meta: Metadata },
    Message(Message),
    Entity(Entity),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Edge {
    relationship_type: String,
    meta: Metadata,
    timestamp: Timestamp,
}

struct ConversationalKG {
    graph: DiGraph<NodeType, Edge>,
    node_indices: HashMap<Uuid, NodeIndex>,
    embeddings_service: EmbeddingService,
}

impl ConversationalKG {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let embedding_config = EmbeddingConfig {
            cache_path: "~/cache".to_string(),
            hugging_face: None,
            preload: EmbeddingPreload::List(vec![OramaModel::Fastembed(
                OramaFastembedModel::MultilingualE5Base,
            )]),
        };

        let embeddings_service = EmbeddingService::try_new(embedding_config).await?;

        Ok(Self {
            graph: DiGraph::new(),
            node_indices: HashMap::new(),
            embeddings_service,
        })
    }

    pub fn add_session(&mut self, user_id: String, meta: Metadata) -> Uuid {
        let session_id = Uuid::new_v4();

        let node_idx = self.graph.add_node(NodeType::Session { user_id, meta });

        self.node_indices.insert(session_id, node_idx);

        session_id
    }

    pub async fn add_message(
        &mut self,
        session_id: Uuid,
        content: String,
        meta: Metadata,
    ) -> Result<Uuid, Box<dyn std::error::Error>> {
        let embeddings = self
            .embeddings_service
            .embed(
                OramaModel::Fastembed(OramaFastembedModel::MultilingualE5Base),
                vec![content],
                None,
            )
            .await?
            .first()
            .unwrap()
            .to_owned();

        let message = Message {
            content,
            timestamp: Utc::now(),
            embedding: embeddings,
            meta,
        };

        let message_id = Uuid::new_v4();
        let message_idx = self.graph.add_node(NodeType::Message(message));
        self.node_indices.insert(message_id, message_idx);

        if let Some(_session_idx) = self.node_indices.get(&session_id) {
            self.graph.add_edge(
                *session_id,
                message_idx,
                Edge {
                    relationship_type: "contains_message".to_string(),
                    meta: HashMap::new(),
                    timestamp: Utc::now(),
                },
            );
        }

        Ok(message_id)
    }

    pub async fn add_entity(&mut self, name: String, entity_type: String, meta: Metadata) -> Uuid {
        let entity = Entity {
            name,
            entity_type,
            first_mentioned: Utc::now(),
            meta,
        };

        let entity_id = Uuid::new_v4();
        let entity_idx = self.graph.add_node(NodeType::Entity(entity));
        self.node_indices.insert(entity_id, entity_idx);

        entity_id
    }

    pub fn add_relationship(
        &mut self,
        from_id: Uuid,
        to_id: Uuid,
        relationship_type: String,
        meta: Metadata,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let from_idx = self
            .node_indices
            .get(&from_id)
            .ok_or("Source node not found")?;
        let to_idx = self
            .node_indices
            .get(&to_id)
            .ok_or("Target node not found")?;

        self.graph.add_edge(
            *from_idx,
            *to_idx,
            Edge {
                relationship_type,
                meta,
                timestamp: Utc::now(),
            },
        );

        Ok(())
    }

    pub async fn semantic_search(
        &self,
        query: String,
        context: Option<Metadata>,
        top_k: usize,
    ) -> Result<SemanticSearchResult, Box<dyn std::error::Error>> {
        let query_embedding = self
            .embeddings_service
            .embed(
                OramaModel::Fastembed(OramaFastembedModel::MultilingualE5Base),
                vec![query],
                None,
            )
            .await?
            .first()
            .unwrap()
            .to_owned();

        let mut similarities: SemanticSearchResult = Vec::new();

        for node_idx in self.graph.node_indices() {
            if let NodeType::Message(message) = &self.graph[node_idx] {
                let similarity =
                    Self::cosine_similarity(&query_embedding, &message.embedding).unwrap();

                if let Some(ref context) = context {
                    let matches_context = context
                        .iter()
                        .all(|(key, value)| message.meta.get(key).map_or(false, |v| v == value));

                    if !matches_context {
                        continue;
                    }
                }

                similarities.push((message.content.clone(), similarity, message.meta.clone()));
            }
        }

        similarities.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
        Ok(similarities.into_iter().take(top_k).collect())
    }

    pub fn get_related_entities(
        &self,
        node_id: Uuid,
        relationship_type: Option<String>,
    ) -> Vec<(Entity, String, Metadata)> {
        let mut related = Vec::new();

        if let Some(node_idx) = self.node_indices.get(&node_id) {
            for neighbor in self.graph.neighbors(*node_id) {
                if let NodeType::Entity(entity) = &self.graph[neighbor] {
                    let edge = self
                        .graph
                        .find_edge(*node_idx, neighbor)
                        .and_then(|e| Some(&self.graph[e]));

                    if let Some(edge) = edge {
                        if relationship_type
                            .as_ref()
                            .map_or(true, |rt| rt == &edge.relationship_type)
                        {
                            related.push((
                                entity.clone(),
                                edge.relationship_type.clone(),
                                edge.meta.clone(),
                            ));
                        }
                    }
                }
            }
        }

        related
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }

        let (mut dot_product, mut norm_a, mut norm_b) = (0.0f32, 0.0f32, 0.0f32);

        for (&val_a, &val_b) in a.iter().zip(b.iter()) {
            dot_product += val_a * val_b;
            norm_a += val_a * val_a;
            norm_b += val_b * val_b;
        }

        if norm_a == 0.0 || norm_b == 0.0 {
            return None;
        }

        Some(dot_product / (norm_a.sqrt() * norm_b.sqrt()))
    }
}
