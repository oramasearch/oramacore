use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use anyhow::{anyhow, Context, Result};
use committed::{
    fields::{
        BoolCommittedFieldStats, NumberCommittedFieldStats, StringCommittedFieldStats,
        StringFilterCommittedFieldStats, VectorCommittedFieldStats,
    },
    CommittedCollection,
};
use dashmap::DashMap;
use debug_panic::debug_panic;
use dump::CollectionInfo;
use merge::{
    merge_bool_field, merge_number_field, merge_string_field, merge_string_filter_field,
    merge_vector_field,
};
use redact::Secret;
use serde::{Deserialize, Serialize};
use tokio::{join, sync::RwLock};
use tracing::{debug, error, info, instrument, trace};
use uncommitted::{
    fields::{
        BoolUncommittedFieldStats, NumberUncommittedFieldStats, StringFilterField,
        StringFilterUncommittedFieldStats, StringUncommittedFieldStats,
        VectorUncommittedFieldStats,
    },
    UncommittedCollection,
};

pub mod stats {
    pub use super::committed::fields::{
        BoolCommittedFieldStats, NumberCommittedFieldStats, StringCommittedFieldStats,
        StringFilterCommittedFieldStats, VectorCommittedFieldStats,
    };
    pub use super::uncommitted::fields::{
        BoolUncommittedFieldStats, NumberUncommittedFieldStats, StringFilterUncommittedFieldStats,
        StringUncommittedFieldStats, VectorUncommittedFieldStats,
    };
}

mod committed;
mod merge;
mod uncommitted;

use crate::{
    ai::{
        llms::{self, LLMService},
        AIService, OramaModel,
    },
    collection_manager::{
        bm25::BM25Scorer,
        sides::{CollectionWriteOperation, Offset, OramaModelSerializable, TypedFieldWrapper},
    },
    file_utils::BufferedFile,
    metrics::{
        commit::FIELD_COMMIT_CALCULATION_TIME,
        search::{
            FILTER_CALCULATION_TIME, FILTER_COUNT_CALCULATION_COUNT, FILTER_PERC_CALCULATION_COUNT,
            MATCHING_COUNT_CALCULTATION_COUNT, MATCHING_PERC_CALCULATION_COUNT,
        },
        CollectionFieldCommitLabels, CollectionLabels,
    },
    nlp::{locales::Locale, NLPService, TextParser},
    offset_storage::OffsetStorage,
    types::{
        ApiKey, CollectionId, DocumentId, FacetDefinition, FacetResult, FieldId, Filter,
        FulltextMode, HybridMode, LanguageDTO, Limit, NumberFilter, Properties, SearchMode,
        SearchModeResult, SearchParams, Similarity, VectorMode,
    },
};

#[derive(Debug)]
pub struct CollectionReader {
    pub(super) id: CollectionId,
    description: Option<String>,
    default_language: LanguageDTO,
    deleted: bool,

    read_api_key: ApiKey,
    ai_service: Arc<AIService>,
    nlp_service: Arc<NLPService>,
    llm_service: Arc<LLMService>,

    document_count: AtomicU64,

    score_fields: DashMap<String, (FieldId, TypedField)>,
    filter_fields: DashMap<String, (FieldId, TypedField)>,

    uncommitted_collection: RwLock<UncommittedCollection>,
    committed_collection: RwLock<CommittedCollection>,
    uncommitted_deleted_documents: RwLock<HashSet<DocumentId>>,

    fields_per_model: DashMap<OramaModel, Vec<FieldId>>,

    text_parser_per_field: DashMap<FieldId, (Locale, Arc<TextParser>)>,

    offset_storage: OffsetStorage,
}

impl CollectionReader {
    pub fn empty(
        id: CollectionId,
        description: Option<String>,
        default_language: LanguageDTO,
        read_api_key: ApiKey,
        ai_service: Arc<AIService>,
        nlp_service: Arc<NLPService>,
        llm_service: Arc<LLMService>,
    ) -> Self {
        Self {
            id,
            description,
            default_language,
            deleted: false,

            read_api_key,
            ai_service,
            nlp_service,
            llm_service,
            document_count: AtomicU64::new(0),
            filter_fields: Default::default(),
            score_fields: Default::default(),

            uncommitted_collection: RwLock::new(UncommittedCollection::new()),
            committed_collection: RwLock::new(CommittedCollection::empty()),
            uncommitted_deleted_documents: Default::default(),

            fields_per_model: Default::default(),

            text_parser_per_field: Default::default(),

            offset_storage: OffsetStorage::new(),
        }
    }

    pub fn try_load(
        ai_service: Arc<AIService>,
        nlp_service: Arc<NLPService>,
        llm_service: Arc<LLMService>,
        data_dir: PathBuf,
    ) -> Result<Self> {
        info!("Loading collection from {:?}", data_dir);

        let collection_info_path = data_dir.join("info.info");
        let current_offset: Offset = BufferedFile::open(collection_info_path.clone())
            .context("Cannot open previous collection info")?
            .read_json_data()
            .context("Cannot read previous info")?;

        info!("Current offset: {:?}", current_offset);

        let collection_info_path = data_dir.join(format!("info-offset-{}.info", current_offset.0));
        let collection_info: CollectionInfo = BufferedFile::open(collection_info_path)
            .context("Cannot open previous collection info")?
            .read_json_data()
            .context("Cannot read previous collection info")?;

        let collection_info: dump::CollectionInfoV2 = match collection_info {
            dump::CollectionInfo::V1(info) => dump::migrate_v1_to_v2(info),
            dump::CollectionInfo::V2(info) => info,
        };
        let read_api_key = ApiKey(Secret::new(collection_info.read_api_key));

        let score_fields: DashMap<String, (FieldId, TypedField)> = Default::default();
        for (field_name, (field_id, field_type)) in collection_info.score_fields {
            let typed_field: TypedField = match field_type {
                dump::TypedField::Text(locale) => TypedField::Text(locale),
                dump::TypedField::Embedding(embedding) => TypedField::Embedding(embedding.model.0),
                dump::TypedField::Number => continue,
                dump::TypedField::Bool => continue,
                dump::TypedField::String => continue,
            };
            score_fields.insert(field_name, (field_id, typed_field));
        }

        let filter_fields: DashMap<String, (FieldId, TypedField)> = Default::default();
        for (field_name, (field_id, field_type)) in collection_info.filter_fields {
            let typed_field: TypedField = match field_type {
                dump::TypedField::Text(_) => continue,
                dump::TypedField::Embedding(_) => continue,
                dump::TypedField::Number => TypedField::Number,
                dump::TypedField::Bool => TypedField::Bool,
                dump::TypedField::String => TypedField::String,
            };
            filter_fields.insert(field_name, (field_id, typed_field));
        }

        let fields_per_model: DashMap<OramaModel, Vec<FieldId>> = Default::default();
        for (orama_model, fields) in collection_info.used_models {
            fields_per_model.insert(orama_model.0, fields);
        }

        let text_parser_per_field: DashMap<FieldId, (Locale, Arc<TextParser>)> = score_fields
            .iter()
            .filter_map(|e| {
                if let TypedField::Text(l) = e.1 {
                    let locale = l;
                    Some((e.0, (locale, nlp_service.get(locale))))
                } else if let TypedField::ArrayText(l) = e.1 {
                    let locale = l;
                    Some((e.0, (locale, nlp_service.get(locale))))
                } else {
                    None
                }
            })
            .collect();

        let committed_collection = CommittedCollection::try_load(
            collection_info.number_field_infos,
            collection_info.bool_field_infos,
            collection_info.string_filter_field_infos,
            collection_info.string_field_infos,
            collection_info.vector_field_infos,
        )
        .context("Cannot load committed collection")?;
        let committed_collection = RwLock::new(committed_collection);

        let offset_storage = OffsetStorage::new();
        offset_storage.set_offset(current_offset);

        Ok(Self {
            id: collection_info.id,
            description: collection_info.description,
            default_language: collection_info.default_language,
            deleted: false,
            read_api_key,
            ai_service,
            nlp_service,
            llm_service,
            document_count: AtomicU64::new(collection_info.document_count),
            fields_per_model,
            text_parser_per_field,
            filter_fields,
            score_fields,

            uncommitted_collection: RwLock::new(UncommittedCollection::new()),
            committed_collection,
            uncommitted_deleted_documents: RwLock::new(HashSet::new()),

            offset_storage,
        })
    }

    pub fn mark_as_deleted(&mut self) {
        self.deleted = true;
    }

    pub fn is_deleted(&self) -> bool {
        self.deleted
    }

    #[inline]
    pub fn check_read_api_key(&self, api_key: ApiKey) -> Result<()> {
        if api_key != self.read_api_key {
            return Err(anyhow!("Invalid read api key"));
        }
        Ok(())
    }

    #[inline]
    pub fn get_id(&self) -> CollectionId {
        self.id
    }

    #[instrument(skip(self, data_dir), fields(coll_id = ?self.id))]
    pub async fn commit(&self, data_dir: PathBuf, force_new: bool) -> Result<()> {
        info!("Committing collection");

        let offset = self.offset_storage.get_offset();
        debug!("Committing with offset: {:?}", offset);

        let collection_info_path = data_dir.join("info.info");
        let previous_offset: Option<Offset> = match BufferedFile::open(collection_info_path.clone())
            .context("Cannot open previous collection info")?
            .read_json_data()
        {
            Ok(offset) => Some(offset),
            Err(_) => None,
        };

        let previous_offset_collection_info_path = previous_offset.map(|previous_offset| {
            data_dir.join(format!("info-offset-{}.info", previous_offset.0))
        });
        let mut current_collection_info = if force_new {
            debug!("Force to create new new one");
            dump::CollectionInfoV2 {
                id: self.id,
                description: self.description.clone(),
                document_count: 0,
                deleted: false,
                default_language: self.default_language,
                filter_fields: Default::default(),
                score_fields: Default::default(),
                read_api_key: self.read_api_key.0.expose_secret().clone(),
                used_models: Default::default(),
                number_field_infos: Default::default(),
                string_field_infos: Default::default(),
                string_filter_field_infos: Default::default(),
                bool_field_infos: Default::default(),
                vector_field_infos: Default::default(),
            }
        } else if let Some(previous_offset_collection_info_path) =
            previous_offset_collection_info_path
        {
            debug!("We will merge new data with previous one");
            let previous_collection_info: CollectionInfo =
                BufferedFile::open(previous_offset_collection_info_path)
                    .context("Cannot open previous collection info")?
                    .read_json_data()
                    .context("Cannot read previous collection info")?;

            match previous_collection_info {
                CollectionInfo::V1(info) => dump::migrate_v1_to_v2(info),
                CollectionInfo::V2(info) => info,
            }
        } else {
            debug!("No previous collection info found, creating a new one");
            dump::CollectionInfoV2 {
                id: self.id,
                description: self.description.clone(),
                document_count: 0,
                deleted: false,
                default_language: self.default_language,
                filter_fields: Default::default(),
                score_fields: Default::default(),
                read_api_key: self.read_api_key.0.expose_secret().clone(),
                used_models: Default::default(),
                number_field_infos: Default::default(),
                string_filter_field_infos: Default::default(),
                string_field_infos: Default::default(),
                bool_field_infos: Default::default(),
                vector_field_infos: Default::default(),
            }
        };

        let committed = self.committed_collection.read().await;
        let uncommitted = self.uncommitted_collection.read().await;

        let mut uncommitted_infos = uncommitted.get_infos();
        debug!("Uncommitted info {:?}", uncommitted_infos);

        let uncommitted_document_deletions = self.uncommitted_deleted_documents.read().await;
        let uncommitted_document_deletions = if !uncommitted_document_deletions.is_empty() {
            info!("Uncommitted document deletion: commit every fields");

            let uncommitted_document_deletions = uncommitted_document_deletions.clone();
            let info = committed.get_keys();

            uncommitted_infos.bool_fields.extend(info.bool_fields);
            uncommitted_infos.number_fields.extend(info.number_fields);
            uncommitted_infos.string_fields.extend(info.string_fields);
            uncommitted_infos.vector_fields.extend(info.vector_fields);

            uncommitted_document_deletions
        } else {
            HashSet::new()
        };

        if uncommitted_infos.is_empty() {
            info!("No uncommitted data to commit");

            // If we create a collection with no data, and a commit is triggered,
            // we need to save the collection info file
            if current_collection_info.score_fields.is_empty()
                && current_collection_info.filter_fields.is_empty()
            {
                let new_offset_collection_info_path =
                    data_dir.join(format!("info-offset-{}.info", offset.0));
                BufferedFile::create_or_overwrite(new_offset_collection_info_path)
                    .context("Cannot create previous collection info")?
                    .write_json_data(&CollectionInfo::V2(current_collection_info))
                    .context("Cannot write previous collection info")?;

                BufferedFile::create_or_overwrite(collection_info_path)
                    .context("Cannot create previous collection info")?
                    .write_json_data(&offset)
                    .context("Cannot write previous collection info")?;
            }

            return Ok(());
        }

        debug!("Merging fields {:?}", uncommitted_infos);

        let mut number_fields = HashMap::new();
        let number_dir = data_dir.join("numbers");
        debug!(
            "Merging number fields {:?}",
            uncommitted_infos.number_fields
        );
        let m = FIELD_COMMIT_CALCULATION_TIME.create(CollectionFieldCommitLabels {
            collection: self.id.to_string().into(),
            field_type: "number",
            side: "read",
        });
        for field_id in uncommitted_infos.number_fields {
            let uncommitted_number_index = uncommitted.number_index.get(&field_id);
            let committed_number_index = committed.number_index.get(&field_id);

            let field_dir = number_dir
                .join(format!("field-{}", field_id.0))
                .join(format!("offset-{}", offset.0));
            let new_committed_number_index = merge_number_field(
                uncommitted_number_index,
                committed_number_index,
                field_dir,
                &uncommitted_document_deletions,
            )
            .with_context(|| {
                format!(
                    "Cannot merge {:?} field for collection {:?}",
                    field_id, self.id
                )
            })?;

            let new_committed_number_index = match new_committed_number_index {
                None => {
                    debug_panic!("Number field is not changed");
                    continue;
                }
                Some(new_committed_number_index) => new_committed_number_index,
            };

            let field_info = new_committed_number_index.get_field_info();
            current_collection_info
                .number_field_infos
                .retain(|(k, _)| k != &field_id);
            current_collection_info
                .number_field_infos
                .push((field_id, field_info));

            number_fields.insert(field_id, new_committed_number_index);

            let field = current_collection_info
                .filter_fields
                .iter_mut()
                .find(|(_, (f, _))| f == &field_id);
            match field {
                Some((_, (_, typed_field))) => {
                    if typed_field != &mut dump::TypedField::Number {
                        error!("Field {:?} is changing type and this is not allowed. before {:?} after {:?}", field_id, typed_field, dump::TypedField::Number);
                        return Err(anyhow!(
                            "Field {:?} is changing type and this is not allowed",
                            field_id
                        ));
                    }
                }
                None => {
                    let field_name = self
                        .filter_fields
                        .iter()
                        .find(|e| e.0 == field_id)
                        .context("Number field not registered")?;
                    let field_name = field_name.key().to_string();
                    current_collection_info
                        .filter_fields
                        .push((field_name, (field_id, dump::TypedField::Number)));
                }
            }
        }
        drop(m);
        debug!("Number fields merged");

        let mut string_fields = HashMap::new();
        let string_dir = data_dir.join("strings");
        debug!(
            "Merging string fields {:?}",
            uncommitted_infos.string_fields
        );
        let m = FIELD_COMMIT_CALCULATION_TIME.create(CollectionFieldCommitLabels {
            collection: self.id.to_string().into(),
            field_type: "string",
            side: "read",
        });
        for field_id in uncommitted_infos.string_fields {
            let uncommitted_string_index = uncommitted.string_index.get(&field_id);
            let committed_string_index = committed.string_index.get(&field_id);

            let field_dir = string_dir
                .join(format!("field-{}", field_id.0))
                .join(format!("offset-{}", offset.0));
            let new_committed_string_index = merge_string_field(
                uncommitted_string_index,
                committed_string_index,
                field_dir,
                &uncommitted_document_deletions,
            )
            .with_context(|| {
                format!(
                    "Cannot merge {:?} field for collection {:?}",
                    field_id, self.id
                )
            })?;

            let new_committed_string_index = match new_committed_string_index {
                None => {
                    debug_panic!("String field is not changed");
                    continue;
                }
                Some(new_committed_string_index) => new_committed_string_index,
            };

            let field_info = new_committed_string_index.get_field_info();
            string_fields.insert(field_id, new_committed_string_index);
            current_collection_info
                .string_field_infos
                .retain(|(k, _)| k != &field_id);
            current_collection_info
                .string_field_infos
                .push((field_id, field_info));

            let field_locale = self
                .text_parser_per_field
                .get(&field_id)
                .map(|e| e.0)
                .context("String field not registered")?;
            let field = current_collection_info
                .score_fields
                .iter_mut()
                .find(|(_, (f, _))| f == &field_id);
            match field {
                Some((_, (_, typed_field))) => {
                    if typed_field != &mut dump::TypedField::Text(field_locale) {
                        error!("Field {:?} is changing type and this is not allowed. before {:?} after {:?}", field_id, typed_field, dump::TypedField::Text(field_locale));
                        return Err(anyhow!(
                            "Field {:?} is changing type and this is not allowed",
                            field_id
                        ));
                    }
                }
                None => {
                    let field_name = self
                        .score_fields
                        .iter()
                        .find(|e| e.0 == field_id)
                        .context("String field not registered")?;
                    let field_name = field_name.key().to_string();
                    current_collection_info
                        .score_fields
                        .push((field_name, (field_id, dump::TypedField::Text(field_locale))));
                }
            }
        }
        drop(m);
        debug!("String fields merged");

        let mut bool_fields = HashMap::new();
        let bool_dir = data_dir.join("bools");
        debug!("Merging bool fields {:?}", uncommitted_infos.bool_fields);
        let m = FIELD_COMMIT_CALCULATION_TIME.create(CollectionFieldCommitLabels {
            collection: self.id.to_string().into(),
            field_type: "bool",
            side: "read",
        });
        for field_id in uncommitted_infos.bool_fields {
            let uncommitted_bool_index = uncommitted.bool_index.get(&field_id);
            let committed_bool_index = committed.bool_index.get(&field_id);

            let field_dir = bool_dir
                .join(format!("field-{}", field_id.0))
                .join(format!("offset-{}", offset.0));
            let new_committed_bool_index = merge_bool_field(
                uncommitted_bool_index,
                committed_bool_index,
                field_dir,
                &uncommitted_document_deletions,
            )
            .with_context(|| {
                format!(
                    "Cannot merge {:?} field for collection {:?}",
                    field_id, self.id
                )
            })?;
            let new_committed_bool_index = match new_committed_bool_index {
                None => {
                    debug_panic!("Bool field is not changed");
                    continue;
                }
                Some(new_committed_bool_index) => new_committed_bool_index,
            };

            let field_info = new_committed_bool_index.get_field_info();
            bool_fields.insert(field_id, new_committed_bool_index);
            current_collection_info
                .bool_field_infos
                .retain(|(k, _)| k != &field_id);
            current_collection_info
                .bool_field_infos
                .push((field_id, field_info));

            let field = current_collection_info
                .filter_fields
                .iter_mut()
                .find(|(_, (f, _))| f == &field_id);
            match field {
                Some((_, (_, typed_field))) => {
                    if typed_field != &mut dump::TypedField::Bool {
                        error!("Field {:?} is changing type and this is not allowed. before {:?} after {:?}", field_id, typed_field, dump::TypedField::Bool);
                        return Err(anyhow!(
                            "Field {:?} is changing type and this is not allowed",
                            field_id
                        ));
                    }
                }
                None => {
                    let field_name = self
                        .filter_fields
                        .iter()
                        .find(|e| e.0 == field_id)
                        .context("Bool field not registered")?;
                    let field_name = field_name.key().to_string();
                    current_collection_info
                        .filter_fields
                        .push((field_name, (field_id, dump::TypedField::Bool)));
                }
            }
        }
        drop(m);
        debug!("Bool fields merged");

        let mut string_filter_fields = HashMap::new();
        let string_filter_dir = data_dir.join("string_filters");
        debug!(
            "Merging string filter fields {:?}",
            uncommitted_infos.string_filter_fields
        );
        let m = FIELD_COMMIT_CALCULATION_TIME.create(CollectionFieldCommitLabels {
            collection: self.id.to_string().into(),
            field_type: "string_filter",
            side: "read",
        });
        for field_id in uncommitted_infos.string_filter_fields {
            let uncommitted_string_filter_index = uncommitted.string_filter_index.get(&field_id);
            let committed_string_filter_index = committed.string_filter_index.get(&field_id);

            let field_dir = string_filter_dir
                .join(format!("field-{}", field_id.0))
                .join(format!("offset-{}", offset.0));
            let new_committed_string_filter_index = merge_string_filter_field(
                uncommitted_string_filter_index,
                committed_string_filter_index,
                field_dir,
                &uncommitted_document_deletions,
            )
            .with_context(|| {
                format!(
                    "Cannot merge {:?} field for collection {:?}",
                    field_id, self.id
                )
            })?;
            let new_committed_string_filter_index = match new_committed_string_filter_index {
                None => {
                    debug_panic!("String filter field is not changed");
                    continue;
                }
                Some(new_committed_string_filter_index) => new_committed_string_filter_index,
            };

            let field_info = new_committed_string_filter_index.get_field_info();
            string_filter_fields.insert(field_id, new_committed_string_filter_index);
            current_collection_info
                .string_filter_field_infos
                .retain(|(k, _)| k != &field_id);
            current_collection_info
                .string_filter_field_infos
                .push((field_id, field_info));

            let field = current_collection_info
                .filter_fields
                .iter_mut()
                .find(|(_, (f, _))| f == &field_id);
            match field {
                Some((_, (_, typed_field))) => {
                    if typed_field != &mut dump::TypedField::String {
                        error!("Field {:?} is changing type and this is not allowed. before {:?} after {:?}", field_id, typed_field, dump::TypedField::String);
                        return Err(anyhow!(
                            "Field {:?} is changing type and this is not allowed",
                            field_id
                        ));
                    }
                }
                None => {
                    let field_name = self
                        .filter_fields
                        .iter()
                        .find(|e| e.0 == field_id)
                        .context("String filter field not registered")?;
                    let field_name = field_name.key().to_string();
                    current_collection_info
                        .filter_fields
                        .push((field_name, (field_id, dump::TypedField::String)));
                }
            }
        }
        drop(m);
        debug!("String filter  fields merged");

        let mut vector_fields = HashMap::new();
        let vector_dir = data_dir.join("vectors");
        debug!(
            "Merging vector fields {:?}",
            uncommitted_infos.vector_fields
        );
        let m = FIELD_COMMIT_CALCULATION_TIME.create(CollectionFieldCommitLabels {
            collection: self.id.to_string().into(),
            field_type: "vector",
            side: "read",
        });
        for field_id in uncommitted_infos.vector_fields {
            let uncommitted_vector_index = uncommitted.vector_index.get(&field_id);
            let committed_vector_index = committed.vector_index.get(&field_id);

            let field_dir = vector_dir
                .join(format!("field-{}", field_id.0))
                .join(format!("offset-{}", offset.0));
            let new_committed_vector_index = merge_vector_field(
                uncommitted_vector_index,
                committed_vector_index,
                field_dir,
                &uncommitted_document_deletions,
            )
            .with_context(|| {
                format!(
                    "Cannot merge {:?} field for collection {:?}",
                    field_id, self.id
                )
            })?;

            let new_committed_vector_index = match new_committed_vector_index {
                None => {
                    debug_panic!("Vector field is not changed");
                    continue;
                }
                Some(new_committed_vector_index) => new_committed_vector_index,
            };

            let field_info = new_committed_vector_index.get_field_info();
            vector_fields.insert(field_id, new_committed_vector_index);
            current_collection_info
                .vector_field_infos
                .retain(|(k, _)| k != &field_id);
            current_collection_info
                .vector_field_infos
                .push((field_id, field_info));

            let field = current_collection_info
                .score_fields
                .iter_mut()
                .find(|(_, (f, _))| f == &field_id);
            match field {
                Some((_, (_, _))) => {
                    // TODO: check if the field is changing type
                }
                None => {
                    let field_name = self
                        .score_fields
                        .iter()
                        .find(|e| e.0 == field_id)
                        .context("Vector field not registered - 1")?;
                    let field_name = field_name.key().to_string();

                    let item = self
                        .fields_per_model
                        .iter()
                        .find(|e| e.value().contains(&field_id))
                        .context("Vector field not registered - 2")?;
                    let orama_model = item.key();

                    let serializable_orama_model = OramaModelSerializable(*orama_model);
                    let el = current_collection_info
                        .used_models
                        .iter_mut()
                        .find(|e| e.0 == serializable_orama_model);
                    if let Some(el) = el {
                        el.1.push(field_id);
                    } else {
                        current_collection_info
                            .used_models
                            .push((serializable_orama_model.clone(), vec![field_id]));
                    }

                    current_collection_info.score_fields.push((
                        field_name,
                        (
                            field_id,
                            dump::TypedField::Embedding(dump::EmbeddingTypedField {
                                model: serializable_orama_model,
                            }),
                        ),
                    ));
                }
            }
        }
        drop(m);
        debug!("Vector fields merged");

        // Read lock ends
        drop(committed);
        drop(uncommitted);

        // The following loop should fast, so the read lock is not held for a long time
        let (mut committed, mut uncommitted) = join!(
            self.committed_collection.write(),
            self.uncommitted_collection.write()
        );
        for (field_id, field) in number_fields {
            uncommitted.number_index.remove(&field_id);
            committed.number_index.insert(field_id, field);
        }
        for (field_id, field) in string_filter_fields {
            uncommitted.string_filter_index.remove(&field_id);
            committed.string_filter_index.insert(field_id, field);
        }
        for (field_id, field) in string_fields {
            uncommitted.string_index.remove(&field_id);
            committed.string_index.insert(field_id, field);
        }
        for (field_id, field) in bool_fields {
            uncommitted.bool_index.remove(&field_id);
            committed.bool_index.insert(field_id, field);
        }
        for (field_id, field) in vector_fields {
            uncommitted.vector_index.remove(&field_id);
            committed.vector_index.insert(field_id, field);
        }
        drop(committed);
        drop(uncommitted);

        let new_offset_collection_info_path =
            data_dir.join(format!("info-offset-{}.info", offset.0));
        BufferedFile::create_or_overwrite(new_offset_collection_info_path)
            .context("Cannot create previous collection info")?
            .write_json_data(&CollectionInfo::V2(current_collection_info))
            .context("Cannot write previous collection info")?;

        BufferedFile::create_or_overwrite(collection_info_path)
            .context("Cannot create previous collection info")?
            .write_json_data(&offset)
            .context("Cannot write previous collection info")?;

        info!("Collection committed");

        Ok(())
    }

    pub fn increment_document_count(&self) {
        self.document_count.fetch_add(1, Ordering::Relaxed);
    }

    pub async fn update(
        &self,
        offset: Offset,
        collection_operation: CollectionWriteOperation,
    ) -> Result<()> {
        match collection_operation {
            CollectionWriteOperation::InsertDocument { .. } => {
                unreachable!("InsertDocument is not managed by the collection");
            }
            CollectionWriteOperation::DeleteDocuments { doc_ids } => {
                let len = doc_ids.len() as u64;
                self.offset_storage.set_offset(offset);
                let mut uncommitted_deleted_documents =
                    self.uncommitted_deleted_documents.write().await;
                uncommitted_deleted_documents.extend(doc_ids);
                self.document_count
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |a| {
                        Some(a.saturating_sub(len))
                    })
                    .map_err(|_| anyhow!("Cannot update document count"))?;
                self.document_count.fetch_sub(len, Ordering::Relaxed);
                info!("Document deleted: {:?}", len);
            }
            CollectionWriteOperation::CreateField {
                field_id,
                field_name,
                field: typed_field,
            } => {
                trace!(collection_id=?self.id, ?field_id, ?field_name, ?typed_field, "Creating field");

                match typed_field {
                    TypedFieldWrapper::Embedding(model) => {
                        self.fields_per_model
                            .entry(model.model.0)
                            .or_default()
                            .push(field_id);
                        self.score_fields.insert(
                            field_name.clone(),
                            (field_id, TypedField::Embedding(model.model.0)),
                        );
                    }
                    TypedFieldWrapper::Text(locale) => {
                        let text_parser = self.nlp_service.get(locale);
                        self.text_parser_per_field
                            .insert(field_id, (locale, text_parser));
                        self.score_fields
                            .insert(field_name.clone(), (field_id, TypedField::Text(locale)));
                    }
                    TypedFieldWrapper::ArrayText(locale) => {
                        let text_parser = self.nlp_service.get(locale);
                        self.text_parser_per_field
                            .insert(field_id, (locale, text_parser));
                        self.score_fields.insert(
                            field_name.clone(),
                            (field_id, TypedField::ArrayText(locale)),
                        );
                    }
                    TypedFieldWrapper::Number => {
                        self.filter_fields
                            .insert(field_name.clone(), (field_id, TypedField::Number));
                    }
                    TypedFieldWrapper::ArrayNumber => {
                        self.filter_fields
                            .insert(field_name.clone(), (field_id, TypedField::ArrayNumber));
                    }
                    TypedFieldWrapper::Bool => {
                        self.filter_fields
                            .insert(field_name.clone(), (field_id, TypedField::Bool));
                    }
                    TypedFieldWrapper::ArrayBoolean => {
                        self.filter_fields
                            .insert(field_name.clone(), (field_id, TypedField::ArrayBoolean));
                    }
                    TypedFieldWrapper::ArrayString => {
                        self.filter_fields
                            .insert(field_name.clone(), (field_id, TypedField::String));
                        self.uncommitted_collection
                            .write()
                            .await
                            .string_filter_index
                            .insert(field_id, StringFilterField::empty());
                    }
                    TypedFieldWrapper::String => {
                        self.filter_fields
                            .insert(field_name.clone(), (field_id, TypedField::ArrayString));
                        self.uncommitted_collection
                            .write()
                            .await
                            .string_filter_index
                            .insert(field_id, StringFilterField::empty());
                    }
                };

                self.offset_storage.set_offset(offset);

                trace!("Field created");
            }
            CollectionWriteOperation::Index(doc_id, field_id, field_op) => {
                trace!(collection_id=?self.id, ?doc_id, ?field_id, ?field_op, "Indexing a new value");

                self.offset_storage.set_offset(offset);

                self.uncommitted_collection
                    .write()
                    .await
                    .insert(field_id, doc_id, field_op)?;

                trace!("Value indexed");
            }
        };

        Ok(())
    }

    #[instrument(skip(self, search_params), level="debug", fields(coll_id = ?self.id))]
    pub async fn search(
        &self,
        search_params: SearchParams,
    ) -> Result<HashMap<DocumentId, f32>, anyhow::Error> {
        info!(search_params = ?search_params, "Start search");
        let SearchParams {
            mode,
            properties,
            boost,
            limit,
            where_filter,
            ..
        } = search_params;

        let uncommitted_deleted_documents = self.uncommitted_deleted_documents.read().await;
        let uncommitted_deleted_documents = uncommitted_deleted_documents.clone();

        let filtered_doc_ids = self
            .calculate_filtered_doc_ids(where_filter, &uncommitted_deleted_documents)
            .await?;
        if let Some(filtered_doc_ids) = &filtered_doc_ids {
            FILTER_PERC_CALCULATION_COUNT.track(
                CollectionLabels {
                    collection: self.id.to_string(),
                },
                filtered_doc_ids.len() as f64 / self.count_documents() as f64,
            );
            FILTER_COUNT_CALCULATION_COUNT.track_usize(
                CollectionLabels {
                    collection: self.id.to_string(),
                },
                filtered_doc_ids.len(),
            );
        }

        // Manage the "auto" search mode. OramaCore will automatically determine
        // wether to use full text search, vector search or hybrid search.
        let search_mode: SearchMode = match mode.clone() {
            SearchMode::Auto(mode_result) => {
                let final_mode: String = self
                    .llm_service
                    .run_known_prompt(
                        llms::KnownPrompts::Autoquery,
                        vec![("query".to_string(), mode_result.term.clone())],
                        // @todo: determine if we want to allow the user to select which LLM to use here.
                        None,
                    )
                    .await?;

                let serilized_final_mode: SearchModeResult = serde_json::from_str(&final_mode)?;

                match serilized_final_mode.mode.as_str() {
                    "fulltext" => SearchMode::FullText(FulltextMode {
                        term: mode_result.term,
                    }),
                    "hybrid" => SearchMode::Hybrid(HybridMode {
                        term: mode_result.term,
                        similarity: Similarity(0.8),
                    }),
                    "vector" => SearchMode::Vector(VectorMode {
                        term: mode_result.term,
                        similarity: Similarity(0.8),
                    }),
                    _ => anyhow::bail!("Invalid search mode"),
                }
            }
            _ => mode,
        };

        let boost = self.calculate_boost(boost);

        let token_scores = match search_mode {
            SearchMode::Default(search_params) | SearchMode::FullText(search_params) => {
                let properties = self.calculate_string_properties(properties)?;
                self.search_full_text(
                    &search_params.term,
                    properties,
                    boost,
                    filtered_doc_ids.as_ref(),
                    &uncommitted_deleted_documents,
                )
                .await?
            }
            SearchMode::Vector(search_params) => {
                self.search_vector(
                    &search_params.term,
                    search_params.similarity.0,
                    filtered_doc_ids.as_ref(),
                    &limit,
                    &uncommitted_deleted_documents,
                )
                .await?
            }
            SearchMode::Hybrid(search_params) => {
                let properties = self.calculate_string_properties(properties)?;

                let (vector, fulltext) = join!(
                    self.search_vector(
                        &search_params.term,
                        search_params.similarity.0,
                        filtered_doc_ids.as_ref(),
                        &limit,
                        &uncommitted_deleted_documents
                    ),
                    self.search_full_text(
                        &search_params.term,
                        properties,
                        boost,
                        filtered_doc_ids.as_ref(),
                        &uncommitted_deleted_documents,
                    )
                );
                let vector = vector?;
                let fulltext = fulltext?;

                // min-max normalization
                let max = vector.values().copied().fold(0.0, f32::max);
                let max = max.max(fulltext.values().copied().fold(0.0, f32::max));
                let min = vector.values().copied().fold(0.0, f32::min);
                let min = min.min(fulltext.values().copied().fold(0.0, f32::min));

                let vector: HashMap<_, _> = vector
                    .into_iter()
                    .map(|(k, v)| (k, (v - min) / (max - min)))
                    .collect();

                let mut fulltext: HashMap<_, _> = fulltext
                    .into_iter()
                    .map(|(k, v)| (k, (v - min) / (max - min)))
                    .collect();

                for (k, v) in vector {
                    let e = fulltext.entry(k).or_default();
                    *e += v;
                }
                fulltext
            }
            SearchMode::Auto(_) => unreachable!(),
        };

        MATCHING_COUNT_CALCULTATION_COUNT.track_usize(
            CollectionLabels {
                collection: self.id.to_string(),
            },
            token_scores.len(),
        );
        MATCHING_PERC_CALCULATION_COUNT.track(
            CollectionLabels {
                collection: self.id.to_string(),
            },
            token_scores.len() as f64 / self.count_documents() as f64,
        );

        info!("token_scores len: {:?}", token_scores.len());
        debug!("token_scores: {:?}", token_scores);

        Ok(token_scores)
    }

    pub fn count_documents(&self) -> u64 {
        self.document_count.load(Ordering::Relaxed)
    }

    fn calculate_boost(&self, boost: HashMap<String, f32>) -> HashMap<FieldId, f32> {
        boost
            .into_iter()
            .filter_map(|(field_name, boost)| {
                let field_id = self.score_fields.get(&field_name)?.0;
                Some((field_id, boost))
            })
            .collect()
    }

    async fn calculate_filtered_doc_ids(
        &self,
        where_filter: HashMap<String, Filter>,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
    ) -> Result<Option<HashSet<DocumentId>>> {
        if where_filter.is_empty() {
            return Ok(None);
        }

        let filter_m = FILTER_CALCULATION_TIME.create(CollectionLabels {
            collection: self.id.to_string(),
        });

        info!(
            "where_filter: {:?} {:?}",
            where_filter, self.uncommitted_collection
        );

        let filters: Result<Vec<_>> = where_filter
            .into_iter()
            .map(|(field_name, value)| {
                // This error should be typed.
                // We could return a formatted message to http
                // so, the user can understand what is wrong
                // TODO: do it
                self.filter_fields
                    .get(&field_name)
                    .with_context(|| format!("Cannot filter by \"{}\": unknown field", &field_name))
                    .map(|e| {
                        let field_name = e.key().clone();
                        let (field_id, field_type) = e.value();
                        (field_name, *field_id, field_type.clone(), value)
                    })
            })
            .collect();
        let mut filters = filters?;
        let (field_name, field_id, field_type, filter) = filters
            .pop()
            .expect("filters is not empty here. it is already checked");

        info!(
            "Filtering on field {:?}({:?}): {:?}",
            field_name, field_type, filter
        );

        let mut doc_ids = get_filtered_document(
            self,
            &field_name,
            field_id,
            &field_type,
            filter,
            uncommitted_deleted_documents,
        )
        .await?;
        for (field_name, field_id, field_type, filter) in filters {
            let doc_ids_for_field = get_filtered_document(
                self,
                &field_name,
                field_id,
                &field_type,
                filter,
                uncommitted_deleted_documents,
            )
            .await?;
            doc_ids = doc_ids.intersection(&doc_ids_for_field).copied().collect();
        }

        info!("Matching doc from filters: {:?}", doc_ids.len());

        drop(filter_m);

        Ok(Some(doc_ids))
    }

    fn calculate_string_properties(&self, properties: Properties) -> Result<Vec<FieldId>> {
        let properties: Vec<_> = match properties {
            Properties::Specified(properties) => {
                let mut r = Vec::with_capacity(properties.len());
                for field_name in properties {
                    let field = self.score_fields.get(&field_name);
                    let field = match field {
                        None => return Err(anyhow!("Unknown field name {}", field_name)),
                        Some(field) => field,
                    };
                    if !matches!(field.1, TypedField::Text(_)) {
                        return Err(anyhow!("Cannot search on non-string field {}", field_name));
                    }
                    r.push(field.0);
                }
                r
            }
            Properties::None | Properties::Star => {
                let mut r = Vec::with_capacity(self.score_fields.len());
                for field in &self.score_fields {
                    if !matches!(field.1, TypedField::Text(_) | TypedField::ArrayText(_)) {
                        continue;
                    }
                    r.push(field.0);
                }
                r
            }
        };

        Ok(properties)
    }

    async fn search_full_text(
        &self,
        term: &str,
        properties: Vec<FieldId>,
        boost: HashMap<FieldId, f32>,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
    ) -> Result<HashMap<DocumentId, f32>> {
        let mut scorer: BM25Scorer<DocumentId> = BM25Scorer::new();

        let mut tokens_cache: HashMap<Locale, Vec<String>> = Default::default();

        let committed_lock = self.committed_collection.read().await;
        let uncommitted_lock = self.uncommitted_collection.read().await;

        for field_id in properties {
            info!(?field_id, "Searching on field");
            let text_parser = self.text_parser_per_field.get(&field_id);
            let (locale, text_parser) = match text_parser.as_ref() {
                None => return Err(anyhow!("No text parser for this field")),
                Some(text_parser) => (text_parser.0, text_parser.1.clone()),
            };
            let tokens = tokens_cache.entry(locale).or_insert_with(|| {
                let a = text_parser.tokenize_and_stem(term);
                a.into_iter()
                    .flat_map(|e| std::iter::once(e.0).chain(e.1.into_iter()))
                    .collect()
            });

            let committed_global_info = committed_lock.global_info(&field_id);
            let uncommitted_global_info = uncommitted_lock.global_info(&field_id);
            let global_info = committed_global_info + uncommitted_global_info;

            committed_lock.fulltext_search(
                tokens,
                vec![field_id],
                &boost,
                filtered_doc_ids,
                &mut scorer,
                &global_info,
                uncommitted_deleted_documents,
            )?;
            uncommitted_lock.fulltext_search(
                tokens,
                vec![field_id],
                &boost,
                filtered_doc_ids,
                &mut scorer,
                &global_info,
                uncommitted_deleted_documents,
            )?;
        }

        Ok(scorer.get_scores())
    }

    async fn search_vector(
        &self,
        term: &str,
        similarity: f32,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        limit: &Limit,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
    ) -> Result<HashMap<DocumentId, f32>> {
        let mut output: HashMap<DocumentId, f32> = HashMap::new();

        let committed_lock = self.committed_collection.read().await;
        let uncommitted_lock = self.uncommitted_collection.read().await;

        info!("fields_per_model: {:?}", self.fields_per_model);

        for e in &self.fields_per_model {
            let model = e.key();
            let fields = e.value();

            info!("Searching on model {:?} on fields {:?}", model, fields);

            let targets = self
                .ai_service
                .embed_query(*model, vec![&term.to_string()])
                .await?;

            for target in targets {
                committed_lock.vector_search(
                    &target,
                    similarity,
                    fields,
                    filtered_doc_ids,
                    limit.0,
                    &mut output,
                    uncommitted_deleted_documents,
                )?;
                uncommitted_lock.vector_search(
                    &target,
                    similarity,
                    fields,
                    filtered_doc_ids,
                    &mut output,
                    uncommitted_deleted_documents,
                )?;
            }
        }

        Ok(output)
    }

    pub async fn calculate_facets(
        &self,
        token_scores: &HashMap<DocumentId, f32>,
        facets: HashMap<String, FacetDefinition>,
    ) -> Result<Option<HashMap<String, FacetResult>>> {
        if facets.is_empty() {
            return Ok(None);
        }

        info!("Computing facets on {:?}", facets.keys());

        let mut res_facets: HashMap<String, FacetResult> = HashMap::new();
        for (field_name, facet) in facets {
            let field_id = self.filter_fields.get(&field_name);
            let field_id = match field_id {
                None => return Err(anyhow!("Unknown field name {}", field_name)),
                Some(field_id) => field_id.0,
            };

            // This calculation is not efficient
            // we have the doc_ids that matches:
            // - filters
            // - search
            // We should use them to calculate the facets
            // Instead here we are building an hashset and
            // iter again on it to filter the doc_ids.
            // We could create a dedicated method in the indexes that
            // accepts the matching doc_ids + facet definition and returns the count
            // TODO: do it
            match facet {
                FacetDefinition::Number(facet) => {
                    let mut values = HashMap::new();

                    let committed = self.committed_collection.read().await;
                    let uncommitted = self.uncommitted_collection.read().await;
                    for range in facet.ranges {
                        let filter = NumberFilter::Between((range.from, range.to));
                        let committed_output =
                            committed.calculate_number_filter(field_id, &filter)?;
                        let uncommitted_output =
                            uncommitted.calculate_number_filter(field_id, &filter)?;

                        let facet = match (committed_output, uncommitted_output) {
                            (Some(committed_output), Some(uncommitted_output)) => committed_output
                                .chain(uncommitted_output)
                                .filter(|doc_id| token_scores.contains_key(doc_id))
                                .collect(),
                            (Some(committed_output), None) => committed_output
                                .filter(|doc_id| token_scores.contains_key(doc_id))
                                .collect(),
                            (None, Some(uncommitted_output)) => uncommitted_output
                                .filter(|doc_id| token_scores.contains_key(doc_id))
                                .collect(),
                            (None, None) => HashSet::new(),
                        };

                        values.insert(format!("{}-{}", range.from, range.to), facet.len());
                    }

                    res_facets.insert(
                        field_name,
                        FacetResult {
                            count: values.len(),
                            values,
                        },
                    );
                }
                FacetDefinition::Bool(facets) => {
                    let mut values = HashMap::new();

                    let committed = self.committed_collection.read().await;
                    let uncommitted = self.uncommitted_collection.read().await;

                    if facets.r#true {
                        let committed_output = committed.calculate_bool_filter(field_id, true)?;
                        let uncommitted_output =
                            uncommitted.calculate_bool_filter(field_id, true)?;
                        let true_facet = match (committed_output, uncommitted_output) {
                            (Some(committed_output), Some(uncommitted_output)) => committed_output
                                .chain(uncommitted_output)
                                .filter(|doc_id| token_scores.contains_key(doc_id))
                                .collect(),
                            (Some(committed_output), None) => committed_output
                                .filter(|doc_id| token_scores.contains_key(doc_id))
                                .collect(),
                            (None, Some(uncommitted_output)) => uncommitted_output
                                .filter(|doc_id| token_scores.contains_key(doc_id))
                                .collect(),
                            (None, None) => HashSet::new(),
                        };

                        values.insert("true".to_string(), true_facet.len());
                    }
                    if facets.r#false {
                        let committed_output = committed.calculate_bool_filter(field_id, false)?;
                        let uncommitted_output =
                            uncommitted.calculate_bool_filter(field_id, false)?;
                        let false_facet = match (committed_output, uncommitted_output) {
                            (Some(committed_output), Some(uncommitted_output)) => committed_output
                                .chain(uncommitted_output)
                                .filter(|doc_id| token_scores.contains_key(doc_id))
                                .collect(),
                            (Some(committed_output), None) => committed_output
                                .filter(|doc_id| token_scores.contains_key(doc_id))
                                .collect(),
                            (None, Some(uncommitted_output)) => uncommitted_output
                                .filter(|doc_id| token_scores.contains_key(doc_id))
                                .collect(),
                            (None, None) => HashSet::new(),
                        };

                        values.insert("false".to_string(), false_facet.len());
                    }

                    res_facets.insert(
                        field_name,
                        FacetResult {
                            count: values.len(),
                            values,
                        },
                    );
                }
            }
        }

        Ok(Some(res_facets))
    }

    pub async fn stats(&self) -> Result<CollectionStats> {
        let mut fields_stats = Vec::new();

        let mut bools: HashMap<
            FieldId,
            (
                Option<BoolCommittedFieldStats>,
                Option<BoolUncommittedFieldStats>,
            ),
        > = HashMap::new();
        let bool_committed_fields_stats =
            self.committed_collection.read().await.get_bool_stats()?;
        bools.extend(
            bool_committed_fields_stats
                .into_iter()
                .map(|(k, v)| (k, (Some(v), None))),
        );
        let bool_uncommitted_fields_stats =
            self.uncommitted_collection.read().await.get_bool_stats();
        for (k, v) in bool_uncommitted_fields_stats {
            let e = bools.entry(k).or_default();
            e.1 = Some(v);
        }
        fields_stats.extend(bools.into_iter().map(|(k, v)| {
            let name = self
                .filter_fields
                .iter()
                .find(|e| e.value().0 == k)
                .map(|e| e.key().to_string())
                .unwrap_or_default();
            FieldStats {
                field_id: k,
                name,
                stats: FieldStatsType::Bool {
                    uncommitted: v.1,
                    committed: v.0,
                },
            }
        }));

        let mut numbers: HashMap<
            FieldId,
            (
                Option<NumberCommittedFieldStats>,
                Option<NumberUncommittedFieldStats>,
            ),
        > = HashMap::new();
        let number_committed_fields_stats =
            self.committed_collection.read().await.get_number_stats()?;
        numbers.extend(
            number_committed_fields_stats
                .into_iter()
                .map(|(k, v)| (k, (Some(v), None))),
        );
        let number_uncommitted_fields_stats =
            self.uncommitted_collection.read().await.get_number_stats();
        for (k, v) in number_uncommitted_fields_stats {
            let e = numbers.entry(k).or_default();
            e.1 = Some(v);
        }
        fields_stats.extend(numbers.into_iter().map(|(k, v)| {
            let name = self
                .filter_fields
                .iter()
                .find(|e| e.value().0 == k)
                .map(|e| e.key().to_string())
                .unwrap_or_default();
            FieldStats {
                field_id: k,
                name,
                stats: FieldStatsType::Number {
                    uncommitted: v.1,
                    committed: v.0,
                },
            }
        }));

        let mut string_filters: HashMap<
            FieldId,
            (
                Option<StringFilterCommittedFieldStats>,
                Option<StringFilterUncommittedFieldStats>,
            ),
        > = HashMap::new();
        let string_filter_committed_fields_stats = self
            .committed_collection
            .read()
            .await
            .get_string_filter_stats();
        string_filters.extend(
            string_filter_committed_fields_stats
                .into_iter()
                .map(|(k, v)| (k, (Some(v), None))),
        );
        let string_filter_uncommitted_fields_stats = self
            .uncommitted_collection
            .read()
            .await
            .get_string_filter_stats();
        for (k, v) in string_filter_uncommitted_fields_stats {
            let e = string_filters.entry(k).or_default();
            e.1 = Some(v);
        }
        fields_stats.extend(string_filters.into_iter().map(|(k, v)| {
            let name = self
                .filter_fields
                .iter()
                .find(|e| e.value().0 == k)
                .map(|e| e.key().to_string())
                .unwrap_or_default();
            FieldStats {
                field_id: k,
                name,
                stats: FieldStatsType::StringFilter {
                    uncommitted: v.1,
                    committed: v.0,
                },
            }
        }));

        let mut strings: HashMap<
            FieldId,
            (
                Option<StringCommittedFieldStats>,
                Option<StringUncommittedFieldStats>,
            ),
        > = HashMap::new();
        let string_committed_fields_stats =
            self.committed_collection.read().await.get_string_stats()?;
        strings.extend(
            string_committed_fields_stats
                .into_iter()
                .map(|(k, v)| (k, (Some(v), None))),
        );
        let string_uncommitted_fields_stats =
            self.uncommitted_collection.read().await.get_string_stats();
        for (k, v) in string_uncommitted_fields_stats {
            let e = strings.entry(k).or_default();
            e.1 = Some(v);
        }
        fields_stats.extend(strings.into_iter().map(|(k, v)| {
            let name = self
                .score_fields
                .iter()
                .find(|e| e.value().0 == k)
                .map(|e| e.key().to_string())
                .unwrap_or_default();
            FieldStats {
                field_id: k,
                name,
                stats: FieldStatsType::String {
                    uncommitted: v.1,
                    committed: v.0,
                },
            }
        }));

        let mut vectors: HashMap<
            FieldId,
            (
                Option<VectorCommittedFieldStats>,
                Option<VectorUncommittedFieldStats>,
            ),
        > = HashMap::new();
        let vector_committed_fields_stats =
            self.committed_collection.read().await.get_vector_stats()?;
        vectors.extend(
            vector_committed_fields_stats
                .into_iter()
                .map(|(k, v)| (k, (Some(v), None))),
        );
        let vector_uncommitted_fields_stats =
            self.uncommitted_collection.read().await.get_vector_stats();
        for (k, v) in vector_uncommitted_fields_stats {
            let e = vectors.entry(k).or_default();
            e.1 = Some(v);
        }
        fields_stats.extend(vectors.into_iter().filter_map(|(k, v)| {
            let name = self
                .score_fields
                .iter()
                .find(|e| e.value().0 == k)?
                .key()
                .to_string();
            Some(FieldStats {
                field_id: k,
                name,
                stats: FieldStatsType::Vector {
                    uncommitted: v.1,
                    committed: v.0,
                },
            })
        }));

        fields_stats.sort_by_key(|e| e.field_id.0);

        Ok(CollectionStats {
            id: self.get_id(),
            description: self.description.clone(),
            default_language: self.default_language,
            document_count: self.document_count.load(Ordering::Relaxed),
            fields_stats,
        })
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Committed {
    pub epoch: u64,
}

mod dump {
    use serde::{Deserialize, Serialize};

    use crate::{
        collection_manager::sides::OramaModelSerializable,
        nlp::locales::Locale,
        types::{CollectionId, FieldId, LanguageDTO},
    };

    use super::committed;

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(tag = "version")]
    pub enum CollectionInfo {
        #[serde(rename = "1")]
        V1(CollectionInfoV1),
        #[serde(rename = "2")]
        V2(CollectionInfoV2),
    }

    pub fn migrate_v1_to_v2(info: CollectionInfoV1) -> CollectionInfoV2 {
        let mut filter_fields: Vec<(String, (FieldId, TypedField))> = vec![];
        let mut score_fields: Vec<(String, (FieldId, TypedField))> = vec![];
        for (field_name, (field_id, field_type)) in info.fields {
            match field_type {
                TypedField::Text(locale) => {
                    score_fields.push((field_name, (field_id, TypedField::Text(locale))));
                }
                TypedField::Embedding(embedding) => {
                    score_fields.push((field_name, (field_id, TypedField::Embedding(embedding))));
                }
                TypedField::Number => {
                    filter_fields.push((field_name, (field_id, TypedField::Number)));
                }
                TypedField::Bool => {
                    filter_fields.push((field_name, (field_id, TypedField::Bool)));
                }
                TypedField::String => {
                    filter_fields.push((field_name, (field_id, TypedField::String)));
                }
            }
        }

        CollectionInfoV2 {
            id: info.id,
            description: info.description,
            document_count: info.document_count,
            deleted: info.deleted,
            default_language: info.default_language,
            filter_fields,
            score_fields,
            read_api_key: info.read_api_key,
            used_models: info.used_models,
            number_field_infos: info.number_field_infos,
            string_field_infos: info.string_field_infos,
            string_filter_field_infos: vec![],
            bool_field_infos: info.bool_field_infos,
            vector_field_infos: info.vector_field_infos,
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct CollectionInfoV1 {
        pub id: CollectionId,
        pub description: Option<String>,
        pub default_language: LanguageDTO,
        #[serde(default)]
        pub deleted: bool,
        pub document_count: u64,
        pub read_api_key: String,
        pub fields: Vec<(String, (FieldId, TypedField))>,
        pub used_models: Vec<(OramaModelSerializable, Vec<FieldId>)>,
        pub number_field_infos: Vec<(FieldId, committed::fields::NumberFieldInfo)>,
        pub string_field_infos: Vec<(FieldId, committed::fields::StringFieldInfo)>,
        pub bool_field_infos: Vec<(FieldId, committed::fields::BoolFieldInfo)>,
        pub vector_field_infos: Vec<(FieldId, committed::fields::VectorFieldInfo)>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct CollectionInfoV2 {
        pub id: CollectionId,
        pub description: Option<String>,
        pub default_language: LanguageDTO,
        #[serde(default)]
        pub deleted: bool,
        pub document_count: u64,
        pub read_api_key: String,
        pub score_fields: Vec<(String, (FieldId, TypedField))>,
        pub filter_fields: Vec<(String, (FieldId, TypedField))>,
        pub used_models: Vec<(OramaModelSerializable, Vec<FieldId>)>,
        pub number_field_infos: Vec<(FieldId, committed::fields::NumberFieldInfo)>,
        #[serde(default)]
        pub string_filter_field_infos: Vec<(FieldId, committed::fields::StringFilterFieldInfo)>,
        pub string_field_infos: Vec<(FieldId, committed::fields::StringFieldInfo)>,
        pub bool_field_infos: Vec<(FieldId, committed::fields::BoolFieldInfo)>,
        pub vector_field_infos: Vec<(FieldId, committed::fields::VectorFieldInfo)>,
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
    pub struct EmbeddingTypedField {
        pub model: OramaModelSerializable,
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
    pub enum TypedField {
        Text(Locale),
        Embedding(EmbeddingTypedField),
        Number,
        Bool,
        String,
    }
}

async fn get_bool_filtered_document(
    reader: &CollectionReader,
    field_id: FieldId,
    filter_bool: bool,
    uncommitted_deleted_documents: &HashSet<DocumentId>,
) -> Result<HashSet<DocumentId>> {
    let lock = reader.uncommitted_collection.read().await;
    let uncommitted_output = lock.calculate_bool_filter(field_id, filter_bool)?;

    let lock = reader.committed_collection.read().await;
    let committed_output = lock.calculate_bool_filter(field_id, filter_bool)?;

    let result = match (uncommitted_output, committed_output) {
        (Some(uncommitted_output), Some(committed_output)) => committed_output
            .chain(uncommitted_output)
            .filter(|doc_id| !uncommitted_deleted_documents.contains(doc_id))
            .collect(),
        (Some(uncommitted_output), None) => uncommitted_output
            .filter(|doc_id| !uncommitted_deleted_documents.contains(doc_id))
            .collect(),
        (None, Some(committed_output)) => committed_output
            .filter(|doc_id| !uncommitted_deleted_documents.contains(doc_id))
            .collect(),
        // This case probable means the field is not a number indexed
        (None, None) => HashSet::new(),
    };

    Ok(result)
}

async fn get_number_filtered_document(
    reader: &CollectionReader,
    field_id: FieldId,
    filter_number: NumberFilter,
    uncommitted_deleted_documents: &HashSet<DocumentId>,
) -> Result<HashSet<DocumentId>> {
    let lock = reader.uncommitted_collection.read().await;
    let uncommitted_output = lock
        .calculate_number_filter(field_id, &filter_number)
        .context("Cannot calculate uncommitted filter")?;

    let lock = reader.committed_collection.read().await;
    let committed_output = lock
        .calculate_number_filter(field_id, &filter_number)
        .context("Cannot calculate committed filter")?;

    let result = match (uncommitted_output, committed_output) {
        (Some(uncommitted_output), Some(committed_output)) => committed_output
            .chain(uncommitted_output)
            .filter(|doc_id| !uncommitted_deleted_documents.contains(doc_id))
            .collect(),
        (Some(uncommitted_output), None) => uncommitted_output
            .filter(|doc_id| !uncommitted_deleted_documents.contains(doc_id))
            .collect(),
        (None, Some(committed_output)) => committed_output
            .filter(|doc_id| !uncommitted_deleted_documents.contains(doc_id))
            .collect(),
        // This case probable means the field is not a number indexed
        (None, None) => HashSet::new(),
    };

    Ok(result)
}

async fn get_string_filtered_document(
    reader: &CollectionReader,
    field_id: FieldId,
    filter_string: String,
    uncommitted_deleted_documents: &HashSet<DocumentId>,
) -> Result<HashSet<DocumentId>> {
    let lock = reader.uncommitted_collection.read().await;
    let uncommitted_output = lock
        .calculate_string_filter(field_id, &filter_string)
        .context("Cannot calculate uncommitted filter")?;

    let lock = reader.committed_collection.read().await;
    let committed_output = lock
        .calculate_string_filter(field_id, &filter_string)
        .context("Cannot calculate committed filter")?;

    let result = match (uncommitted_output, committed_output) {
        (Some(uncommitted_output), Some(committed_output)) => committed_output
            .chain(uncommitted_output)
            .filter(|doc_id| !uncommitted_deleted_documents.contains(doc_id))
            .collect(),
        (Some(uncommitted_output), None) => uncommitted_output
            .filter(|doc_id| !uncommitted_deleted_documents.contains(doc_id))
            .collect(),
        (None, Some(committed_output)) => committed_output
            .filter(|doc_id| !uncommitted_deleted_documents.contains(doc_id))
            .collect(),
        // This case probable means the field is not a number indexed
        (None, None) => HashSet::new(),
    };

    Ok(result)
}

async fn get_filtered_document(
    reader: &CollectionReader,
    field_name: &String,
    field_id: FieldId,
    field_type: &TypedField,
    filter: Filter,
    uncommitted_deleted_documents: &HashSet<DocumentId>,
) -> Result<HashSet<DocumentId>> {
    match (&field_type, filter) {
        (TypedField::Number | TypedField::ArrayNumber, Filter::Number(filter_number)) => {
            get_number_filtered_document(
                reader,
                field_id,
                filter_number,
                uncommitted_deleted_documents,
            )
            .await
        }
        (TypedField::Bool | TypedField::ArrayBoolean, Filter::Bool(filter_bool)) => {
            get_bool_filtered_document(reader, field_id, filter_bool, uncommitted_deleted_documents)
                .await
        }
        (TypedField::String | TypedField::ArrayString, Filter::String(string_filter)) => {
            get_string_filtered_document(
                reader,
                field_id,
                string_filter,
                uncommitted_deleted_documents,
            )
            .await
        }
        _ => {
            anyhow::bail!(
                "Filter on field {:?}({:?}) not supported",
                field_name,
                field_type
            )
        }
    }
}

#[derive(Debug, Clone)]
pub enum TypedField {
    #[allow(dead_code)]
    Embedding(OramaModel),
    Text(Locale),
    ArrayText(Locale),
    Number,
    ArrayNumber,
    Bool,
    ArrayBoolean,
    String,
    ArrayString,
}

#[derive(Serialize, Debug)]
pub struct FieldStats {
    pub name: String,
    pub field_id: FieldId,
    #[serde(flatten)]
    pub stats: FieldStatsType,
}

#[derive(Serialize, Debug)]
pub struct CollectionStats {
    pub id: CollectionId,
    pub description: Option<String>,
    pub default_language: LanguageDTO,
    pub document_count: u64,
    pub fields_stats: Vec<FieldStats>,
}

#[derive(Serialize, Debug)]
#[serde(tag = "type")]
pub enum FieldStatsType {
    #[serde(rename = "bool")]
    Bool {
        uncommitted: Option<BoolUncommittedFieldStats>,
        committed: Option<BoolCommittedFieldStats>,
    },
    #[serde(rename = "number")]
    Number {
        uncommitted: Option<NumberUncommittedFieldStats>,
        committed: Option<NumberCommittedFieldStats>,
    },
    #[serde(rename = "string_filter")]
    StringFilter {
        uncommitted: Option<StringFilterUncommittedFieldStats>,
        committed: Option<StringFilterCommittedFieldStats>,
    },
    #[serde(rename = "string")]
    String {
        uncommitted: Option<StringUncommittedFieldStats>,
        committed: Option<StringCommittedFieldStats>,
    },
    #[serde(rename = "vector")]
    Vector {
        uncommitted: Option<VectorUncommittedFieldStats>,
        committed: Option<VectorCommittedFieldStats>,
    },
}
