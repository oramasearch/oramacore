use serde_json::json;

use crate::{
    tests::utils::{extrapolate_ids_from_result, init_log, TestContext},
    types::LanguageDTO,
};

/// Regular and temporary indexes use the same supported locale on both sides.
#[tokio::test(flavor = "multi_thread")]
async fn test_cjk_indexes_and_temporary_indexes_are_searchable() {
    init_log();
    let context = TestContext::new().await;
    for (language, text, query) in [
        (LanguageDTO::Japanese, "東京、検索。", "検索"),
        (LanguageDTO::Korean, "서울, 검색!", "검색"),
        (LanguageDTO::Chinese, "北京，搜索。", "搜索"),
    ] {
        let collection = context
            .create_collection_with_language(Some(language))
            .await
            .unwrap();
        let index = collection.create_index().await.unwrap();
        index
            .insert_documents(json!([{"id":"regular", "text":text}]).try_into().unwrap())
            .await
            .unwrap();
        let result = collection
            .search(json!({"term":query}).try_into().unwrap())
            .await
            .unwrap();
        assert_eq!(result.hits.len(), 1);
        assert_eq!(extrapolate_ids_from_result(&result), vec!["regular"]);

        let temporary = collection.create_temp_index(index.index_id).await.unwrap();
        temporary
            .insert_documents(
                json!([{"id":"replacement", "text":text}])
                    .try_into()
                    .unwrap(),
            )
            .await
            .unwrap();
        collection
            .replace_index(index.index_id, temporary.index_id)
            .await
            .unwrap();
        context.commit_all().await.unwrap();
        let result = collection
            .search(json!({"term":query}).try_into().unwrap())
            .await
            .unwrap();
        assert_eq!(result.hits.len(), 1);
        assert_eq!(extrapolate_ids_from_result(&result), vec!["replacement"]);
    }
}

#[test]
fn test_malformed_locales_are_rejected_before_index_publication() {
    for value in [
        json!("unknown"),
        json!("JP"),
        json!(""),
        json!(123),
        json!(null),
    ] {
        assert!(serde_json::from_value::<LanguageDTO>(value).is_err());
    }
}
