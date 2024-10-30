use analytics_storage::{
    AnalyticsStorage, AnalyticsStorageConfig, Granularity, OffloadTarget, VersionV1_0Schema,
};
use rand::distributions::Alphanumeric;
use rand::seq::SliceRandom;
use rand::Rng;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

fn main() {
    let mut storage: AnalyticsStorage<VersionV1_0Schema> =
        AnalyticsStorage::try_new(AnalyticsStorageConfig {
            index_id: "my_index_id".to_string(),
            granularity: Some(Granularity::Hour),
            persistence_dir: None,
            offload_after: Some(100),
            offload_to: Some(OffloadTarget::Void),
            buffer_size: Some(100),
        })
        .unwrap();

    let pops = vec!["mxp", "fra"];
    let continents = vec!["asia", "europe", "north america", "south america"];
    let countries = vec!["italy", "usa", "brazil", "china"];
    let mut rng = rand::thread_rng();

    for _ in 0..10_000 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as i64;

        let pop = pops.choose(&mut rand::thread_rng()).unwrap();
        let continent = continents.choose(&mut rand::thread_rng()).unwrap();
        let country = countries.choose(&mut rand::thread_rng()).unwrap();

        let raw_query: String = rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(7)
            .map(char::from)
            .collect();

        storage
            .insert(VersionV1_0Schema {
                timestamp: now,
                pop: pop.to_string(),
                continent: continent.to_string(),
                country: country.to_string(),
                deployment_id: Uuid::new_v4().to_string(),
                id: Uuid::new_v4().to_string(),
                instance_id: Uuid::new_v4().to_string(),
                raw_query: raw_query.clone(),
                raw_search_string: raw_query,
                referer: "".to_string(),
                results_count: rng.gen_range(0..15),
                visitor_id: Uuid::new_v4().to_string(),
            })
            .unwrap();
    }
}
