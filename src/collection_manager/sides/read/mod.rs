mod collection;
mod collections;

pub use collection::CollectionReader;
pub use collections::{CollectionsReader, IndexesConfig};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_reader_sync_send() {
        fn assert_sync_send<T: Sync + Send>() {}
        assert_sync_send::<CollectionsReader>();
        assert_sync_send::<CollectionReader>();
    }
}
