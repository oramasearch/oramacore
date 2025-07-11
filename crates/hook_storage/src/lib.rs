use serde::{Deserialize, Serialize};

#[derive(PartialEq, Eq, Debug, Serialize, Deserialize, Clone, Copy, Hash)]
pub enum HookType {
    BeforeRetrieval,
}

impl HookType {
    fn get_file_name(&self) -> &'static str {
        match self {
            Self::BeforeRetrieval => "before_retrieval",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HookOperation {
    Insert(HookType, String),
    Delete(HookType),
}

mod reader;
mod writer;

pub use reader::{HookReader, HookReaderError};

pub use writer::{HookWriter, HookWriterError};

#[cfg(test)]
mod communication_tests {
    use super::*;
    use fs::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_hook_writer_reader_communication_via_channel() {
        let base_dir = generate_new_path();
        let (tx, mut rx) = mpsc::unbounded_channel::<HookOperation>();

        // Writer sends operations to the channel
        let writer = HookWriter::try_new(
            base_dir.clone(),
            Box::new(move |op| {
                let tx = tx.clone();
                Box::pin(async move {
                    let _ = tx.send(op);
                })
            }),
        )
        .expect("Failed to create HookWriter");

        // Reader receives operations from the channel
        let mut reader =
            HookReader::try_new(base_dir.clone()).expect("Failed to create HookReader");

        // Insert a hook via writer
        let code = "console.log('from_channel');".to_string();
        writer
            .insert_hook(HookType::BeforeRetrieval, code.clone())
            .await
            .expect("insert_hook failed");

        // Receive and apply operation on reader
        let op = rx.recv().await.expect("No operation received");
        reader.update(op).expect("Reader update failed");
        reader.commit().expect("Reader commit failed");
        assert_eq!(
            reader.get_hook_content(HookType::BeforeRetrieval).unwrap(),
            Some(code.clone())
        );

        // Delete the hook via writer
        writer
            .delete_hook(HookType::BeforeRetrieval)
            .await
            .expect("delete_hook failed");
        let op = rx.recv().await.expect("No operation received");
        reader.update(op).expect("Reader update failed");
        reader.commit().expect("Reader commit failed");
        assert_eq!(
            reader.get_hook_content(HookType::BeforeRetrieval).unwrap(),
            None
        );
    }
}
