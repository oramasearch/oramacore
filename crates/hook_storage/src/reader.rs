use std::path::PathBuf;

use anyhow::Context;
use fs::*;
use thiserror::Error;

use crate::{HookOperation, HookType};

#[derive(Error, Debug)]
pub enum HookReaderError {
    #[error("")]
    Io(std::io::Error),
    #[error("")]
    Generic(#[from] anyhow::Error),
}

pub struct HookReader {
    data_dir: PathBuf,
    pending_operations: Vec<HookOperation>,
}

struct Status {
    before_retrieval: (Option<String>, bool),
}

impl Status {
    fn apply_operations(&mut self, ops: &[HookOperation]) {
        for op in ops {
            match op {
                HookOperation::Insert(HookType::BeforeRetrieval, code) => {
                    self.before_retrieval = (Some(code.clone()), true);
                }
                HookOperation::Delete(HookType::BeforeRetrieval) => {
                    self.before_retrieval = (None, true);
                }
            }
        }
    }
}

impl HookReader {
    pub fn try_new(data_dir: PathBuf) -> Result<Self, HookReaderError> {
        create_if_not_exists(&data_dir)?;

        Ok(Self {
            data_dir,
            pending_operations: vec![],
        })
    }

    pub fn update(&mut self, op: HookOperation) -> Result<(), HookReaderError> {
        self.pending_operations.push(op);
        Ok(())
    }

    pub fn commit(&mut self) -> Result<(), HookReaderError> {
        let mut status = Status {
            before_retrieval: (None, false),
        };

        status.apply_operations(&self.pending_operations);
        self.pending_operations = vec![]; // deallocate

        if status.before_retrieval.1 {
            let file_path = self
                .data_dir
                .join(HookType::BeforeRetrieval.get_file_name());
            if let Some(code) = status.before_retrieval.0 {
                // Save the code to the file system
                BufferedFile::create_or_overwrite(file_path)
                    .context("Cannot open file")?
                    .write_text_data(&code)
                    .context("Cannot write code to file")?;
            } else {
                // Remove the file from the file system
                match std::fs::remove_file(&file_path) {
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(HookReaderError::Io(e)),
                }
            }
        }

        Ok(())
    }

    pub fn get_hook_content(&self, hook_type: HookType) -> Result<Option<String>, HookReaderError> {
        let content = self.list()?;
        let a = content
            .into_iter()
            .find(|(t, _)| t == &hook_type)
            .and_then(|(_, c)| c);

        Ok(a)
    }

    /// List all hooks. Returns the content as well if present.
    pub fn list(&self) -> Result<Vec<(HookType, Option<String>)>, HookReaderError> {
        let mut status = Status {
            before_retrieval: (None, false),
        };
        status.apply_operations(&self.pending_operations);

        let mut result = Vec::with_capacity(1);

        if !status.before_retrieval.1 {
            // if not touched
            let file_path = self
                .data_dir
                .join(HookType::BeforeRetrieval.get_file_name());
            let content = if BufferedFile::exists_as_file(&file_path) {
                let content = BufferedFile::open(file_path)
                    .context("Cannot open file")?
                    .read_text_data()
                    .context("Cannot write code to file")?;
                Some(content)
            } else {
                None
            };

            result.push((HookType::BeforeRetrieval, content));
        } else {
            result.push((HookType::BeforeRetrieval, status.before_retrieval.0));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hook_reader_lifecycle() {
        let data_dir = generate_new_path();
        let mut reader = HookReader::try_new(data_dir.clone()).unwrap();
        let code1 = "console.log('hello');".to_string();
        let code2 = "console.log('world');".to_string();

        // Initially, nothing is present
        assert_eq!(
            reader.get_hook_content(HookType::BeforeRetrieval).unwrap(),
            None
        );

        let list = reader.list().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].1, None);

        // Insert (pending)
        reader
            .update(HookOperation::Insert(
                HookType::BeforeRetrieval,
                code1.clone(),
            ))
            .unwrap();
        // Should see the pending value
        assert_eq!(
            reader.get_hook_content(HookType::BeforeRetrieval).unwrap(),
            Some(code1.clone())
        );
        let list = reader.list().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0], (HookType::BeforeRetrieval, Some(code1.clone())));

        // Commit
        reader.commit().unwrap();
        // Should see the committed value
        assert_eq!(
            reader.get_hook_content(HookType::BeforeRetrieval).unwrap(),
            Some(code1.clone())
        );
        let list = reader.list().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0], (HookType::BeforeRetrieval, Some(code1.clone())));

        // Update with new code (pending)
        reader
            .update(HookOperation::Insert(
                HookType::BeforeRetrieval,
                code2.clone(),
            ))
            .unwrap();
        // Should see the new pending value
        assert_eq!(
            reader.get_hook_content(HookType::BeforeRetrieval).unwrap(),
            Some(code2.clone())
        );
        let list = reader.list().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0], (HookType::BeforeRetrieval, Some(code2.clone())));

        // Commit
        reader.commit().unwrap();
        // Delete (pending)
        reader
            .update(HookOperation::Delete(HookType::BeforeRetrieval))
            .unwrap();
        // Should see None (pending delete)
        assert_eq!(
            reader.get_hook_content(HookType::BeforeRetrieval).unwrap(),
            None
        );
        let list = reader.list().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].1, None);

        // Commit
        reader.commit().unwrap();

        assert_eq!(
            reader.get_hook_content(HookType::BeforeRetrieval).unwrap(),
            None
        );
        let list = reader.list().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].1, None);

        // Re-insert after delete
        reader
            .update(HookOperation::Insert(
                HookType::BeforeRetrieval,
                code1.clone(),
            ))
            .unwrap();
        assert_eq!(
            reader.get_hook_content(HookType::BeforeRetrieval).unwrap(),
            Some(code1.clone())
        );
        let list = reader.list().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0], (HookType::BeforeRetrieval, Some(code1.clone())));

        reader.commit().unwrap();

        assert_eq!(
            reader.get_hook_content(HookType::BeforeRetrieval).unwrap(),
            Some(code1.clone())
        );
        let list = reader.list().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0], (HookType::BeforeRetrieval, Some(code1.clone())));
    }
}
