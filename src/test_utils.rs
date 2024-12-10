use std::{fs, path::PathBuf};

use tempdir::TempDir;

pub fn generate_new_path() -> PathBuf {
    let tmp_dir = TempDir::new("test").unwrap();
    let dir = tmp_dir.path().to_path_buf();
    fs::create_dir_all(dir.clone()).unwrap();
    dir
}
