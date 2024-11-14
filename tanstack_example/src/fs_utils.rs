use std::{
    fs::{self, DirEntry},
    io,
    path::{Path, PathBuf},
};

pub fn get_files(dir: PathBuf, allowed_extensions: Vec<String>) -> Vec<DirEntry> {
    let mut files = vec![];
    fn visit_dirs(
        dir: &Path,
        files: &mut Vec<DirEntry>,
        allowed_extensions: &Vec<String>,
    ) -> io::Result<()> {
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    visit_dirs(&path, files, allowed_extensions)?;
                } else if let Some(extension) = path.extension() {
                    let extension = extension.to_string_lossy().to_string();
                    if allowed_extensions.contains(&extension) {
                        files.push(entry);
                    }
                }
            }
        }
        Ok(())
    }
    visit_dirs(&dir, &mut files, &allowed_extensions).unwrap();

    files
}
