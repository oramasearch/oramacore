use std::path::{Path, PathBuf};
use tracing::error;

pub fn get_rule_file_name(id: &str) -> String {
    format!("{id}.rule")
}

pub fn is_rule_file(p: &Path) -> bool {
    p.extension().and_then(|os| os.to_str()) == Some("rule")
}

pub fn remove_rule_file(p: PathBuf) {
    match std::fs::remove_file(p) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            error!(error = ?e, "Cannot remove pin rule file");
        }
    }
}
