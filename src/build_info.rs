static ORAMA_CORE_VERSION: &str = env!("CARGO_PKG_VERSION");
static ORAMA_CORE_COMMIT: Option<&str> = option_env!("GIT_COMMIT");

pub fn get_build_info() -> String {
    let version = ORAMA_CORE_VERSION;
    let commit = ORAMA_CORE_COMMIT.unwrap_or("unknown");
    format!("OramaCore version: {} (commit: {})", version, commit)
}
pub fn get_build_version() -> String {
    let version = ORAMA_CORE_VERSION;
    let commit = ORAMA_CORE_COMMIT.unwrap_or("unknown");
    format!("{}@{}", version, commit)
}
