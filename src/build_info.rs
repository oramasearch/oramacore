static ORAMA_CORE_VERSION: &str = env!("CARGO_PKG_VERSION");
static GITHUB_REF_NAME: Option<&'static str> = option_env!("GITHUB_REF_NAME");

static GITHUB_SHA: Option<&'static str> = option_env!("GITHUB_SHA");

pub fn get_build_info() -> String {
    let version = GITHUB_REF_NAME.unwrap_or(ORAMA_CORE_VERSION);
    let commit = GITHUB_SHA.unwrap_or("unknown");
    format!("OramaCore version: {version} (commit: {commit})")
}
pub fn get_build_version() -> String {
    let version = GITHUB_REF_NAME.unwrap_or(ORAMA_CORE_VERSION);
    let commit = GITHUB_SHA.unwrap_or("unknown");
    format!("{version}@{commit}")
}
