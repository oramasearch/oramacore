static ORAMA_CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

static ORAMA_CORE_COMMIT: Option<&'static str> = option_env!("GIT_COMMIT");
static GITHUB_SHA: Option<&'static str> = option_env!("GITHUB_SHA");

pub fn get_mode() -> &'static str {
    if cfg!(feature = "writer") && cfg!(feature = "reader") {
        "standalone"
    } else if cfg!(feature = "writer") {
        "writer"
    } else if cfg!(feature = "reader") {
        "reader"
    } else {
        "unknown"
    }
}

pub fn get_build_info() -> String {
    let version = ORAMA_CORE_VERSION;
    let commit = GITHUB_SHA.or(ORAMA_CORE_COMMIT).unwrap_or("unknown");
    let mode = get_mode();
    format!("OramaCore version: {version} (commit: {commit}) - {mode}")
}
pub fn get_build_version() -> String {
    let version = ORAMA_CORE_VERSION;
    let commit = GITHUB_SHA.or(ORAMA_CORE_COMMIT).unwrap_or("unknown");

    let mode = get_mode();
    format!("{version}@{commit} - {mode}")
}
