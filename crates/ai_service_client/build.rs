fn main() {
    let cargo_manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let proto_file_path = std::path::Path::new(&cargo_manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("./src/ai_server/service.proto");

    println!("cargo::rerun-if-changed={}", proto_file_path.display());
    tonic_build::compile_protos(proto_file_path).unwrap();
    println!("cargo::rerun-if-changed=build.rs");
}
