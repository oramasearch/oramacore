fn main() {
    tonic_build::compile_protos("./src/ai_server/service.proto").unwrap();

    println!("cargo::rerun-if-changed=build.rs");
}
