fn main() {
    println!("cargo:rustc-link-arg=Metal");
    println!("cargo:rustc-link-arg=Accelerate");
}