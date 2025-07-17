pub mod generic_kv;
mod operation;
pub mod read;
pub mod system_prompts;
pub mod write;

pub use operation::*;

pub fn field_name_to_path(field_name: &str) -> Box<[String]> {
    field_name
        .split('.')
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .into_boxed_slice()
}
pub fn field_names_to_paths(field_names: Vec<String>) -> Box<[Box<[String]>]> {
    let mut output = Vec::with_capacity(field_names.len());
    for name in field_names {
        output.push(field_name_to_path(&name));
    }
    output.into_boxed_slice()
}
