use std::process::Command;
use uniffi_bindgen::{bindings::{KotlinBindingGenerator, PythonBindingGenerator, SwiftBindingGenerator}, generate_bindings};

fn main() {
    let udl_file = "./src/strathweb-phi-engine.udl";
    let out_dir = "./bindings";

    uniffi::generate_scaffolding(udl_file).unwrap();

    generate_bindings(udl_file.into(), None, PythonBindingGenerator, Some(out_dir.into()), None, None, false).unwrap_or_else(|e| eprintln!("Failed to generate Python bindings: {}", e));
    generate_bindings(udl_file.into(), None, KotlinBindingGenerator, Some(out_dir.into()), None, None, false).unwrap_or_else(|e| eprintln!("Failed to generate Kotlin bindings: {}", e));
    generate_bindings(udl_file.into(), None, SwiftBindingGenerator, Some(out_dir.into()), None, None, false).unwrap_or_else(|e| eprintln!("Failed to generate Swift bindings: {}", e));


    let status = Command::new("uniffi-bindgen-cs")
        .arg("--out-dir")
        .arg(out_dir)
        .arg(udl_file)
        .status();

    if status.is_err() {
        eprintln!("Warning: Failed when generating C# bindings, make sure you have uniffi-bindgen-cs installed.");
    }
}
