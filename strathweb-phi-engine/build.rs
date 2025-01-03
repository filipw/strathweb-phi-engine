use std::process::Command;
use uniffi_bindgen::{bindings::TargetLanguage, generate_bindings};

fn main() {
    let udl_file = "./src/strathweb-phi-engine.udl";
    let out_dir = "./bindings";

    uniffi::generate_scaffolding(udl_file).unwrap();
    generate_bindings(
        udl_file.into(),
        None,
        vec![
            TargetLanguage::Swift,
            TargetLanguage::Kotlin,
            TargetLanguage::Python,
        ],
        Some(out_dir.into()),
        None,
        None,
        false,
    )
    .unwrap(); 

    let status = Command::new("uniffi-bindgen-cs")
        .arg("--out-dir")
        .arg(out_dir)
        .arg(udl_file)
        .status();

    if status.is_err() {
        eprintln!("Warning: Failed when generating C# bindings, make sure you have uniffi-bindgen-cs installed.");
    }
}
