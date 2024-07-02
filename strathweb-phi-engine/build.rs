use uniffi_bindgen::{bindings::TargetLanguage, generate_bindings};
use std::process::Command;

fn main() {
    let udl_file = "./src/strathweb-phi-engine.udl";
    let out_dir = "./bindings";

    uniffi::generate_scaffolding(udl_file).unwrap();
    generate_bindings(udl_file.into(), 
    None, 
    vec![TargetLanguage::Swift], 
    Some(out_dir.into()), 
    None,
    None,
    false).unwrap(); 

    Command::new("uniffi-bindgen-cs").arg("--out-dir").arg(out_dir).arg(udl_file).output().expect("Failed when generating C# bindings");
}