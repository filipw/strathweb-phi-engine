use uniffi_bindgen::{bindings::swift::gen_swift::SwiftBindingGenerator, generate_bindings};
use std::process::Command;

fn main() {
    let udl_file = "./src/strathweb-phi-engine.udl";
    let out_dir = "./bindings";

    generate_bindings(udl_file.into(), None, SwiftBindingGenerator {}, Some(out_dir.into()), None, None, true).unwrap();
    uniffi::generate_scaffolding(udl_file).unwrap();

    Command::new("uniffi-bindgen-cs").arg("--out-dir").arg(out_dir).arg(udl_file).output().expect("Failed when generating C# bindings");
}