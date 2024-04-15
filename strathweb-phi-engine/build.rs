use uniffi_bindgen::{bindings::swift::gen_swift::SwiftBindingGenerator, generate_bindings};

fn main() {
    let udl_file = "./src/strathweb-phi-engine.udl";
    let out_dir = "./bindings";

    generate_bindings(udl_file.into(), None, SwiftBindingGenerator {}, Some(out_dir.into()), None, None, true).unwrap();
    uniffi::generate_scaffolding(udl_file).unwrap();
}