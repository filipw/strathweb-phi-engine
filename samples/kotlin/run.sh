#!/bin/bash

rm -rf deps
rm -rf out
mkdir -p deps

cargo build --release --manifest-path ../../strathweb-phi-engine/Cargo.toml
cp ../../strathweb-phi-engine/target/release/libstrathweb_phi_engine.dylib deps/
cp ../../strathweb-phi-engine/bindings/strathweb/phi/engine/strathweb_phi_engine.kt deps/

kotlinc main.kt deps/strathweb_phi_engine.kt -include-runtime -cp lib/jna.jar -d out/main.jar
javac -cp out/main.jar:lib/jna.jar Main.java -d out
jar uf out/main.jar -C out Main.class

java -Djna.library.path=$(pwd)/deps -cp out/main.jar:lib/jna.jar MainKt