#!/bin/bash

rm -f strathweb_phi_engine.dll
rm -f strathweb_phi_engine.py

cargo build --release --manifest-path ../../../strathweb-phi-engine/Cargo.toml
if [[ "$OSTYPE" == "darwin"* ]]; then
    cp ../../../strathweb-phi-engine/target/release/libstrathweb_phi_engine.dylib .
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    cp ../../../strathweb-phi-engine/target/release/libstrathweb_phi_engine.so .
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi
cp ../../../strathweb-phi-engine/bindings/strathweb_phi_engine.py .