#!/bin/bash

rm -f phi-engine-swift-sample

cd ../../
./build-xcframework.sh
cd samples/swift

swiftc *.swift \
    ../../strathweb-phi-engine/bindings/strathweb_phi_engine.swift \
    -I ../../artifacts/swift/strathweb_phi_engine_framework.xcframework/macos-arm64/Headers \
    -L ../../artifacts/swift/strathweb_phi_engine_framework.xcframework/macos-arm64 \
    -lstrathweb_phi_engine \
    -framework Metal \
    -framework MetalPerformanceShaders \
    -framework SystemConfiguration \
    -lc++ \
    -O \
    -o phi-engine-swift-sample

./phi-engine-swift-sample