#!/bin/bash

NAME="strathweb_phi_engine"
HEADERPATH="strathweb-phi-engine/bindings/strathweb_phi_engineFFI.h"
TARGETDIR="strathweb-phi-engine/target"
OUTDIR="artifacts/swift"
RELDIR="release"
STATIC_LIB_NAME="lib${NAME}.a"
NEW_HEADER_DIR="strathweb-phi-engine/bindings/include"

IPHONEOS_DEPLOYMENT_TARGET=18.0 cargo build --manifest-path strathweb-phi-engine/Cargo.toml --target aarch64-apple-ios --release
cargo build --manifest-path strathweb-phi-engine/Cargo.toml --target aarch64-apple-ios-sim --release
MACOSX_DEPLOYMENT_TARGET=14.0 cargo build --manifest-path strathweb-phi-engine/Cargo.toml --target aarch64-apple-darwin --release

mkdir -p "${NEW_HEADER_DIR}"
cp "${HEADERPATH}" "${NEW_HEADER_DIR}/"
cp "strathweb-phi-engine/bindings/strathweb_phi_engineFFI.modulemap" "${NEW_HEADER_DIR}/module.modulemap"

rm -rf "${OUTDIR}/${NAME}_framework.xcframework"

xcodebuild -create-xcframework \
    -library "${TARGETDIR}/aarch64-apple-ios/${RELDIR}/${STATIC_LIB_NAME}" \
    -headers "${NEW_HEADER_DIR}" \
    -library "${TARGETDIR}/aarch64-apple-ios-sim/${RELDIR}/${STATIC_LIB_NAME}" \
    -headers "${NEW_HEADER_DIR}" \
    -library "${TARGETDIR}/aarch64-apple-darwin/${RELDIR}/${STATIC_LIB_NAME}" \
    -headers "${NEW_HEADER_DIR}" \
    -output "${OUTDIR}/${NAME}_framework.xcframework"

rm -rf "packages/swift/Strathweb.Phi.Engine/Libs"
mkdir -p "packages/swift/Strathweb.Phi.Engine/Libs"
cp -R "${OUTDIR}/${NAME}_framework.xcframework" "packages/swift/Strathweb.Phi.Engine/Libs"
cp "strathweb-phi-engine/bindings/strathweb_phi_engine.swift" "packages/swift/Strathweb.Phi.Engine/Sources/Strathweb.Phi.Engine"

swift build -c release --package-path packages/swift/Strathweb.Phi.Engine