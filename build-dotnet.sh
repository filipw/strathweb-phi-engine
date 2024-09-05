#!/bin/bash

OUTDIR="artifacts/csharp"
rm -rf "${OUTDIR}"
cargo build --release --manifest-path strathweb-phi-engine/Cargo.toml
dotnet build packages/csharp -c Release
dotnet pack packages/csharp -c Release -o ${OUTDIR}