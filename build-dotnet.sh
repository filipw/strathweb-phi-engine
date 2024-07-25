#!/bin/bash

OUTDIR="artifacts/csharp"
rm -rf "${OUTDIR}"
cargo build --release --manifest-path strathweb-phi-engine/Cargo.toml
dotnet build --project packages/csharp/Strathweb.Phi.Engine -c Release
dotnet pack -c Release --project packages/csharp/Strathweb.Phi.Engine -o ${OUTDIR}