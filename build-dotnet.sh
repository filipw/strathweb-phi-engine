#!/bin/bash

OUTDIR="artifacts/csharp"
rm -rf "${OUTDIR}"
cargo build --release --manifest-path strathweb-phi-engine/Cargo.toml
dotnet build packages/csharp/Strathweb.Phi.Engine -c Release
dotnet pack packages/csharp/Strathweb.Phi.Engine -c Release -o ${OUTDIR}