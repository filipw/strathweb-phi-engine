@echo off
set OUTDIR=artifacts\csharp
rmdir /S /Q "%OUTDIR%"
cargo build --release --manifest-path strathweb-phi-engine/Cargo.toml
dotnet build packages\csharp\Strathweb.Phi.Engine.slnx -c Release
dotnet pack packages\csharp\Strathweb.Phi.Engine.slnx -c Release -o %OUTDIR%