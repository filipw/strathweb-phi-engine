# Strathweb Phi Engine

A cross-platform library for running Microsoft's [Phi-3](https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/) locally using [candle](https://github.com/huggingface/candle).

Supported platforms:
 - Swift (Swift bindings + XCframework + native library .a/.dylib)
   - macOS (arm64)
   - iOS
 - .NET (NuGet package, C# bindings + native library as .dll, .so, .dylib)
   - Windows x64 & arm64
   - Linux x64 & arm64
   - macOS arm64
- Kotlin (Kotlin bindings + native library as .so/.dylib)
   - macOS arm64
- Python 
 (Kotlin bindings + native library as .so/.dylib)
   - Windows x64 & arm64
   - Linux x64 & arm64
   - macOS arm64

## Building instructions

### Swift

Run `./build-xcframework.sh` to build the xcframework (arm64 Mac required).

Then open `samples/io/phi.engine.sample/phi.engine.sample.xcodeproj` and build the SwiftUI app (iOS) or go to `samples/swift` and run `./run.sh` (macOS).

### C#

Install UniFFI C# bindings generator

```shell
cargo install uniffi-bindgen-cs --git https://github.com/NordSecurity/uniffi-bindgen-cs --tag v0.8.0+v0.25.0
```

Build the Nuget package for your platform:

```shell
./build-dotnet.sh
```

or

```shell
cargo build --release --manifest-path strathweb-phi-engine/Cargo.toml
dotnet build packages/csharp/Strathweb.Phi.Engine -c Release
dotnet pack packages/csharp/Strathweb.Phi.Engine -c Release -o artifacts/csharp
```

Nuget package will be in `artifacts/csharp/Strathweb.Phi.Engine.0.1.0.nupkg`.
(Optional) Run the sample console app:

```shell
cd samples/csharp/console
dotnet run -c Release
```

### Kotlin

Run the sample console app:

```shell
cd samples/kotlin
./run.sh
```

### Python

Run the sample console app:

```shell
cd samples/python/console
./run.sh # or run.bat on Windows
```

or use the Jupyter Notebooks

```shell
cd samples/python/jupyter
./init.sh # or init.bat on Windows
```

Now open the Notebook and run the cells.

## Compatibility notes

### .NET

✅ Tested on Windows arm64

✅ Tested on Windows x64

✅ Tested on Linux arm64

✅ Tested on Linux x64

✅ Tested on macOS arm64. Supports Metal.

### Swift

✅ Tested on macOS arm64. Supports Metal.

✅ Tested on iPad Air M1 8GB RAM

✅ Should work on 6GB RAM iPhones too

❌ Will not work on 4GB RAM iPhones

However, for 4GB RAM iPhones, it's possible to use the (very) low fidelity Q2_K quantized model. Such model is not included in the official Phi-3 release, but I tested [this one from HuggingFace](https://huggingface.co/SanctumAI/Phi-3-mini-4k-instruct-GGUF) on an iPhone 12 mini successfully.

### Kotlin

✅ Tested on macOS arm64. Supports Metal.

### Python

✅ Tested on Windows arm64

✅ Tested on macOS arm64. Supports Metal.

## Blog post

For an announcement post, go [here](https://strathweb.com/2024/07/announcing-strathweb-phi-engine-a-cross-platform-library-for-running-phi-3-anywhere/).