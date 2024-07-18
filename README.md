# Strathweb Phi Engine

A cross-platform library for running Microsoft's [Phi-3](https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/) locally using [candle](https://github.com/huggingface/candle).

Supported platforms:
 - Swift (Swift bindings + XCframework + native library .a/.dylib)
   - macOS (arm64)
   - iOS
 - .NET (C# bindings + native library as .dll, .so, .dylib)
   - Windows x64 & arm64
   - Linux x64 & arm64
   - macOS arm64
- Kotlin (Kotlin bindings + native library as .so/.dylib)
   - macOS arm64

## Building instructions

### Swift

Run `./build-xcframework.sh` to build the xcframework.

Then open `samples/io/phi.engine.sample/phi.engine.sample.xcodeproj` and build the SwiftUI app (iOS) or go to `samples/swift` and run `./run.sh` (macOS).

### C#

Install UniFFI C# bindings generator

```shell
cargo install uniffi-bindgen-cs --git https://github.com/NordSecurity/uniffi-bindgen-cs --tag v0.8.0+v0.25.0
```

Run the sample console app:

```shell
cd strathweb-phi-engine
cargo build --release
cd ../samples/csharp/console
dotnet run -c Release
```

### Kotlin

Run the sample console app:

```shell
cd samples/kotlin
./run.sh
```

## Compatibility notes

### .NET

✅ Tested on Windows arm64

✅ Tested on Linux arm64

✅ Tested on macOS arm64

### Swift

✅ Tested on macOS arm64. Supports Metal.

✅ Tested on iPad Air M1 8GB RAM

✅ Should work on 6GB RAM iPhones too

❌ Will not work on 4GB RAM iPhones

However, for 4GB RAM iPhones, it's possible to use the (very) low fidelity Q2_K quantized model. Such model is not included in the official Phi-3 release, but I tested [this one from HuggingFace](https://huggingface.co/SanctumAI/Phi-3-mini-4k-instruct-GGUF) on an iPhone 12 mini successfully.

### Kotlin

✅ Tested on macOS arm64. Supports Metal.

## Blog post

For a detailed explanation of how this works, check out the blog post [here](https://www.strathweb.com/2024/05/running-microsoft-phi-3-model-in-an-ios-app-with-rust/).