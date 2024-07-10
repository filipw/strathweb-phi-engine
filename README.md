# Strathweb Phi Engine

An cross-platform library for running Microsoft's [Phi-3](https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/) locally using [candle](https://github.com/huggingface/candle).

Supported platforms:
 - iOS
 - .NET (Windows x64 & arm64, Linux x64 & arm64, macOS arm64)

## Building instructions

### iOS

Run `./build-ios.sh` to build the xcframework.
Then open `samples/io/phi.engine.sample/phi.engine.sample.xcodeproj` and build the SwiftUI app.

### C#

Install UniFFI C# bindings generator

```shell
cargo install uniffi-bindgen-cs --git https://github.com/NordSecurity/uniffi-bindgen-cs --tag v0.8.0+v0.25.0
```

Run

```shell
cd strathweb-phi-engine
cargo build --release
cd ../samples/csharp/console
dotnet run -c Release
```

## Blog post

For a detailed explanation of how this works, check out the blog post [here](https://www.strathweb.com/2024/05/running-microsoft-phi-3-model-in-an-ios-app-with-rust/).

## Compatibility notes

### .NET

✅ Tested on Windows arm64

✅ Tested on Linux arm64

✅ Tested on MacOS arm64

### iOS

✅ Tested on iPad Air M1 8GB RAM

✅ Should work on 6GB RAM iPhones too

❌ Will not work on 4GB RAM iPhones

However, for 4GB RAM iPhones, it's possible to use the (very) low fidelity Q2_K quantized model. Such model is not included in the official Phi-3 release, but I tested [this one from HuggingFace](https://huggingface.co/SanctumAI/Phi-3-mini-4k-instruct-GGUF) on an iPhone 12 mini successfully.

This is the setup in the app:

```swift
 self.engine = try! PhiEngine(engineOptions: EngineOptions(cacheDir: FileManager.default.temporaryDirectory.path(), systemInstruction: nil, tokenizerRepo: nil, modelRepo: "SanctumAI/Phi-3-mini-4k-instruct-GGUF", modelFileName: "phi-3-mini-4k-instruct.Q2_K.gguf", modelRevision: "main"), eventHandler: ModelEventsHandler(parent: self))
```

This variant requires the system instruction to be set differently too, as it recognizes the system token, so in other words, for the model to stop the inference correctly, this line in the Rust code:

```rust
let prompt_with_history = format!("<|user|>\nYour overall instructions are: {}<|end|>\n<|assistant|>Understood, I will adhere to these instructions<|end|>{}\n<|assistant|>\n", self.system_instruction, history_prompt);
```

needs to be changed to:

```rust
let prompt_with_history = format!("<|system|>\nYour overall instructions are: {}<|end|>{}\n<|assistant|>\n", self.system_instruction, history_prompt);
```