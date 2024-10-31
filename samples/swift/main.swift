import Foundation

let isQuantizedMode = CommandLine.arguments.contains("--quantized")

if isQuantizedMode {
    print(" 🍃 Quantized mode is enabled.")
} else {
    print(" 💪 Safe tensors mode is enabled.")
}

let modelProvider = isQuantizedMode ? 
    PhiModelProvider.huggingFaceGguf(modelRepo: "microsoft/Phi-3-mini-4k-instruct-gguf", modelFileName: "Phi-3-mini-4k-instruct-q4.gguf", modelRevision: "main") : 
    PhiModelProvider.huggingFace(modelRepo: "microsoft/Phi-3-mini-4k-instruct", modelRevision: "main")

let inferenceOptionsBuilder = InferenceOptionsBuilder()
try! inferenceOptionsBuilder.withTemperature(temperature: 0.9)
try! inferenceOptionsBuilder.withSeed(seed: 146628346)
let inferenceOptions = try! inferenceOptionsBuilder.build()

let cacheDir = FileManager.default.currentDirectoryPath.appending("/.cache")

class ModelEventsHandler: PhiEventHandler {
    func onInferenceStarted() {}

    func onInferenceEnded() {}

    func onInferenceToken(token: String) {
        print(token, terminator: "")
    }

    func onModelLoaded() {
        print("""
 🧠 Model loaded!
****************************************
""")
    }
}

let modelBuilder = PhiEngineBuilder()
try! modelBuilder.withEventHandler(eventHandler: BoxedPhiEventHandler(handler: ModelEventsHandler()))
let gpuEnabled = try! modelBuilder.tryUseGpu()
try! modelBuilder.withModelProvider(modelProvider: modelProvider)

let model = try! modelBuilder.buildStateful(cacheDir: cacheDir, systemInstruction: "You are a hockey poet. Be brief and polite.")

// Run inference
let result = try! model.runInference(promptText: "Write a haiku about ice hockey", inferenceOptions: inferenceOptions)

print("""

****************************************
 📝 Tokens Generated: \(result.tokenCount)
 🖥️ Tokens per second: \(result.tokensPerSecond)
 ⏱️ Duration: \(result.duration)s
 🏎️ GPU enabled: \(gpuEnabled)
""")