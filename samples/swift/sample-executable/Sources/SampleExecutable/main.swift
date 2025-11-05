import Foundation
import Strathweb_Phi_Engine

let (isGgufMode, isPhi4, isCpuMode) = (
    CommandLine.arguments.contains("--gguf"),
    CommandLine.arguments.contains("--phi-4"),
    CommandLine.arguments.contains("--cpu")
)

let formatMode = isGgufMode ? " 🍃 GGUF" : " 💪 Safe tensors"
let modelMode = isPhi4 ? " 🚀 Phi-4" : " 🚗 Phi-3"

print("\(formatMode) mode is enabled.\n\(modelMode) mode is enabled.")

let modelProvider = switch (isGgufMode, isPhi4) {
    case (true, true):
        PhiModelProvider.huggingFaceGguf(
            modelRepo: "microsoft/phi-4-gguf",
            modelFileName: "phi-4-Q4_0.gguf",
            modelRevision: "main"
        )
    case (true, false):
        PhiModelProvider.huggingFaceGguf(
            modelRepo: "microsoft/Phi-3-mini-4k-instruct-gguf",
            modelFileName: "Phi-3-mini-4k-instruct-q4.gguf",
            modelRevision: "main"
        )
    case (false, true):
        PhiModelProvider.huggingFace(
            modelRepo: "microsoft/Phi-4-mini-instruct",
            modelRevision: "main"
        )
    case (false, false):
        PhiModelProvider.huggingFace(
            modelRepo: "microsoft/Phi-3.5-mini-instruct",
            modelRevision: "main"
        )
}

let inferenceOptionsBuilder = InferenceOptionsBuilder()
try! inferenceOptionsBuilder.withTemperature(temperature: 0.9)
try! inferenceOptionsBuilder.withSeed(seed: 146628346)
if isPhi4 {
    try! inferenceOptionsBuilder.withChatFormat(chatFormat: ChatFormat.chatMl)
}
let inferenceOptions = try! inferenceOptionsBuilder.build()

let cacheDir = FileManager.default.currentDirectoryPath.appending("/../../../.cache")

class ModelEventsHandler: PhiEventHandler {
    func onInferenceStarted() {
        print(" ℹ️ Inference started...")
    }
    func onInferenceEnded() {
        print("\n ℹ️ Inference ended.")
    }
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
try! modelBuilder.withEventHandler(eventHandler: ModelEventsHandler())
try! modelBuilder.withModelProvider(modelProvider: modelProvider)

if isPhi4 {
    try! modelBuilder.withTokenizerProvider(tokenizerProvider: .huggingFace(
        tokenizerRepo: "microsoft/phi-4",
        tokenizerFileName: "tokenizer.json"
    ))
}

if !isCpuMode {
    let gpuEnabled = try! modelBuilder.tryUseGpu()
    print(gpuEnabled ? " 🎮 GPU mode enabled." : " 💻 Tried GPU, but falling back to CPU.")
} else {
    print(" 💻 CPU mode enabled.")
}

let model = try! modelBuilder.buildStateful(
    cacheDir: cacheDir,
    systemInstruction: "You are a hockey poet. Be brief and polite."
)

let result = try! model.runInference(
    promptText: "Write a haiku about ice hockey",
    inferenceOptions: inferenceOptions
)

print("""
****************************************
 📝 Tokens Generated: \(result.tokenCount)
 🖥️ Tokens per second: \(result.tokensPerSecond)
 ⏱️ Duration: \(result.duration)s
""")