import Foundation

let inferenceOptionsBuilder = InferenceOptionsBuilder()
try! inferenceOptionsBuilder.withTemperature(temperature: 0.9)
try! inferenceOptionsBuilder.withSeed(seed: 146628346)
let inferenceOptions = try! inferenceOptionsBuilder.build()

let cacheDir = FileManager.default.currentDirectoryPath.appending("/.cache")

class ModelEventsHandler: PhiEventHandler {
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