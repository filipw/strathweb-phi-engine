import strathweb.phi.engine.InferenceOptionsBuilder
import strathweb.phi.engine.PhiEventHandler
import strathweb.phi.engine.PhiEngineBuilder
import strathweb.phi.engine.BoxedPhiEventHandler
import strathweb.phi.engine.PhiModelProvider
import java.io.File

fun main(args: Array<String>) {
    val isNonQuantizedMode = args.contains("--non-quantized")
    if (isNonQuantizedMode) {
        println(" 💪 Safe tensors mode is enabled.")
    } else {
        println(" 🍃 Quantized mode is enabled.")
    }

    val modelProvider = if (isNonQuantizedMode) {
        PhiModelProvider.HuggingFace(
            modelRepo = "microsoft/Phi-3-mini-4k-instruct",
            modelRevision = "main"
        )
    } else {
        PhiModelProvider.HuggingFaceGguf(
            modelRepo = "microsoft/Phi-3-mini-4k-instruct-gguf",
            modelFileName = "Phi-3-mini-4k-instruct-q4.gguf",
            modelRevision = "main"
        )
    }

    val inferenceOptionsBuilder = InferenceOptionsBuilder()
    inferenceOptionsBuilder.withTemperature(0.9)
    inferenceOptionsBuilder.withSeed(146628346.toULong())
    val inferenceOptions = inferenceOptionsBuilder.build()

    val cacheDir = File(System.getProperty("user.dir"), ".cache").absolutePath

    class ModelEventsHandler : PhiEventHandler {
        override fun onInferenceStarted() {}
        
        override fun onInferenceEnded() {}

        override fun onInferenceToken(token: String) {
            print(token)
        }

        override fun onModelLoaded() {
            println(
                """
                🧠 Model loaded!
                ****************************************
                """.trimIndent()
            )
        }
    }

    val modelBuilder = PhiEngineBuilder()
    modelBuilder.withEventHandler(BoxedPhiEventHandler(ModelEventsHandler()))
    modelBuilder.withModelProvider(modelProvider)
    val model = modelBuilder.buildStateful(cacheDir, "You are a hockey poet")

    // Run inference
    val result = model.runInference("Write a haiku about ice hockey", inferenceOptions)

    println(
        """
        
        ****************************************
        📝 Tokens Generated: ${result.tokenCount}
        🖥️ Tokens per second: ${result.tokensPerSecond}
        ⏱️ Duration: ${result.duration}s
        """.trimIndent()
    )
}