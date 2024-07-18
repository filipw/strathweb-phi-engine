import strathweb.phi.engine.InferenceOptionsBuilder
import strathweb.phi.engine.PhiEventHandler
import strathweb.phi.engine.PhiEngineBuilder
import strathweb.phi.engine.BoxedPhiEventHandler
import java.io.File

fun main(args: Array<String>) {
    val inferenceOptionsBuilder = InferenceOptionsBuilder()
    inferenceOptionsBuilder.withTemperature(0.9)
    inferenceOptionsBuilder.withSeed(146628346.toULong())
    val inferenceOptions = inferenceOptionsBuilder.build()

    val cacheDir = File(System.getProperty("user.dir"), ".cache").absolutePath

    class ModelEventsHandler : PhiEventHandler {
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
    val gpuEnabled = modelBuilder.tryUseGpu()
    val model = modelBuilder.buildStateful(cacheDir, "You are a hockey poet")

    // Run inference
    val result = model.runInference("Write a haiku about ice hockey", inferenceOptions)

    println(
        """
        
        ****************************************
        📝 Tokens Generated: ${result.tokenCount}
        🖥️ Tokens per second: ${result.tokensPerSecond}
        ⏱️ Duration: ${result.duration}s
        🏎️ GPU enabled: $gpuEnabled
        """.trimIndent()
    )
}