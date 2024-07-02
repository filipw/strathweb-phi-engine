using uniffi.strathweb_phi_engine;

var inferenceOptions = new InferenceOptions(tokenCount: 100, temperature: 0.9, topP: null, repeatPenalty: 1.0f, repeatLastN: 64, seed: 146628346);
var cacheDir = Path.Combine(Directory.GetCurrentDirectory(), ".cache");
var model = new PhiEngine(new EngineOptions(cacheDir: cacheDir, systemInstruction: null, tokenizerRepo: null, modelRepo: null, modelFileName: null, modelRevision: "5eef2ce24766d31909c0b269fe90c817a8f263fb"), new BoxedPhiEventHandler(new ModelEventsHandler()));

var result = model.RunInference("Write a haiku about ice hockey", inferenceOptions);
Console.WriteLine($"{Environment.NewLine}Tokens Generated: {result.tokenCount}{Environment.NewLine}Token per second: {result.tokensPerSecond}{Environment.NewLine}Duration: {result.duration}s");

class ModelEventsHandler : PhiEventHandler
{
    public void OnInferenceToken(string token)
    {
        Console.Write(token);
    }

    public void OnModelLoaded()
    {
        Console.WriteLine("Model loaded!");
    }
}