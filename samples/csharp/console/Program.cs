using uniffi.strathweb_phi_engine;

var inferenceOptionsBuilder = new InferenceOptionsBuilder();
inferenceOptionsBuilder.WithTemperature(0.9);
inferenceOptionsBuilder.WithTokenCount(100);
var inferenceOptions = inferenceOptionsBuilder.Build();

var cacheDir = Path.Combine(Directory.GetCurrentDirectory(), ".cache");

var modelBuilder = new PhiEngineBuilder();
modelBuilder.WithSystemInstruction("You are a hockey poet");
modelBuilder.WithEventHandler(new BoxedPhiEventHandler(new ModelEventsHandler()));
var model = modelBuilder.Build(cacheDir);

var result = model.RunInference("Write a haiku about ice hockey", inferenceOptions);
Console.WriteLine($"{Environment.NewLine}Tokens Generated: {result.tokenCount}{Environment.NewLine}Tokens per second: {result.tokensPerSecond}{Environment.NewLine}Duration: {result.duration}s");

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