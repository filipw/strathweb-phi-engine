namespace Strathweb.Phi.Engine.Microsoft.Extensions.AI;

public static class PhiEngineExtensions
{
    public static PhiEngineChatClient AsChatClient(this PhiEngine engine, string id, StreamingEventHandler handler, string systemInstruction = null, InferenceOptions inferenceOptions = null) => 
        new(id, engine, handler, systemInstruction, inferenceOptions);
}