using AutoGen.Core;
using uniffi.strathweb_phi_engine;

class LocalStreamingPhiAgent : LocalPhiAgent, IStreamingAgent
{
    private readonly StreamingEventHandler _handler;

    public LocalStreamingPhiAgent(string name, PhiEngine phiEngine, string systemInstruction, StreamingEventHandler handler) : base(name, phiEngine, systemInstruction)
    {
        _handler = handler;
    }
    
    public async IAsyncEnumerable<IMessage> GenerateStreamingReplyAsync(IEnumerable<IMessage> messages,
        GenerateReplyOptions options = null,
        CancellationToken cancellationToken = new CancellationToken())
    {
        var prompt = GetCurrentPrompt(messages);
        var conversationContext = GetConversationContext(messages);
        var inferenceOptions = GetInferenceOptions(options);
        
        var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        Task.Run(() =>
        {
            try
            {
                var response = PhiEngine.RunInference(prompt, conversationContext, inferenceOptions);
            }
            catch (Exception)
            {
                cts.Cancel();
                throw;
            }
        }, cts.Token);
        
        await foreach (var token in _handler.GetInferenceTokensAsync().WithCancellation(cts.Token))
        {
            yield return new TextMessageUpdate(AutoGen.Core.Role.Assistant, token, from: Name);
        }
    }
}