using AutoGen.Core;
using uniffi.strathweb_phi_engine;

class LocalPhiAgent : IAgent
{
    private readonly PhiEngine _phiEngine;
    private readonly string _systemInstruction;

    public string Name { get; }

    public LocalPhiAgent(string name, PhiEngine phiEngine, string systemInstruction)
    {
        _phiEngine = phiEngine;
        Name = name;
        _systemInstruction = systemInstruction;
    }

    public Task<IMessage> GenerateReplyAsync(IEnumerable<IMessage> messages, GenerateReplyOptions? options = null, CancellationToken cancellationToken = default)
    {
        var tail = messages.Last();
        var head = messages.Take(messages.Count() - 1);

        var prompt = tail.GetContent();
        if (string.IsNullOrEmpty(prompt))
        {
            throw new ArgumentException("Prompt cannot be empty");
        }

        var phiEngineMessages = head.Select(m => m switch
        {
            TextMessage message => message.ToConversationMessage(),
            _ => throw new ArgumentException($"Invalid message type: {m.GetType()}")
        }).ToList();

        var context = new ConversationContext(phiEngineMessages, _systemInstruction);

        var inferenceOptionsBuilder = new InferenceOptionsBuilder();
        if (options != null)
        {
            if (options.Temperature != null)
            {
                inferenceOptionsBuilder.WithTemperature(options.Temperature.Value);
            }
        }
        var inferenceOptions = inferenceOptionsBuilder.Build();

        var response = _phiEngine.RunInference(prompt, context, inferenceOptions);
        var textMessage = new TextMessage(AutoGen.Core.Role.Assistant, response.resultText, Name);
        return Task.FromResult(textMessage as IMessage);
    }
}