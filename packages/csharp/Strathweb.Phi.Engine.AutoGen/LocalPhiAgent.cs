using AutoGen.Core;
using AutoGenRole = AutoGen.Core.Role;
using Strathweb.Phi.Engine;

namespace Strathweb.Phi.Engine.AutoGen;

public class LocalPhiAgent : IAgent
{
    protected readonly PhiEngine PhiEngine;
    protected readonly InferenceOptions InferenceOptions;
    private readonly string _systemInstruction;

    public string Name { get; }

    public LocalPhiAgent(string name, PhiEngine phiEngine, string systemInstruction, InferenceOptions inferenceOptions = null)
    {
        PhiEngine = phiEngine;
        Name = name;
        _systemInstruction = systemInstruction;
        InferenceOptions = inferenceOptions ?? new InferenceOptionsBuilder().Build();
    }

    public Task<IMessage> GenerateReplyAsync(IEnumerable<IMessage> messages, GenerateReplyOptions options = null, CancellationToken cancellationToken = default)
    {
        var prompt = GetCurrentPrompt(messages);
        var conversationContext = GetConversationContext(messages);
        var inferenceOptions = GetInferenceOptions(options);

        var response = PhiEngine.RunInference(prompt, conversationContext, inferenceOptions);
        var textMessage = new TextMessage(AutoGenRole.Assistant, response.resultText, Name);
        return Task.FromResult(textMessage as IMessage);
    }

    protected string GetCurrentPrompt(IEnumerable<IMessage> messages)
    {
        var tail = messages.Last();
        var prompt = tail.GetContent();
        return prompt;
    }

    protected ConversationContext GetConversationContext(IEnumerable<IMessage> messages)
    {
        var head = messages.Take(messages.Count() - 1);
        var phiEngineMessages = head.Select(m => m switch
        {
            TextMessage message => message.ToConversationMessage(),
            _ => throw new ArgumentException($"Invalid message type: {m.GetType()}")
        }).ToArray();

        var context = new ConversationContext(phiEngineMessages, _systemInstruction);
        return context;
    }

    protected InferenceOptions GetInferenceOptions(GenerateReplyOptions options)
    {
        if (options != null)
        {
            var inferenceOptionsBuilder = new InferenceOptionsBuilder();
            inferenceOptionsBuilder.FromInferenceOptions(InferenceOptions);

            if (options.Temperature != null)
            {
                inferenceOptionsBuilder.WithTemperature(options.Temperature.Value);
            }

            if (options.MaxToken != null)
            {
                inferenceOptionsBuilder.WithTokenCount(Convert.ToUInt16(options.MaxToken.Value));
            }

            var inferenceOptions = inferenceOptionsBuilder.Build();
            return inferenceOptions;
        }

        return InferenceOptions;
    }
}
