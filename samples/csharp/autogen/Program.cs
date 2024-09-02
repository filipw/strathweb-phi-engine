using AutoGen;
using AutoGen.Core;
using FluentAssertions;
using uniffi.strathweb_phi_engine;

var cacheDir = Path.Combine(Directory.GetCurrentDirectory(), ".cache");
var modelBuilder = new PhiEngineBuilder();
var model = modelBuilder.Build(cacheDir);

var assistantAgent = new LocalPhiAgent("assistant", model, "You convert what user said to all uppercase.")
    .RegisterPrintMessage();

var reply = await assistantAgent.SendAsync("hello world");
reply.Should().BeOfType<TextMessage>();
reply.GetContent().Should().Be("HELLO WORLD");

// to carry on the conversation, pass the previous conversation history to the next call
var conversationHistory = new List<IMessage>
{
    new TextMessage(AutoGen.Core.Role.User, "hello world"), // first message
    reply, // reply from assistant agent
};

reply = await assistantAgent.SendAsync("hello world again", conversationHistory);
reply.Should().BeOfType<TextMessage>();
reply.GetContent().Should().Be("HELLO WORLD AGAIN");


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

public static class IMessageExtensions
{
    public static ConversationMessage ToConversationMessage(this TextMessage message)
    {
        uniffi.strathweb_phi_engine.Role? role = null;
        if (message.Role == AutoGen.Core.Role.User)
        {
            role = uniffi.strathweb_phi_engine.Role.User;
        }
        else if (message.Role == AutoGen.Core.Role.Assistant)
        {
            role = uniffi.strathweb_phi_engine.Role.Assistant;
        }

        if (role == null)
        {
            throw new NotSupportedException("Invalid role");
        }

        return new ConversationMessage(role.Value, message.Content);
    }
}