using AutoGen.Core;
using FluentAssertions;
using uniffi.strathweb_phi_engine;
using Role = AutoGen.Core.Role;

namespace autogen.console;

public partial class Examples
{
    // AutoGen example from
    // https://github.com/microsoft/autogen/blob/main/dotnet/sample/AutoGen.BasicSamples/Example01_AssistantAgent.cs
    public static async Task Example01_AssistantAgent(PhiEngine model, StreamingEventHandler handler)
    {
        var assistantAgent = new LocalStreamingPhiAgent("assistant", model, "You convert what user said to all uppercase.", handler)
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
        
        // expected output:
        // TextMessage from assistant
        // --------------------
        // HELLO WORLD
        // --------------------
        //
        // TextMessage from assistant
        // --------------------
        // HELLO WORLD AGAIN
        // --------------------
        //
    }
}