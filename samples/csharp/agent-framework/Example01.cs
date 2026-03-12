using Microsoft.Agents.AI;
using Strathweb.Phi.Engine;
using Strathweb.Phi.Engine.AgentFramework;
using Strathweb.Phi.Engine.Microsoft.Extensions.AI;

namespace agentframework.console;

public partial class Examples
{
    public static async Task Example01_BasicAgent(PhiEngine model, StreamingEventHandler handler)
    {
        AIAgent agent = model.AsAIAgent(
            id: "Local Phi-3 Demo",
            handler: handler,
            instructions: "You are good at telling jokes.",
            name: "Joker"
        );

        // Invoke the agent and output the text result.
        var response = await agent.RunAsync("Tell me a joke about a pirate.");
        Console.WriteLine(response);

        Console.WriteLine();

        // Invoke the agent with streaming support.
        await foreach (var update in agent.RunStreamingAsync("Tell me another joke about a pirate."))
        {
            Console.Write(update);
        }
    }
}
