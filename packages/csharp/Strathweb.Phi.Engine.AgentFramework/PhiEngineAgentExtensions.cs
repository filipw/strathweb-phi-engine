using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using Strathweb.Phi.Engine.Microsoft.Extensions.AI;

namespace Strathweb.Phi.Engine.AgentFramework;

/// <summary>
/// Provides extension methods for the <see cref="PhiEngine"/> class.
/// </summary>
public static class PhiEngineAgentExtensions
{
    /// <summary>
    /// Creates a new AI agent using the specified model and options.
    /// </summary>
    /// <param name="engine">A <see cref="PhiEngine"/> to use with the agent.</param>
    /// <param name="id">The id of the chat client.</param>
    /// <param name="handler">The streaming event handler.</param>
    /// <param name="instructions">The instructions for the AI agent.</param>
    /// <param name="name">The name of the AI agent.</param>
    /// <param name="description">The description of the AI agent.</param>
    /// <param name="tools">The tools available to the AI agent.</param>
    /// <param name="inferenceOptions">The inference options.</param>
    /// <param name="clientFactory">Provides a way to customize the creation of the underlying <see cref="IChatClient"/> used by the agent.</param>
    /// <param name="loggerFactory">Optional logger factory for enabling logging within the agent.</param>
    /// <param name="services">An optional <see cref="IServiceProvider"/> to use for resolving services required by the <see cref="AIFunction"/> instances being invoked.</param>
    /// <returns>The created <see cref="ChatClientAgent"/> AI agent.</returns>
    public static ChatClientAgent AsAIAgent(
        this PhiEngine engine,
        string id,
        StreamingEventHandler handler,
        string? instructions = null,
        string? name = null,
        string? description = null,
        IList<AITool>? tools = null,
        InferenceOptions? inferenceOptions = null,
        Func<IChatClient, IChatClient>? clientFactory = null,
        ILoggerFactory? loggerFactory = null,
        IServiceProvider? services = null)
    {
        var options = new ChatClientAgentOptions
        {
            Name = name,
            Description = description,
        };

        if (!string.IsNullOrWhiteSpace(instructions))
        {
            options.ChatOptions ??= new();
            options.ChatOptions.Instructions = instructions;
        }

        if (tools is { Count: > 0 })
        {
            options.ChatOptions ??= new();
            options.ChatOptions.Tools = tools;
        }

        IChatClient chatClient = engine.AsChatClient(id, handler, instructions, inferenceOptions);

        if (clientFactory is not null)
        {
            chatClient = clientFactory(chatClient);
        }

        return new ChatClientAgent(chatClient, options, loggerFactory, services);
    }

    /// <summary>
    /// Creates an AI agent from a <see cref="PhiEngine"/> using the ChatClient wrapper.
    /// </summary>
    /// <param name="engine">A <see cref="PhiEngine"/> to use for the agent.</param>
    /// <param name="id">The id of the chat client.</param>
    /// <param name="handler">The streaming event handler.</param>
    /// <param name="options">Full set of options to configure the agent.</param>
    /// <param name="inferenceOptions">The inference options.</param>
    /// <param name="clientFactory">Provides a way to customize the creation of the underlying <see cref="IChatClient"/> used by the agent.</param>
    /// <param name="loggerFactory">Optional logger factory for enabling logging within the agent.</param>
    /// <param name="services">An optional <see cref="IServiceProvider"/> to use for resolving services required by the <see cref="AIFunction"/> instances being invoked.</param>
    /// <returns>An <see cref="ChatClientAgent"/> instance backed by the Phi Engine.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="engine"/> or <paramref name="options"/> is <see langword="null"/>.</exception>
    public static ChatClientAgent AsAIAgent(
        this PhiEngine engine,
        string id,
        StreamingEventHandler handler,
        ChatClientAgentOptions options,
        InferenceOptions? inferenceOptions = null,
        Func<IChatClient, IChatClient>? clientFactory = null,
        ILoggerFactory? loggerFactory = null,
        IServiceProvider? services = null)
    {
        if (engine == null) throw new ArgumentNullException(nameof(engine));
        if (options == null) throw new ArgumentNullException(nameof(options));

        IChatClient chatClient = engine.AsChatClient(id, handler, options.ChatOptions?.Instructions, inferenceOptions);

        if (clientFactory is not null)
        {
            chatClient = clientFactory(chatClient);
        }

        return new ChatClientAgent(chatClient, options, loggerFactory, services);
    }
}
