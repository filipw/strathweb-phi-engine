using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;

namespace Strathweb.Phi.Engine.Microsoft.Extensions.AI;

public class PhiEngineChatClient : IChatClient
{
    private readonly PhiEngine _phiEngine;
    private readonly StreamingEventHandler _handler;
    private readonly ChatClientMetadata _metadata;
    private readonly string _systemInstruction;
    private readonly InferenceOptions _inferenceOptions;

    public void Dispose()
    {
    }
    
    public PhiEngineChatClient(string name, PhiEngine phiEngine, StreamingEventHandler handler, string systemInstruction = null, InferenceOptions inferenceOptions = null)
    {
        _phiEngine = phiEngine;
        _handler = handler;
        _metadata = new ChatClientMetadata(providerName: name);
        _systemInstruction = systemInstruction;
        _inferenceOptions = inferenceOptions ?? new InferenceOptionsBuilder().Build();
    }

    public Task<ChatCompletion> CompleteAsync(IList<ChatMessage> chatMessages, ChatOptions options = null,
        CancellationToken cancellationToken = new())
    {
        if (chatMessages == null || chatMessages.Count == 0) throw new ArgumentException("Messages cannot be empty");

        var prompt = GetCurrentPrompt(chatMessages);
        var conversationContext = GetConversationContext(chatMessages);
        var inferenceOptions = GetInferenceOptions(options);

        var response = _phiEngine.RunInference(prompt, conversationContext, inferenceOptions);
        var textMessage = new ChatMessage(ChatRole.Assistant, response.resultText);
        return Task.FromResult(new ChatCompletion(new[] { textMessage }));
    }

    public async IAsyncEnumerable<StreamingChatCompletionUpdate> CompleteStreamingAsync(IList<ChatMessage> chatMessages, ChatOptions options = null,
        [EnumeratorCancellation]CancellationToken cancellationToken = new())
    {
        if (chatMessages == null || chatMessages.Count == 0) throw new ArgumentException("Messages cannot be empty");
        
        var prompt = GetCurrentPrompt(chatMessages);
        var conversationContext = GetConversationContext(chatMessages);
        var inferenceOptions = GetInferenceOptions(options);

        var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
#pragma warning disable CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed
        Task.Run(() =>
#pragma warning restore CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed
        {
            try
            {
                _phiEngine.RunInference(prompt, conversationContext, inferenceOptions);
            }
            catch (Exception)
            {
                cts.Cancel();
                throw;
            }
        }, cts.Token);

        await foreach (var token in _handler.GetInferenceTokensAsync().WithCancellation(cts.Token))
        {
            yield return new StreamingChatCompletionUpdate
            {
                Role = ChatRole.Assistant,
                Text = token
            };
        }
    }
    
    public object GetService(Type serviceType, object serviceKey = null)
    {
        throw new NotSupportedException("No services can be retrieved");
    }

    public ChatClientMetadata Metadata  => _metadata;
    
    private static string GetCurrentPrompt(IList<ChatMessage> messages)
    {
        var tail = messages.Where(m => m.Role == ChatRole.User).LastOrDefault();
        return tail?.Text;
    }
    
    private ConversationContext GetConversationContext(IList<ChatMessage> messages)
    {
        if (messages.Count > 0)
        {
            var head = messages.Take(messages.Count - 1).ToList();
            var phiEngineMessages = head.Where(m => m.Role != ChatRole.System && m.Role != ChatRole.Tool)
                .Select(m => m.ToConversationMessage()).ToList();
            var systemInstruction = head.LastOrDefault(m => m.Role == ChatRole.System)?.Text ?? _systemInstruction;
            return new ConversationContext(phiEngineMessages, systemInstruction);
        }

        return new ConversationContext([], _systemInstruction);
    }

    private InferenceOptions GetInferenceOptions(ChatOptions options)
    {
        if (options == null) return _inferenceOptions;
        
        var inferenceOptionsBuilder = new InferenceOptionsBuilder();
        inferenceOptionsBuilder.FromInferenceOptions(_inferenceOptions);
        inferenceOptionsBuilder.FromChatOptions(options);
        return inferenceOptionsBuilder.Build();
    }
}