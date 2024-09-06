using AutoGen.Core;
using System.Runtime.CompilerServices;
using Strathweb.Phi.Engine;
using AutoGenRole = AutoGen.Core.Role;

namespace Strathweb.Phi.Engine.AutoGen;

public class LocalStreamingPhiAgent : LocalPhiAgent, IStreamingAgent
{
    private readonly StreamingEventHandler _handler;

    public LocalStreamingPhiAgent(string name, PhiEngine phiEngine, string systemInstruction, StreamingEventHandler handler, InferenceOptions inferenceOptions = null) : base(name, phiEngine, systemInstruction, inferenceOptions)
    {
        _handler = handler;
    }

    public async IAsyncEnumerable<IMessage> GenerateStreamingReplyAsync(IEnumerable<IMessage> messages,
        GenerateReplyOptions options = null,
         [EnumeratorCancellation] CancellationToken cancellationToken = new CancellationToken())
    {
        var prompt = GetCurrentPrompt(messages);
        var conversationContext = GetConversationContext(messages);
        var inferenceOptions = GetInferenceOptions(options);

        var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
#pragma warning disable CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed
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
#pragma warning restore CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed

        await foreach (var token in _handler.GetInferenceTokensAsync().WithCancellation(cts.Token))
        {
            yield return new TextMessageUpdate(AutoGenRole.Assistant, token, from: Name);
        }
    }
}
