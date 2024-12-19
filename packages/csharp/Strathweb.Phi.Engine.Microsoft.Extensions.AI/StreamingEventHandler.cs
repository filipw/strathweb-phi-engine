using System.Threading.Channels;

namespace Strathweb.Phi.Engine.Microsoft.Extensions.AI;

public class StreamingEventHandler : PhiEventHandler
{
    private Channel<string> _tokenChannel;
    private TaskCompletionSource<bool> _inferenceStartedTcs = new();

    public void OnInferenceToken(string token)
    {
        _tokenChannel?.Writer.TryWrite(token);
    }

    public void OnInferenceStarted()
    {
        _tokenChannel = Channel.CreateUnbounded<string>();
        _inferenceStartedTcs.TrySetResult(true);
    }

    public void OnInferenceEnded()
    {
        _tokenChannel?.Writer.Complete();
        _tokenChannel = null;
        _inferenceStartedTcs = new TaskCompletionSource<bool>();
    }

    public void OnModelLoaded()
    {
    }

    public async IAsyncEnumerable<string> GetInferenceTokensAsync()
    {
        await _inferenceStartedTcs.Task;
        await foreach (var token in _tokenChannel.Reader.ReadAllAsync())
        {
            yield return token;
        }
    }
}