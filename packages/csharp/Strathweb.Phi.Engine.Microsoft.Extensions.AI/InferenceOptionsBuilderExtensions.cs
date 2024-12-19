using Microsoft.Extensions.AI;

namespace Strathweb.Phi.Engine.Microsoft.Extensions.AI;

static class InferenceOptionsBuilderExtensions
{
    public static InferenceOptionsBuilder FromChatOptions(this InferenceOptionsBuilder builder, ChatOptions options)
    {
        if (options.Temperature != null)
        {
            builder.WithTemperature(options.Temperature.Value);
        }

        if (options.TopP != null)
        {
            builder.WithTopP(options.TopP.Value);
        }

        if (options.TopK != null)
        {
            builder.WithTopK((ulong)options.TopK.Value);
        }

        if (options.MaxOutputTokens != null)
        {
            builder.WithTokenCount((ushort)options.MaxOutputTokens.Value);
        }

        if (options.FrequencyPenalty != null)
        {
            builder.WithRepeatPenalty(options.FrequencyPenalty.Value);
        }

        if (options.Seed != null)
        {
            builder.WithSeed((ulong)options.Seed);
        }

        return builder;
    }

    public static InferenceOptionsBuilder FromInferenceOptions(this InferenceOptionsBuilder builder,
        InferenceOptions options)
    {
        builder.WithTemperature(options.temperature);
        builder.WithTokenCount(options.tokenCount);

        if (options.topP != null)
        {
            builder.WithTopP(options.topP.Value);
        }

        if (options.topK != null)
        {
            builder.WithTopK(options.topK.Value);
        }

        builder.WithRepeatPenalty(options.repeatPenalty);
        builder.WithRepeatLastN(options.repeatLastN);
        builder.WithSeed(options.seed);

        return builder;
    }
}