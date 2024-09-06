using AutoGen.Core;
using Strathweb.Phi.Engine;
using AutoGenRole = AutoGen.Core.Role;

namespace Strathweb.Phi.Engine.AutoGen;

public static class IMessageExtensions
{
    public static ConversationMessage ToConversationMessage(this TextMessage message)
    {
        Role? role = null;
        if (message.Role == AutoGenRole.User)
        {
            role = Role.User;
        }
        else if (message.Role == AutoGenRole.Assistant)
        {
            role = Role.Assistant;
        }

        if (role == null)
        {
            throw new NotSupportedException("Invalid role");
        }

        return new ConversationMessage(role.Value, message.Content);
    }
}

static class InferenceOptionsBuilderExtensions
{
    public static InferenceOptionsBuilder FromInferenceOptions(this InferenceOptionsBuilder builder, InferenceOptions options)
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