using AutoGen.Core;
using uniffi.strathweb_phi_engine;

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