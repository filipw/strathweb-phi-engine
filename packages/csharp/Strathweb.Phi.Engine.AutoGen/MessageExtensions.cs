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