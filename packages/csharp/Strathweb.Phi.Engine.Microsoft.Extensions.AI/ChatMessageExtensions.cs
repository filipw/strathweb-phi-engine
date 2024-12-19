using Microsoft.Extensions.AI;

namespace Strathweb.Phi.Engine.Microsoft.Extensions.AI;

public static class ChatMessageExtensions
{
    public static ConversationMessage ToConversationMessage(this ChatMessage message)
    {
        Role? role = null;
        if (message.Role == ChatRole.User)
        {
            role = Role.User;
        }
        else if (message.Role == ChatRole.Assistant)
        {
            role = Role.Assistant;
        }

        if (role == null)
        {
            throw new NotSupportedException("Invalid role");
        }

        return new ConversationMessage(role.Value, message.Text);
    }
}