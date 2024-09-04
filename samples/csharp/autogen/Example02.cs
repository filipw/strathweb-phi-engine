using AutoGen.Core;
using FluentAssertions;
using uniffi.strathweb_phi_engine;
using Role = AutoGen.Core.Role;

namespace autogen.console;

public partial class Examples
{
    // AutoGen example from
    // https://github.com/microsoft/autogen/blob/main/dotnet/sample/AutoGen.BasicSamples/Example02_TwoAgent_MathChat.cs
    public static async Task Example02_TwoAgent_MathChat(PhiEngine model, StreamingEventHandler handler)
    {
        // teacher agent - asks questions and checks answers
        var teacher = new LocalStreamingPhiAgent("teacher", model,
                @"You are a teacher that asks a pre-school math question for student. Ask a question but do not provide the answer. As soon as student provides the answer, check the answer.
        If the answer is correct, praise the student and stop the conversation by saying [COMPLETE].
        If the answer is wrong, you ask student to fix it.", handler)
            .RegisterMiddleware(async (msgs, option, agent, _) =>
            {
                var reply = await agent.GenerateReplyAsync(msgs, option);
                if (reply.GetContent()?.ToLower().Contains("[complete]") is true)
                {
                    return new TextMessage(Role.Assistant, GroupChatExtension.TERMINATE, from: reply.From);
                }

                return reply;
            })
            .RegisterPrintMessage();

        // student agent - answers the math questions
        var student = new LocalStreamingPhiAgent("student", model,
                "You are a student that answer question from teacher", handler)
            .RegisterPrintMessage();

        // start the conversation
        var conversation = await student.InitiateChatAsync(
            receiver: teacher,
            message: "Hey teacher, please create a math question for me.",
            maxRound: 10);

        // expected output
        // TextMessage from teacher
        // --------------------
        // Great! Here's a question for you: 
        // 
        // "There are 5 apples in one basket and 3 apples in another basket. How many apples are there in total?"
        // 
        // Once you provide your answer, I will check it to see if it's correct!
        // --------------------
        // 
        // TextMessage from student
        // --------------------
        // The total number of apples is 8. This is calculated by adding the number of apples in each basket together: 5 apples + 3 apples = 8 apples.
        // --------------------
        // 
        // TextMessage from teacher
        // --------------------
        // content: [GROUPCHAT_TERMINATE]
        // --------------------

        conversation.Count().Should().BeLessThan(10);
        conversation.Last().IsGroupChatTerminateMessage().Should().BeTrue();
    }
}