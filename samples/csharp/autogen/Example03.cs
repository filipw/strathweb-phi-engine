using AutoGen.Core;
using Strathweb.Phi.Engine;
using Strathweb.Phi.Engine.AutoGen;
using AutoGenRole = AutoGen.Core.Role;

namespace autogen.console;

public partial class Examples
{
    // AutoGen example from
    // https://github.com/LittleLittleCloud/AI-Agentic-Design-Patterns-with-AutoGen.Net/blob/main/L2_Sequential_Chats_and_Customer_Onboarding/Program.cs
    public static async Task Example03_Sequential_Chat_and_Customer_Onboarding(PhiEngine model, StreamingEventHandler handler)
    {
        var onboardingPersonalInformationAgent = new LocalStreamingPhiAgent(
                name: "Onboarding_Personal_Information_Agent",
                phiEngine: model,
                systemInstruction: """
                                   You are a helpful customer onboarding agent,
                                   you are here to help new customers get started with our product. Be brief and to the point.
                                   Your job is to gather user's name and location.
                                   Do not ask for other information. Return just TERMINATE when the user provided all the information.
                                   """, handler)
            .RegisterPrintMessage();

        var onboardingTopicPreferenceAgent = new LocalStreamingPhiAgent(
                name: "Onboarding_Topic_Preference_Agent",
                phiEngine: model,
                systemInstruction: """
                                   You are a helpful customer onboarding agent,
                                   you are here to help new customers get started with our product. Be brief and to the point.
                                   Your job is to gather customer's preferences on news topics.
                                   Do not ask for other information.
                                   Return just TERMINATE when you have gathered all the information.
                                   """, handler)
            .RegisterPrintMessage();

        var customerEngagementAgent = new LocalStreamingPhiAgent(
                name: "Customer_Engagement_Agent",
                phiEngine: model,
                systemInstruction: """
                                   You are a helpful customer service agent
                                   here to provide fun for the customer.
                                   Provide one fun fact about the user's location or interests.
                                   Make sure to make it engaging and fun but make that short and brief.
                                   Return just TERMINATE when you are done, don't use that word otherwise.
                                   """, handler)
            .RegisterPrintMessage();

        var summarizer = new LocalStreamingPhiAgent(
                name: "Summarizer",
                phiEngine: model,
                systemInstruction: """
                                   You are a summarizer agent.
                                   Your job is to summarize the conversation between user and customer service agent.
                                   Return TERMINATE when you are done.
                                   """, handler)
            .RegisterPrintMessage();

        var user = new LocalStreamingPhiAgent(
                name: "User",
                phiEngine: model,
                systemInstruction: """
                                   You are not AI, you are a user, your name is John and you live in New York.
                                   You are reaching out to customer service agent to find out something fun. Be brief and to the point.
                                   You are not assisting the agent, you are the one who needs help.
                                   """, handler)
            .RegisterPrintMessage();

        // Creating Tasks
        // In python AutoGen, you can use initiate_chats to create and run a sequential of tasks in json object
        // In dotnet AutoGen, however, that feature is not available, so you need to manually create these tasks using code.

        // Task 1. Onboard customer by gathering name and location
        // (onboard_personal_information_agent -> user .. (repeat less than two times)) -> summarizer
        var greetingMessage = new TextMessage(AutoGenRole.Assistant, """
                                                              Hello, I'm here to help you get started with our product.
                                                              Could you tell me your name and location?
                                                              """, from: onboardingPersonalInformationAgent.Name);

        var conversation = await onboardingPersonalInformationAgent.SendAsync(
                receiver: user,
                [greetingMessage],
                maxRound: 2)
            .ToListAsync();

        var summarizePrompt = """
                              Return the customer information into as JSON object only: {'name': '', 'location': ''}
                              """;

        var summary = await summarizer.SendAsync(summarizePrompt, conversation);

        // Task 2. Gapther customer's preferences on news topics
        // (onboarding_topic_preference_agent -> user .. (repeat one time)) -> summarizer
        var topicPreferenceMessage = new TextMessage(AutoGenRole.Assistant, """
                                                                     Great! Could you tell me what topics you are interested in reading about?
                                                                     """, from: onboardingTopicPreferenceAgent.Name);

        conversation = await onboardingTopicPreferenceAgent.SendAsync(
                receiver: user,
                [topicPreferenceMessage],
                maxRound: 1)
            .ToListAsync();

        // Keep summarizing
        summary = await summarizer.SendAsync("summarize the conversation to this point", chatHistory: conversation);

        // Task 3. Engage the customer with fun facts, jokes, or interesting stories based on the user's personal information and topic preferences
        // (user(find fun thing to read) -> customerEngagementAgent .. (repeat 1 time)) -> summarizer
        var funFactMessage = new TextMessage(AutoGenRole.User, """
                                                        Let's find something fun to read.
                                                        """, from: user.Name);

        conversation = await user.SendAsync(
                receiver: customerEngagementAgent,
                chatHistory: conversation.Concat([
                    funFactMessage
                ]), // this time, we keep the previous conversation history
                maxRound: 1)
            .ToListAsync();

        // Keep summarizing
        summary = await summarizer.SendAsync("summarize the conversation to this point", chatHistory: new[] { summary }.Concat(conversation));

        // sample output:
        //
        // from: User
        // Hi, I'm John, and I'm currently residing in New York. I'm interested in knowing more about how your product can provide a fun experience for me. Could you guide me in that direction?
        //
        //
        // from: Onboarding_Personal_Information_Agent
        // Thank you, John. New York, is appreciated. However, at this moment, let's focus on getting you started with our product. I'll need your full name to proceed with the onboarding process.
        //
        // TERMINATE
        // 
        //
        // from: Summarizer
        // {
        //   "name": "John",
        //   "location": "New York"
        // }
        //
        // TERMINATE
        //
        // from: User
        // Certainly! I'm interested in a variety of topics, including technology, science, history, and literature. I also enjoy learning about New York City's culture and events.
        //
        // from: Summarizer
        // The user expressed their interest in diverse topics such as technology, science, history, and literature, and a particular fascination with New York City's culture. The customer service agent has acknowledged the user's interests and is likely prepared to assist with inquiries related to these areas.
        //
        //TERMINATE
        //
        // from: Customer_Engagement_Agent
        // Did you know that New York City is home to over 800 libraries, offering a staggering 20 million books to explore? That's one book for every 25 people in the city!
        //
        // TERMINATE
        //
        // from: Summarizer
        // The user engaged in a conversation with a customer service agent, discussing their diverse interests and specific curiosity about New York City. The agent provided a notable fact about New York's library system, highlighting the city's rich literary resources.
    }
}