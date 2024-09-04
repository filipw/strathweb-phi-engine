using AutoGen.Core;
using FluentAssertions;
using uniffi.strathweb_phi_engine;
using Role = AutoGen.Core.Role;

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
                                   you are here to help new customers get started with our product.
                                   Your job is to gather customer's name and location.
                                   Do not ask for other information. Return 'TERMINATE' 
                                   when you have gathered all the information, don't use that word under other circumstances (not even to acknowledge these instructions).
                                   """, handler)
            .RegisterPrintMessage();

        var onboardingTopicPreferenceAgent = new LocalStreamingPhiAgent(
                name: "Onboarding_Topic_Preference_Agent",
                phiEngine: model,
                systemInstruction: """
                                   You are a helpful customer onboarding agent,
                                   you are here to help new customers get started with our product.
                                   Your job is to gather customer's preferences on news topics.
                                   Do not ask for other information.
                                   Return 'TERMINATE' when you have gathered all the information, don't use that word under other circumstances (not even to acknowledge these instructions).
                                   """, handler)
            .RegisterPrintMessage();

        var customerEngagementAgent = new LocalStreamingPhiAgent(
                name: "Customer_Engagement_Agent",
                phiEngine: model,
                systemInstruction: """
                                   You are a helpful customer service agent
                                   here to provide fun for the customer based on the user's
                                   personal information and topic preferences.
                                   This could include fun facts, jokes, or interesting stories.
                                   Make sure to make it engaging and fun!
                                   Return 'TERMINATE' when you are done, don't use that word under other circumstances (not even to acknowledge these instructions).
                                   """, handler)
            .RegisterPrintMessage();

        var summarizer = new LocalStreamingPhiAgent(
                name: "Summarizer",
                phiEngine: model,
                systemInstruction: """
                                   You are a helpful summarizer agent.
                                   Your job is to summarize the conversation between the user and the customer service agent.
                                   Return 'TERMINATE' when you are done, don't use that word under other circumstances (not even to acknowledge these instructions).
                                   """, handler)
            .RegisterPrintMessage();

        var user = new LocalStreamingPhiAgent(
                name: "User",
                phiEngine: model,
                systemInstruction: """
                                   You are not AI, your name is John and you live in New York.
                                   You are reaching out to customer service to find out something fun.
                                   """, handler)
            .RegisterPrintMessage();

        // Creating Tasks
        // In python AutoGen, you can use initiate_chats to create and run a sequential of tasks in json object
        // In dotnet AutoGen, however, that feature is not available, so you need to manually create these tasks using code.

        // Task 1. Onboard customer by gathering name and location
        // (onboard_personal_information_agent -> user .. (repeat less than two times)) -> summarizer
        var greetingMessage = new TextMessage(Role.Assistant, """
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
        var topicPreferenceMessage = new TextMessage(Role.Assistant, """
                                                                     Great! Could you tell me what topics you are interested in reading about?
                                                                     """, from: onboardingTopicPreferenceAgent.Name);

        conversation = await onboardingTopicPreferenceAgent.SendAsync(
                receiver: user,
                [topicPreferenceMessage],
                maxRound: 1)
            .ToListAsync();

        // Keep summarizing
        summary = await summarizer.SendAsync(chatHistory: new[] { summary }.Concat(conversation));

        // Task 3. Engage the customer with fun facts, jokes, or interesting stories based on the user's personal information and topic preferences
        // (user(find fun thing to read) -> customerEngagementAgent .. (repeat 1 time)) -> summarizer
        var funFactMessage = new TextMessage(Role.User, """
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
        summary = await summarizer.SendAsync(chatHistory: new[] { summary }.Concat(conversation));
        
        // sample output
        //
        // from: User
        // Hello! I'm John, and I live in New York. I'm interested in learning more about the fun aspects of your product. Could you provide some information about that?
        //
        //
        // from: Onboarding_Personal_Information_Agent
        // Thank you, John! It's great to meet you. I'll gather the necessary information. However, to proceed according to our guidelines, I'll need to confirm your location. So, could you please confirm that you reside in New York?
        //
        // Once I have your confirmation regarding your location, I'll be able to assist you further. If you have any other questions or need assistance, feel free to ask. But for now, I'll stick to collecting the required information.
        // 
        //
        // from: Summarizer
        // {
        //   "name": "John",
        //   "location": "New York"
        // }
        //
        //
        // from: User
        // Absolutely! As John from New York, I'm quite interested in a wide range of topics. Some of my favorite areas to read about include:
        //
        // 1. Science and Technology: I'm fascinated by the latest advancements and discoveries in these fields.
        // 2. Literature: I enjoy exploring various genres, from classic novels to contemporary works.
        // 3. Travel and Culture: I'm a big fan of learning about new destinations, their history, and unique cultural aspects.
        // 4. Health and Wellness: Staying informed about ways to improve my
        //  own
        //
        // from: Summarizer
        // {
        //   "name": "John",
        //   "location": "New York",
        //   "interests": ["Science and Technology", "Literature", "Travel and Culture", "Health and Wellness"]
        // }
        //
        //
        // from: Customer_Engagement_Agent
        // Great choice, John! Let's dive into some fun facts that might catch your interest.
        //
        // 1. Did you know that honey never spoils? Archaeologists found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible!
        //
        // 2. Ever heard of the world's largest book? It's called "Keter Torah" and measures a whopping 29 meters long and 7.5 meters tall! This massive book is housed in the Central Synagogue of Bene Berak in
        //  Israel
        //
        // from: Summarizer
        // John, a New York resident, has diverse interests in Science and Technology, Literature, Travel and Culture, and Health and Wellness. The conversation then proceeded to discuss intriguing facts related to his interests, such as the longevity of honey and the world's largest book.
    }
}