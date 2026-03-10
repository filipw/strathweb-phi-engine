using Strathweb.Phi.Engine;
using Strathweb.Phi.Engine.Microsoft.Extensions.AI;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;
using ChatRole = Microsoft.Extensions.AI.ChatRole;

var cacheDir = Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "..", ".cache");

var handler = new StreamingEventHandler();
var modelBuilder = new PhiEngineBuilder();
modelBuilder.WithEventHandler(handler);
var model = modelBuilder.Build(cacheDir);

var chatClient = model.AsChatClient("Local Phi-3 Demo", handler,
    systemInstruction: "Return the input in uppercase.");
// or alternatively
//var chatClient = new PhiEngineChatClient("Local Phi-3 Demo", model, handler, systemInstruction: "You convert what user said to all uppercase.");

var message = new ChatMessage(ChatRole.User, "hello world");
var response = await chatClient.GetResponseAsync([message]);

Console.WriteLine(response);

Console.WriteLine();
await foreach (var update in chatClient.GetStreamingResponseAsync([
                   new ChatMessage(ChatRole.System, "you are an ice hockey poet"),
                   new ChatMessage(ChatRole.User, "write a haiku")
               ]))
{
    Console.Write(update);
}