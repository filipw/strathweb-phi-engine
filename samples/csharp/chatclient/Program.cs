using Strathweb.Phi.Engine;
using Strathweb.Phi.Engine.Microsoft.Extensions.AI;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;
using ChatRole = Microsoft.Extensions.AI.ChatRole;

var cacheDir = Path.Combine(Directory.GetCurrentDirectory(), ".cache");

var handler = ;
var modelBuilder = new PhiEngineBuilder();
modelBuilder.WithEventHandler(new StreamingEventHandler());
var model = modelBuilder.Build(cacheDir);

var chatClient = model.AsChatClient("Local Phi-3 Demo", handler,
    systemInstruction: "You convert what user said to all uppercase.");
// or alternatively
//var chatClient = new PhiEngineChatClient("Local Phi-3 Demo", model, handler, systemInstruction: "You convert what user said to all uppercase.");

var message = new ChatMessage(ChatRole.User, "hello world");
var response = await chatClient.CompleteAsync([message]);

Console.WriteLine(response);

Console.WriteLine();
await foreach (var update in chatClient.CompleteStreamingAsync([
                   new ChatMessage(ChatRole.System, "you are an ice hockey poet"),
                   new ChatMessage(ChatRole.User, "write a haiku")
               ]))
{
    Console.Write(update);
}