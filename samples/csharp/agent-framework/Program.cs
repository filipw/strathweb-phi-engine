using System.IO;
using Microsoft.Agents.AI;
using Spectre.Console;
using agentframework.console;
using Strathweb.Phi.Engine;
using Strathweb.Phi.Engine.AgentFramework;
using Strathweb.Phi.Engine.Microsoft.Extensions.AI;

GlobalPhiEngine.EnableTracing();

var cacheDir = Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "..", ".cache");
var handler = new StreamingEventHandler();
var modelBuilder = new PhiEngineBuilder();
modelBuilder.WithEventHandler(handler);
var model = modelBuilder.Build(cacheDir);

var demo = AnsiConsole.Prompt(
    new SelectionPrompt<string>()
        .Title("Choose the [green]demo[/] to run?")
        .AddChoices(new[]
        {
            "Basic Agent", "Function Calling"
        }));

switch (demo)
{
    case "Basic Agent":
        await Examples.Example01_BasicAgent(model, handler);
        break;
    default:
        Console.WriteLine("Nothing selected!");
        break;
}
