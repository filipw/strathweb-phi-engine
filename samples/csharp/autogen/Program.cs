using Spectre.Console;
using uniffi.strathweb_phi_engine;
using autogen.console;

var cacheDir = Path.Combine(Directory.GetCurrentDirectory(), ".cache");

var handler = new StreamingEventHandler();
var modelBuilder = new PhiEngineBuilder();
modelBuilder.WithEventHandler(new BoxedPhiEventHandler(handler));
var model = modelBuilder.Build(cacheDir);

var demo = AnsiConsole.Prompt(
    new SelectionPrompt<string>()
        .Title("Choose the [green]demo[/] to run?")
        .AddChoices(new[]
        {
            "Basic Assistant Agent", "Two Agent Math Chat", "Sequential Chat and Customer Onboarding"
        }));

switch (demo)
{
    case "Basic Assistant Agent":
        await Examples.Example01_AssistantAgent(model, handler);
        break;
    case "Two Agent Math Chat":
        await Examples.Example02_TwoAgent_MathChat(model, handler);
        break;
    case "Sequential Chat and Customer Onboarding":
        await Examples.Example03_Sequential_Chat_and_Customer_Onboarding(model, handler);
        break;
    default:
        Console.WriteLine("Nothing selected!");
        break;
}