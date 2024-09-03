using Spectre.Console;
using uniffi.strathweb_phi_engine;
using autogen.console;

var cacheDir = Path.Combine(Directory.GetCurrentDirectory(), ".cache");
var modelBuilder = new PhiEngineBuilder();
var model = modelBuilder.Build(cacheDir);

var demo = AnsiConsole.Prompt(
    new SelectionPrompt<string>()
        .Title("Choose the [green]demo[/] to run?")
        .AddChoices(new[]
        {
            "Assistant Agent", "Two Agent Match Chat"
        }));

switch (demo)
{
    case "Assistant Agent":
        await Examples.Example01_AssistantAgent(model);
        break;
    case "Two Agent Match Chat":
        await Examples.Example02_TwoAgent_MathChat(model);
        break;
    default:
        Console.WriteLine("Nothing selected!");
        break;
}