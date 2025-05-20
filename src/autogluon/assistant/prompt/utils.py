from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from rich import print


def write_prompt_to_file(prompt, output_file):
    try:
        with open(output_file, "w") as file:
            file.write(prompt)
        print(f"[bold green]Prompt successfully written to[/bold green] {output_file}")
    except Exception as e:
        print(f"[bold red]Error writing to file:[/bold red] {str(e)}")


def generate_chat_prompt(prompt, system_prompt=""):
    chat_prompt = ChatPromptTemplate.from_messages(
        [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
    )
    return chat_prompt
