from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


def write_prompt_to_file(prompt, output_file):
    try:
        with open(output_file, "w") as file:
            file.write(prompt)
        print(f"Prompt successfully written to {output_file}")
    except Exception as e:
        print(f"Error writing to file: {str(e)}")


def generate_chat_prompt(prompt, system_prompt=""):
    chat_prompt = ChatPromptTemplate.from_messages(
        [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
    )
    return chat_prompt
