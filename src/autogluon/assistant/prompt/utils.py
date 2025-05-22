
def write_prompt_to_file(prompt, output_file):
    try:
        with open(output_file, "w") as file:
            file.write(prompt)
        print(f"Prompt successfully written to {output_file}")
    except Exception as e:
        print(f"Error writing to file: {str(e)}")
