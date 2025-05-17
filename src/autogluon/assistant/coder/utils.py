import re


def write_code_script(script, output_code_file):
    if script:
        # Save the extracted script to the output file
        save_script(script, output_code_file)
        print(f"Python script extracted and saved to {output_code_file}")
    else:
        print("No Python script found in the response.")


def write_retrieved_context(retrieved_context, output_context_path):
    if retrieved_context:
        # Save the extracted script to the output file
        save_retrieved_context(retrieved_context, output_context_path)
        print(f"Context retrieved and saved to {output_context_path}")
    else:
        print("No retrieved context.")


def save_retrieved_context(retrieved_context, output_file):
    with open(output_file, "w") as file:
        for context in retrieved_context:
            file.write(context)


def save_script(script, output_file):
    with open(output_file, "w") as file:
        file.write(script)


def extract_python_script(response):
    # Look for Python code blocks in the response
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)

    if matches:
        return matches[0].strip()
    else:
        print(f"No python script found in reponse, return the full response instead: {response}")
        return response


def extract_bash_script(response):
    # Look for Bash code blocks in the response
    pattern = r"```bash\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    else:
        print(f"No bash script found in reponse, return the full response instead: {response}")
        return response


def extract_script(response, language):
    if language == "python":
        return extract_python_script(response)
    elif language == "bash":
        return extract_bash_script(response)
    else:
        raise ValueError(f"Unsupported mode: {language}")


def read_prompt(file_path):
    with open(file_path, "r") as file:
        return file.read().strip()
