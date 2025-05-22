import re


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