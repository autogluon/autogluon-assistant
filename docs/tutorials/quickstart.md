# Quickstart Guide

This quickstart guide will help you get up and running with AutoGluon-Assistant in just a few minutes.

## Installation

First, install AutoGluon-Assistant:

```bash
pip install autogluon.assistant
```

## Basic Usage

Here's a simple example to get you started:

```python
from autogluon.assistant import Assistant

# Initialize the assistant
assistant = Assistant()

# Ask for help with AutoML
response = assistant.chat("How do I train a tabular model with AutoGluon?")
print(response)
```

## Interactive Chat

You can have an interactive conversation with the assistant:

```python
# Start a conversation
assistant = Assistant()

# Ask multiple questions
questions = [
    "What is AutoGluon?",
    "How do I prepare my data for training?",
    "What are the best practices for hyperparameter tuning?"
]

for question in questions:
    response = assistant.chat(question)
    print(f"Q: {question}")
    print(f"A: {response}")
    print("-" * 50)
```

## Code Generation

AutoGluon-Assistant can generate code for your specific use cases:

```python
# Request code generation
prompt = "Generate code to train a binary classification model on a CSV file"
code = assistant.generate_code(prompt)
print(code)
```

## Next Steps

Now that you've got the basics down, explore more advanced features:

- Learn about custom configurations (coming soon)
- Explore integration patterns (coming soon)
- Check out real-world examples (coming soon)

## Need Help?

If you run into any issues:

1. Check the [API Reference](../api/index.rst) for detailed documentation
2. Browse the examples for common use cases (coming soon)
3. Visit our [GitHub repository](https://github.com/autogluon/autogluon-assistant) for support
