# Condensed: ---

Summary: This tutorial covers the implementation of Qwen3-0.6B, a 0.6B parameter language model with 32,768 context length. It demonstrates how to load and use the model with Transformers, focusing on its unique thinking/non-thinking mode capabilities. The guide provides code for basic inference, mode switching, deployment via API endpoints (SGLang/vLLM), and agentic use with tools. Key functionalities include parsing thinking content, managing conversation history, and optimizing generation parameters. The tutorial offers best practices for sampling parameters, output length management, and format standardization for different tasks like math problems and multiple-choice questions.

*This is a condensed version that preserves essential implementation details and context.*

# Qwen3-0.6B Implementation Guide

## Model Overview

**Qwen3-0.6B** is a causal language model with:
- 0.6B parameters (0.44B non-embedding)
- 28 layers
- 16 attention heads for Q, 8 for KV (GQA)
- 32,768 context length
- Unique capability to switch between thinking and non-thinking modes

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    torch_dtype="auto",
    device_map="auto"
)

# Prepare input
prompt = "Give me a short introduction to large language model."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # Toggle thinking mode
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate response
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# Parse thinking content
try:
    index = len(output_ids) - output_ids[::-1].index(151668)  # Find </think> token
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
```

## Deployment Options

### API Endpoints
```shell
# SGLang
python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --reasoning-parser qwen3

# vLLM
vllm serve Qwen/Qwen3-0.6B --enable-reasoning --reasoning-parser deepseek_r1
```

## Thinking vs Non-Thinking Mode

### Thinking Mode (`enable_thinking=True`)
- Default mode
- Generates reasoning in `<think>...</think>` blocks
- Recommended parameters: `Temperature=0.6`, `TopP=0.95`, `TopK=20`, `MinP=0`
- **Warning**: Do not use greedy decoding (can cause endless repetitions)

### Non-Thinking Mode (`enable_thinking=False`)
- Disables reasoning blocks for more efficient responses
- Recommended parameters: `Temperature=0.7`, `TopP=0.8`, `TopK=20`, `MinP=0`

### Dynamic Mode Switching

```python
# Switch modes via user input with /think or /no_think
class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        
        return response
```

## Agentic Use

```python
from qwen_agent.agents import Assistant

# Define LLM configuration
llm_cfg = {
    'model': 'Qwen3-0.6B',
    'model_server': 'http://localhost:8000/v1',  # Custom endpoint
    'api_key': 'EMPTY',
}

# Define tools
tools = [
    {'mcpServers': {
        'time': {
            'command': 'uvx',
            'args': ['mcp-server-time', '--local-timezone=Asia/Shanghai']
        },
        "fetch": {
            "command": "uvx",
            "args": ["mcp-server-fetch"]
        }
    }},
    'code_interpreter',  # Built-in tool
]

# Create agent
bot = Assistant(llm=llm_cfg, function_list=tools)

# Generate response
messages = [{'role': 'user', 'content': 'https://qwenlm.github.io/blog/ Introduce the latest developments of Qwen'}]
for responses in bot.run(messages=messages):
    pass
```

## Best Practices

1. **Sampling Parameters**:
   - Thinking mode: `Temperature=0.6`, `TopP=0.95`, `TopK=20`, `MinP=0`
   - Non-thinking mode: `Temperature=0.7`, `TopP=0.8`, `TopK=20`, `MinP=0`
   - Set `presence_penalty` between 0-2 to reduce repetitions (may cause language mixing)

2. **Output Length**: Use 32,768 tokens for most queries; 38,912 for complex problems

3. **Format Standardization**:
   - Math problems: Include "Please reason step by step, and put your final answer within \boxed{}."
   - Multiple-choice: Add JSON structure for standardized responses

4. **History Management**: Don't include thinking content in conversation history