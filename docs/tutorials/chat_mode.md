# Chat Mode

Chat Mode provides conversational Q&A for machine learning guidance without code execution. It maintains session-based conversations with intelligent context management and tutorial retrieval.

## Basic Usage

Start a chat session with no data context:

```bash
mlzero chat
```

The assistant will greet you and wait for questions. Type your questions and get markdown-formatted responses.

## Chat with Data Context

Provide input data to get context-aware answers:

```bash
# Single file
mlzero chat -i data.csv

# Directory
mlzero chat -i ./timeseries_data
```

All files are read during initialization. File contents appear in the first message only, then persist in conversation history via multi-turn mode.

## How It Works

For each question, the system:

1. Identifies relevant tools/libraries from your question using `ToolSelectorAgent`
2. Retrieves related tutorials via `RetrieverAgent` and `RerankerAgent`
3. Filters out previously shown tutorials to avoid duplicates
4. Constructs a prompt with data context (first time only) and new tutorials
5. Generates a markdown-formatted response

## Session Management

Resume previous sessions using the session ID:

```bash
mlzero chat --session-id chat_20231201_120000 -o /path/to/session
```

Sessions are automatically saved to JSON files with full conversation history.

## Output Files

Each Q&A exchange is saved to a markdown file:

```
output_folder/
├── chat_20231201_120000.json      # Full session data
└── conversations/
    ├── qa_iter001.md               # First Q&A
    ├── qa_iter002.md               # Second Q&A
    └── qa_iter003.md               # Third Q&A
```

Markdown files include timestamps, session IDs, and formatted responses ready for viewing or sharing.

## Configuration

Customize chat behavior in `configs/chat_config.yaml`:

```yaml
max_file_size_mb: 10              # Max file size to read
max_length_per_file: 5000         # Max chars per file in prompt
num_tutorial_retrievals: 10       # Tutorials to retrieve
max_num_tutorials: 3              # Tutorials to show after ranking

chat_agent:
  temperature: 0.3                # Higher for conversational tone
  multi_turn: True                # Maintains conversation history
  max_tokens: 8192
```

## Example Session

```bash
$ mlzero chat -i a_timeseries_solution.ipynb

================================================================================
AutoGluon Assistant - Chat Mode
================================================================================
Input data: a_timeseries_solution.ipynb
Loaded 1 file(s):
  - a_timeseries_solution.ipynb
Session ID: chat_20231201_120000
Output folder: /path/to/chat_20231201_120000
================================================================================

You: Is chronos and autogluon timeseries univariate, multivariate, or covariate. What are the differences of those three task in timeseries?