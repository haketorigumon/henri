# Henri

A [pedagogical](TUTORIAL.md) Claude Code clone built in Python.

Henri is a minimal but complete AI coding assistant that demonstrates the core architecture of tools like Claude Code: streaming LLM responses, tool execution, and permission management.

## Features

- **Multiple LLM providers** - AWS Bedrock, Google Gemini, Vertex AI, Ollama (local)
- **Streaming responses** - Real-time token streaming
- **Tool system** - bash, file read/write capabilities
- **Permission management** - Prompts before potentially dangerous operations
- **Clean architecture** - Easy to understand and extend

## Installation

```bash
pip install -e .
brew install ripgrep  # for the grep tool
```

## Usage

```bash
# AWS Bedrock (default)
henri

# Google Gemini
henri --provider google

# Vertex AI
henri --provider vertex

# Ollama (local)
henri --provider ollama
```

### Provider Setup

**AWS Bedrock** (default):
- Configure AWS credentials (`aws configure` or environment variables)
- Ensure access to Claude models in your region

**Google Gemini**:
- Set `GOOGLE_API_KEY` for Google AI API, or
- Set `GOOGLE_CLOUD_PROJECT` for Vertex AI

**Vertex AI**:
- Set `GOOGLE_CLOUD_PROJECT`

**Ollama**:
- Install and run [Ollama](https://ollama.ai) locally
- Pull a model: `ollama pull qwen3-coder:30b`

### Example Session

```
> What files are in the current directory?

▶ bash(command='ls -la')
┌──────────────────────────────────────┐
│ total 16                             │
│ drwxr-xr-x  5 user staff  160 Jan  3 │
│ -rw-r--r--  1 user staff  123 Jan  3 │
│ ...                                  │
└──────────────────────────────────────┘

There are 5 files in the current directory...
```

### Permission System

When Henri wants to execute a tool that requires permission (like `bash` or `write_file`), you'll be prompted:

- `y` - Allow this specific execution
- `n` - Deny this execution
- `a` - Always allow this tool for the session
- `A` - Allow all tools for the session

## Architecture

```
henri/
├── messages.py      # Core data types (Message, ToolCall, ToolResult)
├── providers/
│   ├── base.py      # Provider abstract base class
│   ├── bedrock.py   # AWS Bedrock
│   ├── google.py    # Google Gemini
│   ├── vertex.py    # Vertex AI
│   └── ollama.py    # Ollama
├── tools/
│   └── base.py      # Tool base class + built-in tools
├── permissions.py   # Permission management
├── agent.py         # Main conversation loop
└── cli.py           # Entry point
```

### Adding New Tools

Create a new tool by subclassing `Tool`:

```python
from henri.tools.base import Tool

class MyTool(Tool):
    name = "my_tool"
    description = "Does something useful"
    parameters = {
        "type": "object",
        "properties": {
            "arg1": {"type": "string", "description": "First argument"},
        },
        "required": ["arg1"],
    }
    requires_permission = True  # Set to False for safe operations

    def execute(self, arg1: str) -> str:
        # Your implementation here
        return f"Result: {arg1}"
```

Then add it to the tools list in `agent.py` or pass it when creating the Agent.

### Adding New Providers

Subclass `Provider` and implement the `stream()` method:

```python
from henri.providers.base import Provider, StreamEvent

class MyProvider(Provider):
    name = "my_provider"

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool],
        system: str = "",
    ) -> AsyncIterator[StreamEvent]:
        # Your implementation
        yield StreamEvent(text="Hello")
        yield StreamEvent(stop_reason="end_turn")
```

Then register it in `providers/__init__.py`.

## Configuration

```bash
# Provider selection
henri --provider bedrock|google|vertex|ollama

# Model override
henri --model <model-id>

# Provider-specific options
henri --region us-east-1             # AWS Bedrock
henri --region us-east5              # Vertex AI
henri --host http://localhost:11434  # Ollama

# Limit turns (for benchmarking)
henri --max-turns 10                 # Stop after 10 turns (default: 20)
```

On exit, Henri prints metrics: `Turns: X | Tokens: Y in, Z out`

### Links

- [Supported AWS Bedrock models](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html)
- [Google Cloud Model Garden](https://console.cloud.google.com/vertex-ai/model-garden)
- [Ollama model library](https://ollama.ai/library)

## License

MIT
