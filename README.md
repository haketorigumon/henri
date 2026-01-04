# Henri

A pedagogical Claude Code clone built in Python.

Henri is a minimal but complete AI coding assistant that demonstrates the core architecture of tools like Claude Code: streaming LLM responses, tool execution, and permission management.

## Features

- **Streaming responses** from Claude via AWS Bedrock
- **Tool system** with bash, file read/write capabilities
- **Permission management** - prompts before potentially dangerous operations
- **Clean architecture** - easy to understand and extend

## Installation

```bash
pip install -e .
```

## Usage

```bash
henri
```

Or:

```bash
python -m henri.cli
```

### Requirements

- Python 3.11+
- AWS credentials configured with Bedrock access
- Access to Claude models in your AWS region

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
│   └── bedrock.py   # AWS Bedrock provider with streaming
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

Implement a provider with a `stream()` method that yields `StreamEvent` objects:

```python
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

## Configuration

Command-line options:

```bash
henri --model anthropic.claude-sonnet-4-20250514-v1:0 --region us-east-1
```

- [supported AWS models](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html)

## License

MIT
