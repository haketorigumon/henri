"""Configuration defaults for Henri."""

# Default provider
DEFAULT_PROVIDER = "bedrock"

# AWS Bedrock defaults
DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_BEDROCK_REGION = "us-east-1"

# Google Cloud defaults
DEFAULT_GOOGLE_MODEL = "gemini-2.5-flash"
DEFAULT_GOOGLE_LOCATION = "us-central1"

# Ollama defaults
DEFAULT_OLLAMA_MODEL = "qwen3-coder:30b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Vertex AI defaults (for Claude models)
DEFAULT_VERTEX_MODEL = "claude-sonnet-4-5"
DEFAULT_VERTEX_REGION = "us-east5"
