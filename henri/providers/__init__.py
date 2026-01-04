"""LLM providers for Henri."""

from henri.config import DEFAULT_PROVIDER

from .base import Provider, StreamEvent
from .bedrock import BedrockProvider
from .google import GoogleProvider
from .ollama import OllamaProvider
from .vertex import VertexProvider

# Registry of available providers
PROVIDERS: dict[str, type[Provider]] = {
    "bedrock": BedrockProvider,
    "google": GoogleProvider,
    "ollama": OllamaProvider,
    "vertex": VertexProvider,
}


def create_provider(name: str = DEFAULT_PROVIDER, **kwargs) -> Provider:
    """Create a provider instance by name.

    Args:
        name: Provider name ("bedrock", "google", "ollama")
        **kwargs: Provider-specific arguments (model_id, region, host, etc.)

    Returns:
        Configured provider instance

    Raises:
        ValueError: If provider name is unknown
    """
    if name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")

    return PROVIDERS[name](**kwargs)


__all__ = [
    "Provider",
    "StreamEvent",
    "BedrockProvider",
    "GoogleProvider",
    "OllamaProvider",
    "VertexProvider",
    "PROVIDERS",
    "create_provider",
]
