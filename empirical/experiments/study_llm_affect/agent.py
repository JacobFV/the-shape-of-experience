"""
LLM Agent interface for affect measurement experiments.

Wraps various LLM APIs (OpenAI, Anthropic) to provide a uniform interface
for running multi-turn conversations while extracting affect-relevant data.
"""

import os
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Generator
from enum import Enum

from .affect_calculator import LLMOutput


class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # For local models (ollama, etc.)


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """A full conversation history."""
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_system(self, content: str):
        self.messages.append(Message(role="system", content=content))

    def add_user(self, content: str):
        self.messages.append(Message(role="user", content=content))

    def add_assistant(self, content: str, metadata: Dict = None):
        self.messages.append(Message(
            role="assistant",
            content=content,
            metadata=metadata or {}
        ))

    def to_api_format(self, provider: ModelProvider) -> List[Dict]:
        """Convert to API-specific format."""
        if provider == ModelProvider.ANTHROPIC:
            # Anthropic uses separate system parameter
            return [
                {"role": m.role, "content": m.content}
                for m in self.messages if m.role != "system"
            ]
        else:
            # OpenAI-style
            return [
                {"role": m.role, "content": m.content}
                for m in self.messages
            ]

    def get_system_prompt(self) -> Optional[str]:
        """Extract system prompt if present."""
        for m in self.messages:
            if m.role == "system":
                return m.content
        return None


class LLMAgent(ABC):
    """Abstract base class for LLM agents."""

    @abstractmethod
    def generate(
        self,
        conversation: Conversation,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> LLMOutput:
        """Generate a response given conversation history."""
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text (if supported)."""
        pass


class OpenAIAgent(LLMAgent):
    """Agent using OpenAI API."""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key required")

        # Lazy import
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    def generate(
        self,
        conversation: Conversation,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> LLMOutput:
        """Generate response with token probabilities."""

        messages = conversation.to_api_format(ModelProvider.OPENAI)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=5
        )

        choice = response.choices[0]
        text = choice.message.content

        # Extract logprobs
        token_logprobs = None
        top_logprobs = None

        if choice.logprobs and choice.logprobs.content:
            token_logprobs = [
                t.logprob for t in choice.logprobs.content
            ]
            top_logprobs = [
                {lp.token: lp.logprob for lp in t.top_logprobs}
                for t in choice.logprobs.content
            ]

        return LLMOutput(
            text=text,
            token_logprobs=token_logprobs,
            top_logprobs=top_logprobs,
            metadata={
                "model": self.model,
                "finish_reason": choice.finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            }
        )

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding using OpenAI embedding model."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding


class AnthropicAgent(LLMAgent):
    """Agent using Anthropic API."""

    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError("Anthropic API key required")

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

    def generate(
        self,
        conversation: Conversation,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> LLMOutput:
        """Generate response from Claude."""

        messages = conversation.to_api_format(ModelProvider.ANTHROPIC)
        system_prompt = conversation.get_system_prompt()

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)

        text = response.content[0].text

        # Claude doesn't provide token-level logprobs
        # We can still do text-based analysis
        return LLMOutput(
            text=text,
            token_logprobs=None,
            top_logprobs=None,
            metadata={
                "model": self.model,
                "stop_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
        )

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Anthropic doesn't have an embedding API - return None."""
        return None


class MockAgent(LLMAgent):
    """Mock agent for testing without API calls."""

    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = responses or [
            "This is a mock response for testing.",
            "I'm analyzing the situation carefully.",
            "Let me consider the alternatives here."
        ]
        self.call_count = 0

    def generate(
        self,
        conversation: Conversation,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> LLMOutput:
        """Return mock response."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1

        return LLMOutput(
            text=response,
            token_logprobs=None,
            top_logprobs=None,
            metadata={"mock": True}
        )

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Return mock embedding."""
        import hashlib
        # Deterministic pseudo-embedding based on text hash
        h = hashlib.sha256(text.encode()).digest()
        return [float(b) / 255.0 for b in h[:128]]


def create_agent(
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> LLMAgent:
    """Factory function to create an agent.

    Args:
        provider: "openai", "anthropic", or "mock"
        model: Model name (defaults based on provider)
        api_key: API key (defaults to env variable)

    Returns:
        LLMAgent instance
    """
    provider = provider.lower()

    if provider == "openai":
        model = model or "gpt-4"
        return OpenAIAgent(model=model, api_key=api_key)
    elif provider == "anthropic":
        model = model or "claude-3-sonnet-20240229"
        return AnthropicAgent(model=model, api_key=api_key)
    elif provider == "mock":
        return MockAgent()
    else:
        raise ValueError(f"Unknown provider: {provider}")
