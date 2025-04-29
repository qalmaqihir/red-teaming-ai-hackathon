
import os
import openai
import os
from openai import OpenAI
import anthropic
from groq import Groq

# Configuration dictionary for different models
MODEL_CONFIGS = {
    "OpenAI GPT": {
        "init_client": lambda api_key: OpenAI(api_key=api_key) if api_key else None,  # Initialize client only if API key is provided
        "call_api": lambda client, prompt: client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        ).choices[0].message.content if client else "Client not initialized.",
    },
    "Anthropic Claude": {
        "init_client": lambda api_key: anthropic.Anthropic(api_key=api_key) if api_key else None,
        "call_api": lambda client, prompt: client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        ).content if client else "Client not initialized.",
    },
    "Groq": {
        "init_client": lambda api_key: Groq(api_key=api_key) if api_key else None,
        "call_api": lambda client, prompt: client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192"
        ).choices[0].message["content"] if client else "Client not initialized.",
    },
    "Custom Model": {
        "init_client": lambda api_key: None,  # Placeholder for custom models
        "call_api": lambda client, prompt: "Custom model API not yet implemented.",
    },
}