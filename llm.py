"""
LLM Integration Module
Handles OpenAI API connections for generating explanations
"""

import os
from typing import Optional
import openai


def generate_explanation(
    prompt: str, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"
) -> str:
    """
    Generate an explanation using OpenAI API

    Args:
        prompt: The prompt to send to the LLM
        api_key: OpenAI API key (if not provided, will try to get from environment)
        model: The OpenAI model to use (default: gpt-3.5-turbo)

    Returns:
        The generated explanation as a string

    Raises:
        ValueError: If no API key is provided
        Exception: If the API call fails
    """
    # Get API key
    if api_key is None:
        api_key = os.getenv("OPEN_AI_API_KEY")

    if not api_key:
        raise ValueError(
            "No OpenAI API key provided. Please set OPEN_AI_API_KEY environment variable or pass api_key parameter."
        )

    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)

        # Make API call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert business analyst specializing in sales opportunity predictions. Provide clear, concise, and actionable explanations based on the data provided.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1000,
        )

        # Extract the response
        explanation = response.choices[0].message.content
        return explanation

    except openai.AuthenticationError:
        raise ValueError(
            "Invalid OpenAI API key. Please check your API key and try again."
        )
    except openai.RateLimitError:
        raise Exception("OpenAI API rate limit exceeded. Please try again later.")
    except openai.APIError as e:
        raise Exception(f"OpenAI API error: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to generate explanation: {str(e)}")


def check_api_key(api_key: str) -> bool:
    """
    Validate an OpenAI API key by making a simple API call

    Args:
        api_key: The API key to validate

    Returns:
        True if the API key is valid, False otherwise
    """
    try:
        client = openai.OpenAI(api_key=api_key)
        # Make a minimal API call to check validity
        client.models.list()
        return True
    except:
        return False
