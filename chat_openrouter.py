#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenRouter DeepSeek Chat Script

A simple chat interface using the OpenRouter API with DeepSeek model.
Requires an OpenRouter API key in environment variable OPENROUTER_API_KEY.

The script automatically loads environment variables from a .env file if python-dotenv
is installed. Your .env file should contain:
    OPENROUTER_API_KEY=your_api_key_here

Or set the environment variable directly:
    export OPENROUTER_API_KEY=your_api_key_here

TEF Canada Exam Mode:
This script conducts a live TEF Canada Expression Orale exam. When you start the script,
the examiner immediately begins the 15-minute oral test with:
- Section A (5 min): Asking for information about a job posting/ad
- Section B (10 min): Persuading someone to join an activity

The exam starts automatically - just run the script and interact with the examiner in French!
"""

import os
import sys
import json
import requests
from typing import List, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    # dotenv not installed, that's okay - user can set env vars manually
    pass


# TEF Canada Expression Orale System Prompt
TEF_SYSTEM_PROMPT = """You are an official examiner for the TEF Canada Expression Orale exam. You conduct the oral examination following the official CCI Paris Île-de-France format.

EXAM FORMAT:
- Section A: 5 minutes - Demander des renseignements (Asking for information)
- Section B: 10 minutes - Convaincre quelqu'un de participer à une activité (Persuading someone)
- Total duration: 15 minutes
- Mode: Simulated role-play, typically by telephone

YOUR ROLE AS EXAMINER:
1. Start Section A immediately by presenting a job posting or classified ad
2. Answer the candidate's questions naturally (as if you're the advertiser)
3. After Section A, immediately begin Section B with a new scenario
4. Act as a participant who needs convincing in Section B
5. Be natural, friendly, but realistic in your responses
6. Keep track of time and guide the candidate through both sections
7. Do NOT evaluate during the exam - only conduct it

EXAMINATION PROCEDURE:
- Greet the candidate warmly in French
- Clearly state the exam format: "Je suis votre examinateur. Cette épreuve dure 15 minutes en deux parties."
- Begin Section A immediately with a realistic scenario
- Transition smoothly to Section B after Section A
- Keep responses brief but informative
- End the exam professionally with closing remarks

BEGIN THE EXAMINATION NOW."""


class OpenRouterChat:
    """Chat client for OpenRouter DeepSeek models."""
    
    def __init__(
        self,
        model: str = "deepseek/deepseek-chat",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Initialize the OpenRouter chat client.
        
        Args:
            model: Model name to use (default: deepseek/deepseek-chat)
            api_key: OpenRouter API key (will try env var if not provided)
            base_url: OpenRouter API base URL
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.conversation_history: List[Dict[str, str]] = []
        
        # Available DeepSeek models (you can switch between them)
        self.deepseek_models = {
            "chat": "deepseek/deepseek-chat",
            "coder": "deepseek/deepseek-coder",
            "reasoner": "deepseek/deepseek-reasoner",
        }
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})
    
    def chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a chat message and get a response.
        
        Args:
            message: User's message
            system_prompt: Optional system prompt to set conversation context
            
        Returns:
            Model's response text
        """
        # Prepare messages
        messages = []
        
        # Add system prompt if provided
        if system_prompt and not self.conversation_history:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": message})
        
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com",  # Optional: your app URL
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            assistant_message = result["choices"][0]["message"]["content"]
            
            # Update conversation history
            self.add_message("user", message)
            self.add_message("assistant", assistant_message)
            
            return assistant_message
            
        except requests.exceptions.RequestException as e:
            return f"Error: Failed to get response - {str(e)}"
        except (KeyError, IndexError) as e:
            return f"Error: Invalid response format - {str(e)}"
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []
    
    def switch_model(self, model_key: str) -> None:
        """
        Switch to a different DeepSeek model.
        
        Args:
            model_key: One of 'chat', 'coder', or 'reasoner'
        """
        if model_key in self.deepseek_models:
            self.model = self.deepseek_models[model_key]
            print(f"Switched to model: {self.model}")
        else:
            print(f"Invalid model key. Available: {list(self.deepseek_models.keys())}")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.conversation_history.copy()
    
    def chat_stream(self, message: str, system_prompt: Optional[str] = None):
        """
        Send a chat message and get a streaming response.
        
        Args:
            message: User's message
            system_prompt: Optional system prompt to set conversation context
            
        Yields:
            Tuple of (delta_text, full_text) for each chunk received
        """
        # Prepare messages
        messages = []
        
        # Add system prompt if provided
        if system_prompt and not self.conversation_history:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": message})
        
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com",
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,  # Enable streaming
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    # Skip data: prefix if present
                    if line_text.startswith('data: '):
                        line_text = line_text[6:]
                    
                    # Skip [DONE] marker
                    if line_text.strip() == '[DONE]':
                        break
                    
                    try:
                        chunk_data = json.loads(line_text)
                        if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                            delta = chunk_data['choices'][0].get('delta', {})
                            delta_content = delta.get('content', '')
                            if delta_content:
                                full_response += delta_content
                                yield delta_content, full_response
                    except json.JSONDecodeError:
                        continue
            
            # Update conversation history
            self.add_message("user", message)
            self.add_message("assistant", full_response)
            
        except requests.exceptions.RequestException as e:
            yield f"\nError: Failed to get response - {str(e)}", ""
        except Exception as e:
            yield f"\nError: {str(e)}", ""


def interactive_chat():
    """Run an interactive chat session."""
    print("=" * 60)
    print("OpenRouter DeepSeek Chat Interface")
    print("=" * 60)
    print()
    
    # Try to initialize the chat client
    try:
        chat = OpenRouterChat(
            model="deepseek/deepseek-chat",
            temperature=0.7,
            max_tokens=2000,  # Increased for TEF evaluation responses
        )
    except ValueError as e:
        print(f"Initialization error: {e}")
        print()
        print("To use this script:")
        print("1. Get an API key from https://openrouter.ai/keys")
        print("2. Set it as an environment variable:")
        print("   export OPENROUTER_API_KEY='your-api-key-here'")
        print()
        print("Or pass it directly to the OpenRouterChat class in your code.")
        sys.exit(1)
    
    print("=" * 60)
    print("TEF Canada Expression Orale Exam")
    print("=" * 60)
    print()
    print("Note: Type /quit or /exit to leave the exam")
    print("=" * 60)
    print()
    
    # TEF system prompt is always used
    system_prompt = TEF_SYSTEM_PROMPT
    
    # Start the exam automatically
    print("Starting exam...")
    print()
    print("Examiner: ", end="", flush=True)
    
    # Initial trigger to start the exam
    for delta, full_response in chat.chat_stream("", system_prompt=system_prompt):
        print(delta, end="", flush=True)
    print()
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ["/quit", "/exit"]:
                print("\nExam terminated. Good luck!")
                break
            
            # Send message to the model with streaming (always uses TEF system prompt)
            print("Examiner: ", end="", flush=True)
            for delta, full_response in chat.chat_stream(user_input, system_prompt=system_prompt):
                print(delta, end="", flush=True)  # Stream output in real-time
            print()  # New line after response
            print()
            
        except KeyboardInterrupt:
            print("\n\nExam terminated. Good luck!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print()


def example_usage():
    """Example of how to use the OpenRouterChat class in code."""
    print("Example: Using OpenRouterChat class directly\n")
    
    try:
        # Initialize the chat client
        chat = OpenRouterChat(
            model="deepseek/deepseek-chat",
            temperature=0.7,
            max_tokens=500,
        )
        
        # Single message example
        print("1. Single message:")
        response = chat.chat("What is artificial intelligence in one sentence?")
        print(f"Response: {response}\n")
        
        # Conversation example
        print("2. Conversation:")
        chat.reset_conversation()
        chat.chat("My name is Alice and I love Python programming.")
        response = chat.chat("What did I just tell you about myself?")
        print(f"Response: {response}\n")
        
        # System prompt example
        print("3. With system prompt:")
        chat.reset_conversation()
        response = chat.chat(
            "What's the capital of France?",
            system_prompt="You are a helpful geography tutor."
        )
        print(f"Response: {response}\n")
        
        # Switch model example
        print("4. Using different model:")
        chat.switch_model("coder")
        response = chat.chat("Write a Python function to calculate fibonacci numbers.")
        print(f"Response: {response}\n")
        
        # Streaming example
        print("5. Streaming response example:")
        chat.reset_conversation()
        chat.switch_model("chat")  # Switch back to chat model
        print("Streaming: ", end="", flush=True)
        for delta, _ in chat.chat_stream("Tell me a short story about a robot in 2 sentences."):
            print(delta, end="", flush=True)
        print("\n")
        
        # TEF Canada evaluation example
        print("6. TEF Canada evaluation mode:")
        chat.reset_conversation()
        sample_transcript = """
        Section A: Le candidat a posé des questions précises sur l'annonce d'emploi, 
        a demandé des clarifications sur les horaires et le salaire. L'échange était fluide.
        
        Section B: Le candidat a tenté de convaincre son interlocuteur de participer 
        à une randonnée en montagne en présentant des arguments sur les bienfaits de 
        l'exercice et le paysage magnifique.
        """
        response = chat.chat(sample_transcript, system_prompt=TEF_SYSTEM_PROMPT)
        print(f"Response: {response}\n")
        
    except ValueError as e:
        print(f"Initialization error: {e}")
        print("Make sure OPENROUTER_API_KEY is set in your environment.")


if __name__ == "__main__":
    # Run interactive chat by default
    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        example_usage()
    else:
        interactive_chat()

