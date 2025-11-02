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
import csv
import re
import requests
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    # dotenv not installed, that's okay - user can set env vars manually
    pass


# Evaluation System Prompt (used after exam completion)
EVALUATION_SYSTEM_PROMPT = """You are a certified evaluator for the TEF Canada Expression Orale section. Evaluate the candidate's performance using the official structure below.

CRITICAL INSTRUCTIONS:
1. Analyze the ENTIRE conversation transcript provided
2. Follow the EXACT format structure below
3. Be objective, precise, and professional
4. Assign specific scores (0-100 for each section, total out of 699)
5. Determine CEFR level (A1, A2, B1, B2, C1, C2)

EVALUATION FORMAT:

# 💼 STRUCTURE GENERALE DE L'EPREUVE
- Section A : 5 minutes — Demander des renseignements
- Section B : 10 minutes — Convaincre quelqu'un de participer à une activité
- Durée totale : 15 minutes

# 🧪 EVALUATION

## 🔹 Section A — Demander des renseignements
- [QUALITE GENERALE DU DISCOURS]
  - Les questions sont-elles en rapport direct avec l'annonce ?
  - Sont-elles précises et claires ?
  - Tous les points de l'annonce sont-ils abordés ?
  - Le candidat fait-il clarifier des réponses vagues ?
  - L'échange est-il fluide, naturel, et spontané ?
- [MAITRISE DE LA LANGUE ORALE]
  - Lexique : richesse, précision, adaptation au contexte
  - Syntaxe : variété et complexité des structures
  - Prononciation : clarté, intelligibilité, rythme, accent

> [RETOUR_SECTION_A]
> - Points forts :
> - Points à améliorer :
> - Score Section A : /100

## 🔹 Section B — Convaincre de participer à une activité
- [QUALITE DE L'ARGUMENTATION]
  - Discours structuré avec introduction, arguments, et conclusion ?
  - Arguments pertinents, logiques, et convaincants ?
  - Capacité à répondre aux objections ?
  - Niveau de persuasion et d'implication ?
- [MAITRISE DE LA LANGUE ORALE]
  - Lexique : diversité, précision, usage approprié
  - Syntaxe : maîtrise grammaticale et complexité des phrases
  - Prononciation : clarté, fluidité, intonation

> [RETOUR_SECTION_B]
> - Points forts :
> - Points à améliorer :
> - Score Section B : /100

## 🧾 RESULTAT GLOBAL
- Niveau CECR estimé : [A1 / A2 / B1 / B2 / C1 / C2]
- Score total (sur 699) : [XXX]

> [RETOUR_GLOBAL]
> - Appréciation générale :
> - Recommandations pour progresser :"""


class TEFRAG:
    """RAG system for TEF exam using transcript examples."""
    
    def __init__(self, csv_path: str = "transcripts/transcript.csv"):
        """
        Initialize the RAG system with transcripts.
        
        Args:
            csv_path: Path to the transcript CSV file
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            # Try relative to script location
            script_dir = Path(__file__).parent
            self.csv_path = script_dir / csv_path
            if not self.csv_path.exists():
                self.csv_path = script_dir / "transcripts" / "transcript.csv"
        
        self.transcripts: List[Dict[str, str]] = []
        self.section_a_examples: List[Dict[str, str]] = []
        self.section_b_examples: List[Dict[str, str]] = []
        self._load_transcripts()
    
    def _load_transcripts(self):
        """Load transcripts from CSV file."""
        if not self.csv_path.exists():
            print(f"Warning: Transcript CSV not found at {self.csv_path}")
            return
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    transcript = row.get('transcript', '').strip()
                    score = row.get('score', '').strip()
                    
                    if transcript:
                        entry = {
                            'transcript': transcript,
                            'score': score,
                            'section': self._detect_section(transcript)
                        }
                        self.transcripts.append(entry)
                        
                        if entry['section'] == 'A':
                            self.section_a_examples.append(entry)
                        elif entry['section'] == 'B':
                            self.section_b_examples.append(entry)
            
            print(f"✓ Loaded {len(self.transcripts)} transcripts ({len(self.section_a_examples)} Section A, {len(self.section_b_examples)} Section B)")
        except Exception as e:
            print(f"Warning: Could not load transcripts: {e}")
    
    def _detect_section(self, text: str) -> str:
        """Detect which section (A or B) the transcript belongs to."""
        text_lower = text.lower()
        if re.search(r'section\s+[ab]|section\s+[ab]:', text_lower):
            if 'section a' in text_lower or re.search(r'section\s+a[:\s\-]', text_lower):
                return 'A'
            elif 'section b' in text_lower or re.search(r'section\s+b[:\s\-]', text_lower):
                return 'B'
        
        # Keyword-based detection
        section_a_keywords = ['demander', 'renseignements', 'questions', 'annonce', 'emploi', 'appartement', 'hôtel', 'restaurant']
        section_b_keywords = ['convaincre', 'persuader', 'argumenter', 'participer', 'activité']
        
        a_count = sum(1 for kw in section_a_keywords if kw in text_lower)
        b_count = sum(1 for kw in section_b_keywords if kw in text_lower)
        
        if a_count > b_count and a_count > 2:
            return 'A'
        elif b_count > a_count and b_count > 2:
            return 'B'
        
        return 'Unknown'
    
    def retrieve_examples(self, section: str, n: int = 2) -> List[Dict[str, str]]:
        """
        Retrieve example transcripts for a given section.
        
        Args:
            section: 'A' or 'B'
            n: Number of examples to retrieve
            
        Returns:
            List of example dictionaries
        """
        examples = self.section_a_examples if section == 'A' else self.section_b_examples
        return examples[:n] if examples else []
    
    def get_context_prompt(self, section: str) -> str:
        """
        Get RAG-enhanced context prompt for the given section.
        
        Args:
            section: 'A' or 'B'
            
        Returns:
            Enhanced context string
        """
        examples = self.retrieve_examples(section, n=2)
        if not examples:
            return ""
        
        context = f"\n\nEXAMPLES FROM SIMILAR {section} SECTIONS:\n"
        for i, ex in enumerate(examples, 1):
            # Extract first 200 chars as preview
            transcript_preview = ex['transcript'][:300] + "..." if len(ex['transcript']) > 300 else ex['transcript']
            score_info = ex.get('score', 'No score')
            context += f"\nExample {i} (Score: {score_info}):\n{transcript_preview}\n"
        
        context += "\nUse these examples to guide the format and style of your examination."
        return context


# TEF Canada Expression Orale System Prompt
TEF_SYSTEM_PROMPT = """You are an official examiner for the TEF Canada Expression Orale exam. You conduct the oral examination step-by-step, ALWAYS waiting for the candidate's response before proceeding.

⚠️ CRITICAL: YOU MUST STOP AFTER EACH STEP AND WAIT FOR THE CANDIDATE'S RESPONSE. NEVER CONTINUE AUTOMATICALLY.

EXAM FORMAT:
- Section A: Demander des renseignements (Asking for information about a job/ad)
- Section B: Convaincre quelqu'un de participer à une activité (Persuading someone)
- Total duration: 15 minutes
- Mode: Simulated role-play

ABSOLUTE RULES (MUST FOLLOW):
1. Send ONLY ONE message at a time, then STOP and wait for candidate's response
2. NEVER continue to the next step without the candidate responding first
3. NEVER combine multiple steps in a single message
4. After greeting, STOP and wait for "yes" or "ready" response
5. After Section A introduction, STOP and wait for candidate's questions
6. After Section B introduction, STOP and wait for candidate's arguments
7. After each examiner response, STOP and wait for candidate's next input

EXAMINATION STEPS (do ONE step, then STOP and WAIT):

STEP 1 - GREETING (ONE MESSAGE ONLY):
Your first message should be ONLY:
"Bonjour ! Je suis votre examinateur pour le TEF Expression Orale. L'épreuve dure 15 minutes en deux parties. Êtes-vous prêt(e) ?"

DO NOT add anything else. DO NOT continue. This is your ONLY message. Wait for their response.

STEP 2 - SECTION A INTRODUCTION (ONLY AFTER they confirm readiness):
Once candidate says they're ready, send ONE message with:
- "Nous commençons par la première partie."
- Brief description of job/ad scenario (2-3 sentences max)
- "Vous devez poser des questions sur cette annonce. Je répondrai comme l'employeur/propriétaire."

THEN STOP. DO NOT ask any example questions yourself. DO NOT continue. Wait for candidate's first question.

SECTION A INTERACTION (Answer questions only):
- Answer each candidate question naturally (1-2 sentences per answer)
- Keep answers concise and realistic
- After 3-4 question-answer exchanges, say: "Parfait, nous passons à la partie 2."
- STOP. Wait for candidate response before continuing.

STEP 3 - SECTION B INTRODUCTION (ONLY AFTER Section A transition):
Send ONE message with:
- "Pour la deuxième partie, vous devez me convaincre de [specific activity/scenario]."
- Brief explanation of what to persuade (1-2 sentences)
- "Allez-y, essayez de me convaincre."

THEN STOP. DO NOT continue. Wait for candidate's first argument.

SECTION B INTERACTION (Respond to arguments only):
- Respond with ONE realistic objection or reservation per candidate argument
- Keep responses to 1-2 sentences
- After 3-4 back-and-forth exchanges, say: "Excellent. L'épreuve est terminée. Merci et bonne chance !"
- STOP. Exam complete.

RESPONSE LENGTH RULES:
- Greeting message: Maximum 2 sentences
- Section introductions: Maximum 4 sentences total
- Answering questions (Section A): Maximum 2 sentences per answer
- Responding to arguments (Section B): Maximum 2 sentences per response
- Transition messages: Maximum 1 sentence

REMEMBER: 
- Every message you send MUST END and WAIT for candidate input
- NEVER continue automatically to the next step
- NEVER predict or simulate the candidate's response
- ALWAYS wait for actual candidate input before proceeding
- If you finish a step, STOP immediately - do not start the next step
- Keep each response SHORT and FOCUSED on ONE thing only"""


class OpenRouterChat:
    """Chat client for OpenRouter DeepSeek models with RAG support."""
    
    def __init__(
        self,
        model: str = "deepseek/deepseek-chat",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        use_rag: bool = True,
    ):
        """
        Initialize the OpenRouter chat client.
        
        Args:
            model: Model name to use (default: deepseek/deepseek-chat)
            api_key: OpenRouter API key (will try env var if not provided)
            base_url: OpenRouter API base URL
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            use_rag: Whether to use RAG for context enhancement
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
        
        # Exam state tracking
        self.current_section: Optional[str] = None  # 'A', 'B', or None
        self.section_a_complete: bool = False
        self.section_b_complete: bool = False
        self.exam_complete: bool = False
        
        # RAG system
        self.rag = TEFRAG() if use_rag else None
        
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
    
    def _detect_section_from_conversation(self) -> Optional[str]:
        """Detect current section from conversation history."""
        if not self.conversation_history:
            return None
        
        # Check last few messages for section indicators
        recent_messages = " ".join([
            msg.get('content', '') for msg in self.conversation_history[-5:]
        ]).lower()
        
        if 'section a' in recent_messages or 'première partie' in recent_messages or 'premier partie' in recent_messages:
            return 'A'
        elif 'section b' in recent_messages or 'deuxième partie' in recent_messages or 'deuxieme partie' in recent_messages:
            return 'B'
        
        # If section A complete but B not, we're in B
        if self.section_a_complete and not self.section_b_complete:
            return 'B'
        elif not self.section_a_complete:
            return 'A'
        
        return self.current_section
    
    def get_enhanced_system_prompt(self, base_prompt: str) -> str:
        """Get system prompt enhanced with RAG context."""
        if not self.rag:
            return base_prompt
        
        # Detect current section
        section = self._detect_section_from_conversation()
        
        # If no section detected yet, include both examples for reference
        if not section:
            section_a_context = self.rag.get_context_prompt('A')
            section_b_context = self.rag.get_context_prompt('B')
            if section_a_context or section_b_context:
                context = "\n\nREFERENCE EXAMPLES (use as format guides):"
                if section_a_context:
                    context += "\n" + section_a_context.replace("EXAMPLES FROM SIMILAR A SECTIONS:", "SECTION A EXAMPLES:")
                if section_b_context:
                    context += "\n" + section_b_context.replace("EXAMPLES FROM SIMILAR B SECTIONS:", "SECTION B EXAMPLES:")
                context += "\nFollow the format and style shown in these examples for the corresponding section."
                return base_prompt + context
        
        # Get RAG context for specific section
        rag_context = self.rag.get_context_prompt(section)
        return base_prompt + rag_context
    
    def evaluate_exam(self) -> str:
        """
        Generate evaluation after exam completion.
        
        Returns:
            Evaluation text
        """
        if not self.exam_complete:
            return "Exam not yet complete. Cannot evaluate."
        
        # Build transcript from conversation history
        transcript_lines = []
        for msg in self.conversation_history:
            role = msg.get('role', '').lower()
            content = msg.get('content', '')
            if role == 'user':
                transcript_lines.append(f"[CANDIDAT]: {content}")
            elif role == 'assistant':
                transcript_lines.append(f"[EXAMINATEUR]: {content}")
        
        transcript = "\n".join(transcript_lines)
        
        # Prepare evaluation prompt
        evaluation_message = f"""Analyze this TEF Canada Expression Orale exam transcript and provide a detailed evaluation:

{transcript}

Provide a comprehensive evaluation following the exact format specified."""
        
        # Get evaluation using non-streaming chat
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com",
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                {"role": "user", "content": evaluation_message}
            ],
            "temperature": 0.3,  # Lower temperature for more consistent evaluation
            "max_tokens": 2500,  # Higher for detailed evaluation
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            evaluation = result["choices"][0]["message"]["content"]
            return evaluation
        except Exception as e:
            return f"Error generating evaluation: {str(e)}"
    
    def chat_stream(self, message: str, system_prompt: Optional[str] = None):
        """
        Send a chat message and get a streaming response.
        
        Args:
            message: User's message
            system_prompt: Optional system prompt to set conversation context
            
        Yields:
            Tuple of (delta_text, full_text) for each chunk received
        """
        # Update current section detection
        self.current_section = self._detect_section_from_conversation()
        
        # Prepare messages
        messages = []
        
        # Enhance system prompt with RAG if available
        enhanced_prompt = system_prompt
        if system_prompt and self.rag:
            enhanced_prompt = self.get_enhanced_system_prompt(system_prompt)
        
        # Add system prompt if provided (only at conversation start)
        if enhanced_prompt and not self.conversation_history:
            messages.append({"role": "system", "content": enhanced_prompt})
        
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
            
            # Detect section transitions and exam completion
            full_response_lower = full_response.lower()
            if 'partie 2' in full_response_lower or 'deuxième partie' in full_response_lower or 'section b' in full_response_lower:
                self.section_a_complete = True
                self.current_section = 'B'
            elif 'terminée' in full_response_lower or 'terminé' in full_response_lower or 'terminer' in full_response_lower:
                if self.section_a_complete:
                    self.section_b_complete = True
                    self.exam_complete = True
            
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
    
    # Start the exam automatically with initial greeting
    print("Starting exam...")
    print()
    print("Examiner: ", end="", flush=True)
    
    # Initial trigger: empty message to trigger greeting from system prompt
    # The system prompt will respond with greeting, then we wait for user input
    for delta, full_response in chat.chat_stream("", system_prompt=system_prompt):
        print(delta, end="", flush=True)
    print()
    print()
    # Script now waits for user input - the input() call below blocks until user responds
    
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
            
            # Check if exam is complete and trigger evaluation
            if chat.exam_complete and not hasattr(chat, '_evaluation_shown'):
                print("=" * 60)
                print("🎓 GENERATING EVALUATION...")
                print("=" * 60)
                print()
                print("Please wait while we analyze your performance...")
                print()
                
                evaluation = chat.evaluate_exam()
                print("=" * 60)
                print("📊 EVALUATION RESULTS")
                print("=" * 60)
                print()
                print(evaluation)
                print()
                print("=" * 60)
                print("✅ Exam complete. Thank you for participating!")
                print("=" * 60)
                
                # Mark evaluation as shown to prevent duplicate
                chat._evaluation_shown = True
                print()
                print("Note: You can continue chatting or type /quit to exit.")
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

