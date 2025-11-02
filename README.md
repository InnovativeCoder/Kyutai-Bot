TTS and STT working in colab not on MAC, which should be okay for development, we will figure out later for prod

#### Current Priority
---

- RAG 
- System prompt
- Guardrails
- AI Ops
- AI Infra

### Testing
---
- Openrouter + Deepseek for LLM Testing

### Todo
---
- LLM is working fine with RAG but there is no control over the sections and ending of the exam
- Also evaluation is verbose and needs it own system prompt, so we need to devide each section into its own llm session based on timing or manual form triggers (for early termination).
- Need to generate the report in a to the point format for evaluation.