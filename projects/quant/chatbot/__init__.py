"""
Módulo de chatbot con RAG para análisis financiero cuantitativo

Este módulo proporciona un asistente conversacional que puede:
- Responder preguntas sobre finanzas cuantitativas
- Explicar cómo se calculan métricas en la aplicación
- Buscar en el código fuente para dar respuestas precisas
- Mantener contexto de conversación
"""

from .chat_engine import ChatEngine
from .memory.conversation_memory import ConversationMemory
from .code_indexer import CodeIndexer
from .prompts.finance_prompts import (
    WELCOME_MESSAGE,
    EXAMPLE_QUESTIONS,
    SYSTEM_PROMPT
)

__version__ = "0.1.0"

__all__ = [
    'ChatEngine',
    'ConversationMemory',
    'CodeIndexer',
    'WELCOME_MESSAGE',
    'EXAMPLE_QUESTIONS',
    'SYSTEM_PROMPT'
]
