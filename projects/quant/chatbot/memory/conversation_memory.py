"""
Sistema de memoria para conversaciones del chatbot.

Mejoras:
- Truncado de mensajes largos en get_context_string() para no exceder contexto LLM
- Formato de historial más limpio y legible para el prompt
"""
from typing import List, Dict, Optional
from datetime import datetime

# Longitud máxima por mensaje al incluirlo en el contexto del prompt
MAX_CONTEXT_MSG_LENGTH = 500


class ConversationMemory:
    """
    Gestiona el historial de conversación y contexto para el ChatEngine.
    """

    def __init__(self, max_messages: int = 20):
        """
        Args:
            max_messages: Número máximo de mensajes a mantener en memoria (FIFO)
        """
        self.max_messages = max_messages
        self.messages: List[Dict] = []
        self.session_start = datetime.now()
        self.metadata: Dict = {}

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Añade un mensaje al historial.

        Args:
            role: 'user' o 'assistant'
            content: Contenido del mensaje
            metadata: Información adicional (ej: fuentes citadas)
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        self.messages.append(message)

        # FIFO: mantener solo los últimos max_messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self, last_n: Optional[int] = None) -> List[Dict]:
        """
        Obtiene mensajes del historial.

        Args:
            last_n: Número de mensajes recientes a devolver (None = todos)
        """
        if last_n:
            return self.messages[-last_n:]
        return self.messages

    def get_context_string(self, last_n: int = 10) -> str:
        """
        Obtiene el contexto de la conversación como string formateado para el prompt.
        Los mensajes muy largos se truncan para no exceder el contexto del LLM.

        Args:
            last_n: Número de mensajes recientes a incluir (se cuentan por turno)

        Returns:
            String con el historial formateado o vacío si no hay mensajes previos suficientes
        """
        # Tomar last_n mensajes, pero excluir el último (que acaba de añadirse y es la pregunta actual)
        relevant = self.messages[:-1]  # Excluir el último mensaje (la pregunta actual del usuario)
        recent_messages = relevant[-last_n:] if len(relevant) >= 1 else []

        if not recent_messages:
            return ""

        context_lines = []
        for msg in recent_messages:
            role_label = "Usuario" if msg['role'] == 'user' else "WarrenAI"
            content = msg['content']

            # Truncar mensajes muy largos para no saturar el prompt
            if len(content) > MAX_CONTEXT_MSG_LENGTH:
                content = content[:MAX_CONTEXT_MSG_LENGTH] + "... [truncado]"

            context_lines.append(f"**{role_label}:** {content}")

        return "\n\n".join(context_lines)

    def clear(self):
        """Limpia el historial de mensajes"""
        self.messages = []
        self.session_start = datetime.now()

    def set_metadata(self, key: str, value):
        """Guarda metadata de la sesión"""
        self.metadata[key] = value

    def get_metadata(self, key: str, default=None):
        """Obtiene metadata de la sesión"""
        return self.metadata.get(key, default)

    def get_session_duration(self) -> float:
        """Retorna duración de la sesión en segundos"""
        return (datetime.now() - self.session_start).total_seconds()

    def to_langchain_format(self) -> List[Dict[str, str]]:
        """
        Convierte el historial al formato estándar de LangChain.

        Returns:
            Lista de dicts {'role': ..., 'content': ...}
        """
        return [
            {'role': msg['role'], 'content': msg['content']}
            for msg in self.messages
        ]

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return f"ConversationMemory(messages={len(self.messages)}, max={self.max_messages})"
