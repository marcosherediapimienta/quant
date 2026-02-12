"""
Sistema de memoria para conversaciones del chatbot
"""
from typing import List, Dict, Optional
from datetime import datetime


class ConversationMemory:
    """
    Gestiona el historial de conversación y contexto
    """
    
    def __init__(self, max_messages: int = 20):
        """
        Args:
            max_messages: Número máximo de mensajes a mantener en memoria
        """
        self.max_messages = max_messages
        self.messages: List[Dict] = []
        self.session_start = datetime.now()
        self.metadata: Dict = {}
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Añade un mensaje al historial
        
        Args:
            role: 'user' o 'assistant'
            content: Contenido del mensaje
            metadata: Información adicional (ej: código citado, fuentes)
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.messages.append(message)
        
        # Mantener solo los últimos max_messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self, last_n: Optional[int] = None) -> List[Dict]:
        """
        Obtiene mensajes del historial
        
        Args:
            last_n: Número de mensajes recientes a devolver (None = todos)
        """
        if last_n:
            return self.messages[-last_n:]
        return self.messages
    
    def get_context_string(self, last_n: int = 5) -> str:
        """
        Obtiene el contexto de la conversación como string
        
        Args:
            last_n: Número de mensajes recientes a incluir
        """
        recent_messages = self.get_messages(last_n)
        context_lines = []
        
        for msg in recent_messages:
            role_label = "Usuario" if msg['role'] == 'user' else "Asistente"
            context_lines.append(f"{role_label}: {msg['content']}")
        
        return "\n".join(context_lines)
    
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
        Convierte el historial al formato de LangChain
        
        Returns:
            Lista de dicts con formato {'role': ..., 'content': ...}
        """
        return [
            {'role': msg['role'], 'content': msg['content']}
            for msg in self.messages
        ]
