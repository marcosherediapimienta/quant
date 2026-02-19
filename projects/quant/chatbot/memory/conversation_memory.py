from typing import List, Dict, Optional
from datetime import datetime

try:
    from ..tools.config import MAX_CONTEXT_MSG_LENGTH
except ImportError:
    from tools.config import MAX_CONTEXT_MSG_LENGTH

class ConversationMemory:
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages: List[Dict] = []
        self.session_start = datetime.now()
        self.metadata: Dict = {}

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        self.messages.append(message)

        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self, last_n: Optional[int] = None) -> List[Dict]:

        if last_n:
            return self.messages[-last_n:]
        return self.messages

    def get_context_string(self, last_n: int = 10) -> str:
        relevant = self.messages[:-1]  
        recent_messages = relevant[-last_n:] if len(relevant) >= 1 else []

        if not recent_messages:
            return ""

        context_lines = []
        for msg in recent_messages:
            role_label = "User" if msg['role'] == 'user' else "WarrenAI"
            content = msg['content']

            if len(content) > MAX_CONTEXT_MSG_LENGTH:
                content = content[:MAX_CONTEXT_MSG_LENGTH] + "... [truncated]"

            context_lines.append(f"**{role_label}:** {content}")

        return "\n\n".join(context_lines)

    def clear(self):
        self.messages = []
        self.session_start = datetime.now()

    def set_metadata(self, key: str, value):
        self.metadata[key] = value

    def get_metadata(self, key: str, default=None):
        return self.metadata.get(key, default)

    def get_session_duration(self) -> float:
        return (datetime.now() - self.session_start).total_seconds()

    def to_langchain_format(self) -> List[Dict[str, str]]:
        return [
            {'role': msg['role'], 'content': msg['content']}
            for msg in self.messages
        ]

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return f"ConversationMemory(messages={len(self.messages)}, max={self.max_messages})"
