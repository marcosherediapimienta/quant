import logging
from typing import Optional, List, Dict

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

try:
    from .code_indexer import CodeIndexer
    from .memory.conversation_memory import ConversationMemory
    from .prompts.finance_prompts import (
        SYSTEM_PROMPT_BASE,
        RAG_PROMPT_TEMPLATE,
        WELCOME_MESSAGE,
        QUERY_ENHANCEMENT_PROMPTS
    )
    from .tools.config import (
        EMBEDDING_MODEL, EMBEDDING_DEVICE, CHUNK_SIZE, CHUNK_OVERLAP,
        RETRIEVER_K, MAX_QUERY_ENHANCEMENTS, SOURCE_PREVIEW_LENGTH,
        HISTORY_WINDOW, DEFAULT_MODEL, DEFAULT_TEMPERATURE,
        MAX_CONVERSATION_MESSAGES,
    )
except ImportError:
    from code_indexer import CodeIndexer
    from memory.conversation_memory import ConversationMemory
    from prompts.finance_prompts import (
        SYSTEM_PROMPT_BASE,
        RAG_PROMPT_TEMPLATE,
        WELCOME_MESSAGE,
        QUERY_ENHANCEMENT_PROMPTS
    )
    from tools.config import (
        EMBEDDING_MODEL, EMBEDDING_DEVICE, CHUNK_SIZE, CHUNK_OVERLAP,
        RETRIEVER_K, MAX_QUERY_ENHANCEMENTS, SOURCE_PREVIEW_LENGTH,
        HISTORY_WINDOW, DEFAULT_MODEL, DEFAULT_TEMPERATURE,
        MAX_CONVERSATION_MESSAGES,
    )

logger = logging.getLogger(__name__)

class ChatEngine:
    def __init__(
        self,
        api_key: str,
        project_root: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        enable_rag: bool = True
    ):
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.enable_rag = enable_rag
        self.llm = ChatGroq(
            api_key=api_key,
            model=model,
            temperature=temperature
        )
        self.conversation_memory = ConversationMemory(max_messages=MAX_CONVERSATION_MESSAGES)
        self.vectorstore = None
        self.retriever = None
        self.chain = None

        if enable_rag and project_root:
            self._setup_rag(project_root)
        else:
            self._setup_simple_chain()

        logger.info("ChatEngine initialized (model: %s, RAG: %s)", model, self.enable_rag and self.vectorstore is not None)

    @staticmethod
    def _create_embeddings() -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': EMBEDDING_DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )

    def _setup_rag(self, project_root: str):

        try:
            logger.info("Indexing source code...")

            indexer = CodeIndexer(project_root)
            documents = indexer.index_project(extensions=['.py'])

            if not documents:
                logger.warning("No documents found for indexing. Falling back to no-RAG mode.")
                self._setup_simple_chain()
                return

            langchain_docs = []
            for doc in documents:
                content = f"# {doc['type'].upper()}: {doc['name']}\n"
                content += f"# File: {doc['file']}\n\n"
                content += doc['content']

                if doc['metadata'].get('docstring'):
                    content += f"\n\n# Documentation:\n{doc['metadata']['docstring']}"

                langchain_docs.append(Document(
                    page_content=content,
                    metadata={
                        'name': doc['name'],
                        'type': doc['type'],
                        'file': doc['file']
                    }
                ))

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len
            )
            split_docs = text_splitter.split_documents(langchain_docs)

            logger.info("Creating embeddings for %d chunks...", len(split_docs))

            embeddings = self._create_embeddings()

            self.vectorstore = FAISS.from_documents(split_docs, embeddings)
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": RETRIEVER_K}
            )
            self._build_rag_chain()

            logger.info("RAG system configured successfully")

        except Exception as e:
            logger.exception("Error configuring RAG, falling back to simple chain")
            self._setup_simple_chain()

    def _build_rag_chain(self):
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        def format_docs(docs: List[Document]) -> str:
            if not docs:
                return "No relevant source code was found for this question."
            return "\n\n---\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {
                "context": (
                    RunnableLambda(lambda x: x["question"])
                    | self.retriever
                    | format_docs
                ),
                "question": RunnableLambda(lambda x: x["question"]),
                "history": RunnableLambda(lambda x: x.get("history", "No previous history"))
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        self.chain = self.rag_chain

    def _setup_simple_chain(self):
        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_BASE),
            ("human", "Recent history:\n{history}\n\nQuestion: {question}")
        ])

        self.chain = (
            simple_prompt
            | self.llm
            | StrOutputParser()
        )

    def respond(
        self,
        message: str,
        context: Optional[Dict] = None    
    ) -> Dict[str, any]:

        self.conversation_memory.add_message('user', message)

        enhanced_query = self._enhance_query(message)

        history_context = self.conversation_memory.get_context_string(last_n=HISTORY_WINDOW)
        if not history_context:
            history_context = "No previous history."

        chain_input = {
            "question": enhanced_query,
            "history": history_context
        }

        try:
            if self.vectorstore and self.retriever:
                response_text = self.chain.invoke(chain_input)
                docs = self.retriever.invoke(enhanced_query)
                sources = self._format_sources(docs)

            else:
                result = self.chain.invoke(chain_input)
                response_text = result
                sources = []

            self.conversation_memory.add_message('assistant', response_text, {
                'sources': sources
            })

            return {
                'response': response_text,
                'sources': sources,
                'has_rag': bool(self.vectorstore),
                'session_duration': self.conversation_memory.get_session_duration()
            }

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.exception("Error generating chatbot response")
            return {
                'response': "Sorry, there was an error processing your question. Please try again.",
                'sources': [],
                'error': error_msg
            }

    def _enhance_query(self, query: str) -> str:

        query_lower = query.lower()

        matched_enhancements = []
        for topic, enhancement in QUERY_ENHANCEMENT_PROMPTS.items():
            if topic in query_lower:
                matched_enhancements.append(enhancement)

        if matched_enhancements:
            combined = " | ".join(matched_enhancements[:MAX_QUERY_ENHANCEMENTS])
            return f"{query}\n\n[Search context: {combined}]"

        return query

    def _format_sources(self, documents: List) -> List[Dict]:
        sources = []
        seen = set()

        for doc in documents:
            file_path = doc.metadata.get('file', 'unknown')
            doc_type = doc.metadata.get('type', 'code')
            name = doc.metadata.get('name', 'unknown')
            key = f"{file_path}:{name}"

            if key in seen:
                continue
            seen.add(key)

            sources.append({
                'file': file_path,
                'type': doc_type,
                'name': name,
                'preview': doc.page_content[:SOURCE_PREVIEW_LENGTH] + '...' if len(doc.page_content) > SOURCE_PREVIEW_LENGTH else doc.page_content
            })

        return sources

    def get_welcome_message(self) -> str:
        return WELCOME_MESSAGE

    def clear_memory(self):
        self.conversation_memory.clear()
        logger.debug("Conversation memory cleared")

    def get_history(self, last_n: Optional[int] = None) -> List[Dict]:
        return self.conversation_memory.get_messages(last_n)

    def save_vectorstore(self, path: str):

        if self.vectorstore:
            self.vectorstore.save_local(path)
            logger.info("Vectorstore saved to %s", path)
        else:
            logger.warning("No vectorstore to save")

    def load_vectorstore(self, path: str):

        try:
            embeddings = self._create_embeddings()
            self.vectorstore = FAISS.load_local(
                path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
            self._build_rag_chain()

            logger.info("Vectorstore loaded from %s and RAG chain rebuilt", path)
        except Exception as e:
            logger.exception("Error loading vectorstore from %s", path)
            self._setup_simple_chain()
