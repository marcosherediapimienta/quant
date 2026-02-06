"""
Motor principal del chatbot con RAG (Retrieval Augmented Generation)
"""
from typing import Optional, List, Dict
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

try:
    # Imports relativos (cuando se usa como paquete)
    from .code_indexer import CodeIndexer
    from .memory.conversation_memory import ConversationMemory
    from .prompts.finance_prompts import (
        SYSTEM_PROMPT,
        SYSTEM_PROMPT_BASE,
        RAG_PROMPT_TEMPLATE, 
        WELCOME_MESSAGE,
        QUERY_ENHANCEMENT_PROMPTS
    )
except ImportError:
    # Imports absolutos (cuando se ejecuta directamente)
    from code_indexer import CodeIndexer
    from memory.conversation_memory import ConversationMemory
    from prompts.finance_prompts import (
        SYSTEM_PROMPT,
        SYSTEM_PROMPT_BASE,
        RAG_PROMPT_TEMPLATE, 
        WELCOME_MESSAGE,
        QUERY_ENHANCEMENT_PROMPTS
    )


class ChatEngine:
    """
    Motor de chatbot con capacidad de búsqueda en código fuente (RAG)
    """
    
    def __init__(
        self, 
        api_key: str,
        project_root: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        enable_rag: bool = True
    ):
        """
        Inicializa el motor del chatbot
        
        Args:
            api_key: API key de Groq
            project_root: Ruta raíz del proyecto para indexar código (opcional)
            model: Modelo de Groq a usar
            temperature: Temperatura del modelo (0-1)
            enable_rag: Si activar RAG (búsqueda en código)
        """
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.enable_rag = enable_rag
        
        # Inicializar LLM
        self.llm = ChatGroq(
            api_key=api_key,
            model=model,
            temperature=temperature
        )
        
        # Memoria de conversación personalizada
        self.conversation_memory = ConversationMemory(max_messages=20)
        
        # Sistema RAG (si está habilitado y hay project_root)
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        
        if enable_rag and project_root:
            self._setup_rag(project_root)
        else:
            # Chain simple sin RAG
            self._setup_simple_chain()
        
        print(f"✓ ChatEngine inicializado (RAG: {self.enable_rag and self.vectorstore is not None})")
    
    def _setup_rag(self, project_root: str):
        """
        Configura el sistema RAG (indexación de código)
        
        Args:
            project_root: Ruta raíz del proyecto
        """
        try:
            print("Indexando código fuente...")
            
            # Indexar código
            indexer = CodeIndexer(project_root)
            documents = indexer.index_project(extensions=['.py'])
            
            if not documents:
                print("⚠ No se encontraron documentos para indexar. Usando modo sin RAG.")
                self._setup_simple_chain()
                return
            
            # Convertir a documentos de LangChain
            langchain_docs = []
            for doc in documents:
                content = f"# {doc['type'].upper()}: {doc['name']}\n"
                content += f"# Archivo: {doc['file']}\n\n"
                content += doc['content']
                
                if doc['metadata'].get('docstring'):
                    content += f"\n\n# Documentación:\n{doc['metadata']['docstring']}"
                
                langchain_docs.append(Document(
                    page_content=content,
                    metadata={
                        'name': doc['name'],
                        'type': doc['type'],
                        'file': doc['file']
                    }
                ))
            
            # Split de documentos si son muy grandes
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            split_docs = text_splitter.split_documents(langchain_docs)
            
            print(f"Creando embeddings para {len(split_docs)} chunks...")
            
            # Crear embeddings (modelo ligero y local)
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # Modelo pequeño (80MB)
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Crear vectorstore
            self.vectorstore = FAISS.from_documents(split_docs, embeddings)
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 4}  # Top 4 resultados más relevantes
            )
            
            # Crear prompt personalizado para RAG
            prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
            
            # Crear chain con la nueva API (LCEL - LangChain Expression Language)
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            self.rag_chain = (
                {
                    "context": self.retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            self.chain = self.rag_chain
            
            print("✓ Sistema RAG configurado correctamente")
            
        except Exception as e:
            print(f"⚠ Error configurando RAG: {e}")
            print("Usando modo sin RAG como fallback.")
            self._setup_simple_chain()
    
    def _setup_simple_chain(self):
        """
        Configura un chain simple sin RAG (para cuando falla la indexación)
        """
        # Prompt simple sin contexto RAG
        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_BASE),
            ("human", "{input}")
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
        """
        Genera una respuesta al mensaje del usuario
        
        Args:
            message: Mensaje del usuario
            context: Contexto adicional (ej: resultados de análisis previos)
        
        Returns:
            Dict con 'response', 'sources' (si hay RAG), y metadata
        """
        # Guardar mensaje del usuario en memoria
        self.conversation_memory.add_message('user', message)
        
        # Enriquecer query si es necesario
        enhanced_query = self._enhance_query(message)
        
        # Añadir historial al mensaje
        history_context = self.conversation_memory.get_context_string(last_n=5)
        if history_context:
            full_query = f"Historial de conversación:\n{history_context}\n\nPregunta actual: {enhanced_query}"
        else:
            full_query = enhanced_query
        
        try:
            if self.vectorstore and self.retriever:
                # Modo RAG - invocar chain con retrieval
                response_text = self.chain.invoke(full_query)
                
                # Obtener documentos recuperados manualmente para sources
                docs = self.retriever.invoke(full_query)
                sources = self._format_sources(docs)
                
            else:
                # Modo simple
                result = self.chain.invoke({"input": full_query})
                response_text = result
                sources = []
            
            # Guardar respuesta en memoria
            self.conversation_memory.add_message('assistant', response_text, {
                'sources': sources if sources else []
            })
            
            return {
                'response': response_text,
                'sources': sources,
                'has_rag': bool(self.vectorstore),
                'session_duration': self.conversation_memory.get_session_duration()
            }
            
        except Exception as e:
            error_msg = f"Error generando respuesta: {str(e)}"
            print(f"⚠ {error_msg}")
            return {
                'response': "Lo siento, hubo un error procesando tu pregunta. Por favor intenta de nuevo.",
                'sources': [],
                'error': error_msg
            }
    
    def _enhance_query(self, query: str) -> str:
        """
        Mejora la query del usuario detectando temas clave
        
        Args:
            query: Query original del usuario
        
        Returns:
            Query mejorada con contexto adicional
        """
        query_lower = query.lower()
        
        # Detectar temas y añadir contexto
        for topic, enhancement in QUERY_ENHANCEMENT_PROMPTS.items():
            if topic in query_lower:
                return f"{query}\n\n(Contexto relevante: {enhancement})"
        
        return query
    
    def _format_sources(self, documents: List) -> List[Dict]:
        """
        Formatea documentos fuente para respuesta
        
        Args:
            documents: Documentos recuperados por RAG
        
        Returns:
            Lista de dicts con información de fuentes
        """
        sources = []
        seen = set()
        
        for doc in documents:
            file_path = doc.metadata.get('file', 'unknown')
            doc_type = doc.metadata.get('type', 'code')
            name = doc.metadata.get('name', 'unknown')
            
            # Evitar duplicados
            key = f"{file_path}:{name}"
            if key in seen:
                continue
            seen.add(key)
            
            sources.append({
                'file': file_path,
                'type': doc_type,
                'name': name,
                'preview': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
            })
        
        return sources
    
    def get_welcome_message(self) -> str:
        """Retorna mensaje de bienvenida"""
        return WELCOME_MESSAGE
    
    def clear_memory(self):
        """Limpia la memoria de conversación"""
        self.conversation_memory.clear()
        print("✓ Memoria de conversación limpiada")
    
    def get_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """
        Obtiene historial de conversación
        
        Args:
            last_n: Número de mensajes recientes (None = todos)
        """
        return self.conversation_memory.get_messages(last_n)
    
    def save_vectorstore(self, path: str):
        """
        Guarda el vectorstore en disco para no tener que reindexar
        
        Args:
            path: Ruta donde guardar
        """
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"✓ Vectorstore guardado en {path}")
        else:
            print("⚠ No hay vectorstore para guardar")
    
    def load_vectorstore(self, path: str):
        """
        Carga un vectorstore previamente guardado
        
        Args:
            path: Ruta del vectorstore guardado
        """
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            self.vectorstore = FAISS.load_local(
                path, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
            print(f"✓ Vectorstore cargado desde {path}")
        except Exception as e:
            print(f"⚠ Error cargando vectorstore: {e}")
