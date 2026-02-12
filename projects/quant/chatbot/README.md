# Chatbot Module

Módulo de chatbot con RAG (Retrieval Augmented Generation) para la aplicación de análisis financiero cuantitativo.

## Características

- **LLM**: Usa Groq con modelos Llama 3 (gratis, rápido)
- **RAG**: Indexa el código fuente para responder preguntas específicas sobre implementación
- **Memoria**: Mantiene contexto de conversación
- **Prompts especializados**: Optimizado para finanzas cuantitativas

## Instalación

```bash
pip install -r requirements.txt
```

## Uso básico

### 1. Sin RAG (solo conversación)

```python
from quant.chatbot import ChatEngine

# Inicializar
engine = ChatEngine(
    api_key="tu-groq-api-key",
    enable_rag=False
)

# Conversar
response = engine.respond("¿Qué es el Sharpe ratio?")
print(response['response'])
```

### 2. Con RAG (busca en el código)

```python
from quant.chatbot import ChatEngine
from pathlib import Path

# Ruta del proyecto a indexar
project_root = Path(__file__).parent.parent  # quant/projects/quant/

# Inicializar con RAG
engine = ChatEngine(
    api_key="tu-groq-api-key",
    project_root=str(project_root),
    enable_rag=True
)

# Hacer pregunta sobre implementación
response = engine.respond("¿Cómo se calcula el VaR en esta app?")
print(response['response'])
print("\nFuentes:")
for source in response['sources']:
    print(f"  - {source['file']}: {source['name']}")
```

### 3. Gestión de memoria

```python
# Ver historial
history = engine.get_history(last_n=5)

# Limpiar memoria
engine.clear_memory()
```

### 4. Guardar/cargar índice (evita reindexar)

```python
# Primera vez: indexar y guardar
engine = ChatEngine(api_key="...", project_root="...", enable_rag=True)
engine.save_vectorstore("./vectorstore_cache")

# Siguientes veces: cargar índice
engine = ChatEngine(api_key="...", enable_rag=False)
engine.load_vectorstore("./vectorstore_cache")
```

## Obtener API Key de Groq (Gratis)

1. Ve a https://console.groq.com
2. Crea una cuenta (gratis)
3. Ve a "API Keys"
4. Crea una nueva key
5. Copia y guarda la key

**Límites gratuitos:**
- 14,400 requests por día
- 30 requests por minuto
- Modelos: Llama 3 70B, Mixtral 8x7B, Gemma 7B

## Modelos disponibles

| Modelo | Parámetros | Velocidad | Contexto |
|--------|------------|-----------|----------|
| `llama3-70b-8192` | 70B | Muy rápido | 8K tokens |
| `llama3-8b-8192` | 8B | Ultra rápido | 8K tokens |
| `mixtral-8x7b-32768` | 47B | Rápido | 32K tokens |
| `gemma-7b-it` | 7B | Ultra rápido | 8K tokens |

## Estructura del módulo

```
chatbot/
├── __init__.py              # Exports principales
├── chat_engine.py           # Motor principal con RAG
├── code_indexer.py          # Indexación de código
├── requirements.txt         # Dependencias
├── README.md               # Esta documentación
├── prompts/
│   ├── __init__.py
│   └── finance_prompts.py  # Prompts especializados
└── memory/
    ├── __init__.py
    └── conversation_memory.py  # Gestión de historial
```

## Ejemplos de preguntas

El chatbot puede responder:

**Sobre métricas:**
- "¿Qué es el Sharpe ratio?"
- "¿Cómo interpreto un beta de 1.5?"
- "¿Qué diferencia hay entre VaR y Expected Shortfall?"

**Sobre implementación (con RAG):**
- "¿Cómo se calcula el VaR en esta app?"
- "Muéstrame cómo se hace la regresión CAPM"
- "¿Qué métodos usa el optimizador de portafolios?"

**Interpretación:**
- "Tengo un Sharpe de 0.8, ¿es bueno?"
- "Mi portafolio tiene un VaR del 5%, ¿qué significa?"

## Troubleshooting

### Error: "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### Error: "FAISS not installed"
```bash
pip install faiss-cpu
```

### El chatbot no encuentra código
- Verifica que `project_root` apunta a `quant/projects/quant/`
- Asegúrate de que hay archivos `.py` en esa ruta

### Respuestas lentas
- Usa un modelo más pequeño: `llama3-8b-8192`
- Reduce `k` en el retriever (línea 145 de chat_engine.py)

## Próximas mejoras

- [ ] Soporte para múltiples sesiones de usuario
- [ ] Cache de respuestas frecuentes
- [ ] Análisis de sentiment para mejorar respuestas
- [ ] Integración con análisis en tiempo real
- [ ] Explicaciones con gráficos generados
