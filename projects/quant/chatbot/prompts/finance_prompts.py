"""
Prompts especializados para el chatbot de finanzas cuantitativas
"""

SYSTEM_PROMPT_BASE = """Eres WarrenAI, un asistente experto en finanzas cuantitativas e inversión, especializado en la aplicación Gala Analytics.

**Tu conocimiento incluye:**
- CAPM (Capital Asset Pricing Model) y análisis de riesgo/retorno
- Métricas de riesgo: Sharpe Ratio, Sortino Ratio, VaR, Expected Shortfall (ES), Maximum Drawdown
- Análisis de portafolios: optimización de Markowitz, frontera eficiente, CML, tangency portfolio
- Análisis macroeconómico y correlaciones con factores (VIX, tasas, spreads)
- Valuación de empresas: ratios fundamentales, P/E, P/B, EV/EBITDA, análisis comparativo
- Implementación técnica: Python, NumPy, Pandas, statsmodels, scipy, FAISS
- Estrategias de inversión: momentum, value, factor investing, diversificación

**Tu función:**
1. Responder preguntas sobre métricas financieras y su interpretación
2. Explicar CÓMO se calculan las métricas en esta aplicación específica
3. Ayudar a interpretar resultados del análisis
4. Proporcionar contexto sobre el código fuente cuando sea relevante
5. Guiar al usuario en la toma de decisiones de inversión basadas en datos

**Estilo de respuesta:**
- Clara, concisa y técnicamente precisa
- Usa ejemplos numéricos cuando sea útil
- Cuando expliques cálculos, menciona las fórmulas matemáticas relevantes usando notación estándar
- Si citas código o implementación, referencia de dónde viene
- Si no estás seguro, admítelo honestamente
- Usa listas y headers para organizar respuestas largas
"""

# Prompt con contexto para RAG (incluye system prompt + historial + código)
RAG_PROMPT_TEMPLATE = SYSTEM_PROMPT_BASE + """
---

**Historial de conversación reciente:**
{history}

---

**Código fuente relevante encontrado en Gala Analytics:**
{context}

---

**Pregunta actual del usuario:** {question}

**Respuesta** (sé específico sobre la implementación si el contexto de código lo permite, cita las fórmulas cuando sea relevante):"""

# Prompt del sistema con contexto (versión legacy, mantenida por compatibilidad)
SYSTEM_PROMPT = SYSTEM_PROMPT_BASE + """

**Contexto de código:**
{context}

**Conversación:**
"""

QUERY_ENHANCEMENT_PROMPTS = {
    # Métricas de riesgo
    'sharpe': "Busca información sobre Sharpe ratio, retorno ajustado por riesgo, desviación estándar anualizada, risk-free rate",
    'sortino': "Busca información sobre Sortino ratio, downside deviation, retornos negativos, semidesviación",
    'var': "Busca información sobre Value at Risk, percentiles, método histórico, paramétrico, Monte Carlo",
    'es': "Busca información sobre Expected Shortfall, CVaR, tail risk, pérdida esperada en cola",
    'cvar': "Busca información sobre CVaR, Expected Shortfall, tail risk, pérdida esperada en cola",
    'drawdown': "Busca información sobre Maximum Drawdown, peak, trough, recuperación, calmar ratio",
    'volatilidad': "Busca información sobre volatilidad, desviación estándar, GARCH, rolling volatility",
    'volatility': "Busca información sobre volatilidad, desviación estándar, GARCH, rolling volatility",

    # CAPM y factores
    'capm': "Busca información sobre CAPM, cálculo de beta, alpha, regresión lineal, retornos excedentes, Security Market Line",
    'beta': "Busca información sobre beta, sensibilidad al mercado, regresión, covarianza, correlación",
    'alpha': "Busca información sobre alpha de Jensen, retorno anormal, CAPM, regresión sobre mercado",
    'sml': "Busca información sobre Security Market Line, CAPM, retorno esperado vs beta",
    'cml': "Busca información sobre Capital Market Line, portafolio tangente, frontera eficiente, Sharpe máximo",

    # Portfolio
    'portfolio': "Busca información sobre optimización de portafolio, pesos óptimos, frontera eficiente, Markowitz",
    'markowitz': "Busca información sobre modelo de Markowitz, media-varianza, frontera eficiente, diversificación",
    'frontera': "Busca información sobre frontera eficiente, optimización, media-varianza, portafolio óptimo",
    'optimiz': "Busca información sobre optimización de portafolio, scipy optimize, restricciones de pesos",
    'diversif': "Busca información sobre diversificación, correlación entre activos, riesgo idiosincrático",
    'weights': "Busca información sobre pesos óptimos, tangency portfolio, equal weight, minimum variance",
    'pesos': "Busca información sobre pesos óptimos, tangency portfolio, igual peso, mínima varianza",

    # Valuación
    'valuation': "Busca información sobre análisis fundamental, ratios financieros, P/E, P/B, EV/EBITDA",
    'valuación': "Busca información sobre análisis fundamental, ratios financieros, P/E, P/B, EV/EBITDA",
    'per': "Busca información sobre P/E ratio, price-to-earnings, valoración relativa",
    'p/e': "Busca información sobre P/E ratio, price-to-earnings, valoración relativa",
    'dcf': "Busca información sobre DCF, flujos de caja descontados, WACC, terminal value",
    'wacc': "Busca información sobre WACC, costo de capital, costo de deuda, estructura de capital",

    # Macro
    'macro': "Busca información sobre factores macroeconómicos, correlaciones, regresión múltiple, HAC",
    'vix': "Busca información sobre VIX, volatilidad implícita, fear index, correlación con mercado",
    'inflaci': "Busca información sobre inflación, IPC, tasas reales, impacto en portafolio",
    'tasas': "Busca información sobre tasas de interés, yield curve, duration, bonos",
    'correlaci': "Busca información sobre correlación entre activos, matriz de correlación, diversificación",

    # Retornos y estadística
    'retorno': "Busca información sobre cálculo de retornos, log-returns, simple returns, retorno anualizado",
    'return': "Busca información sobre cálculo de retornos, log-returns, simple returns, retorno anualizado",
    'normal': "Busca información sobre distribución normal, kurtosis, skewness, test de normalidad",
    'skewness': "Busca información sobre asimetría, skewness, distribución de retornos",
    'kurtosis': "Busca información sobre curtosis, fat tails, distribución de retornos extremos",
}

WELCOME_MESSAGE = """**WarrenAI** — Asistente de Análisis Cuantitativo

¿En qué puedo ayudarte?"""

# Ejemplos de preguntas frecuentes
EXAMPLE_QUESTIONS = [
    "¿Cómo se calcula el Sharpe ratio?",
    "¿Qué significa un VaR del 5%?",
    "¿Cómo interpreto un beta de 1.5?",
    "¿Qué diferencia hay entre VaR y Expected Shortfall?",
    "¿Cómo optimiza el portafolio la aplicación?",
    "Explícame el modelo CAPM",
    "¿Cómo se calcula el alpha en esta app?",
    "¿Qué es el Maximum Drawdown?",
    "¿Cómo funciona la frontera eficiente de Markowitz?",
    "¿Qué indica el Sortino ratio?",
]
