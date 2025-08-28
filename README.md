# 📊 Quantitative Finance - Finanzas Cuantitativas

Repositorio simple de proyectos de finanzas cuantitativas para aprender y practicar.

## 🎯 **Qué Contiene**

- **01_portfolio_optimization**: Optimización de carteras con datos reales
- **02_risk_management**: Gestión de riesgo (próximamente)
- **03_time_series**: Análisis de series temporales (próximamente)
- **04_options**: Preciado de opciones (próximamente)
- **05_backtesting**: Backtesting de estrategias (próximamente)

## 🚀 **Cómo Empezar**

### 1. **Clonar y configurar**
```bash
git clone https://github.com/tu-usuario/Quantitative-Finance.git
cd Quantitative-Finance
```

### 2. **Crear entorno virtual**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. **Instalar dependencias globales**
```bash
pip install -r requirements.txt
```

### 4. **Ejecutar ejemplos**
```bash
cd projects/01_portfolio_optimization
python examples.py
```

## 📈 **Proyecto 1: Optimización de Carteras**

**Funcionalidades:**
- Descarga datos reales de Yahoo Finance
- Optimiza carteras (Markowitz, Sharpe, Risk Parity)
- Calcula métricas de riesgo (VaR, CVaR, Drawdown)
- Genera fronteras eficientes

**Ejemplo rápido:**
```python
from portfolio import Portfolio
from optimizer import PortfolioOptimizer
from data_loader import MarketDataLoader

# Cargar datos
loader = MarketDataLoader()
prices, returns = loader.download_data(['AAPL', 'MSFT', 'GOOGL'])

# Optimizar cartera
optimizer = PortfolioOptimizer(returns)
weights, _ = optimizer.maximize_sharpe_ratio()

# Crear cartera
portfolio = Portfolio(returns, weights)
print(portfolio.get_summary_stats())
```

## 🛠️ **Dependencias Principales**

- pandas, numpy, scipy
- matplotlib, seaborn
- yfinance (datos de mercado)
- scikit-learn

## 📚 **Conceptos Aprendidos**

- Teoría Moderna de Portafolios
- Optimización de carteras
- Gestión de riesgo cuantitativa
- Análisis de datos financieros

## ⚠️ **Importante**

Este repositorio es **solo para aprendizaje**. Los resultados no garantizan rendimiento futuro.

---

⭐ **¡Dale una estrella si te resulta útil!**
