# Quant

Built for portfolio management, risk analysis, CAPM modeling, company valuation, macroeconomic analysis and AI assistance.

## Modules

### Portfolio Management (`pm`)

Investment portfolio analysis.

- **Optimization**: Markowitz, Black-Litterman, risk parity, score-based weighting.
- **Risk**: VaR (historical, parametric, Monte Carlo), Expected Shortfall (CVaR), Sharpe, Sortino, Treynor, maximum drawdown, tracking error, Information Ratio, rolling correlations, distribution analysis (skewness, kurtosis).
- **CAPM**: beta, Jensen's alpha, significance testing (HAC robust), Security Market Line, multi-asset CAPM.
- **Valuation**: multiples (P/E, P/B, P/S, EV/EBITDA, PEG), profitability (ROE, ROA, margins), financial health, growth, efficiency, composite scoring, buy/sell signals, price target estimation.

### Macro (`macro`)

Macroeconomic analysis and its impact on portfolios.

- Multi-factor regression with HAC robust standard errors
- Factor categories: volatility, interest rates, credit, currencies, commodities
- Implied yield curve and credit spreads
- Rolling correlations and macro situation dashboard

### Chatbot — WarrenAI

AI-powered financial assistant with Retrieval-Augmented Generation (RAG).

- LLM: Groq (Llama 3.3 70B) via LangChain
- Project code indexing with FAISS + sentence-transformers
- Conversational memory for persistent context
- Prompts specialized in quantitative finance

## Project Structure

```
quant/
└── projects/quant/
    ├── pm/                             # Portfolio Management
    │   ├── utils/
    │   │   ├── analysis/
    │   │   │   ├── capm/               # CAPM model
    │   │   │   ├── portfolio/          # Portfolio optimization
    │   │   │   ├── risk_metrics/       # Risk metrics
    │   │   │   └── valuation/          # Fundamental valuation
    │   │   ├── data/                   # Data loading & processing
    │   │   ├── tools/                  # Configuration (config.py)
    │   │   └── visualizations/         # Charts & plots
    │   ├── portfolio_config.ipynb
    │   ├── capm_analysis.ipynb
    │   ├── risk_analysis.ipynb
    │   ├── valuation_analysis.ipynb
    │   └── buy_sell_analysis.ipynb
    │
    ├── macro/                          # Macroeconomic analysis
    │   ├── utils/
    │   │   ├── analyzers/
    │   │   ├── components/
    │   │   ├── reporters/
    │   │   ├── tools/                  # Configuration (config.py)
    │   │   └── visualizations/
    │   └── macro_analysis.ipynb
    │
    └── chatbot/                        # WarrenAI
        ├── chat_engine.py
        ├── code_indexer.py
        ├── memory/
        └── prompts/
```

## Installation

```bash
git clone https://github.com/marcosherediapimienta/quant.git
cd quant
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Main Dependencies

| Library | Purpose |
|---|---|
| `numpy`, `pandas` | Data structures and numerical computing |
| `scipy`, `statsmodels` | Statistics, regressions, tests |
| `scikit-learn` | Optimization and machine learning |
| `yfinance` | Market data (stocks, indices, ETFs, currencies, commodities) |
| `matplotlib`, `seaborn` | Visualization |
| `jupyter` | Interactive notebooks |

The chatbot requires additional dependencies — see `projects/quant/chatbot/requirements.txt`.

## Usage

Each module includes a Jupyter notebook with full examples:

| Notebook | Description |
|---|---|
| `pm/portfolio_config.ipynb` | Portfolio configuration and optimization |
| `pm/capm_analysis.ipynb` | CAPM analysis and efficient frontier |
| `pm/risk_analysis.ipynb` | Risk metrics: VaR, CVaR, ratios, drawdown |
| `pm/valuation_analysis.ipynb` | Fundamental company valuation |
| `pm/buy_sell_analysis.ipynb` | Buy and sell signals |
| `macro/macro_analysis.ipynb` | Macroeconomic factor analysis |

## Data Sources

- **Yahoo Finance** (via yfinance) — historical and real-time prices for stocks, indices, ETFs, commodities and currencies.
- **FRED** — Treasury yields and macroeconomic data.

## License

GNU Affero General Public License v3 — see [LICENSE](LICENSE).
