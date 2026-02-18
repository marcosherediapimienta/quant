SYSTEM_PROMPT_BASE = """You are WarrenAI, an expert assistant in quantitative finance and investment, specialized in the Gala Analytics application.

**Your knowledge includes:**

*CAPM & Benchmark Analysis:*
- CAPM (Capital Asset Pricing Model): beta, Jensen's alpha, alpha significance testing (HAC robust t-stats, p-values)
- Treynor Ratio (excess return per unit of systematic risk)
- Security Market Line (SML) and Capital Market Line (CML)
- Tracking Error, Information Ratio, R-squared
- Multi-asset CAPM analysis across multiple tickers

*Risk Metrics:*
- Sharpe Ratio, Sortino Ratio (annualized, rolling)
- Value at Risk (VaR): historical, parametric, and Monte Carlo methods
- Expected Shortfall (ES / CVaR): historical, parametric, and Monte Carlo methods
- Maximum Drawdown and recovery analysis
- Volatility (annualized, rolling)
- Distribution moments: skewness, kurtosis, normality tests
- Rolling metrics over configurable windows

*Portfolio Optimization:*
- Markowitz mean-variance optimization with efficient frontier
- Tangency portfolio (maximum Sharpe ratio)
- Risk Parity: equal risk contribution across assets
- Black-Litterman model: combining market equilibrium with investor views
- Score-based and score-risk-adjusted weighting from fundamental analysis
- Equal weight baseline
- Ledoit-Wolf covariance shrinkage for robust estimation
- Weight constraints (min/max per asset)

*Fundamental / Company Valuation:*
- Valuation multiples: P/E (trailing & forward), P/B, P/S, EV/EBITDA, EV/Revenue, PEG, FCF yield, earnings yield
- Profitability metrics: ROE, ROA, gross/operating/net margins
- Financial health: current ratio, debt/equity, interest coverage
- Growth metrics: revenue growth, earnings growth, FCF growth
- Efficiency metrics: asset turnover, inventory turnover, DSO, DIO, revenue per employee
- Composite scoring system with weighted aggregation across categories
- Buy/sell signal generation and price target estimation

*Macroeconomic Analysis:*
- Multi-factor regression with HAC (Newey-West) robust standard errors
- Factor categories: volatility (VIX), interest rates (2Y–30Y), credit (HYG, LQD), currencies (DXY), commodities (gold, oil, copper), global indices
- Yield curve analysis: slope, inversion detection, implied yield curve
- Credit spread analysis: HY vs Treasury, HY vs IG (risk appetite)
- Inflation signals, global bond analysis, risk sentiment assessment
- Rolling correlations and sensitivity analysis
- Factor collinearity detection (VIF)
- Risk decomposition: systematic vs idiosyncratic
- Macro situation dashboard with overall risk scoring

*Technical Implementation:*
- Python, NumPy, Pandas, statsmodels, scipy, scikit-learn, FAISS
- Data from Yahoo Finance (yfinance) and FRED
- Covariance estimation with Ledoit-Wolf shrinkage (sklearn)
- Optimization via scipy.optimize.minimize with multi-start

**Your role:**
1. Answer questions about financial metrics and their interpretation
2. Explain HOW metrics are calculated in this specific application
3. Help interpret analysis results
4. Provide context about the source code when relevant
5. Guide the user in making data-driven investment decisions

**Response style:**
- Clear, concise, and technically precise
- Use numerical examples when helpful
- When explaining calculations, mention the relevant mathematical formulas using standard notation
- If you reference code or implementation, cite where it comes from
- If you are unsure, admit it honestly
- Use lists and headers to organize long responses
"""

RAG_PROMPT_TEMPLATE = SYSTEM_PROMPT_BASE + """
---

**Recent conversation history:**
{history}

---

**Relevant source code found in Gala Analytics:**
{context}

---

**Current user question:** {question}

**Response** (be specific about the implementation if the code context allows it, cite formulas when relevant):"""

SYSTEM_PROMPT = SYSTEM_PROMPT_BASE + """

**Code context:**
{context}

**Conversation:**
"""

QUERY_ENHANCEMENT_PROMPTS = {
    # Risk-adjusted ratios
    'sharpe': "Search for information about Sharpe ratio, risk-adjusted return, annualized standard deviation, risk-free rate",
    'sortino': "Search for information about Sortino ratio, downside deviation, negative returns, semi-deviation",
    'treynor': "Search for information about Treynor ratio, systematic risk, beta, excess return per unit of market risk",
    'information': "Search for information about Information Ratio, tracking error, active return, benchmark comparison",
    'tracking': "Search for information about Tracking Error, benchmark deviation, active risk, Information Ratio",

    # Tail risk
    'var': "Search for information about Value at Risk, percentiles, historical method, parametric, Monte Carlo",
    'es': "Search for information about Expected Shortfall, CVaR, tail risk, expected tail loss",
    'cvar': "Search for information about CVaR, Expected Shortfall, tail risk, expected tail loss",
    'monte carlo': "Search for information about Monte Carlo simulation, VaR, Expected Shortfall, random sampling, simulated returns",
    'tail': "Search for information about tail risk, fat tails, Expected Shortfall, extreme losses",

    # Drawdown & volatility
    'drawdown': "Search for information about Maximum Drawdown, peak, trough, recovery, calmar ratio",
    'volatilidad': "Search for information about volatility, standard deviation, GARCH, rolling volatility",
    'volatility': "Search for information about volatility, standard deviation, GARCH, rolling volatility",
    'rolling': "Search for information about rolling metrics, rolling Sharpe, rolling beta, rolling correlation, window size",

    # Distribution
    'normal': "Search for information about normal distribution, kurtosis, skewness, normality test",
    'skewness': "Search for information about skewness, return distribution asymmetry",
    'kurtosis': "Search for information about kurtosis, fat tails, extreme return distribution",
    'distribution': "Search for information about return distribution, moments, skewness, kurtosis, normality testing",

    # CAPM & benchmark
    'capm': "Search for information about CAPM, beta calculation, alpha, linear regression, excess returns, Security Market Line, Treynor ratio",
    'beta': "Search for information about beta, market sensitivity, regression, covariance, correlation, R-squared",
    'alpha': "Search for information about Jensen's alpha, abnormal return, CAPM, significance test, HAC t-statistic, p-value",
    'significan': "Search for information about alpha significance, HAC robust standard errors, t-statistic, p-value, Newey-West",
    'r-squared': "Search for information about R-squared, coefficient of determination, model fit, explained variance",
    'r_squared': "Search for information about R-squared, coefficient of determination, model fit, explained variance",
    'sml': "Search for information about Security Market Line, CAPM, expected return vs beta, undervalued overvalued",
    'cml': "Search for information about Capital Market Line, tangency portfolio, efficient frontier, maximum Sharpe",

    # Portfolio optimization
    'portfolio': "Search for information about portfolio optimization, optimal weights, efficient frontier, Markowitz, risk parity, Black-Litterman",
    'markowitz': "Search for information about Markowitz model, mean-variance, efficient frontier, diversification, covariance matrix",
    'frontera': "Search for information about efficient frontier, optimization, mean-variance, optimal portfolio",
    'frontier': "Search for information about efficient frontier, optimization, mean-variance, optimal portfolio",
    'optimiz': "Search for information about portfolio optimization, scipy optimize, weight constraints, multi-start",
    'diversif': "Search for information about diversification, asset correlation, idiosyncratic risk",
    'weights': "Search for information about optimal weights, tangency portfolio, equal weight, minimum variance, risk parity",
    'pesos': "Search for information about optimal weights, tangency portfolio, equal weight, minimum variance, risk parity",
    'risk parity': "Search for information about risk parity, equal risk contribution, inverse volatility, portfolio optimization",
    'black-litterman': "Search for information about Black-Litterman model, market equilibrium, investor views, posterior returns",
    'black litterman': "Search for information about Black-Litterman model, market equilibrium, investor views, posterior returns",
    'ledoit': "Search for information about Ledoit-Wolf shrinkage, covariance estimation, regularization, sample covariance",
    'shrinkage': "Search for information about Ledoit-Wolf shrinkage, covariance matrix, robust estimation, sklearn",
    'covariance': "Search for information about covariance matrix, Ledoit-Wolf shrinkage, correlation, portfolio risk",

    # Fundamental / valuation
    'valuation': "Search for information about fundamental analysis, valuation multiples, scoring system, P/E, P/B, EV/EBITDA, FCF yield",
    'valuación': "Search for information about fundamental analysis, valuation multiples, scoring system, P/E, P/B, EV/EBITDA, FCF yield",
    'fundamental': "Search for information about fundamental analysis, company scoring, profitability, financial health, growth, efficiency",
    'per': "Search for information about P/E ratio, price-to-earnings, relative valuation, forward P/E, trailing P/E",
    'p/e': "Search for information about P/E ratio, price-to-earnings, relative valuation, forward P/E, trailing P/E",
    'peg': "Search for information about PEG ratio, price-earnings-growth, earnings growth, valuation",
    'fcf': "Search for information about free cash flow yield, FCF, cash generation, valuation metric",
    'profitab': "Search for information about profitability metrics, ROE, ROA, gross margin, operating margin, net margin",
    'roe': "Search for information about Return on Equity, profitability, shareholder returns, ROA",
    'margin': "Search for information about profit margins, gross margin, operating margin, net margin, profitability",
    'health': "Search for information about financial health, current ratio, debt-to-equity, interest coverage, leverage",
    'debt': "Search for information about debt-to-equity ratio, leverage, financial health, interest coverage",
    'growth': "Search for information about growth metrics, revenue growth, earnings growth, FCF growth",
    'efficien': "Search for information about efficiency metrics, asset turnover, inventory turnover, DSO, DIO, revenue per employee",
    'score': "Search for information about fundamental scoring system, weighted aggregation, valuation score, composite score",
    'signal': "Search for information about buy/sell signals, signal determination, price target, fundamental scoring",
    'price target': "Search for information about price target calculation, valuation score, upside/downside potential",
    'dcf': "Search for information about DCF, discounted cash flows, WACC, terminal value",
    'wacc': "Search for information about WACC, cost of capital, cost of debt, capital structure",

    # Macro
    'macro': "Search for information about macroeconomic factors, multi-factor regression, HAC, risk decomposition, macro situation",
    'vix': "Search for information about VIX, implied volatility, fear index, market correlation, risk sentiment",
    'inflaci': "Search for information about inflation, CPI, real rates, portfolio impact, inflation signals",
    'inflation': "Search for information about inflation, CPI, real rates, portfolio impact, inflation signals",
    'tasas': "Search for information about interest rates, yield curve, duration, bonds, term structure",
    'rates': "Search for information about interest rates, yield curve, duration, bonds, term structure",
    'yield curve': "Search for information about yield curve, term structure, slope, inversion, implied yield curve, recession signal",
    'credit spread': "Search for information about credit spreads, HY vs Treasury, HY vs IG, risk appetite, credit conditions",
    'credit': "Search for information about credit conditions, HYG, LQD, credit spreads, default risk",
    'spread': "Search for information about credit spreads, yield curve spread, HY vs IG, risk premium",
    'correlaci': "Search for information about asset correlation, correlation matrix, diversification, rolling correlation",
    'correlation': "Search for information about asset correlation, correlation matrix, diversification, rolling correlation",
    'regression': "Search for information about multi-factor regression, HAC standard errors, betas, t-statistics, R-squared",
    'hac': "Search for information about HAC robust standard errors, Newey-West, heteroscedasticity, autocorrelation",
    'collinear': "Search for information about factor collinearity, VIF, variance inflation factor, multicollinearity",
    'vif': "Search for information about VIF, variance inflation factor, collinearity, multicollinearity detection",
    'sensitivity': "Search for information about macro sensitivity analysis, factor exposure, rolling betas, factor loadings",
    'risk decomp': "Search for information about risk decomposition, systematic vs idiosyncratic risk, factor contributions",
    'situation': "Search for information about macro situation dashboard, yield curve analysis, credit conditions, risk sentiment, overall risk score",

    # Returns
    'retorno': "Search for information about return calculation, log-returns, simple returns, annualized return",
    'return': "Search for information about return calculation, log-returns, simple returns, annualized return",
}

WELCOME_MESSAGE = """**WarrenAI** — Quantitative Analysis Assistant

How can I help you?"""

EXAMPLE_QUESTIONS = [
    "How is the Sharpe ratio calculated?",
    "What does a 5% VaR mean?",
    "How do I interpret a beta of 1.5?",
    "What is the difference between VaR and Expected Shortfall?",
    "How does the app optimize the portfolio?",
    "Explain the CAPM model",
    "How is alpha calculated in this app?",
    "What is Maximum Drawdown?",
    "How does the Markowitz efficient frontier work?",
    "What does the Sortino ratio indicate?",
    "What is the Treynor ratio?",
    "How does Risk Parity work?",
    "What is the Black-Litterman model?",
    "How does the valuation scoring system work?",
    "What macro factors does the app analyze?",
    "How is the yield curve analyzed?",
    "What is a credit spread?",
    "How does the alpha significance test work?",
]
