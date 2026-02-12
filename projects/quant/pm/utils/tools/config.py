# ============================================================================
# BENCHMARKS
# ============================================================================
BENCHMARKS = {
    'SP500': '^GSPC',
    'NASDAQ': '^IXIC',
    'DOW': '^DJI',
    'MSCI_WORLD': 'EUNL.DE',
    'STOXX50': '^STOXX50E',
    'VIX': '^VIX',
    'RUSSELL2000': '^RUT',
    'FTSE100': '^FTSE',
    'DAX': '^GDAXI',
    'NIKKEI': '^N225',
    'CAC40': '^FCHI',
    'EUROSTOXX': '^STOXX',
    'IBEX35': '^IBEX'
}

BENCHMARK_CURRENCIES = {
    'SP500': 'USD',
    'NASDAQ': 'USD',
    'DOW': 'USD',
    'MSCI_WORLD': 'EUR',
    'STOXX50': 'EUR',
    'VIX': 'USD',
    'RUSSELL2000': 'USD',
    'FTSE100': 'GBP',
    'DAX': 'EUR',
    'NIKKEI': 'JPY',
    'CAC40': 'EUR',
    'EUROSTOXX': 'EUR',
    'IBEX35': 'EUR'
}

# ============================================================================
# CONFIGURACIÓN GLOBAL DE FECHAS PARA ANÁLISIS
# ============================================================================
ANALYSIS_DATES = {
    'start_date': '2020-01-01',       # Fecha inicial por defecto para todos los análisis
    'end_date': '2025-12-24',         # Fecha final por defecto
    'use_current_date_as_end': True,  # Si True, ignora end_date y usa fecha actual
    'default_lookback_years': 5,      # Solo se usa si start_date está vacío
}

# ============================================================================
# CONFIGURACIÓN DE DESCARGA (yfinance)
# ============================================================================
DOWNLOAD_DEFAULTS = {
    'auto_adjust': True,
    'group_by': 'ticker',
    'threads': True,
    'progress': True
}

YFINANCE_COLUMNS = {
    'adj_close': 'Adj Close',
    'close': 'Close',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'volume': 'Volume',
}

# ============================================================================
# CONFIGURACIÓN DE ANÁLISIS DE RIESGO
# ============================================================================
RISK_ANALYSIS = {
    'annual_factor': 252,             
    'risk_free_rate': 0.045,          
    'default_confidence_levels': (0.95, 0.99),
    'default_confidence_level': 0.95,
    'monte_carlo': {
        'n_simulations': 10000,
        'seed': 42
    },
    'rolling': {
        'default_window': 252
    }
}

# ============================================================================
# PARÁMETROS ESTADÍSTICOS
# ============================================================================
STATISTICAL_DEFAULTS = {
    'significance_level': 0.05,
    'min_observations': 30,
}

# ============================================================================
# CONFIGURACIÓN NUMÉRICA
# ============================================================================
NUMERICAL_DEFAULTS = {
    'epsilon': 1e-12,
    'tolerance': 1e-6,
    'weight_tolerance': 1e-6,
}

# ============================================================================
# CONFIGURACIÓN DE OPTIMIZACIÓN
# ============================================================================
OPTIMIZATION_DEFAULTS = {
    'method': 'SLSQP',
    'max_iterations': 1000,
    'tolerance': 1e-6,
    'frontier_points': 60,
}

# ============================================================================
# UMBRALES DE INTERPRETACIÓN - RATIOS
# ============================================================================
RATIO_INTERPRETATION = {
    'sharpe': {
        'excellent': 2.0,
        'very_good': 1.0,
        'acceptable': 0.5,
        'poor': 0.0
    },
    'sortino': {
        'excellent': 2.0,
        'good': 1.0,
        'acceptable': 0.5,
        'poor': 0.0
    },
    'calmar': {
        'excellent': 1.0,
        'good': 0.5,
        'poor': 0.0
    }
}

DRAWDOWN_RISK_LEVELS = {
    'low': 10,
    'moderate': 20,
    'high': 30,
}

# ============================================================================
# UMBRALES DE VALORACIÓN
# ============================================================================
VALUATION_THRESHOLDS = {
    'profitability': {
        'roic': {'excellent': 0.20, 'good': 0.15, 'fair': 0.10, 'poor': 0.05},
        'roe': {'excellent': 0.25, 'good': 0.15, 'fair': 0.10, 'poor': 0.05},
        'roa': {'excellent': 0.15, 'good': 0.10, 'fair': 0.05, 'poor': 0.02},
        'gross_margin': {'excellent': 0.50, 'good': 0.35, 'fair': 0.20, 'poor': 0.10},
        'operating_margin': {'excellent': 0.25, 'good': 0.15, 'fair': 0.10, 'poor': 0.05},
        'net_margin': {'excellent': 0.20, 'good': 0.10, 'fair': 0.05, 'poor': 0.02},
    },
    'financial_health': {
        'debt_ebitda': {'excellent': 1.0, 'good': 2.0, 'fair': 3.0, 'poor': 5.0},
        'debt_equity': {'excellent': 0.3, 'good': 0.5, 'fair': 1.0, 'poor': 2.0},
        'current_ratio': {'excellent': 2.5, 'good': 2.0, 'fair': 1.5, 'poor': 1.0},
        'interest_coverage': {'excellent': 10.0, 'good': 5.0, 'fair': 3.0, 'poor': 1.5},
    },
    'valuation_multiples': {
        'pb_ratio': {'cheap': 1.5, 'fair': 3.0, 'expensive': 5.0, 'very_expensive': 8.0},
    },
    'efficiency': {
        'asset_turnover': {'excellent': 1.5, 'good': 1.0, 'fair': 0.7, 'poor': 0.4},
        'inventory_turnover': {'excellent': 12.0, 'good': 8.0, 'fair': 5.0, 'poor': 3.0},
        'dso': {'excellent': 30, 'good': 45, 'fair': 60, 'poor': 90},
        'dio': {'excellent': 30, 'good': 60, 'fair': 90, 'poor': 120},
    }
}

# ============================================================================
# RANGOS PARA SCORING (usado en cálculos normalizados de métricas)
# ============================================================================
SCORING_RANGES = {
    'growth': {
        'revenue': {'min': -0.20, 'max': 0.40},    # -20% a +40%
        'earnings': {'min': -0.30, 'max': 0.50},   # -30% a +50%
    },
    'profitability': {
        'margin': {'min': -0.10, 'max': 0.50},     # -10% a +50%
    }
}

# ============================================================================
# PESOS PARA SCORING DE VALORACIÓN
# ============================================================================
SCORING_WEIGHTS = {
    'profitability': {
        'roic': 0.30,
        'roe': 0.20,
        'operating_margin': 0.25,
        'net_margin': 0.25,
    },
    'financial_health': {
        'debt_ebitda': 0.25,
        'debt_equity': 0.20,
        'current_ratio': 0.20,
        'net_cash_ebitda': 0.20,
        'free_cash_flow': 0.15,
    }
}

# ============================================================================
# UMBRALES DE ALERTAS
# ============================================================================
ALERT_THRESHOLDS = {
    'profitability': {
        'roic_low': 0.08,
        'operating_margin_low': 0.05,
    },
    'financial_health': {
        'debt_ebitda_danger': 4,
        'debt_ebitda_warning': 3,
        'current_ratio_low': 1.0,
    },
    'growth': {
        'revenue_decline_significant': -0.10,    # Caída >10% en ingresos
        'earnings_decline_strong': -0.20,        # Caída >20% en beneficios
        'growth_too_high': 0.50,                 # Crecimiento >50% (insostenible?)
        'revenue_decline_mild': -0.05,           # Caída >5% en ingresos
        'earnings_vs_revenue_multiple': 2,       # Earnings crecen >2x revenue
        'high_earnings_growth_threshold': 0.20   # >20% crecimiento earnings
    },
    'efficiency': {
        'dso_high': 60,                          # DSO >60 días (cobro lento)
        'dio_high': 90,                          # DIO >90 días (inventario lento)
        'asset_turnover_low': 0.5                # Asset turnover <0.5x
    },
    'valuation': {
        'pe_very_high': 40,                      # P/E >40x (sobrevalorado?)
        'pe_negative': 0,                        # P/E negativo (pérdidas)
        'ev_ebitda_high': 20,                    # EV/EBITDA >20x
        'peg_high': 2,                           # PEG >2 (caro vs crecimiento)
        'fcf_yield_negative': 0                  # FCF yield negativo
    }
}

# ============================================================================
# CONFIGURACIÓN DE PORTFOLIO
# ============================================================================
PORTFOLIO_CONFIG = {
    'selection': {
        'min_score': 60.0,
        'max_companies': 10,
        'max_per_sector': 3,
        'default_method': 'total_score',
    },
    'scoring_weights': {
        'balanced': {
            'profitability': 0.25,
            'health': 0.25,
            'growth': 0.20,
            'valuation': 0.30
        },
        'value': {
            'total': 0.5,
            'valuation': 0.5
        },
        'growth': {
            'total': 0.5,
            'growth': 0.5
        }
    },
    'optimization': {
        'default_method': 'score_risk_adjusted',
        'risk_free_rate': 0.045,
        'annual_trading_days': 252,
        'min_data_points': 30,
        'scipy_method': 'SLSQP',
    },
    'dates': {
        'default_lookback_years': ANALYSIS_DATES['default_lookback_years'],
        'date_format': '%Y-%m-%d',
        'start_date': ANALYSIS_DATES['start_date'],
        'end_date': ANALYSIS_DATES['end_date'],
        'use_current_date_as_end': ANALYSIS_DATES['use_current_date_as_end']
    },
    'defaults': {
        'sector_name': 'Unknown',
        'price_column': 'Close'
    }
}

# ============================================================================
# CONFIGURACIÓN DE SEÑALES DE TRADING (BUY/SELL)
# ============================================================================
TRADING_SIGNALS_CONFIG = {
    'start_date': ANALYSIS_DATES['start_date'],
    'end_date': ANALYSIS_DATES['end_date'],
    'use_current_date_as_end': ANALYSIS_DATES['use_current_date_as_end'],
    'default_lookback_years': ANALYSIS_DATES['default_lookback_years'],
}

# ============================================================================
# CONFIGURACIÓN DE ÍNDICES Y ETFs
# ============================================================================
INDEX_CONFIG = {
    'urls': {
        'sp500': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        'nasdaq100': 'https://en.wikipedia.org/wiki/Nasdaq-100',
        'dow30': 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average',
    },
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'etf_mapping': {
        'SPY': 'SP500',
        'VOO': 'SP500',
        'IVV': 'SP500',
        'QQQ': 'NASDAQ100',
        'DIA': 'DOW30'
    },
    'supported_indices': ['SP500', 'NASDAQ100', 'DOW30', 'RUSSELL1000'],
    'validation': {
        'nasdaq_min_tickers': 50,
        'dow_min_tickers': 20,
        'dow_max_tickers': 35,
        'sp500_min_tickers': 400,
    },
    'fallback': {
        'sp500_top100': [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
            'UNH', 'JNJ', 'XOM', 'V', 'JPM', 'WMT', 'PG', 'MA', 'HD', 'CVX',
            'LLY', 'ABBV', 'MRK', 'AVGO', 'KO', 'PEP', 'COST', 'ADBE', 'MCD',
            'CSCO', 'ACN', 'TMO', 'ABT', 'NFLX', 'DHR', 'VZ', 'NKE', 'WFC',
            'DIS', 'CMCSA', 'TXN', 'PM', 'CRM', 'BMY', 'NEE', 'RTX', 'ORCL',
            'INTC', 'UPS', 'HON', 'QCOM', 'BA', 'LOW', 'SPGI', 'BLK', 'LMT',
            'AMD', 'INTU', 'CAT', 'DE', 'GE', 'AXP', 'ISRG', 'GILD', 'NOW',
            'BKNG', 'ADI', 'PLD', 'MDLZ', 'SYK', 'ADP', 'REGN', 'TJX', 'VRTX',
            'CB', 'SBUX', 'CI', 'TMUS', 'PYPL', 'MMC', 'SO', 'ZTS', 'SCHW',
            'MO', 'BSX', 'DUK', 'AMT', 'PGR', 'LRCX', 'EOG', 'ITW', 'BDX',
            'C', 'SLB', 'NOC', 'CME', 'MMM', 'USB', 'HUM', 'PNC', 'FI', 'TGT'
        ],
        'russell_additional': [
            'SNOW', 'CRWD', 'ZS', 'DDOG', 'NET', 'OKTA', 'FTNT',
            'IONQ', 'SMR', 'NIO', 'RIVN', 'LCID'
        ]
    }
}

# ============================================================================
# UMBRALES DE INTERPRETACIÓN PARA REPORTERS
# ============================================================================

# Umbrales para interpretación de skewness
SKEWNESS_THRESHOLDS = {
    'positive': 0.5,      # > 0.5 = asimetría positiva
    'negative': -0.5      # < -0.5 = asimetría negativa
}

# Umbrales para interpretación de kurtosis
KURTOSIS_THRESHOLDS = {
    'leptokurtic': 3,     # > 3 = colas pesadas
    'platykurtic': -1     # < -1 = colas ligeras
}

# Umbrales para interpretación de Sortino
SORTINO_THRESHOLDS = {
    'excellent': 2.0,
    'good': 1.0,
    'acceptable': 0.5
}

# Umbrales para interpretación de Tracking Error (%)
TRACKING_ERROR_THRESHOLDS = {
    'very_close': 2,      # < 2% muy cercano al benchmark
    'moderate': 5,        # < 5% desviación moderada
    'active': 10          # < 10% gestión activa notable
}

# Umbrales para interpretación de Information Ratio
INFORMATION_RATIO_THRESHOLDS = {
    'excellent': 0.5,
    'positive': 0,
    'slightly_below': -0.5
}

# Umbrales para interpretación de Beta
BETA_THRESHOLDS = {
    'aggressive': 1.2,    # > 1.2 = agresivo
    'market': 0.8         # > 0.8 = similar al mercado
}

# Umbrales para interpretación de Alpha (%)
ALPHA_THRESHOLDS = {
    'excellent': 5,       # > 5% excelente
    'positive': 0,        # > 0% positivo
    'slightly_below': -5  # > -5% ligeramente por debajo
}

# Niveles de riesgo VaR
VAR_RISK_LEVELS = {
    'low': 2,             # < 2% bajo
    'moderate': 5,        # < 5% moderado
    'high': 10            # < 10% alto, >= 10% muy alto
}

# ============================================================================
# CONFIGURACIÓN ESPECÍFICA DE CAPM
# ============================================================================

# Número de puntos para CML (Capital Market Line)
CML_POINTS = 100

# Factor de extensión para CML (extiende el eje de volatilidad más allá de la frontera)
CML_EXTENSION_FACTOR = 1.2

# Configuración para SML (Security Market Line)
SML_CONFIG = {
    'max_beta': 2.0,           # Beta máximo para graficar SML
    'n_points': 100            # Puntos para generar línea SML
}

# Umbrales para identificación de activos
ALPHA_THRESHOLDS_IDENTIFICATION = {
    'outperformer': 0.0,       # Alpha mínimo para considerar outperformer
    'underperformer': 0.0      # Alpha máximo para considerar underperformer
}

# Umbral mínimo para mostrar pesos en reportes
MIN_WEIGHT_DISPLAY = 0.001     # 0.1% - no mostrar pesos menores

# Número de activos a mostrar en rankings
TOP_ASSETS_DISPLAY = 10

# ============================================================================
# CONFIGURACIÓN DE VALORACIÓN DE EMPRESAS
# ============================================================================

# PESOS DE AGREGACIÓN DE SCORES
# Basados en metodologías de análisis fundamental estándar (CFA, Graham & Dodd)
SCORE_AGGREGATION_WEIGHTS = {
    'total': {
        'valuation': 0.40,      # Valoración (P/E, EV/EBITDA, etc.)
        'fundamental': 0.50,    # Calidad del negocio (ROE, FCF, etc.)
        'technical': 0.10       # Momentum y tendencia
    },
    'fundamental': {
        'profitability': 0.35,  # Rentabilidad (ROE, ROIC, márgenes)
        'health': 0.35,         # Salud financiera (deuda, liquidez)
        'growth': 0.30          # Crecimiento (revenue, earnings)
    }
}

# RANGOS DE SCORING PARA PROFITABILIDAD
# Basados en percentiles históricos del S&P 500
PROFITABILITY_SCORING_RANGES = {
    'roic': {'min': -0.10, 'max': 0.30},         # -10% a 30% (Buffett: >15% es excelente)
    'roe': {'min': -0.10, 'max': 0.35},          # -10% a 35% (Graham: >15% es sólido)
    'operating_margin': {'min': -0.10, 'max': 0.30},  # -10% a 30%
    'net_margin': {'min': -0.10, 'max': 0.25}    # -10% a 25%
}

# CONFIGURACIÓN DE SCORING PARA SALUD FINANCIERA
# Basados en ratios de Moody's, S&P para rating crediticio
FINANCIAL_HEALTH_SCORING = {
    'weights': {
        'debt_ebitda': 0.25,        # Capacidad de pago de deuda
        'debt_equity': 0.20,        # Apalancamiento
        'current_ratio': 0.20,      # Liquidez de corto plazo
        'net_cash_ebitda': 0.20,    # Posición de caja vs deuda
        'free_cash_flow': 0.15      # Generación de caja
    },
    'ranges': {
        'debt_ebitda': {'min': 0, 'max': 6},      # 0x a 6x (S&P: >4x es alto riesgo)
        'debt_equity': {'min': 0, 'max': 3},      # 0% a 300%
        'current_ratio': {'min': 0.5, 'max': 3.0}, # 0.5 a 3.0
        'net_cash_ebitda': {'min': -3, 'max': 3}  # -3x a 3x
    }
}

# CONFIGURACIÓN DE SCORING PARA EFICIENCIA
# Basados en benchmarks de industria (McKinsey, BCG)
EFFICIENCY_SCORING = {
    'weights': {
        'asset_turnover': 0.40,     # Eficiencia en uso de activos
        'dso': 0.30,                # Días de cobro
        'dio': 0.30                 # Días de inventario
    },
    'ranges': {
        'asset_turnover': {'min': 0.2, 'max': 2.0},  # 0.2x a 2x
        'dso': {'min': 20, 'max': 90},               # 20 a 90 días
        'dio': {'min': 20, 'max': 120}               # 20 a 120 días
    }
}

# CONFIGURACIÓN DE SCORING PARA VALORACIÓN
# Basados en Damodaran (NYU) y medias históricas del mercado
VALUATION_SCORING = {
    'weights': {
        'pe_ttm': 0.20,             # P/E ratio
        'ev_ebitda': 0.20,         # EV/EBITDA
        'pb_ratio': 0.15,          # Price/Book
        'fcf_yield': 0.25,         # FCF Yield
        'peg_ratio': 0.20          # PEG (P/E to Growth): menor = mejor valoración
    },
    'ranges': {
        'pe_ttm': {'min': 5, 'max': 40},            # 5x a 40x (media ~16x)
        'ev_ebitda': {'min': 4, 'max': 25},         # 4x a 25x (media ~12x)
        'pb_ratio': {'min': 1.5, 'max': 8},         # 1.5x a 8x
        'fcf_yield': {'min': -0.02, 'max': 0.12},   # -2% a 12%
        'peg_ratio': {'min': 0.5, 'max': 3.5}       # PEG <1 barato, ~1 fair, >2 caro
    }
}

# PESOS PARA SCORE DE CRECIMIENTO
GROWTH_SCORING_WEIGHTS = {
    'revenue': 0.50,                # Crecimiento de ingresos
    'earnings': 0.50                # Crecimiento de beneficios
}

# CONFIGURACIÓN DE SEÑALES DE TRADING (BUY/SELL/HOLD)
# Basados en análisis cuantitativo de señales exitosas
TRADING_SIGNAL_RULES = {
    'buy': {
        'strong': {                              # COMPRA FUERTE
            'valuation_min': 60,
            'fundamental_min': 80,
            'confidence_base': 75,
            'confidence_max': 95,
            'valuation_weight': 0.4,
            'fundamental_weight': 0.3
        },
        'moderate': {                            # COMPRA MODERADA
            'valuation_min': 60,
            'fundamental_min': 70,
            'confidence_base': 65,
            'confidence_max': 85,
            'valuation_weight': 0.4,
            'fundamental_weight': 0.2
        },
        'value': {                               # COMPRA POR VALOR
            'valuation_min': 60,
            'fundamental_min': 60,
            'confidence_base': 55,
            'confidence_max': 75,
            'valuation_weight': 0.4
        },
        'quality': {                             # COMPRA POR CALIDAD
            'valuation_min': 45,
            'fundamental_min': 85,
            'confidence_base': 60,
            'confidence_max': 75,
            'fundamental_weight': 0.5
        }
    },
    'sell': {
        'strong': {                              # VENTA FUERTE
            'valuation_max': 20,
            'fundamental_max': 60,
            'confidence_base': 70,
            'confidence_max': 95,
            'valuation_weight': 0.9,
            'fundamental_weight': 0.5
        },
        'moderate': {                            # VENTA MODERADA
            'valuation_max': 40,
            'fundamental_max': 65,
            'confidence_base': 55,
            'confidence_max': 85,
            'valuation_weight': 0.8,
            'fundamental_weight': 0.4
        },
        'overvalued': {                          # VENTA POR SOBREVALORACIÓN
            'valuation_max': 15,
            'confidence_base': 60,
            'confidence_max': 75,
            'valuation_weight': 1.0
        }
    },
    'hold': {
        'mixed_quality_price': {                 # Calidad pero cara
            'valuation_max': 40,
            'fundamental_min': 85,
            'confidence': 50
        },
        'mixed_moderate': {
            'valuation_max': 45,
            'fundamental_min': 75,
            'confidence': 50
        },
        'default': {
            'confidence': 50
        }
    }
}

# CONFIGURACIÓN DE PRICE TARGET
# Basados en metodologías DCF y múltiplos comparables
PRICE_TARGET_CONFIG = {
    'peg_method': {
        'fair_peg': 1.0,        # Peter Lynch: PEG = 1.0 es fair value
        'adjustment_divisor_bear': 100, # Ajuste si sobrevalorado
        'adjustment_divisor_bull': 200  # Ajuste si infravalorado
    },
    'pe_method': {
        'growth_multiplier': 1.5,                # Fair P/E = Growth × 1.5
        'pe_weight': 0.3,
        'growth_weight': 0.7,
        'fair_multiplier_base': 0.7,
        'fair_multiplier_range': 0.6,
        'adjustment_divisor': 333,
        'earnings_growth_threshold': 2
    },
    'analyst_method': {
        'confidence_base': 0.5,
        'confidence_range': 0.5,
    },
    'score_method': {
        'adjustment_divisor_bear': 250,
        'adjustment_divisor_bull': 333
    }
}

# CONFIGURACIÓN DE OVERALL VALUATION
# Lógica de consenso para determinar si empresa está cara o barata
OVERALL_VALUATION_LOGIC = {
    'thresholds': {
        'pe': {
            'cheap': 15,                         # P/E <15 = barato
            'expensive': 25,                     # P/E >25 = caro
            'min_valid': 0,
            'max_valid': 100
        },
        'ev_ebitda': {
            'cheap': 10,                         # EV/EBITDA <10 = barato
            'expensive': 15,                     # EV/EBITDA >15 = caro
            'min_valid': 0,
            'max_valid': 100
        },
        'fcf_yield': {
            'cheap': 0.06,                       # FCF Yield >6% = barato
            'expensive': 0.02,                   # FCF Yield <2% = caro
            'min_valid': -0.5,
            'max_valid': 0.5
        },
        'pb': {
            'cheap': 2,                          # P/B <2 = barato
            'expensive': 5,                      # P/B >5 = caro
            'min_valid': 0,
            'max_valid': 50
        }
    },
    'voting': {
        'min_valid_metrics': 2,
        'min_votes_for_decision': 2
    }
}

# ============================================================================
# VALUATION MULTIPLES - CONSTANTES INTERMEDIAS
# ============================================================================
# Valores de fallback para clasificación cuando no hay thresholds personalizados
# Estos valores son medias/intermedios entre umbrales cheap y expensive

VALUATION_MULTIPLES_FALLBACKS = {
    'pe_ratio': {
        'fair': 18,              # Media entre cheap (12) y expensive (25)
        'very_expensive': 35     # P/E >35x considerado muy caro
    },
    'ev_ebitda': {
        'fair': 12,              # Media entre cheap (10) y expensive (15)
        'very_expensive': 20     # EV/EBITDA >20x considerado muy caro
    },
    'fcf_yield': {
        'good': 0.05,            # FCF Yield 5% es bueno
        'fair': 0.03             # FCF Yield 3% es fair
    }
}

# CONFIGURACIÓN DE ANÁLISIS DE SECTOR
SECTOR_ANALYSIS_CONFIG = {
    'max_peers': 10,
    'percentile_thresholds': {
        'top_performer': 80,
        'above_average': 60,
        'average': 40,
        'below_average': 20
    },
    'percentile_labels': {
        'top': 'Top performer del sector',
        'above': 'Por encima del promedio',
        'average': 'En el promedio del sector',
        'below': 'Por debajo del promedio',
        'bottom': 'Rezagado del sector'
    }
}

# Peers por sector (yfinance suele devolver "Technology", "Financial Services", etc.)
SECTOR_PEERS = {
    'Technology': ['MSFT', 'AAPL', 'GOOGL', 'META', 'AMZN', 'NVDA', 'ORCL', 'ADBE', 'CRM'],
    'Financial Services': ['JPM', 'BAC', 'GS', 'MS', 'C', 'WFC', 'BLK'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK'],
    'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX'],
    'Communication Services': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ'],
    'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'PM'],
    'Industrials': ['CAT', 'HON', 'UPS', 'UNP', 'BA', 'GE', 'DE'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC'],
    'Basic Materials': ['LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM'],
}

# CONFIGURACIÓN DE REPORTES
REPORTING_CONFIG = {
    'top_opportunities': 5,
    'max_reasons_display': 2,
    'line_width': 80,
    'days_per_year': 365
}

# ============================================================================
# SCORE INTERPRETATION CONFIGURATION
# ============================================================================

SCORE_INTERPRETATION = {
    'default_na_score': 50,              # Score por defecto si no hay datos
    
    # Umbrales para reason_generator (interpretación textual)
    'reasons': {
        'valuation': {
            'excellent': 70,             # Valoración muy atractiva
            'good': 55,                  # Valoración razonable
            'poor': 45,                  # Valoración elevada
            'very_poor': 30              # Sobrevalorada
        },
        'fundamental': {
            'exceptional': 80,           # Calidad excepcional
            'good': 70,                  # Sólidos fundamentales
            'weak': 50                   # Fundamentales débiles
        },
        'profitability': {
            'excellent': 80,             # Excelente rentabilidad
            'poor': 40                   # Rentabilidad baja
        },
        'health': {
            'excellent': 80,             # Balance sólido
            'poor': 40                   # Riesgo financiero
        },
        'growth': {
            'high': 75,                  # Alto crecimiento
            'low': 35                    # Crecimiento limitado
        }
    },
    
    # Umbrales para formatters (emojis y visuales)
    'visual': {
        'excellent': 80,                 # 🟢 Verde
        'good': 60,                      # 🟡 Amarillo
        'fair': 40                       # 🟠 Naranja
                                         # <40 🔴 Rojo
    }
}

# ============================================================================
# COMPANY ANALYSIS CONFIGURATION (Valuation Module)
# ============================================================================

# Pesos por defecto para análisis de empresa
COMPANY_ANALYSIS_WEIGHTS = {
    'default': {
        'profitability': 0.25,
        'financial_health': 0.25,
        'growth': 0.20,
        'efficiency': 0.15,
        'valuation': 0.15
    }
}

# Perfiles de inversión predefinidos
INVESTMENT_PROFILES = {
    'balanced': {
        'profitability': 0.30,
        'financial_health': 0.30,
        'growth': 0.15,
        'efficiency': 0.10,
        'valuation': 0.15,
        'description': 'Para inversores que buscan equilibrio'
    },
    'value': {
        'profitability': 0.20,
        'financial_health': 0.25,
        'growth': 0.10,
        'efficiency': 0.10,
        'valuation': 0.35,  # Énfasis en valoración
        'description': 'Para inversores de valor (bajo P/E, alto dividend yield)'
    },
    'growth': {
        'profitability': 0.15,
        'financial_health': 0.15,
        'growth': 0.45,  # Énfasis en crecimiento
        'efficiency': 0.10,
        'valuation': 0.15,
        'description': 'Para inversores de crecimiento (alto revenue growth)'
    },
    'quality': {
        'profitability': 0.35,  # Énfasis en rentabilidad
        'financial_health': 0.35,  # y salud financiera
        'growth': 0.15,
        'efficiency': 0.10,
        'valuation': 0.05,
        'description': 'Para inversores que priorizan calidad (alto ROE, bajo debt)'
    }
}

# Umbrales de conclusión para scores
CONCLUSION_THRESHOLDS = {
    'excellent': 80,
    'good': 65,
    'fair': 50,
    'weak': 35,
    'labels': {
        'excellent': 'EXCELLENT',
        'good': 'GOOD',
        'fair': 'FAIR',
        'weak': 'WEAK',
        'critical': 'CRITICAL',
        'insufficient': 'INSUFFICIENT DATA'
    }
}

# ============================================================================
# CONSTANTES DE ACCESO RÁPIDO (para imports directos)
# ============================================================================

# Factores de anualización
ANNUAL_FACTOR = RISK_ANALYSIS['annual_factor']

# Niveles de confianza
DEFAULT_CONFIDENCE_LEVEL = RISK_ANALYSIS['default_confidence_level']
DEFAULT_CONFIDENCE_LEVELS = RISK_ANALYSIS['default_confidence_levels']

# Monte Carlo
MONTE_CARLO_SIMULATIONS = RISK_ANALYSIS['monte_carlo']['n_simulations']
MONTE_CARLO_SEED = RISK_ANALYSIS['monte_carlo']['seed']

# Rolling windows
ROLLING_WINDOW = RISK_ANALYSIS['rolling']['default_window']

# Estadística
SIGNIFICANCE_LEVEL = STATISTICAL_DEFAULTS['significance_level']
MIN_OBSERVATIONS = STATISTICAL_DEFAULTS['min_observations']

# Numérico
EPSILON = NUMERICAL_DEFAULTS['epsilon']
TOLERANCE = NUMERICAL_DEFAULTS['tolerance']

# Optimización
OPTIMIZATION_METHOD = OPTIMIZATION_DEFAULTS['method']
FRONTIER_POINTS = OPTIMIZATION_DEFAULTS['frontier_points']

# Columnas yfinance
ADJ_CLOSE_COL = YFINANCE_COLUMNS['adj_close']
CLOSE_COL = YFINANCE_COLUMNS['close']

# Portfolio
PORTFOLIO_MIN_SCORE = PORTFOLIO_CONFIG['selection']['min_score']
PORTFOLIO_MAX_COMPANIES = PORTFOLIO_CONFIG['selection']['max_companies']
PORTFOLIO_LOOKBACK_YEARS = PORTFOLIO_CONFIG['dates']['default_lookback_years']
PORTFOLIO_RISK_FREE_RATE = PORTFOLIO_CONFIG['optimization']['risk_free_rate']

# Fechas de análisis
ANALYSIS_START_DATE = ANALYSIS_DATES['start_date']
ANALYSIS_END_DATE = ANALYSIS_DATES['end_date']
USE_CURRENT_DATE = ANALYSIS_DATES['use_current_date_as_end']
ANALYSIS_LOOKBACK_YEARS = ANALYSIS_DATES['default_lookback_years']

DEFAULT_NA_SCORE = SCORE_INTERPRETATION['default_na_score']