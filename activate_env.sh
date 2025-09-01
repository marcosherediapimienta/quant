#!/bin/bash
# Script para activar el entorno virtual del proyecto Quantitative Finance

echo "🚀 Activando entorno virtual Quantitative Finance..."
echo "📁 Directorio: $(pwd)"

# Verificar que existe el entorno virtual
if [ ! -d "venv_quant" ]; then
    echo "❌ Entorno virtual no encontrado."
    echo "💡 Ejecuta primero: python3 -m venv venv_quant && pip install -r requirements.txt"
    exit 1
fi

# Activar entorno virtual
source venv_quant/bin/activate

echo "✅ Entorno virtual activado!"
echo "📊 Versiones instaladas:"
echo "   - Python: $(python --version)"
echo "   - Pandas: $(python -c 'import pandas as pd; print(pd.__version__)')"
echo "   - NumPy: $(python -c 'import numpy as np; print(np.__version__)')"
echo "   - Matplotlib: $(python -c 'import matplotlib; print(matplotlib.__version__)')"
echo "   - Statsmodels: $(python -c 'import statsmodels; print(statsmodels.__version__)')"
echo "   - YFinance: $(python -c 'import yfinance as yf; print(yf.__version__)')"

echo ""
echo "🎯 Para usar el módulo portfolio_analysis:"
echo "   cd projects/quant"
echo "   python -c \"import portfolio_analysis; help(portfolio_analysis)\""
echo ""
echo "🧪 Para ejecutar pruebas:"
echo "   cd projects/quant/test"
echo "   python test_portfolio_analysis.py"
echo ""
echo "🌟 ¡Entorno listo para análisis cuantitativo!"

# Mantener la sesión activa
exec bash
