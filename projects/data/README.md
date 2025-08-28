# 📊 Sistema de Gestión de Datos

Sistema simple y eficiente para gestionar datos financieros.

## 🏗️ Estructura

```
projects/data/
├── data_manager.py          # Clase principal del Data Manager
├── config/                  # Configuración del sistema
│   └── data_config.yaml    # Configuración básica
└── test/                    # Scripts de prueba
    └── test_data_manager.py # Prueba básica del sistema
```

## 🚀 Uso Básico

```python
from data_manager import DataManager

# Inicializar
dm = DataManager()

# Descargar datos directamente
symbols = ['AAPL', 'MSFT', 'GOOGL']
prices = dm.download_market_data(symbols, start_date="2024-01-01")

# Calcular retornos
returns = dm.calculate_returns(prices)

# Obtener resumen
summary = dm.get_data_summary(prices, returns)
```

## ⚙️ Configuración

Edita `config/data_config.yaml` para ajustar:
- Calidad de datos
- Configuración de caché
- Parámetros de descarga
- Nivel de logging

## 🧪 Pruebas

Para probar el sistema:

```bash
cd test
python test_data_manager.py
```

## 🎯 Características

- **Simple**: Solo lo esencial, sin complejidades innecesarias
- **Flexible**: Soporta cualquier instrumento de Yahoo Finance
- **Robusto**: Manejo de errores y sistema de caché
- **Eficiente**: Descarga y procesamiento optimizado

## 📊 Tipos de Activos Soportados

- **Stocks**: AAPL, MSFT, GOOGL, TSLA, etc.
- **ETFs**: SPY, QQQ, VTI, GLD, TLT, etc.
- **Crypto**: BTC-USD, ETH-USD, etc.
- **Bonds**: BND, AGG, LQD, etc.
- **Commodities**: GLD, SLV, USO, etc.
- **Cualquier instrumento disponible en Yahoo Finance**

## 🔧 Personalización

El sistema es completamente configurable desde `config/data_config.yaml` sin necesidad de modificar código.
