# Changelog - Módulo de Análisis de Valoración de Stocks v4

## Resumen de Cambios

Este documento describe los cambios realizados para refactorizar el Módulo de Análisis de Valoración de Stocks de la v3 a la v4, implementando mejoras específicas en el cálculo de FCF TTM, EV corporativo, manejo de equity no positivo, y peers para servicios de crédito/pagos.

## Nuevas Características

### 1. FCF TTM Robusto
- **Antes**: Usaba `info["freeCashflow"]` directamente
- **Ahora**: Calcula FCF_TTM = Σ(CFO_quarter últimos 4) – Σ(Capex_quarter últimos 4) desde `stock.quarterly_cashflow`
- **Fallback**: Si falla, usa datos anuales (`stock.cashflow`) sin inventar datos
- **Manejo de signo**: Gestiona correctamente el signo de capex (muchas veces viene negativo)

### 2. EV Corporativo
- **Nuevo**: Calcula EV_corp = MarketCap + Deuda_corp – Caja_corp
- **Fuente**: Extrae caja y deuda desde el balance (trimestral con fallback anual)
- **Exclusión**: No usa saldos fiduciarios de customer funds
- **Métrica derivada**: FCF/EV Yield cuando EV_corp > 0

### 3. Manejo de Equity ≤ 0
- **Detección**: Si equity del último balance ≤ 0, marca P/B y ROE como NM
- **Exclusión**: P/B y ROE no entran en el scoring cuando son NM
- **Renderizado**: Muestra "NM" en lugar de "N/D" en el reporte

### 4. Peers para Credit Services/Payments
- **Nuevos peers**: ["V","MA","AXP","COF","DFS","SYF","SQ","FI","FIS","GPN","ADYEN.AS"]
- **Industrias**: "Credit Services" y "Payments"
- **Fallback**: Mantiene el sistema de fallback sector/universe existente

### 5. Fecha Real de Fundamentales
- **Antes**: Fecha fija del 31/12 del año anterior
- **Ahora**: Toma la columna más reciente de `quarterly_financials`, `quarterly_cashflow`, o `quarterly_balance_sheet`
- **Fallback**: Si no hay datos trimestrales, usa la más reciente anual

### 6. Reporte Enriquecido
- **Nuevas líneas**:
  - "EV (corp): $..." (si disponible)
  - "FCF/EV Yield: ...%" (si ev_corp > 0)
  - "FCF TTM: $..."
- **Renderizado NM**: P/B y ROE muestran "NM" cuando corresponda
- **Conteo correcto**: `peers_used` excluye el propio ticker

## Cambios Técnicos Detallados

### Nuevas Funciones Helper

```python
# Constantes para extraer datos de yfinance
CF_CFO_KEYS = [...]  # Claves para Cash Flow from Operations
CF_CAPEX_KEYS = [...]  # Claves para Capital Expenditures
BS_EQUITY_KEYS = [...]  # Claves para Total Stockholder Equity
BS_CASH_KEYS = [...]  # Claves para Cash and Cash Equivalents
BS_DEBT_KEYS = [...]  # Claves para Total Debt

# Funciones de extracción
_get_row_sum_last_n(df, keys, n=4)  # Suma últimos n períodos
_get_row_last(df, keys)  # Último valor disponible
_compute_fcf_ttm(stock)  # FCF TTM robusto
_get_equity_last(stock)  # Equity del último balance
_get_cash_debt_last(stock)  # Caja y deuda del balance
_compute_ev_corporativo(info, stock)  # EV corporativo
_equity_nonpositive(stock)  # Verifica si equity ≤ 0
_latest_fund_date(stock)  # Fecha real de fundamentales
```

### Modificaciones en Funciones Existentes

#### `fetch_snapshot()`
- Calcula `fcf_ttm` usando `_compute_fcf_ttm()`
- Calcula `ev_corp` usando `_compute_ev_corporativo()`
- Calcula `fcf_ev_yield` cuando `ev_corp > 0`
- Marca `pb` y `roe` como `np.nan` si `equity_nonpositive`
- Usa `_latest_fund_date()` para fecha de fundamentales
- Añade campos: `fcf_ttm`, `fcf_ev_yield`, `ev_corp`, `equity_nonpositive`

#### `fetch_peers()`
- Añade peers para "Credit Services" y "Payments"
- Mantiene fallback sector/universe existente

#### `compute_score()`
- Corrige conteo de `peers_used` (excluye propio ticker)
- Añade `ev_corp` a `basics`
- Añade `fcf_ttm` y `fcf_ev_yield` a `valuation`
- Añade `meta.equity_nonpositive`

#### `render_stock_report()`
- Actualiza header a "v4"
- Añade líneas para EV (corp), FCF TTM, FCF/EV Yield
- Implementa renderizado "NM" para P/B y ROE cuando `equity_nonpositive`
- Mejora formateo de métricas

## Estructura de Datos Actualizada

### Nuevos Campos en Resultado

```python
result = {
    'basics': {
        'ev_corp': float,  # Enterprise Value corporativo
        # ... campos existentes
    },
    'valuation': {
        'fcf_ttm': float,  # FCF TTM calculado
        'fcf_ev_yield': float,  # FCF/EV Yield
        # ... campos existentes
    },
    'meta': {
        'equity_nonpositive': bool  # True si equity ≤ 0
    }
}
```

## Pruebas Recomendadas

### Pruebas Manuales
1. **PYPL**: Verificar FCF Yield mejorado y peers de Credit Services
2. **HPQ**: Verificar manejo de equity ≤ 0 (P/B y ROE como NM)
3. **Peers**: Verificar conteo correcto excluyendo propio ticker
4. **Reporte**: Verificar "EV (corp)" y "FCF/EV Yield" cuando procede

### Casos de Prueba
- Stocks con equity positivo vs negativo
- Industrias Credit Services/Payments vs otras
- Datos trimestrales vs anuales disponibles
- Múltiples stocks en comparación

## Compatibilidad

- **API Pública**: Mantiene firma de funciones existentes
- **Campos existentes**: No se modifican, solo se añaden nuevos
- **Comportamiento**: Mejora la precisión sin romper funcionalidad existente

## Archivos Modificados

- `projects/quant/tests/test5.py`: Refactorización completa a v4
- `test_v4.py`: Script de prueba para nuevas funcionalidades
- `CHANGELOG_v4.md`: Este documento

## Conclusión

La v4 representa una mejora significativa en la precisión y robustez del análisis de valoración, especialmente en:

1. **Cálculo de FCF**: Más preciso usando datos trimestrales
2. **EV corporativo**: Mejor representación del valor empresarial
3. **Manejo de casos extremos**: Equity negativo manejado correctamente
4. **Peers específicos**: Mejor comparación para industrias especializadas
5. **Datos actualizados**: Fechas reales de fundamentales

Todos los cambios son retrocompatibles y mantienen la API existente.
