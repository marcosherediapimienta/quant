from typing import Dict, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from ..tools.config import (
    MACRO_TRANSFORMS,
    MACRO_SPREADS,
    YIELD_SCALE,
)

@dataclass
class TransformConfig:
    """Configuración de transformación para un factor."""
    factor_name: str
    is_yield: bool
    is_diff: bool
    is_log_return: bool
    yield_scale: float = None


class MacroTransformCalculator:
    """
    Transforma factores macro a retornos/cambios.
    
    Responsabilidad única: Aplicar transformaciones apropiadas a series macro.
    
    Transformaciones:
    - Log returns: Para precios y commodities
    - Diferencias: Para tasas de interés
    - Scaling: Para yields (bps a decimal)
    - Business day resampling: Alineación temporal
    - Spreads: Diferencias y ratios entre factores
    """
    
    def __init__(
        self,
        transform_config: Dict = None,
        spreads_config: Dict = None,
        yield_scale: float = None
    ):
        """
        Inicializa el transformador.
        
        Args:
            transform_config: Config de transformaciones (None = usar default)
            spreads_config: Config de spreads (None = usar default)
            yield_scale: Factor de escala para yields (None = usar config)
        """
        self.transform_config = transform_config if transform_config is not None else MACRO_TRANSFORMS
        self.spreads_config = spreads_config if spreads_config is not None else MACRO_SPREADS
        self.yield_scale = yield_scale if yield_scale is not None else YIELD_SCALE
    
    def calculate_log_returns(self, series: pd.Series) -> pd.Series:
        """
        Calcula log returns de una serie de precios.
        
        Formula: r_t = ln(P_t / P_{t-1})
        
        Args:
            series: Serie de precios
            
        Returns:
            Serie de log returns
            
        Ventajas:
        - Simétricos: ln(1.1) ≈ -ln(0.9)
        - Aditivos en el tiempo: r_total = Σ r_t
        - Aproximación lineal para retornos pequeños
        """
        invalid_values = (series <= 0).sum()
        if invalid_values > 0:
            print(f"[Macro] {series.name}: {invalid_values} valores <= 0, reemplazados por NaN")

        series_clean = series.copy()
        series_clean[series_clean <= 0] = np.nan
        return np.log(series_clean).diff()
    
    def to_business_daily(self, series: pd.Series) -> pd.Series:
        """
        Resamplea serie a días hábiles (business days).
        
        Args:
            series: Serie temporal
            
        Returns:
            Serie con frecuencia de días hábiles, forward filled
            
        Uso:
        - Alinea datos macro (mensuales/semanales) con retornos diarios
        - Forward fill: usa el último valor disponible
        """
        return series.asfreq('B').ffill()
    
    def scale_yield(self, series: pd.Series) -> pd.Series:
        """
        Escala yields (tasas) al formato decimal.
        
        Args:
            series: Serie de yields (ej: en basis points)
            
        Returns:
            Serie escalada (ej: de bps a decimal)
            
        Ejemplo:
        - Input: 250 bps
        - Output: 0.025 (2.5%)
        """
        return series / self.yield_scale
    
    def transform_single_factor(
        self,
        factor_name: str,
        series: pd.Series
    ) -> pd.Series:
        """
        Aplica transformación apropiada a un factor.
        
        Args:
            factor_name: Nombre del factor
            series: Serie temporal del factor
            
        Returns:
            Serie transformada
            
        Lógica:
        1. Yields → scale + diff (si es diff_factor)
        2. Diff factors → diff
        3. Log return factors → log returns
        4. Otros → sin transformación
        """
        series = series.dropna()
        if len(series) == 0:
            return series

        if factor_name in self.transform_config.get('yield_factors', []):
            series = self.scale_yield(series)
            if factor_name in self.transform_config.get('diff_factors', []):
                return series.diff()
            return series

        if factor_name in self.transform_config.get('diff_factors', []):
            return series.diff()

        if factor_name in self.transform_config.get('log_return_factors', []):
            return self.calculate_log_returns(series)

        return series
    
    def transform_all_factors(
        self,
        factors_data: Dict[str, pd.Series],
        target_index: pd.DatetimeIndex = None,
        fill_method: str = 'ffill'
    ) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:
        """
        Transforma todos los factores y los alinea en un DataFrame.
        
        Args:
            factors_data: Dict {nombre: serie}
            target_index: Índice objetivo (None = unión de todos)
            fill_method: Método de relleno ('ffill', 'bfill', None)
            
        Returns:
            Tuple (dict_transformado, dataframe_alineado)
            
        Proceso:
        1. Transforma cada factor individualmente
        2. Resamplea a business days
        3. Alinea todos a un índice común
        4. Rellena valores faltantes
        """
        transformed = {}

        for name, series in factors_data.items():
            try:
                trans = self.transform_single_factor(name, series)
                trans = self.to_business_daily(trans)
                transformed[name] = trans
            except Exception as e:
                print(f"[Macro] Error transformando {name}: {e}")

        if target_index is None:
            all_indices = [s.index for s in transformed.values() if len(s) > 0]
            if not all_indices:
                return transformed, pd.DataFrame()
            target_index = pd.DatetimeIndex(sorted(set().union(*all_indices)))

        df = pd.DataFrame(index=target_index)
        for name, series in transformed.items():
            aligned = series.reindex(df.index)
            if fill_method:
                if fill_method == 'ffill':
                    aligned = aligned.ffill()
                elif fill_method == 'bfill':
                    aligned = aligned.bfill()
            df[name] = aligned
        
        return transformed, df
    
    def calculate_spread(
        self,
        data: pd.DataFrame,
        spread_name: str
    ) -> pd.Series:
        """
        Calcula un spread entre dos factores.
        
        Args:
            data: DataFrame con factores
            spread_name: Nombre del spread (debe estar en spreads_config)
            
        Returns:
            Serie del spread calculado
            
        Tipos de spread:
        - 'diff': long - short (ej: spread de tasas)
        - 'ratio': long / short (ej: ratio de commodities)
        
        Ejemplos:
        - TED spread: LIBOR - T-Bill
        - Yield curve: 10Y - 2Y
        - Credit spread: Corporate - Treasury
        """
        if spread_name not in self.spreads_config:
            raise ValueError(f"Spread '{spread_name}' no encontrado en config")
        
        config = self.spreads_config[spread_name]

        if 'long' in config and 'short' in config:
            long_col = config['long']
            short_col = config['short']
        elif 'risky' in config and 'safe' in config:
            long_col = config['risky']
            short_col = config['safe']
        else:
            raise ValueError(f"Config de spread inválida: {spread_name}")
        
        if long_col not in data.columns or short_col not in data.columns:
            raise ValueError(f"Columnas no encontradas: {long_col}, {short_col}")

        transform = config.get('transform', 'diff')
        
        if transform == 'diff':
            spread = data[long_col] - data[short_col]
        elif transform == 'ratio':
            spread = data[long_col] / data[short_col]
        else:
            raise ValueError(f"Transform no soportado: {transform}")
        
        spread.name = spread_name
        return spread
    
    def calculate_all_spreads(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calcula todos los spreads configurados.
        
        Args:
            data: DataFrame con factores base
            
        Returns:
            DataFrame con todos los spreads
            
        Uso:
        - Genera variables derivadas útiles
        - Ejemplo: yield curve spreads, credit spreads, commodity ratios
        """
        spreads = {}
        for spread_name in self.spreads_config.keys():
            try:
                spread = self.calculate_spread(data, spread_name)
                spreads[spread_name] = spread
            except Exception as e:
                print(f"[Macro] Error calculando spread {spread_name}: {e}")
        
        return pd.DataFrame(spreads)
    
    def align_to_portfolio(
        self,
        macro_data: pd.DataFrame,
        portfolio_returns: pd.Series
    ) -> pd.DataFrame:
        """
        Alinea factores macro al índice de retornos del portfolio.
        
        Args:
            macro_data: DataFrame con factores macro
            portfolio_returns: Serie de retornos del portfolio
            
        Returns:
            DataFrame alineado y forward filled
            
        Uso:
        - Paso final antes de regresiones
        - Asegura que factores y retornos tengan mismo índice
        """
        return macro_data.reindex(portfolio_returns.index).ffill()