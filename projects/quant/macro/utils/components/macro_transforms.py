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
    factor_name: str
    is_yield: bool
    is_diff: bool
    is_log_return: bool
    yield_scale: float = None

class MacroTransformCalculator:
    def __init__(
        self,
        transform_config: Dict = None,
        spreads_config: Dict = None,
        yield_scale: float = None
    ):
        self.transform_config = transform_config if transform_config is not None else MACRO_TRANSFORMS
        self.spreads_config = spreads_config if spreads_config is not None else MACRO_SPREADS
        self.yield_scale = yield_scale if yield_scale is not None else YIELD_SCALE
    
    def calculate_log_returns(self, series: pd.Series) -> pd.Series:
        invalid_values = (series <= 0).sum()
        if invalid_values > 0:
            print(f"[Macro] {series.name}: {invalid_values} valores <= 0, reemplazados por NaN")

        series_clean = series.copy()
        series_clean[series_clean <= 0] = np.nan
        return np.log(series_clean).diff()
    
    def to_business_daily(self, series: pd.Series) -> pd.Series:
        if len(series) == 0:
            return series
        try:
            return series.asfreq('B').ffill()
        except (ValueError, AttributeError):
            # Fallback para series sin freq inferible (ej: datos de FRED)
            bdays = pd.bdate_range(start=series.index.min(), end=series.index.max())
            return series.reindex(bdays).ffill()
    
    def scale_yield(self, series: pd.Series) -> pd.Series:
        return series / self.yield_scale
    
    def transform_single_factor(
        self,
        factor_name: str,
        series: pd.Series
    ) -> pd.Series:

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
 
        return macro_data.reindex(portfolio_returns.index).ffill()