from typing import Dict, List
import pandas as pd
import warnings
# Suprimir warnings de Pandas4Warning de yfinance
warnings.filterwarnings('ignore', category=pd.errors.Pandas4Warning, module='yfinance')
import yfinance as yf

class MacroDataLoader:

    def __init__(self, factors: Dict[str, str] = None):
        self.factors = factors if factors is not None else {}
    
    def download(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        progress: bool = False
    ) -> pd.DataFrame:

        try:
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=progress,
                auto_adjust=True,
                threads=True
            )
            
            if data.empty:
                raise ValueError("No se descargaron datos")
            
            return data
            
        except Exception as e:
            raise RuntimeError(f"Error en descarga: {e}")
    
    def download_single(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        progress: bool = False
    ) -> pd.Series:

        data = self.download([ticker], start_date, end_date, progress)
        
        if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
            return data['Close'].dropna()
        return data.squeeze()