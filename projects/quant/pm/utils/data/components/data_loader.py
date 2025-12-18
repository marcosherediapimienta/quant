from typing import List
import pandas as pd
import yfinance as yf
from ...tools.config import DOWNLOAD_DEFAULTS 

class DataLoader:
    
    def download(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        auto_adjust: bool = None,
        group_by: str = None,
        threads: bool = None,
        progress: bool = None
    ) -> pd.DataFrame:

        auto_adjust = auto_adjust if auto_adjust is not None else DOWNLOAD_DEFAULTS['auto_adjust']
        group_by = group_by if group_by is not None else DOWNLOAD_DEFAULTS['group_by']
        threads = threads if threads is not None else DOWNLOAD_DEFAULTS['threads']
        progress = progress if progress is not None else DOWNLOAD_DEFAULTS['progress']
        
        print(f"Período: {start_date} → {end_date}")
        
        try:
            data = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                auto_adjust=auto_adjust,
                group_by=group_by,
                threads=threads,
                progress=progress
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
        **kwargs
    ) -> pd.DataFrame:
        return self.download([ticker], start_date, end_date, **kwargs)