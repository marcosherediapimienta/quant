# projects/quant/pm/utils/analysis/portfolio/components/selector.py
import pandas as pd
from typing import List, Dict
from ...valuation.metrics.score_extractor import ScoreExtractor
from ....tools.config import PORTFOLIO_CONFIG

class CompanySelector:

    def __init__(
        self,
        min_score: float = 0.0,
        max_companies: int = 0,
        max_per_sector: int = 0,
        default_sector: str = ''
    ):
        config = PORTFOLIO_CONFIG['selection']
        defaults = PORTFOLIO_CONFIG['defaults']
        
        self.min_score = min_score if min_score > 0 else config['min_score']
        self.max_companies = max_companies if max_companies > 0 else config['max_companies']
        self.max_per_sector = max_per_sector if max_per_sector > 0 else config['max_per_sector']
        self.default_sector = default_sector if default_sector else defaults['sector_name']
        self.score_extractor = ScoreExtractor()
        self.scoring_weights = PORTFOLIO_CONFIG['scoring_weights']
    
    def select(
        self,
        analysis_results: Dict[str, Dict],
        method: str = ''
    ) -> List[str]:

        if not method:
            method = PORTFOLIO_CONFIG['selection']['default_method']
        
        df = self._to_dataframe(analysis_results)
        df = df[df['total'] >= self.min_score].copy()
        
        if df.empty:
            return []
        
        df = self._score_by_method(df, method)
        selected = self._apply_diversification(df)
        
        return selected[:self.max_companies]
    
    def _to_dataframe(self, results: Dict) -> pd.DataFrame:
        rows = []
        for ticker, analysis in results.items():
            if not analysis.get('success'):
                continue
            
            scores = analysis.get('scores', {})
            rows.append({
                'ticker': ticker,
                'sector': analysis.get('sector', self.default_sector),
                'total': scores.get('total', 0),
                'profitability': self.score_extractor.extract_profitability(analysis),
                'health': self.score_extractor.extract_health(analysis),
                'growth': self.score_extractor.extract_growth(analysis),
                'valuation': self.score_extractor.extract_valuation(analysis),
            })
        
        return pd.DataFrame(rows)
    
    def _score_by_method(self, df: pd.DataFrame, method: str) -> pd.DataFrame:

        if method == 'balanced':
            w = self.scoring_weights['balanced']
            df['final_score'] = (
                df['profitability'] * w['profitability'] +
                df['health'] * w['health'] +
                df['growth'] * w['growth'] +
                df['valuation'] * w['valuation']
            )
        elif method == 'value':
            w = self.scoring_weights['value']
            df['final_score'] = df['total'] * w['total'] + df['valuation'] * w['valuation']
        elif method == 'growth':
            w = self.scoring_weights['growth']
            df['final_score'] = df['total'] * w['total'] + df['growth'] * w['growth']
        else:  
            df['final_score'] = df['total']
        
        return df.sort_values('final_score', ascending=False)
    
    def _apply_diversification(self, df: pd.DataFrame) -> List[str]:
        selected = []
        sector_count = {}
        
        for _, row in df.iterrows():
            if len(selected) >= self.max_companies:
                break
            
            sector = row['sector']
            if sector_count.get(sector, 0) < self.max_per_sector:
                selected.append(row['ticker'])
                sector_count[sector] = sector_count.get(sector, 0) + 1
        
        return selected