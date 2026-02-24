import numpy as np
import pandas as pd
from typing import Dict, List
from .company_analyzer import CompanyAnalyzer, AnalysisWeights, ConclusionThresholds

_SCORE_CATEGORIES = ('profitability', 'financial_health', 'growth', 'efficiency', 'valuation')
_STAT_CATEGORIES = (*_SCORE_CATEGORIES, 'total')

class ComparisonAnalyzer:
    def __init__(self, company_analyzer: CompanyAnalyzer = None):
        self.company_analyzer = company_analyzer or CompanyAnalyzer()
    
    @classmethod
    def with_config(
        cls,
        weights: AnalysisWeights = None,
        conclusion_thresholds: ConclusionThresholds = None,
        **kwargs
    ) -> 'ComparisonAnalyzer':

        analyzer = CompanyAnalyzer(
            weights=weights,
            conclusion_thresholds=conclusion_thresholds,
            **kwargs
        )
        return cls(company_analyzer=analyzer)
    
    def compare(self, tickers: List[str]) -> Dict:
        results = self.company_analyzer.analyze_multiple(tickers)
        valid_results = {
            t: r for t, r in results.items() 
            if r.get('success', False)
        }
        
        if not valid_results:
            return {
                'error': 'Could not analyze the given companies',
                'individual_results': results,
                'success': False
            }

        ranking = self._create_ranking(valid_results)
        category_leaders = self._identify_leaders(valid_results)
        group_stats = self._calculate_group_stats(valid_results)
        
        return {
            'individual_results': results,
            'valid_count': len(valid_results),
            'ranking': ranking,
            'category_leaders': category_leaders,
            'group_stats': group_stats,
            'summary_df': self.company_analyzer.get_summary_df(results),
            'success': True
        }
    
    @staticmethod
    def _create_ranking(results: Dict[str, Dict]) -> List[Dict]:
        ranking_data = []
        
        for ticker, result in results.items():
            score = result.get('scores', {}).get('total')
            if pd.notna(score):
                ranking_data.append({
                    'ticker': ticker,
                    'name': result.get('company_name', ticker),
                    'score': score,
                    'conclusion': result.get('conclusion', {}),
                    'conclusion_overall': result.get('conclusion', {}).get('overall', 'N/A')
                })

        ranking_data.sort(key=lambda x: x['score'], reverse=True)

        for i, item in enumerate(ranking_data, 1):
            item['rank'] = i
        
        return ranking_data
    
    @staticmethod
    def _identify_leaders(results: Dict[str, Dict]) -> Dict:
        leaders = {}
        
        for cat in _SCORE_CATEGORIES:
            scores = [
                (ticker, result.get('company_name', ticker), result.get('scores', {}).get(cat))
                for ticker, result in results.items()
                if pd.notna(result.get('scores', {}).get(cat))
            ]
            
            if scores:
                scores_sorted = sorted(scores, key=lambda x: x[2], reverse=True)
                leaders[cat] = {
                    'best': {'ticker': scores_sorted[0][0], 'name': scores_sorted[0][1], 'score': scores_sorted[0][2]},
                    'worst': {'ticker': scores_sorted[-1][0], 'name': scores_sorted[-1][1], 'score': scores_sorted[-1][2]}
                }
        
        return leaders
    
    @staticmethod
    def _calculate_group_stats(results: Dict[str, Dict]) -> Dict:
        stats = {}
        
        for cat in _STAT_CATEGORIES:
            scores = [
                result.get('scores', {}).get(cat)
                for result in results.values()
                if pd.notna(result.get('scores', {}).get(cat))
            ]
            
            if scores:
                stats[cat] = {
                    'count': len(scores),
                    'mean': float(np.mean(scores)),
                    'median': float(np.median(scores)),
                    'std': float(np.std(scores)) if len(scores) > 1 else 0.0,
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores))
                }
        
        return stats
