from datetime import datetime, timedelta
from typing import Tuple

class DateCalculator:
    @staticmethod
    def get_current_date_str(fmt: str = '%Y-%m-%d') -> str:
        return datetime.now().strftime(fmt)

    @staticmethod
    def get_lookback_date_from_years(years: int, fmt: str = '%Y-%m-%d') -> str:
        days = 365 * years
        start = datetime.now() - timedelta(days=days)
        return start.strftime(fmt)

    @staticmethod
    def get_lookback_date_from_days(days: int, fmt: str = '%Y-%m-%d') -> str:
        start = datetime.now() - timedelta(days=days)
        return start.strftime(fmt)
        
    @staticmethod
    def get_date_range(
        start_date: str = '',
        end_date: str = '',
        lookback_years: int = 5,
        fmt: str = '%Y-%m-%d'
    ) -> Tuple[str, str]:

        if not end_date:
            end_date = DateCalculator.get_current_date_str(fmt)
        
        if not start_date:
            start_date = DateCalculator.get_lookback_date_from_years(lookback_years, fmt)
        
        return start_date, end_date