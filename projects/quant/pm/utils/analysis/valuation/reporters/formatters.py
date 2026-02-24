import numpy as np
import pandas as pd
from dataclasses import dataclass
from ....tools.config import SCORE_INTERPRETATION

@dataclass
class FormatConfig:
    decimal_places: int = 1
    currency_symbol: str = "$"
    percentage_decimals: int = 1
    bar_width: int = 20
    line_width: int = 65

def _to_finite_float(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None

def fmt_pct(value, decimals: int = 1, na_str: str = "N/A") -> str:

    if pd.isna(value):
        return na_str
    v = _to_finite_float(value)
    if v is None:
        return na_str
    return f"{v * 100:.{decimals}f}%"

def fmt_num(value, decimals: int = 2, na_str: str = "N/A") -> str:

    if pd.isna(value):
        return na_str
    v = _to_finite_float(value)
    if v is None:
        return na_str
    return f"{v:.{decimals}f}"

def fmt_money(value, currency: str = "$", na_str: str = "N/A") -> str:
  
    if pd.isna(value):
        return na_str
    v = _to_finite_float(value)
    if v is None:
        return na_str
    
    abs_val = abs(v)
    sign = "-" if v < 0 else ""
    
    if abs_val >= 1e12:
        return f"{sign}{currency}{abs_val/1e12:.2f}T"
    if abs_val >= 1e9:
        return f"{sign}{currency}{abs_val/1e9:.2f}B"
    if abs_val >= 1e6:
        return f"{sign}{currency}{abs_val/1e6:.2f}M"
    if abs_val >= 1e3:
        return f"{sign}{currency}{abs_val/1e3:.2f}K"
    
    return f"{sign}{currency}{abs_val:,.0f}"

def fmt_multiple(value, suffix: str = "x", decimals: int = 2, na_str: str = "N/A") -> str:

    if pd.isna(value):
        return na_str
    v = _to_finite_float(value)
    if v is None:
        return na_str
    return f"{v:.{decimals}f}{suffix}"

def score_bar(score, width: int = 20, fill_char: str = "█", empty_char: str = "░") -> str:

    if pd.isna(score):
        return "[" + "?" * width + "]"
    
    score = max(0, min(100, score))  
    filled = int(score / 100 * width)
    return "[" + fill_char * filled + empty_char * (width - filled) + "]"

def score_emoji(score) -> str:

    if pd.isna(score):
        return "❓"

    thresh = SCORE_INTERPRETATION['visual']
    
    if score >= thresh['excellent']: 
        return "🟢"
    if score >= thresh['good']:  
        return "🟡"
    if score >= thresh['fair']:  
        return "🟠"
    return "🔴"  
    
def classification_emoji(classification: str) -> str:

    mapping = {
        'excellent': '🟢',
        'good': '🟡',
        'fair': '🟠',
        'poor': '🔴',
        'cheap': '🟢',
        'very_expensive': '🔴',
        'expensive': '🟠',
        'N/A': '⚪'
    }
    return mapping.get(classification, '⚪')

def separator(char: str = "─", width: int = 65) -> str:
    return char * width

def header(text: str, char: str = "=", width: int = 65) -> str:
    line = char * width
    return f"{line}\n{text}\n{line}"