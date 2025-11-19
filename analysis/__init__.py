"""
Analysis module for ELECTRA NLP artifact detection.

Contains tools for:
- Error analysis
- Contrast set evaluation
- Visualization
"""

from .error_analysis import analyze_predictions, ErrorAnalyzer
from .contrast_sets import evaluate_contrast_set, ContrastSetEvaluator
from .visualization import plot_performance_breakdown, plot_error_types

__all__ = [
    "analyze_predictions",
    "ErrorAnalyzer",
    "evaluate_contrast_set",
    "ContrastSetEvaluator",
    "plot_performance_breakdown",
    "plot_error_types",
]
