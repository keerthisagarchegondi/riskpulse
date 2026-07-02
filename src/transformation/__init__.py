"""Data transformation module - Cleaning, normalization, and feature engineering."""

from src.transformation.aggregator import (
    AggregationResult,
    IncrementalAggregator,
    RunningStatistics,
    TimeWindowAggregator,
    WindowSpec,
    WindowType,
)
from src.transformation.cleaner import CleaningMetrics, CleaningResult, DataCleaner
from src.transformation.feature_engineer import (
    FeatureEngineer,
    FeatureMetrics,
    FeatureResult,
)
from src.transformation.normalizer import (
    DataNormalizer,
    NormalizationMetrics,
    NormalizationResult,
    get_normalizer,
    reset_normalizer,
)

__all__ = [
    "AggregationResult",
    "CleaningMetrics",
    "CleaningResult",
    "DataCleaner",
    "DataNormalizer",
    "FeatureEngineer",
    "FeatureMetrics",
    "FeatureResult",
    "IncrementalAggregator",
    "NormalizationMetrics",
    "NormalizationResult",
    "RunningStatistics",
    "TimeWindowAggregator",
    "WindowSpec",
    "WindowType",
    "get_normalizer",
    "reset_normalizer",
]
