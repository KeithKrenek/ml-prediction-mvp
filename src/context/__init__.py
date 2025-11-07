"""
Real-time Context Integration Module

This module provides real-time context gathering from multiple sources including:
- News APIs (NewsAPI, RSS feeds)
- Trending topics (Google Trends)
- Stock market data (yfinance)
- Political events calendar

The context is used to improve prediction timing and content generation.
"""

from .context_gatherer import RealTimeContextGatherer, ContextCache

__all__ = ['RealTimeContextGatherer', 'ContextCache']
