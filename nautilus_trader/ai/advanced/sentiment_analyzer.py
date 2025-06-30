# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Market sentiment analysis using AI.
"""

from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
from decimal import Decimal
import statistics

from nautilus_trader.ai.analyzers.base import BaseAIAnalyzer
from nautilus_trader.ai.config import AIAnalyzerConfig
from nautilus_trader.ai.utils.prompts import SentimentAnalysisPrompts
from nautilus_trader.model.identifiers import InstrumentId


class SentimentSource(str, Enum):
    """Sources of sentiment data."""
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    ANALYST_REPORTS = "analyst_reports"
    MARKET_DATA = "market_data"
    MIXED = "mixed"


class SentimentLevel(str, Enum):
    """Sentiment levels."""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


class MarketSentiment:
    """
    Market sentiment data.
    
    Parameters
    ----------
    instrument_id : InstrumentId
        Instrument identifier
    sentiment : SentimentLevel
        Overall sentiment level
    score : float
        Sentiment score (-1 to 1)
    confidence : float
        Confidence in sentiment (0 to 1)
    sources : dict
        Sentiment by source
    timestamp : datetime
        Analysis timestamp
    metadata : dict
        Additional metadata
    """
    
    def __init__(
        self,
        instrument_id: InstrumentId,
        sentiment: SentimentLevel,
        score: float,
        confidence: float,
        sources: Dict[SentimentSource, float],
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.instrument_id = instrument_id
        self.sentiment = sentiment
        self.score = score
        self.confidence = confidence
        self.sources = sources
        self.timestamp = timestamp
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "instrument_id": str(self.instrument_id),
            "sentiment": self.sentiment.value,
            "score": self.score,
            "confidence": self.confidence,
            "sources": {k.value: v for k, v in self.sources.items()},
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class SentimentAnalyzer(BaseAIAnalyzer):
    """
    AI-powered sentiment analyzer.
    
    This analyzer:
    - Analyzes multiple sentiment sources
    - Combines sentiment signals
    - Tracks sentiment trends
    - Identifies sentiment divergences
    - Provides trading signals based on sentiment
    """
    
    def __init__(
        self,
        config: AIAnalyzerConfig,
        cache: Optional[Any] = None,
    ) -> None:
        """Initialize sentiment analyzer."""
        super().__init__(config, cache)
        self.prompts = SentimentAnalysisPrompts()
        self._sentiment_history: Dict[InstrumentId, List[MarketSentiment]] = {}
        
    async def analyze_sentiment(
        self,
        instrument_id: InstrumentId,
        text_data: Optional[Dict[str, List[str]]] = None,
        market_indicators: Optional[Dict[str, Any]] = None,
        lookback_hours: int = 24,
    ) -> MarketSentiment:
        """
        Analyze market sentiment from various sources.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            Instrument to analyze
        text_data : dict, optional
            Text data by source type
        market_indicators : dict, optional
            Market-based sentiment indicators
        lookback_hours : int
            Hours to look back for trend
            
        Returns
        -------
        MarketSentiment
            Analyzed sentiment
        """
        # Prepare sentiment inputs
        inputs = self._prepare_sentiment_inputs(
            instrument_id,
            text_data,
            market_indicators,
        )
        
        # Get historical sentiment
        historical = self._get_recent_sentiment(
            instrument_id,
            lookback_hours,
        )
        
        # Build analysis prompt
        prompt = self.prompts.build_sentiment_analysis_prompt(
            instrument_id=str(instrument_id),
            text_sources=inputs.get("text_sources", {}),
            market_indicators=inputs.get("market_indicators", {}),
            historical_sentiment=[s.to_dict() for s in historical],
        )
        
        # Get AI analysis
        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self.prompts.SENTIMENT_SYSTEM_PROMPT,
            temperature=0.3,
        )
        
        # Parse response
        result = self._parse_json_response(response.content)
        
        # Create sentiment object
        sentiment = MarketSentiment(
            instrument_id=instrument_id,
            sentiment=SentimentLevel(result["sentiment"]),
            score=result["score"],
            confidence=result["confidence"],
            sources={
                SentimentSource(k): v
                for k, v in result.get("sources", {}).items()
            },
            timestamp=datetime.utcnow(),
            metadata={
                "model": response.model,
                "analysis_factors": result.get("analysis_factors", []),
                "key_drivers": result.get("key_drivers", []),
            },
        )
        
        # Store in history
        if instrument_id not in self._sentiment_history:
            self._sentiment_history[instrument_id] = []
        self._sentiment_history[instrument_id].append(sentiment)
        
        # Limit history size
        max_history = 1000
        if len(self._sentiment_history[instrument_id]) > max_history:
            self._sentiment_history[instrument_id] = \
                self._sentiment_history[instrument_id][-max_history:]
        
        return sentiment
        
    async def analyze_sentiment_trend(
        self,
        instrument_id: InstrumentId,
        lookback_hours: int = 72,
    ) -> Dict[str, Any]:
        """
        Analyze sentiment trend over time.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            Instrument to analyze
        lookback_hours : int
            Hours to analyze
            
        Returns
        -------
        Dict[str, Any]
            Trend analysis results
        """
        # Get historical sentiment
        historical = self._get_recent_sentiment(
            instrument_id,
            lookback_hours,
        )
        
        if len(historical) < 2:
            return {
                "trend": "insufficient_data",
                "strength": 0.0,
                "turning_points": [],
                "forecast": "neutral",
            }
        
        # Calculate trend metrics
        scores = [s.score for s in historical]
        timestamps = [s.timestamp for s in historical]
        
        # Simple trend calculation
        recent_avg = statistics.mean(scores[-10:]) if len(scores) >= 10 else statistics.mean(scores)
        older_avg = statistics.mean(scores[:-10]) if len(scores) > 10 else scores[0]
        
        trend_direction = "improving" if recent_avg > older_avg else "deteriorating"
        trend_strength = abs(recent_avg - older_avg)
        
        # Identify turning points
        turning_points = self._identify_turning_points(scores, timestamps)
        
        # Build trend analysis prompt
        prompt = self.prompts.build_sentiment_trend_prompt(
            instrument_id=str(instrument_id),
            historical_sentiment=[s.to_dict() for s in historical],
            trend_metrics={
                "direction": trend_direction,
                "strength": trend_strength,
                "recent_average": recent_avg,
                "older_average": older_avg,
                "volatility": statistics.stdev(scores) if len(scores) > 1 else 0,
            },
        )
        
        # Get AI analysis
        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self.prompts.SENTIMENT_SYSTEM_PROMPT,
            temperature=0.4,
        )
        
        # Parse response
        analysis = self._parse_json_response(response.content)
        
        # Add calculated metrics
        analysis.update({
            "calculated_trend": trend_direction,
            "calculated_strength": trend_strength,
            "turning_points": turning_points,
            "data_points": len(historical),
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        return analysis
        
    async def detect_sentiment_divergence(
        self,
        instrument_id: InstrumentId,
        price_data: List[Tuple[datetime, Decimal]],
        lookback_hours: int = 48,
    ) -> Dict[str, Any]:
        """
        Detect divergence between price and sentiment.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            Instrument to analyze
        price_data : list
            Price history [(timestamp, price)]
        lookback_hours : int
            Hours to analyze
            
        Returns
        -------
        Dict[str, Any]
            Divergence analysis
        """
        # Get sentiment history
        sentiment_history = self._get_recent_sentiment(
            instrument_id,
            lookback_hours,
        )
        
        if len(sentiment_history) < 2 or len(price_data) < 2:
            return {
                "divergence_detected": False,
                "type": "none",
                "strength": 0.0,
                "description": "Insufficient data",
            }
        
        # Align price and sentiment data
        aligned_data = self._align_price_sentiment(
            price_data,
            sentiment_history,
        )
        
        # Calculate divergence
        divergence = self._calculate_divergence(aligned_data)
        
        # Build divergence prompt
        prompt = self.prompts.build_sentiment_divergence_prompt(
            instrument_id=str(instrument_id),
            price_trend=divergence["price_trend"],
            sentiment_trend=divergence["sentiment_trend"],
            correlation=divergence["correlation"],
            aligned_data=aligned_data,
        )
        
        # Get AI analysis
        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self.prompts.SENTIMENT_SYSTEM_PROMPT,
            temperature=0.3,
        )
        
        # Parse response
        analysis = self._parse_json_response(response.content)
        
        # Add calculated metrics
        analysis.update({
            "calculated_divergence": divergence,
            "timestamp": datetime.utcnow().isoformat(),
            "model": response.model,
        })
        
        return analysis
        
    async def generate_sentiment_signals(
        self,
        instrument_id: InstrumentId,
        current_sentiment: MarketSentiment,
        price_level: Decimal,
        risk_tolerance: str = "moderate",  # "conservative", "moderate", "aggressive"
    ) -> Dict[str, Any]:
        """
        Generate trading signals based on sentiment.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            Instrument to analyze
        current_sentiment : MarketSentiment
            Current sentiment
        price_level : Decimal
            Current price
        risk_tolerance : str
            Risk tolerance level
            
        Returns
        -------
        Dict[str, Any]
            Trading signals and recommendations
        """
        # Get sentiment history and trend
        trend_analysis = await self.analyze_sentiment_trend(
            instrument_id,
            lookback_hours=48,
        )
        
        # Build signal generation prompt
        prompt = self.prompts.build_sentiment_signal_prompt(
            instrument_id=str(instrument_id),
            current_sentiment=current_sentiment.to_dict(),
            sentiment_trend=trend_analysis,
            current_price=str(price_level),
            risk_tolerance=risk_tolerance,
        )
        
        # Get AI signals
        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self.prompts.SENTIMENT_SYSTEM_PROMPT,
            temperature=0.3,
        )
        
        # Parse response
        signals = self._parse_json_response(response.content)
        
        # Add metadata
        signals.update({
            "instrument_id": str(instrument_id),
            "sentiment_score": current_sentiment.score,
            "sentiment_level": current_sentiment.sentiment.value,
            "timestamp": datetime.utcnow().isoformat(),
            "model": response.model,
        })
        
        return signals
        
    def _prepare_sentiment_inputs(
        self,
        instrument_id: InstrumentId,
        text_data: Optional[Dict[str, List[str]]],
        market_indicators: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Prepare inputs for sentiment analysis."""
        inputs = {}
        
        # Process text data
        if text_data:
            inputs["text_sources"] = {}
            for source, texts in text_data.items():
                # Limit texts per source
                inputs["text_sources"][source] = texts[:10]
                
        # Process market indicators
        if market_indicators:
            inputs["market_indicators"] = market_indicators
        else:
            # Default market indicators
            inputs["market_indicators"] = {
                "volume_trend": "normal",
                "volatility": "moderate",
                "price_action": "neutral",
            }
            
        return inputs
        
    def _get_recent_sentiment(
        self,
        instrument_id: InstrumentId,
        hours: int,
    ) -> List[MarketSentiment]:
        """Get recent sentiment history."""
        if instrument_id not in self._sentiment_history:
            return []
            
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            s for s in self._sentiment_history[instrument_id]
            if s.timestamp > cutoff
        ]
        
    def _identify_turning_points(
        self,
        scores: List[float],
        timestamps: List[datetime],
    ) -> List[Dict[str, Any]]:
        """Identify sentiment turning points."""
        if len(scores) < 3:
            return []
            
        turning_points = []
        
        for i in range(1, len(scores) - 1):
            # Local maximum
            if scores[i] > scores[i-1] and scores[i] > scores[i+1]:
                turning_points.append({
                    "type": "peak",
                    "timestamp": timestamps[i].isoformat(),
                    "score": scores[i],
                    "index": i,
                })
            # Local minimum
            elif scores[i] < scores[i-1] and scores[i] < scores[i+1]:
                turning_points.append({
                    "type": "trough",
                    "timestamp": timestamps[i].isoformat(),
                    "score": scores[i],
                    "index": i,
                })
                
        return turning_points
        
    def _align_price_sentiment(
        self,
        price_data: List[Tuple[datetime, Decimal]],
        sentiment_history: List[MarketSentiment],
    ) -> List[Dict[str, Any]]:
        """Align price and sentiment data by timestamp."""
        aligned = []
        
        for sentiment in sentiment_history:
            # Find closest price
            closest_price = None
            min_diff = timedelta(hours=1)  # Max 1 hour difference
            
            for timestamp, price in price_data:
                diff = abs(timestamp - sentiment.timestamp)
                if diff < min_diff:
                    min_diff = diff
                    closest_price = price
                    
            if closest_price is not None:
                aligned.append({
                    "timestamp": sentiment.timestamp.isoformat(),
                    "price": float(closest_price),
                    "sentiment_score": sentiment.score,
                    "sentiment_level": sentiment.sentiment.value,
                })
                
        return aligned
        
    def _calculate_divergence(
        self,
        aligned_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate price-sentiment divergence."""
        if len(aligned_data) < 2:
            return {
                "price_trend": "flat",
                "sentiment_trend": "flat",
                "correlation": 0.0,
                "divergence_strength": 0.0,
            }
            
        # Extract prices and scores
        prices = [d["price"] for d in aligned_data]
        scores = [d["sentiment_score"] for d in aligned_data]
        
        # Calculate trends (simple linear)
        price_trend = "up" if prices[-1] > prices[0] else "down"
        sentiment_trend = "up" if scores[-1] > scores[0] else "down"
        
        # Calculate correlation
        if len(prices) > 2:
            # Simple correlation calculation
            mean_price = statistics.mean(prices)
            mean_score = statistics.mean(scores)
            
            numerator = sum(
                (p - mean_price) * (s - mean_score)
                for p, s in zip(prices, scores)
            )
            
            denominator_p = sum((p - mean_price) ** 2 for p in prices) ** 0.5
            denominator_s = sum((s - mean_score) ** 2 for s in scores) ** 0.5
            
            correlation = (
                numerator / (denominator_p * denominator_s)
                if denominator_p > 0 and denominator_s > 0
                else 0
            )
        else:
            correlation = 0.0
            
        # Calculate divergence strength
        divergence_strength = 0.0
        if (price_trend == "up" and sentiment_trend == "down") or \
           (price_trend == "down" and sentiment_trend == "up"):
            divergence_strength = 1.0 - abs(correlation)
            
        return {
            "price_trend": price_trend,
            "sentiment_trend": sentiment_trend,
            "correlation": correlation,
            "divergence_strength": divergence_strength,
        }