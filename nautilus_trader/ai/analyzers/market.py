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
AI-powered market analysis using DeepSeek.
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from decimal import Decimal

from nautilus_trader.ai.config import AIAnalyzerConfig
from nautilus_trader.ai.providers.base import AIProvider
from nautilus_trader.ai.utils.prompts import MarketAnalysisPrompts
from nautilus_trader.ai.models.responses import (
    AIResponseAdapter,
    TrendAnalysisResponse,
    SignalGenerationResponse,
    PatternRecognitionResponse,
    VolatilityAnalysisResponse,
)
from nautilus_trader.common.component import Logger
from nautilus_trader.model.data import Bar, QuoteTick, TradeTick
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.enums import OrderSide


class TrendAnalysis:
    """Market trend analysis result."""
    
    def __init__(
        self,
        direction: str,  # "bullish", "bearish", "neutral"
        strength: float,  # 0.0 to 1.0
        confidence: float,  # 0.0 to 1.0
        support_levels: List[float],
        resistance_levels: List[float],
        reasoning: str,
    ) -> None:
        self.direction = direction
        self.strength = strength
        self.confidence = confidence
        self.support_levels = support_levels
        self.resistance_levels = resistance_levels
        self.reasoning = reasoning
        self.timestamp = datetime.now(timezone.utc)


class Signal:
    """Trading signal from AI analysis."""
    
    def __init__(
        self,
        side: OrderSide,
        confidence: float,
        entry_price: Decimal,
        stop_loss: Decimal,
        take_profit: Decimal,
        reasoning: str,
        timeframe: str,
    ) -> None:
        self.side = side
        self.confidence = confidence
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.reasoning = reasoning
        self.timeframe = timeframe
        self.timestamp = datetime.now(timezone.utc)


class Pattern:
    """Technical pattern detected by AI."""
    
    def __init__(
        self,
        name: str,
        pattern_type: str,  # "continuation", "reversal"
        confidence: float,
        target_price: Optional[Decimal],
        description: str,
    ) -> None:
        self.name = name
        self.pattern_type = pattern_type
        self.confidence = confidence
        self.target_price = target_price
        self.description = description
        self.timestamp = datetime.now(timezone.utc)


class MarketAnalyzer:
    """
    AI-powered market analyzer using DeepSeek.
    
    This analyzer provides:
    - Trend analysis and prediction
    - Pattern recognition
    - Signal generation
    - Market sentiment analysis
    
    Parameters
    ----------
    provider : AIProvider
        The AI provider to use.
    config : AIAnalyzerConfig
        Analyzer configuration.
    
    """
    
    def __init__(
        self,
        provider: AIProvider,
        config: AIAnalyzerConfig,
    ) -> None:
        self._provider = provider
        self._config = config
        self._log = Logger(self.__class__.__name__)
        self._prompts = MarketAnalysisPrompts()
        
        # Cache for results
        self._cache: Dict[str, Any] = {}
        
    async def analyze_trends(
        self,
        instrument_id: InstrumentId,
        bars: List[Bar],
        additional_context: Optional[str] = None,
    ) -> Optional[TrendAnalysisResponse]:
        """
        Analyze market trends using AI.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument to analyze.
        bars : list[Bar]
            Historical price bars.
        additional_context : str, optional
            Additional market context.
            
        Returns
        -------
        TrendAnalysisResponse or None
            The validated trend analysis result.
        
        """
        # Prepare market data
        market_data = self._prepare_bar_data(bars[-100:])  # Last 100 bars
        
        # Build prompt
        prompt = self._prompts.build_trend_analysis_prompt(
            instrument_id=str(instrument_id),
            market_data=market_data,
            additional_context=additional_context,
        )
        
        # Get AI analysis
        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self._prompts.SYSTEM_PROMPT,
            temperature=0.3,  # Lower temperature for more consistent analysis
        )
        
        # Parse and validate response using Pydantic
        analysis = AIResponseAdapter.safe_parse(
            response_type="trend_analysis",
            raw_response=response.content,
        )
        
        if not analysis:
            self._log.error("Failed to parse trend analysis response")
            
        return analysis
            
    async def generate_signals(
        self,
        instrument_id: InstrumentId,
        bars: List[Bar],
        trend_analysis: TrendAnalysisResponse,
        risk_tolerance: float = 0.02,  # 2% risk default
    ) -> Optional[SignalGenerationResponse]:
        """
        Generate trading signals based on analysis.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument to trade.
        bars : list[Bar]
            Recent price bars.
        trend_analysis : TrendAnalysisResponse
            Current trend analysis.
        risk_tolerance : float
            Risk tolerance as percentage.
            
        Returns
        -------
        SignalGenerationResponse or None
            Validated trading signal if conditions are met.
        
        """
        # Only generate signals with high confidence
        if trend_analysis.confidence < 0.7:
            return None
            
        current_bar = bars[-1]
        
        # Prepare context
        context = {
            "current_price": float(current_bar.close),
            "trend": trend_analysis.direction,
            "trend_strength": trend_analysis.strength,
            "support_levels": trend_analysis.support_levels,
            "resistance_levels": trend_analysis.resistance_levels,
            "risk_tolerance": risk_tolerance,
        }
        
        # Build prompt
        prompt = self._prompts.build_signal_generation_prompt(
            instrument_id=str(instrument_id),
            context=context,
        )
        
        # Get AI recommendation
        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self._prompts.SYSTEM_PROMPT,
            temperature=0.2,  # Very low temperature for consistent signals
        )
        
        # Parse and validate response using Pydantic
        signal = AIResponseAdapter.safe_parse(
            response_type="signal_generation",
            raw_response=response.content,
        )
        
        if signal and not signal.generate_signal:
            return None
            
        return signal
            
    async def identify_patterns(
        self,
        instrument_id: InstrumentId,
        bars: List[Bar],
        min_confidence: float = 0.7,
    ) -> Optional[PatternRecognitionResponse]:
        """
        Identify technical patterns using AI.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument to analyze.
        bars : list[Bar]
            Historical price bars.
        min_confidence : float
            Minimum confidence threshold (handled by Pydantic).
            
        Returns
        -------
        PatternRecognitionResponse or None
            Validated pattern detection results.
        
        """
        # Prepare data
        market_data = self._prepare_bar_data(bars[-50:])  # Last 50 bars for patterns
        
        # Build prompt
        prompt = self._prompts.build_pattern_recognition_prompt(
            instrument_id=str(instrument_id),
            market_data=market_data,
        )
        
        # Get AI analysis
        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self._prompts.SYSTEM_PROMPT,
            temperature=0.4,
        )
        
        # Parse and validate response using Pydantic
        # The Pydantic model automatically filters patterns with confidence < 0.6
        pattern_response = AIResponseAdapter.safe_parse(
            response_type="pattern_recognition",
            raw_response=response.content,
        )
        
        if not pattern_response:
            self._log.error("Failed to parse pattern recognition response")
            
        return pattern_response
        
    async def analyze_volatility(
        self,
        instrument_id: InstrumentId,
        bars: List[Bar],
        quotes: Optional[List[QuoteTick]] = None,
    ) -> Optional[VolatilityAnalysisResponse]:
        """
        Analyze market volatility and predict future volatility.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument to analyze.
        bars : list[Bar]
            Historical price bars.
        quotes : list[QuoteTick], optional
            Recent quote ticks for spread analysis.
            
        Returns
        -------
        VolatilityAnalysisResponse or None
            Validated volatility analysis results.
        
        """
        # Calculate basic volatility metrics
        returns = []
        for i in range(1, len(bars)):
            ret = (float(bars[i].close) - float(bars[i-1].close)) / float(bars[i-1].close)
            returns.append(ret)
            
        # Historical volatility (annualized)
        import numpy as np
        hist_vol = np.std(returns) * np.sqrt(252)
        
        # Prepare context
        context = {
            "historical_volatility": hist_vol,
            "recent_price_action": self._prepare_bar_data(bars[-20:]),
            "spread_analysis": self._analyze_spreads(quotes) if quotes else None,
        }
        
        # Build prompt
        prompt = self._prompts.build_volatility_analysis_prompt(
            instrument_id=str(instrument_id),
            context=context,
        )
        
        # Get AI analysis
        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self._prompts.SYSTEM_PROMPT,
            temperature=0.3,
        )
        
        # Parse and validate response using Pydantic
        volatility_analysis = AIResponseAdapter.safe_parse(
            response_type="volatility_analysis",
            raw_response=response.content,
        )
        
        if not volatility_analysis:
            self._log.error("Failed to parse volatility analysis response")
            
        return volatility_analysis
            
    def _prepare_bar_data(self, bars: List[Bar]) -> List[Dict[str, Any]]:
        """Prepare bar data for AI analysis."""
        return [
            {
                "timestamp": bar.ts_event,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            for bar in bars
        ]
        
    def _analyze_spreads(self, quotes: List[QuoteTick]) -> Dict[str, float]:
        """Analyze bid-ask spreads."""
        if not quotes:
            return {}
            
        spreads = [
            float(q.ask_price) - float(q.bid_price)
            for q in quotes
        ]
        
        return {
            "avg_spread": sum(spreads) / len(spreads),
            "max_spread": max(spreads),
            "min_spread": min(spreads),
            "spread_volatility": np.std(spreads) if len(spreads) > 1 else 0.0,
        }