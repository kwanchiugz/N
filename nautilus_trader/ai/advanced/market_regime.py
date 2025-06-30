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
Market regime detection using AI and statistical methods.
"""

from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from nautilus_trader.ai.analyzers.base import BaseAIAnalyzer
from nautilus_trader.ai.config import AIAnalyzerConfig
from nautilus_trader.ai.utils.prompts import MarketRegimePrompts
from nautilus_trader.model.data import Bar
from nautilus_trader.model.identifiers import InstrumentId


class RegimeType(str, Enum):
    """Market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"
    TRANSITIONING = "transitioning"


class RegimeTransition:
    """
    Represents a market regime transition.
    
    Parameters
    ----------
    from_regime : RegimeType
        Previous regime
    to_regime : RegimeType
        New regime
    timestamp : datetime
        Transition time
    confidence : float
        Transition confidence
    indicators : dict
        Indicators that triggered transition
    """
    
    def __init__(
        self,
        from_regime: RegimeType,
        to_regime: RegimeType,
        timestamp: datetime,
        confidence: float,
        indicators: Dict[str, Any],
    ) -> None:
        self.from_regime = from_regime
        self.to_regime = to_regime
        self.timestamp = timestamp
        self.confidence = confidence
        self.indicators = indicators
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from_regime": self.from_regime.value,
            "to_regime": self.to_regime.value,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "indicators": self.indicators,
        }


class MarketRegimeDetector(BaseAIAnalyzer):
    """
    AI-powered market regime detection.
    
    This detector identifies:
    - Market regimes (trending, ranging, volatile)
    - Regime transitions
    - Regime characteristics
    - Optimal strategies per regime
    """
    
    def __init__(
        self,
        config: AIAnalyzerConfig,
        cache: Optional[Any] = None,
    ) -> None:
        """Initialize regime detector."""
        super().__init__(config, cache)
        self.prompts = MarketRegimePrompts()
        self._current_regime: Optional[RegimeType] = None
        self._regime_history: List[RegimeTransition] = []
        
    async def detect_regime(
        self,
        instrument_id: InstrumentId,
        bars: List[Bar],
        additional_indicators: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Detect current market regime.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            Instrument to analyze
        bars : List[Bar]
            Historical bars (at least 100)
        additional_indicators : dict, optional
            Additional technical indicators
            
        Returns
        -------
        Dict[str, Any]
            Regime detection results
        """
        # Calculate regime indicators
        indicators = self._calculate_regime_indicators(bars)
        
        # Add any additional indicators
        if additional_indicators:
            indicators.update(additional_indicators)
            
        # Build detection prompt
        prompt = self.prompts.build_regime_detection_prompt(
            instrument_id=str(instrument_id),
            indicators=indicators,
            recent_bars=self._prepare_bar_data(bars[-20:]),
            current_regime=self._current_regime.value if self._current_regime else None,
        )
        
        # Get AI analysis
        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self.prompts.REGIME_SYSTEM_PROMPT,
            temperature=0.3,
        )
        
        # Parse response
        result = self._parse_json_response(response.content)
        
        # Update current regime
        new_regime = RegimeType(result["regime"])
        
        # Check for transition
        if self._current_regime and new_regime != self._current_regime:
            transition = RegimeTransition(
                from_regime=self._current_regime,
                to_regime=new_regime,
                timestamp=datetime.utcnow(),
                confidence=result.get("confidence", 0.5),
                indicators=result.get("key_indicators", {}),
            )
            self._regime_history.append(transition)
            
        self._current_regime = new_regime
        
        # Add metadata
        result.update({
            "instrument_id": str(instrument_id),
            "timestamp": datetime.utcnow().isoformat(),
            "indicators": indicators,
            "model": response.model,
        })
        
        return result
        
    async def analyze_regime_stability(
        self,
        instrument_id: InstrumentId,
        bars: List[Bar],
        lookback_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Analyze regime stability and transition probability.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            Instrument to analyze
        bars : List[Bar]
            Historical bars
        lookback_days : int
            Days to look back for stability analysis
            
        Returns
        -------
        Dict[str, Any]
            Stability analysis results
        """
        # Get recent regime history
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        recent_transitions = [
            t for t in self._regime_history
            if t.timestamp > cutoff
        ]
        
        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(
            bars,
            recent_transitions,
            lookback_days,
        )
        
        # Build analysis prompt
        prompt = self.prompts.build_regime_stability_prompt(
            current_regime=self._current_regime.value if self._current_regime else "unknown",
            stability_metrics=stability_metrics,
            recent_transitions=[t.to_dict() for t in recent_transitions],
        )
        
        # Get AI analysis
        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self.prompts.REGIME_SYSTEM_PROMPT,
            temperature=0.3,
        )
        
        # Parse response
        analysis = self._parse_json_response(response.content)
        
        # Add calculated metrics
        analysis.update({
            "stability_metrics": stability_metrics,
            "transition_count": len(recent_transitions),
            "current_regime": self._current_regime.value if self._current_regime else None,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        return analysis
        
    async def recommend_strategy_adjustments(
        self,
        current_regime: RegimeType,
        strategy_type: str,  # "trend_following", "mean_reversion", "breakout", etc.
        current_parameters: Dict[str, Any],
        performance_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Recommend strategy adjustments based on regime.
        
        Parameters
        ----------
        current_regime : RegimeType
            Current market regime
        strategy_type : str
            Type of trading strategy
        current_parameters : dict
            Current strategy parameters
        performance_metrics : dict, optional
            Recent performance metrics
            
        Returns
        -------
        Dict[str, Any]
            Recommended adjustments
        """
        # Build recommendation prompt
        prompt = self.prompts.build_strategy_adjustment_prompt(
            regime=current_regime.value,
            strategy_type=strategy_type,
            current_parameters=current_parameters,
            performance_metrics=performance_metrics,
        )
        
        # Get AI recommendations
        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self.prompts.REGIME_SYSTEM_PROMPT,
            temperature=0.4,
        )
        
        # Parse response
        recommendations = self._parse_json_response(response.content)
        
        # Add metadata
        recommendations.update({
            "current_regime": current_regime.value,
            "strategy_type": strategy_type,
            "timestamp": datetime.utcnow().isoformat(),
            "model": response.model,
        })
        
        return recommendations
        
    async def predict_regime_change(
        self,
        instrument_id: InstrumentId,
        bars: List[Bar],
        forecast_horizon: int = 5,  # days
    ) -> Dict[str, Any]:
        """
        Predict potential regime changes.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            Instrument to analyze
        bars : List[Bar]
            Historical bars
        forecast_horizon : int
            Days to forecast ahead
            
        Returns
        -------
        Dict[str, Any]
            Regime change predictions
        """
        # Calculate predictive indicators
        indicators = self._calculate_predictive_indicators(bars)
        
        # Build prediction prompt
        prompt = self.prompts.build_regime_prediction_prompt(
            instrument_id=str(instrument_id),
            current_regime=self._current_regime.value if self._current_regime else "unknown",
            indicators=indicators,
            forecast_horizon=forecast_horizon,
            recent_bars=self._prepare_bar_data(bars[-50:]),
        )
        
        # Get AI prediction
        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self.prompts.REGIME_SYSTEM_PROMPT,
            temperature=0.4,
        )
        
        # Parse response
        prediction = self._parse_json_response(response.content)
        
        # Add metadata
        prediction.update({
            "instrument_id": str(instrument_id),
            "current_regime": self._current_regime.value if self._current_regime else None,
            "forecast_horizon": forecast_horizon,
            "timestamp": datetime.utcnow().isoformat(),
            "model": response.model,
        })
        
        return prediction
        
    def _calculate_regime_indicators(self, bars: List[Bar]) -> Dict[str, Any]:
        """Calculate indicators for regime detection."""
        if len(bars) < 20:
            return {}
            
        # Convert to pandas for easier calculation
        df = pd.DataFrame([
            {
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            for bar in bars
        ])
        
        # Price indicators
        returns = df["close"].pct_change()
        
        # Trend indicators
        sma_20 = df["close"].rolling(20).mean()
        sma_50 = df["close"].rolling(50).mean() if len(df) >= 50 else sma_20
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        
        # Volatility indicators
        volatility = returns.rolling(20).std()
        atr = self._calculate_atr(df, 14)
        
        # Volume indicators
        volume_sma = df["volume"].rolling(20).mean()
        volume_ratio = df["volume"] / volume_sma
        
        # Momentum indicators
        rsi = self._calculate_rsi(df["close"], 14)
        
        # Market structure
        higher_highs = (df["high"] > df["high"].shift(1)).rolling(10).sum()
        lower_lows = (df["low"] < df["low"].shift(1)).rolling(10).sum()
        
        return {
            "trend": {
                "price_vs_sma20": float((df["close"].iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1] * 100),
                "sma20_vs_sma50": float((sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1] * 100) if len(df) >= 50 else 0,
                "macd": float(ema_12.iloc[-1] - ema_26.iloc[-1]),
                "trend_strength": float(abs(returns.rolling(20).mean().iloc[-1]) / volatility.iloc[-1]) if volatility.iloc[-1] > 0 else 0,
            },
            "volatility": {
                "current_volatility": float(volatility.iloc[-1]),
                "volatility_percentile": float(volatility.iloc[-1] / volatility.quantile(0.75)) if len(volatility) > 20 else 1.0,
                "atr": float(atr.iloc[-1]) if not atr.empty else 0,
                "volatility_trend": float((volatility.iloc[-1] - volatility.iloc[-5]) / volatility.iloc[-5] * 100) if len(volatility) > 5 and volatility.iloc[-5] > 0 else 0,
            },
            "momentum": {
                "rsi": float(rsi.iloc[-1]) if not rsi.empty else 50,
                "price_momentum": float(returns.rolling(10).mean().iloc[-1] * 252) if len(returns) > 10 else 0,
                "volume_momentum": float(volume_ratio.rolling(5).mean().iloc[-1]) if len(volume_ratio) > 5 else 1.0,
            },
            "structure": {
                "higher_highs_ratio": float(higher_highs.iloc[-1] / 10) if not higher_highs.empty else 0.5,
                "lower_lows_ratio": float(lower_lows.iloc[-1] / 10) if not lower_lows.empty else 0.5,
                "range_percent": float((df["high"].iloc[-20:].max() - df["low"].iloc[-20:].min()) / df["close"].iloc[-1] * 100),
            },
        }
        
    def _calculate_stability_metrics(
        self,
        bars: List[Bar],
        transitions: List[RegimeTransition],
        lookback_days: int,
    ) -> Dict[str, Any]:
        """Calculate regime stability metrics."""
        # Time-based metrics
        if transitions:
            regime_durations = []
            for i in range(len(transitions)):
                if i == 0:
                    # First transition - we don't know previous duration
                    continue
                duration = (transitions[i].timestamp - transitions[i-1].timestamp).days
                regime_durations.append(duration)
                
            avg_duration = np.mean(regime_durations) if regime_durations else lookback_days
            
        else:
            avg_duration = lookback_days  # No transitions means stable
            
        # Price-based stability
        prices = [float(bar.close) for bar in bars[-lookback_days:]]
        price_volatility = np.std(prices) / np.mean(prices) if prices else 0
        
        return {
            "transition_frequency": len(transitions) / lookback_days * 30,  # Per month
            "average_regime_duration": avg_duration,
            "current_regime_duration": (datetime.utcnow() - transitions[-1].timestamp).days if transitions else lookback_days,
            "price_stability": 1.0 - min(price_volatility, 1.0),
            "regime_certainty": np.mean([t.confidence for t in transitions]) if transitions else 1.0,
        }
        
    def _calculate_predictive_indicators(self, bars: List[Bar]) -> Dict[str, Any]:
        """Calculate indicators for regime prediction."""
        # Get standard indicators
        indicators = self._calculate_regime_indicators(bars)
        
        # Add predictive indicators
        if len(bars) >= 50:
            df = pd.DataFrame([
                {"close": float(bar.close), "volume": float(bar.volume)}
                for bar in bars
            ])
            
            # Divergence indicators
            price_roc = df["close"].pct_change(10).iloc[-1]
            volume_roc = df["volume"].pct_change(10).iloc[-1]
            
            # Momentum divergence
            price_momentum = df["close"].diff(5).rolling(5).mean()
            volume_momentum = df["volume"].diff(5).rolling(5).mean()
            
            indicators["predictive"] = {
                "price_volume_divergence": float(price_roc - volume_roc) if not pd.isna(price_roc) and not pd.isna(volume_roc) else 0,
                "momentum_divergence": float(price_momentum.iloc[-1] / price_momentum.std()) if price_momentum.std() > 0 else 0,
                "volatility_expansion": float(indicators["volatility"]["volatility_trend"]),
                "trend_exhaustion": float(abs(indicators["momentum"]["rsi"] - 50) / 50),  # 0-1 scale
            }
            
        return indicators
        
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
        
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def _prepare_bar_data(self, bars: List[Bar]) -> List[Dict[str, Any]]:
        """Prepare bar data for prompts."""
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