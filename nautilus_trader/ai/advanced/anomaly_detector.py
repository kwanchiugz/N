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
Market anomaly detection using AI and statistical methods.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from datetime import datetime, timedelta
from decimal import Decimal
import numpy as np
import pandas as pd

from nautilus_trader.ai.analyzers.base import BaseAIAnalyzer
from nautilus_trader.ai.config import AIAnalyzerConfig
from nautilus_trader.ai.utils.prompts import AnomalyDetectionPrompts
from nautilus_trader.model.data import Bar, Trade, QuoteTick
from nautilus_trader.model.identifiers import InstrumentId


class AnomalyType(str, Enum):
    """Types of market anomalies."""
    PRICE_SPIKE = "price_spike"
    VOLUME_SURGE = "volume_surge"
    SPREAD_WIDENING = "spread_widening"
    LIQUIDITY_DROP = "liquidity_drop"
    CORRELATION_BREAK = "correlation_break"
    PATTERN_ANOMALY = "pattern_anomaly"
    MICROSTRUCTURE = "microstructure"
    FLASH_CRASH = "flash_crash"
    FAT_FINGER = "fat_finger"
    MARKET_MANIPULATION = "market_manipulation"


class AnomalySeverity(str, Enum):
    """Anomaly severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MarketAnomaly:
    """
    Detected market anomaly.
    
    Parameters
    ----------
    anomaly_id : str
        Unique anomaly identifier
    instrument_id : InstrumentId
        Affected instrument
    anomaly_type : AnomalyType
        Type of anomaly
    severity : AnomalySeverity
        Severity level
    confidence : float
        Detection confidence (0-1)
    start_time : datetime
        Anomaly start time
    end_time : datetime, optional
        Anomaly end time
    description : str
        Anomaly description
    metrics : dict
        Anomaly metrics
    affected_instruments : list, optional
        Other affected instruments
    """
    
    def __init__(
        self,
        anomaly_id: str,
        instrument_id: InstrumentId,
        anomaly_type: AnomalyType,
        severity: AnomalySeverity,
        confidence: float,
        start_time: datetime,
        end_time: Optional[datetime],
        description: str,
        metrics: Dict[str, Any],
        affected_instruments: Optional[List[InstrumentId]] = None,
    ) -> None:
        self.anomaly_id = anomaly_id
        self.instrument_id = instrument_id
        self.anomaly_type = anomaly_type
        self.severity = severity
        self.confidence = confidence
        self.start_time = start_time
        self.end_time = end_time
        self.description = description
        self.metrics = metrics
        self.affected_instruments = affected_instruments or []
        self.is_active = end_time is None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "anomaly_id": self.anomaly_id,
            "instrument_id": str(self.instrument_id),
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "description": self.description,
            "metrics": self.metrics,
            "affected_instruments": [str(i) for i in self.affected_instruments],
            "is_active": self.is_active,
        }


class AnomalyDetector(BaseAIAnalyzer):
    """
    AI-powered market anomaly detector.
    
    This detector identifies:
    - Price anomalies (spikes, crashes)
    - Volume anomalies
    - Liquidity issues
    - Market manipulation patterns
    - Technical glitches
    - Cross-market anomalies
    """
    
    def __init__(
        self,
        config: AIAnalyzerConfig,
        cache: Optional[Any] = None,
    ) -> None:
        """Initialize anomaly detector."""
        super().__init__(config, cache)
        self.prompts = AnomalyDetectionPrompts()
        self._active_anomalies: Dict[str, MarketAnomaly] = {}
        self._anomaly_history: List[MarketAnomaly] = []
        self._detection_thresholds = self._initialize_thresholds()
        
    async def detect_anomalies(
        self,
        instrument_id: InstrumentId,
        bars: List[Bar],
        trades: Optional[List[Trade]] = None,
        quotes: Optional[List[QuoteTick]] = None,
        reference_data: Optional[Dict[str, Any]] = None,
    ) -> List[MarketAnomaly]:
        """
        Detect anomalies in market data.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            Instrument to analyze
        bars : List[Bar]
            Historical bars
        trades : List[Trade], optional
            Recent trades
        quotes : List[QuoteTick], optional
            Recent quotes
        reference_data : dict, optional
            Reference data for comparison
            
        Returns
        -------
        List[MarketAnomaly]
            Detected anomalies
        """
        # Calculate anomaly indicators
        indicators = self._calculate_anomaly_indicators(
            bars,
            trades,
            quotes,
        )
        
        # Statistical anomaly detection
        statistical_anomalies = self._detect_statistical_anomalies(
            bars,
            indicators,
        )
        
        # Build detection prompt
        prompt = self.prompts.build_anomaly_detection_prompt(
            instrument_id=str(instrument_id),
            indicators=indicators,
            statistical_anomalies=statistical_anomalies,
            recent_bars=self._prepare_bar_data(bars[-50:]),
            trades_summary=self._summarize_trades(trades) if trades else None,
            quotes_summary=self._summarize_quotes(quotes) if quotes else None,
            reference_data=reference_data,
        )
        
        # Get AI analysis
        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self.prompts.ANOMALY_SYSTEM_PROMPT,
            temperature=0.2,  # Low temperature for consistent detection
        )
        
        # Parse response
        result = self._parse_json_response(response.content)
        
        # Create anomaly objects
        anomalies = []
        for anomaly_data in result.get("anomalies", []):
            anomaly = MarketAnomaly(
                anomaly_id=self._generate_anomaly_id(),
                instrument_id=instrument_id,
                anomaly_type=AnomalyType(anomaly_data["type"]),
                severity=AnomalySeverity(anomaly_data["severity"]),
                confidence=anomaly_data["confidence"],
                start_time=datetime.fromisoformat(anomaly_data["start_time"]),
                end_time=datetime.fromisoformat(anomaly_data["end_time"]) 
                    if anomaly_data.get("end_time") else None,
                description=anomaly_data["description"],
                metrics=anomaly_data.get("metrics", {}),
                affected_instruments=[
                    InstrumentId.from_str(i) 
                    for i in anomaly_data.get("affected_instruments", [])
                ],
            )
            
            anomalies.append(anomaly)
            
            # Update active anomalies
            if anomaly.is_active:
                self._active_anomalies[anomaly.anomaly_id] = anomaly
            
            # Add to history
            self._anomaly_history.append(anomaly)
            
        return anomalies
        
    async def analyze_anomaly_impact(
        self,
        anomaly: MarketAnomaly,
        market_data: Dict[str, Any],
        positions: Optional[Dict[InstrumentId, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze the impact of an anomaly.
        
        Parameters
        ----------
        anomaly : MarketAnomaly
            Anomaly to analyze
        market_data : dict
            Current market data
        positions : dict, optional
            Current positions
            
        Returns
        -------
        Dict[str, Any]
            Impact analysis
        """
        # Build impact analysis prompt
        prompt = self.prompts.build_anomaly_impact_prompt(
            anomaly=anomaly.to_dict(),
            market_data=market_data,
            positions=self._prepare_positions(positions) if positions else None,
        )
        
        # Get AI analysis
        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self.prompts.ANOMALY_SYSTEM_PROMPT,
            temperature=0.3,
        )
        
        # Parse response
        impact = self._parse_json_response(response.content)
        
        # Add metadata
        impact.update({
            "anomaly_id": anomaly.anomaly_id,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "model": response.model,
        })
        
        return impact
        
    async def recommend_anomaly_response(
        self,
        anomaly: MarketAnomaly,
        current_strategy: str,
        risk_limits: Dict[str, Any],
        market_conditions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Recommend response to detected anomaly.
        
        Parameters
        ----------
        anomaly : MarketAnomaly
            Detected anomaly
        current_strategy : str
            Current trading strategy
        risk_limits : dict
            Risk limits and parameters
        market_conditions : dict, optional
            Current market conditions
            
        Returns
        -------
        Dict[str, Any]
            Response recommendations
        """
        # Analyze historical responses
        similar_anomalies = self._find_similar_anomalies(
            anomaly,
            lookback_days=90,
        )
        
        # Build recommendation prompt
        prompt = self.prompts.build_anomaly_response_prompt(
            anomaly=anomaly.to_dict(),
            current_strategy=current_strategy,
            risk_limits=risk_limits,
            market_conditions=market_conditions,
            similar_anomalies=[
                {"anomaly": a.to_dict(), "outcome": self._get_anomaly_outcome(a)}
                for a in similar_anomalies[:5]
            ],
        )
        
        # Get AI recommendations
        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self.prompts.ANOMALY_SYSTEM_PROMPT,
            temperature=0.3,
        )
        
        # Parse response
        recommendations = self._parse_json_response(response.content)
        
        # Add metadata
        recommendations.update({
            "anomaly_id": anomaly.anomaly_id,
            "timestamp": datetime.utcnow().isoformat(),
            "model": response.model,
        })
        
        return recommendations
        
    async def monitor_anomaly_resolution(
        self,
        anomaly_id: str,
        current_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Monitor active anomaly for resolution.
        
        Parameters
        ----------
        anomaly_id : str
            Anomaly ID to monitor
        current_data : dict
            Current market data
            
        Returns
        -------
        Dict[str, Any]
            Resolution status
        """
        # Get active anomaly
        anomaly = self._active_anomalies.get(anomaly_id)
        if not anomaly:
            return {
                "status": "not_found",
                "message": f"Anomaly {anomaly_id} not found or inactive",
            }
        
        # Build monitoring prompt
        prompt = self.prompts.build_anomaly_monitoring_prompt(
            anomaly=anomaly.to_dict(),
            current_data=current_data,
            duration_minutes=(datetime.utcnow() - anomaly.start_time).total_seconds() / 60,
        )
        
        # Get AI analysis
        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self.prompts.ANOMALY_SYSTEM_PROMPT,
            temperature=0.2,
        )
        
        # Parse response
        status = self._parse_json_response(response.content)
        
        # Update anomaly if resolved
        if status.get("is_resolved", False):
            anomaly.end_time = datetime.utcnow()
            anomaly.is_active = False
            del self._active_anomalies[anomaly_id]
            
        # Add metadata
        status.update({
            "anomaly_id": anomaly_id,
            "monitoring_timestamp": datetime.utcnow().isoformat(),
            "model": response.model,
        })
        
        return status
        
    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize detection thresholds."""
        return {
            "price_spike": {
                "min_change_pct": 3.0,  # 3% change
                "time_window_seconds": 60,
                "volume_multiplier": 2.0,
            },
            "volume_surge": {
                "multiplier": 5.0,  # 5x average volume
                "sustained_periods": 3,
            },
            "spread_widening": {
                "multiplier": 3.0,  # 3x average spread
                "min_spread_bps": 10,  # 10 basis points
            },
            "liquidity_drop": {
                "depth_reduction_pct": 50,  # 50% reduction
                "quote_frequency_drop": 0.3,  # 70% reduction
            },
        }
        
    def _calculate_anomaly_indicators(
        self,
        bars: List[Bar],
        trades: Optional[List[Trade]],
        quotes: Optional[List[QuoteTick]],
    ) -> Dict[str, Any]:
        """Calculate indicators for anomaly detection."""
        if len(bars) < 20:
            return {}
            
        # Convert to pandas for calculations
        df = pd.DataFrame([
            {
                "ts_event": bar.ts_event,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            for bar in bars
        ])
        
        # Price indicators
        returns = df["close"].pct_change()
        log_returns = np.log(df["close"] / df["close"].shift(1))
        
        # Volatility indicators
        rolling_std = returns.rolling(20).std()
        realized_vol = np.sqrt(252) * rolling_std
        
        # Volume indicators
        volume_ma = df["volume"].rolling(20).mean()
        volume_std = df["volume"].rolling(20).std()
        volume_zscore = (df["volume"] - volume_ma) / volume_std
        
        # Price range indicators
        true_range = pd.concat([
            df["high"] - df["low"],
            abs(df["high"] - df["close"].shift()),
            abs(df["low"] - df["close"].shift())
        ], axis=1).max(axis=1)
        
        atr = true_range.rolling(14).mean()
        
        # Microstructure indicators from trades/quotes
        microstructure = self._calculate_microstructure_indicators(
            trades,
            quotes,
        )
        
        return {
            "price": {
                "current": float(df["close"].iloc[-1]),
                "return_1bar": float(returns.iloc[-1]) if not pd.isna(returns.iloc[-1]) else 0,
                "return_5bar": float(df["close"].iloc[-1] / df["close"].iloc[-6] - 1) if len(df) >= 6 else 0,
                "max_return": float(returns.abs().max()) if len(returns) > 0 else 0,
                "volatility": float(realized_vol.iloc[-1]) if not pd.isna(realized_vol.iloc[-1]) else 0,
                "volatility_percentile": float(realized_vol.iloc[-1] / realized_vol.quantile(0.95)) 
                    if len(realized_vol) > 20 and realized_vol.quantile(0.95) > 0 else 1.0,
            },
            "volume": {
                "current": float(df["volume"].iloc[-1]),
                "zscore": float(volume_zscore.iloc[-1]) if not pd.isna(volume_zscore.iloc[-1]) else 0,
                "ratio_to_avg": float(df["volume"].iloc[-1] / volume_ma.iloc[-1]) 
                    if volume_ma.iloc[-1] > 0 else 1.0,
                "surge_detected": bool(volume_zscore.iloc[-1] > 3) if not pd.isna(volume_zscore.iloc[-1]) else False,
            },
            "range": {
                "current_range": float(df["high"].iloc[-1] - df["low"].iloc[-1]),
                "atr": float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0,
                "range_expansion": float((df["high"].iloc[-1] - df["low"].iloc[-1]) / atr.iloc[-1]) 
                    if atr.iloc[-1] > 0 else 1.0,
            },
            "microstructure": microstructure,
        }
        
    def _detect_statistical_anomalies(
        self,
        bars: List[Bar],
        indicators: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Detect anomalies using statistical methods."""
        anomalies = []
        
        # Price spike detection
        if abs(indicators["price"]["return_1bar"]) > 0.03:  # 3% move
            anomalies.append({
                "type": "price_spike",
                "severity": "high" if abs(indicators["price"]["return_1bar"]) > 0.05 else "medium",
                "metrics": {
                    "return": indicators["price"]["return_1bar"],
                    "direction": "up" if indicators["price"]["return_1bar"] > 0 else "down",
                },
            })
            
        # Volume surge detection
        if indicators["volume"]["zscore"] > 3:
            anomalies.append({
                "type": "volume_surge",
                "severity": "high" if indicators["volume"]["zscore"] > 5 else "medium",
                "metrics": {
                    "zscore": indicators["volume"]["zscore"],
                    "ratio": indicators["volume"]["ratio_to_avg"],
                },
            })
            
        # Volatility expansion
        if indicators["price"]["volatility_percentile"] > 1.5:
            anomalies.append({
                "type": "volatility_expansion",
                "severity": "medium",
                "metrics": {
                    "volatility": indicators["price"]["volatility"],
                    "percentile": indicators["price"]["volatility_percentile"],
                },
            })
            
        return anomalies
        
    def _calculate_microstructure_indicators(
        self,
        trades: Optional[List[Trade]],
        quotes: Optional[List[QuoteTick]],
    ) -> Dict[str, Any]:
        """Calculate microstructure indicators."""
        indicators = {
            "bid_ask_spread": None,
            "quote_frequency": None,
            "trade_frequency": None,
            "order_imbalance": None,
        }
        
        # Quote-based indicators
        if quotes and len(quotes) > 0:
            spreads = [
                float(q.ask_price - q.bid_price) / float(q.bid_price) * 10000  # bps
                for q in quotes[-100:]  # Last 100 quotes
            ]
            
            if spreads:
                indicators["bid_ask_spread"] = {
                    "current": spreads[-1],
                    "average": np.mean(spreads),
                    "max": max(spreads),
                }
                
            # Quote frequency (quotes per minute)
            if len(quotes) > 1:
                time_span = (quotes[-1].ts_event - quotes[0].ts_event) / 1e9 / 60  # minutes
                indicators["quote_frequency"] = len(quotes) / time_span if time_span > 0 else 0
                
        # Trade-based indicators
        if trades and len(trades) > 0:
            # Trade frequency
            if len(trades) > 1:
                time_span = (trades[-1].ts_event - trades[0].ts_event) / 1e9 / 60  # minutes
                indicators["trade_frequency"] = len(trades) / time_span if time_span > 0 else 0
                
            # Order imbalance
            buy_volume = sum(float(t.size) for t in trades if t.aggressor_side.value == "BUYER")
            sell_volume = sum(float(t.size) for t in trades if t.aggressor_side.value == "SELLER")
            total_volume = buy_volume + sell_volume
            
            if total_volume > 0:
                indicators["order_imbalance"] = (buy_volume - sell_volume) / total_volume
                
        return indicators
        
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
        
    def _summarize_trades(self, trades: List[Trade]) -> Dict[str, Any]:
        """Summarize trade data."""
        if not trades:
            return {}
            
        return {
            "count": len(trades),
            "total_volume": sum(float(t.size) for t in trades),
            "avg_size": np.mean([float(t.size) for t in trades]),
            "price_range": {
                "min": min(float(t.price) for t in trades),
                "max": max(float(t.price) for t in trades),
            },
            "time_span_seconds": (trades[-1].ts_event - trades[0].ts_event) / 1e9,
        }
        
    def _summarize_quotes(self, quotes: List[QuoteTick]) -> Dict[str, Any]:
        """Summarize quote data."""
        if not quotes:
            return {}
            
        spreads = [float(q.ask_price - q.bid_price) for q in quotes]
        
        return {
            "count": len(quotes),
            "avg_spread": np.mean(spreads),
            "max_spread": max(spreads),
            "avg_bid_size": np.mean([float(q.bid_size) for q in quotes]),
            "avg_ask_size": np.mean([float(q.ask_size) for q in quotes]),
            "time_span_seconds": (quotes[-1].ts_event - quotes[0].ts_event) / 1e9,
        }
        
    def _generate_anomaly_id(self) -> str:
        """Generate unique anomaly ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        count = len(self._anomaly_history)
        return f"ANOM_{timestamp}_{count:04d}"
        
    def _find_similar_anomalies(
        self,
        anomaly: MarketAnomaly,
        lookback_days: int,
    ) -> List[MarketAnomaly]:
        """Find similar historical anomalies."""
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        
        similar = []
        for historical in self._anomaly_history:
            if historical.start_time < cutoff:
                continue
                
            if historical.anomaly_id == anomaly.anomaly_id:
                continue
                
            # Check similarity
            if (historical.anomaly_type == anomaly.anomaly_type and
                historical.instrument_id == anomaly.instrument_id):
                similar.append(historical)
                
        # Sort by similarity score (could be enhanced)
        similar.sort(key=lambda a: abs(a.confidence - anomaly.confidence))
        
        return similar
        
    def _get_anomaly_outcome(self, anomaly: MarketAnomaly) -> Dict[str, Any]:
        """Get outcome of historical anomaly."""
        # This would be enhanced with actual outcome tracking
        return {
            "duration_minutes": (
                (anomaly.end_time - anomaly.start_time).total_seconds() / 60
                if anomaly.end_time else None
            ),
            "resolved": anomaly.end_time is not None,
            "severity_confirmed": True,  # Placeholder
        }
        
    def _prepare_positions(self, positions: Dict[InstrumentId, Any]) -> List[Dict[str, Any]]:
        """Prepare position data for analysis."""
        return [
            {
                "instrument": str(inst_id),
                "size": float(pos.quantity) if hasattr(pos, "quantity") else 0,
                "value": float(pos.quantity * pos.last_px) 
                    if hasattr(pos, "quantity") and hasattr(pos, "last_px") else 0,
            }
            for inst_id, pos in positions.items()
        ]