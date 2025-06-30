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
AI-enhanced trading strategy with DeepSeek integration.

This strategy combines traditional technical analysis with AI-powered
market analysis and risk management.
"""

import asyncio
from decimal import Decimal
from typing import Optional

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.core.message import Event
from nautilus_trader.indicators.atr import AverageTrueRange
from nautilus_trader.indicators.ema import ExponentialMovingAverage
from nautilus_trader.indicators.rsi import RelativeStrengthIndex
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import OrderBookDeltas
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import LimitOrder
from nautilus_trader.model.position import Position
from nautilus_trader.trading.strategy import Strategy

from nautilus_trader.ai.analyzers.market import MarketAnalyzer
from nautilus_trader.ai.analyzers.risk import RiskAnalyzer
from nautilus_trader.ai.config import AIConfig
from nautilus_trader.common.event_bus import get_event_bus
from nautilus_trader.events.trading_events import (
    EventType,
    StrategyStartedEvent,
    StrategyStoppedEvent,
    OrderFilledEvent as TradingOrderFilledEvent,
    AIAnalysisCompletedEvent,
)


class AIStrategyConfig(StrategyConfig):
    """Configuration for AI-enhanced trading strategy."""
    
    instrument_id: InstrumentId
    bar_type: BarType
    ai_config: AIConfig
    trade_size: Decimal
    risk_per_trade: Decimal = Decimal("0.02")  # 2% risk per trade
    use_ai_signals: bool = True
    use_ai_risk_management: bool = True
    max_positions: int = 3
    ema_fast_period: int = 12
    ema_slow_period: int = 26
    rsi_period: int = 14
    atr_period: int = 14
    min_confidence: float = 0.7  # Minimum AI confidence for signals


class AIEnhancedStrategy(Strategy):
    """
    AI-enhanced trading strategy using DeepSeek for market analysis.
    
    This strategy combines:
    - Traditional technical indicators (EMA, RSI, ATR)
    - AI-powered market analysis and trend detection
    - AI-assisted risk management
    - Dynamic position sizing based on AI confidence
    """
    
    def __init__(self, config: AIStrategyConfig) -> None:
        """Initialize the AI-enhanced strategy."""
        super().__init__(config)
        
        # Configuration
        self.instrument_id = config.instrument_id
        self.bar_type = config.bar_type
        self.trade_size = config.trade_size
        self.risk_per_trade = config.risk_per_trade
        self.use_ai_signals = config.use_ai_signals
        self.use_ai_risk_management = config.use_ai_risk_management
        self.max_positions = config.max_positions
        self.min_confidence = config.min_confidence
        
        # Technical indicators
        self.ema_fast = ExponentialMovingAverage(config.ema_fast_period)
        self.ema_slow = ExponentialMovingAverage(config.ema_slow_period)
        self.rsi = RelativeStrengthIndex(config.rsi_period)
        self.atr = AverageTrueRange(config.atr_period)
        
        # AI components
        self.market_analyzer: Optional[MarketAnalyzer] = None
        self.risk_analyzer: Optional[RiskAnalyzer] = None
        self.ai_config = config.ai_config
        
        # State tracking
        self.last_ai_analysis = None
        self.ai_analysis_task = None
        self.position_count = 0
        
        # Event bus for decoupled architecture
        self._event_bus = get_event_bus()
        
    def on_start(self) -> None:
        """Actions to be performed on strategy start."""
        self.log.info("Starting AI-enhanced strategy")
        
        # Initialize AI components
        if self.ai_config:
            self.market_analyzer = MarketAnalyzer(
                config=self.ai_config.analyzer_config,
                cache=self.cache,
            )
            self.risk_analyzer = RiskAnalyzer(
                config=self.ai_config.analyzer_config,
                cache=self.cache,
            )
            
        # Publish strategy started event
        event = StrategyStartedEvent(
            strategy_id=self.id,
            trader_id=self.trader_id,
            config={
                "instrument": str(self.instrument_id),
                "ai_enabled": bool(self.ai_config),
                "risk_per_trade": str(self.risk_per_trade),
                "max_positions": self.max_positions,
                "min_confidence": self.min_confidence,
            }
        )
        self._event_bus.publish(event)
            
        # Subscribe to market data
        self.subscribe_bars(self.bar_type)
        self.subscribe_quote_ticks(self.instrument_id)
        
        # Request historical bars for indicators
        self.request_bars(self.bar_type, 100)
        
    def on_stop(self) -> None:
        """Actions to be performed on strategy stop."""
        self.log.info("Stopping AI-enhanced strategy")
        
        # Cancel any pending AI analysis
        if self.ai_analysis_task and not self.ai_analysis_task.done():
            self.ai_analysis_task.cancel()
            
        # Publish strategy stopped event
        event = StrategyStoppedEvent(
            strategy_id=self.id,
            reason="User requested stop",
            final_stats={
                "final_position_count": self.position_count,
                "total_trades": self._total_trades if hasattr(self, '_total_trades') else 0,
            }
        )
        self._event_bus.publish(event)
            
        # Unsubscribe from data
        self.unsubscribe_bars(self.bar_type)
        self.unsubscribe_quote_ticks(self.instrument_id)
        
    def on_bar(self, bar: Bar) -> None:
        """Handle bar data."""
        # Update indicators
        self.ema_fast.update_raw(bar.close.as_double())
        self.ema_slow.update_raw(bar.close.as_double())
        self.rsi.update_raw(bar.close.as_double())
        self.atr.update_raw(
            bar.high.as_double(),
            bar.low.as_double(),
            bar.close.as_double(),
        )
        
        # Check if indicators are ready
        if not self._indicators_ready():
            return
            
        # Check position limits
        if self.position_count >= self.max_positions:
            return
            
        # Get current positions
        positions = self.cache.positions_open(
            venue=self.instrument_id.venue,
            instrument_id=self.instrument_id,
        )
        
        # Run AI analysis if enabled
        if self.use_ai_signals and self.market_analyzer:
            self._schedule_ai_analysis(bar)
            
        # Generate trading signals
        signal = self._generate_signal(bar)
        
        if signal and not positions:
            self._execute_signal(signal, bar)
            
    def on_quote_tick(self, tick: QuoteTick) -> None:
        """Handle quote tick data."""
        # Monitor spread for execution quality
        spread = (tick.ask_price - tick.bid_price).as_double()
        
        # Update risk management if needed
        if self.use_ai_risk_management and self.risk_analyzer:
            positions = self.cache.positions_open(
                venue=self.instrument_id.venue,
                instrument_id=self.instrument_id,
            )
            
            for position in positions:
                self._check_position_risk(position, tick)
                
    def on_order_filled(self, event: OrderFilled) -> None:
        """Handle order filled events."""
        self.log.info(f"Order filled: {event}")
        
        # Update position count
        positions = self.cache.positions_open(
            venue=self.instrument_id.venue,
            instrument_id=self.instrument_id,
        )
        self.position_count = len(positions)
        
        # Publish order filled event
        trading_event = TradingOrderFilledEvent(
            strategy_id=self.id,
            order_id=event.order_id,
            side=event.order_side,
            price=event.last_px,
            quantity=event.last_qty,
            commission=event.commission if hasattr(event, 'commission') else None,
        )
        self._event_bus.publish(trading_event)
        
        # Track AI confidence if available
        if self.last_ai_analysis:
            trading_event.data["ai_confidence"] = self.last_ai_analysis.get("confidence")
            
    def _indicators_ready(self) -> bool:
        """Check if all indicators are ready."""
        return (
            self.ema_fast.initialized and
            self.ema_slow.initialized and
            self.rsi.initialized and
            self.atr.initialized
        )
        
    def _schedule_ai_analysis(self, bar: Bar) -> None:
        """Schedule AI analysis in background."""
        # Cancel previous task if still running
        if self.ai_analysis_task and not self.ai_analysis_task.done():
            return
            
        # Create new analysis task
        self.ai_analysis_task = asyncio.create_task(
            self._run_ai_analysis(bar)
        )
        
    async def _run_ai_analysis(self, bar: Bar) -> None:
        """Run AI market analysis."""
        try:
            # Prepare market data
            bars = self.cache.bars(self.bar_type)
            if len(bars) < 20:
                return
                
            # Run market analysis
            trend_analysis = await self.market_analyzer.analyze_trends(
                instrument_id=self.instrument_id,
                bars=bars[-50:],  # Last 50 bars
            )
            
            if trend_analysis:
                # Convert Pydantic model to dict for backward compatibility
                self.last_ai_analysis = {
                    "direction": trend_analysis.direction,
                    "strength": trend_analysis.strength,
                    "confidence": trend_analysis.confidence,
                    "support_levels": trend_analysis.support_levels,
                    "resistance_levels": trend_analysis.resistance_levels,
                    "reasoning": trend_analysis.reasoning,
                }
                self.log.info(f"AI Analysis: direction={trend_analysis.direction}, confidence={trend_analysis.confidence}")
                
                # Publish AI analysis completed event
                event = AIAnalysisCompletedEvent(
                    strategy_id=self.id,
                    analysis_type="market_trend",
                    result=self.last_ai_analysis,
                    confidence=trend_analysis.confidence,
                    ai_provider="deepseek",
                )
                self._event_bus.publish(event)
            
        except Exception as e:
            self.log.error(f"AI analysis failed: {e}")
            
    def _generate_signal(self, bar: Bar) -> Optional[dict]:
        """Generate trading signal combining technical and AI analysis."""
        # Technical analysis
        ema_fast_value = self.ema_fast.value
        ema_slow_value = self.ema_slow.value
        rsi_value = self.rsi.value
        
        # Basic technical signals
        is_uptrend = ema_fast_value > ema_slow_value
        is_downtrend = ema_fast_value < ema_slow_value
        is_oversold = rsi_value < 30
        is_overbought = rsi_value > 70
        
        signal = None
        
        # Combine with AI analysis if available
        if self.use_ai_signals and self.last_ai_analysis:
            ai_confidence = self.last_ai_analysis.get("confidence", 0)
            ai_direction = self.last_ai_analysis.get("direction", "neutral")
            
            if ai_confidence >= self.min_confidence:
                if ai_direction == "bullish" and is_uptrend and not is_overbought:
                    signal = {
                        "side": OrderSide.BUY,
                        "confidence": ai_confidence,
                        "reason": "AI bullish + technical uptrend",
                    }
                elif ai_direction == "bearish" and is_downtrend and not is_oversold:
                    signal = {
                        "side": OrderSide.SELL,
                        "confidence": ai_confidence,
                        "reason": "AI bearish + technical downtrend",
                    }
        else:
            # Fallback to pure technical signals
            if is_uptrend and is_oversold:
                signal = {
                    "side": OrderSide.BUY,
                    "confidence": 0.6,
                    "reason": "Technical: Uptrend + oversold",
                }
            elif is_downtrend and is_overbought:
                signal = {
                    "side": OrderSide.SELL,
                    "confidence": 0.6,
                    "reason": "Technical: Downtrend + overbought",
                }
                
        return signal
        
    def _execute_signal(self, signal: dict, bar: Bar) -> None:
        """Execute trading signal."""
        # Calculate position size based on confidence
        confidence = signal.get("confidence", 0.5)
        position_size = self.trade_size * Decimal(str(confidence))
        
        # Calculate stop loss using ATR
        atr_value = self.atr.value
        stop_distance = Decimal(str(atr_value * 2))  # 2x ATR stop
        
        if signal["side"] == OrderSide.BUY:
            entry_price = bar.close
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * Decimal("2"))  # 2:1 RR
        else:
            entry_price = bar.close
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * Decimal("2"))  # 2:1 RR
            
        # Create and submit order
        order = self.order_factory.limit(
            instrument_id=self.instrument_id,
            order_side=signal["side"],
            quantity=self.instrument.make_qty(position_size),
            price=self.instrument.make_price(entry_price),
            time_in_force=TimeInForce.GTC,
        )
        
        self.submit_order(order)
        
        self.log.info(
            f"Signal executed: {signal['side']} "
            f"@ {entry_price} (confidence: {confidence:.2f})"
        )
        
    def _check_position_risk(self, position: Position, tick: QuoteTick) -> None:
        """Check position risk using AI if enabled."""
        if not self.risk_analyzer:
            return
            
        # Run risk check asynchronously
        asyncio.create_task(
            self._run_risk_check(position, tick)
        )
        
    async def _run_risk_check(self, position: Position, tick: QuoteTick) -> None:
        """Run AI risk analysis on position."""
        try:
            risk_assessment = await self.risk_analyzer.assess_position_risk(
                position=position,
                current_price=tick.bid_price if position.is_long else tick.ask_price,
                market_volatility=self.atr.value if self.atr.initialized else None,
            )
            
            if risk_assessment:
                # Take action based on risk assessment
                risk_score = risk_assessment.risk_score
                
                if risk_score > 0.8:  # High risk
                    self.log.warning(f"High risk detected for position {position.id}: {risk_score}")
                    
                    # Check if any recommendation suggests closing
                    close_recommendations = [
                        r for r in risk_assessment.recommendations 
                        if "close" in r.lower() or "exit" in r.lower()
                    ]
                    
                    if close_recommendations:
                        self._close_position(position, "AI risk management")
                    
        except Exception as e:
            self.log.error(f"Risk check failed: {e}")
            
    def _close_position(self, position: Position, reason: str) -> None:
        """Close a position."""
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL if position.is_long else OrderSide.BUY,
            quantity=position.quantity,
            time_in_force=TimeInForce.IOC,
        )
        
        self.submit_order(order)
        
        self.log.info(f"Closing position {position.id}: {reason}")
        
        # Publish risk action event
        from nautilus_trader.events.trading_events import RiskLimitExceededEvent
        event = RiskLimitExceededEvent(
            strategy_id=self.id,
            risk_type="position_risk",
            risk_value=1.0,  # Max risk score that triggered closure
            threshold=0.8,
            action_taken="position_closed",
            position_id=str(position.id),
            reason=reason,
        )
        self._event_bus.publish(event)