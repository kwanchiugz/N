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
Audit event handler for event-driven security logging.
"""

from typing import Optional

from nautilus_trader.auth.audit import SecurityAuditLogger, SecurityEventType
from nautilus_trader.common.event_bus import EventBus, get_event_bus
from nautilus_trader.events.trading_events import (
    EventType,
    TradingEvent,
    StrategyStartedEvent,
    StrategyStoppedEvent,
    OrderFilledEvent,
    AIAnalysisCompletedEvent,
    RiskLimitExceededEvent,
)


class AuditEventHandler:
    """
    Handles trading events and logs them for security audit.
    
    This decouples strategies from audit logging by subscribing
    to events via the event bus.
    """
    
    def __init__(
        self,
        audit_logger: Optional[SecurityAuditLogger] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """
        Initialize the audit event handler.
        
        Parameters
        ----------
        audit_logger : SecurityAuditLogger, optional
            The audit logger to use. If None, creates a new one.
        event_bus : EventBus, optional
            The event bus to subscribe to. If None, uses global bus.
        """
        self._audit_logger = audit_logger or SecurityAuditLogger(
            log_dir="~/.nautilus/audit_logs",
            enable_alerts=True,
        )
        self._event_bus = event_bus or get_event_bus()
        
        # Subscribe to relevant events
        self._subscribe_to_events()
        
    def _subscribe_to_events(self) -> None:
        """Subscribe to events that need audit logging."""
        # Strategy lifecycle events
        self._event_bus.subscribe(
            EventType.STRATEGY_STARTED,
            self._handle_strategy_started,
            priority=10,  # High priority for audit
        )
        self._event_bus.subscribe(
            EventType.STRATEGY_STOPPED,
            self._handle_strategy_stopped,
            priority=10,
        )
        
        # Trading events
        self._event_bus.subscribe(
            EventType.ORDER_FILLED,
            self._handle_order_filled,
            priority=10,
        )
        self._event_bus.subscribe(
            EventType.ORDER_REJECTED,
            self._handle_order_rejected,
            priority=10,
        )
        
        # Risk events
        self._event_bus.subscribe(
            EventType.RISK_LIMIT_EXCEEDED,
            self._handle_risk_limit_exceeded,
            priority=10,
        )
        
        # AI events
        self._event_bus.subscribe(
            EventType.AI_ANALYSIS_COMPLETED,
            self._handle_ai_analysis_completed,
            priority=5,  # Lower priority, informational
        )
        
    def _handle_strategy_started(self, event: StrategyStartedEvent) -> None:
        """Handle strategy started event."""
        self._audit_logger.log_event({
            "event_type": SecurityEventType.STRATEGY_START.value,
            "strategy_id": str(event.strategy_id),
            "trader_id": str(event.trader_id),
            "config": event.config,
            "timestamp": event.timestamp.isoformat(),
        })
        
    def _handle_strategy_stopped(self, event: TradingEvent) -> None:
        """Handle strategy stopped event."""
        self._audit_logger.log_event({
            "event_type": SecurityEventType.STRATEGY_STOP.value,
            "strategy_id": event.data.get("strategy_id"),
            "reason": event.data.get("reason"),
            "final_stats": event.data.get("final_stats"),
            "timestamp": event.timestamp.isoformat(),
        })
        
    def _handle_order_filled(self, event: OrderFilledEvent) -> None:
        """Handle order filled event."""
        self._audit_logger.log_event({
            "event_type": SecurityEventType.TRADE_EXECUTED.value,
            "strategy_id": str(event.strategy_id),
            "order_id": str(event.order_id),
            "side": event.data.get("side"),
            "price": event.data.get("price"),
            "quantity": event.data.get("quantity"),
            "commission": event.data.get("commission"),
            "timestamp": event.timestamp.isoformat(),
        })
        
    def _handle_order_rejected(self, event: TradingEvent) -> None:
        """Handle order rejected event."""
        self._audit_logger.log_event({
            "event_type": SecurityEventType.ORDER_REJECTED.value,
            "strategy_id": event.data.get("strategy_id"),
            "order_id": event.data.get("order_id"),
            "reason": event.data.get("reason"),
            "timestamp": event.timestamp.isoformat(),
            "severity": "WARNING",
        })
        
    def _handle_risk_limit_exceeded(self, event: RiskLimitExceededEvent) -> None:
        """Handle risk limit exceeded event."""
        self._audit_logger.log_event({
            "event_type": SecurityEventType.RISK_LIMIT_EXCEEDED.value,
            "strategy_id": str(event.strategy_id),
            "limit_type": event.limit_type,
            "current_value": event.current_value,
            "limit_value": event.limit_value,
            "action_taken": event.action_taken,
            "timestamp": event.timestamp.isoformat(),
            "severity": "CRITICAL",
        })
        
    def _handle_ai_analysis_completed(self, event: AIAnalysisCompletedEvent) -> None:
        """Handle AI analysis completed event."""
        # Only log high-confidence AI decisions for audit
        if event.confidence >= 0.8:
            self._audit_logger.log_event({
                "event_type": "AI_DECISION",
                "strategy_id": str(event.strategy_id),
                "analysis_type": event.analysis_type,
                "confidence": event.confidence,
                "ai_provider": event.ai_provider,
                "timestamp": event.timestamp.isoformat(),
            })
            
    def close(self) -> None:
        """Close the audit logger."""
        if self._audit_logger:
            self._audit_logger.close()


# Global audit handler instance
_audit_handler: Optional[AuditEventHandler] = None


def initialize_audit_handler(
    audit_logger: Optional[SecurityAuditLogger] = None,
    event_bus: Optional[EventBus] = None,
) -> AuditEventHandler:
    """
    Initialize the global audit handler.
    
    This should be called once at application startup.
    """
    global _audit_handler
    if _audit_handler is None:
        _audit_handler = AuditEventHandler(audit_logger, event_bus)
    return _audit_handler


def get_audit_handler() -> Optional[AuditEventHandler]:
    """Get the global audit handler instance."""
    return _audit_handler