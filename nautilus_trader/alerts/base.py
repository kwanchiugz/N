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
Base classes for the alerting system.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import asyncio
import uuid
from collections import defaultdict

from nautilus_trader.common.component import Logger
from nautilus_trader.config import NautilusConfig
from nautilus_trader.storage import StorageProvider
from nautilus_trader.common.event_bus import EventBus, get_event_bus
from nautilus_trader.events.trading_events import EventType, TradingEvent


class AlertLevel(str, Enum):
    """Alert severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Alert:
    """
    Represents an alert message.
    
    Parameters
    ----------
    alert_id : str
        Unique alert identifier
    level : AlertLevel
        Alert severity level
    title : str
        Alert title
    message : str
        Alert message content
    source : str
        Source of the alert
    timestamp : datetime
        Alert creation time
    metadata : dict, optional
        Additional alert metadata
    """
    
    def __init__(
        self,
        alert_id: str,
        level: AlertLevel,
        title: str,
        message: str,
        source: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.alert_id = alert_id
        self.level = level
        self.title = title
        self.message = message
        self.source = source
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.metadata = metadata or {}
        
    @classmethod
    def create(
        cls,
        level: AlertLevel,
        title: str,
        message: str,
        source: str,
        **metadata,
    ) -> "Alert":
        """Create a new alert with auto-generated ID."""
        return cls(
            alert_id=str(uuid.uuid4()),
            level=level,
            title=title,
            message=message,
            source=source,
            metadata=metadata,
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
        
    def __str__(self) -> str:
        return f"[{self.level.value.upper()}] {self.title}: {self.message}"


class AlertChannel(ABC):
    """
    Abstract base class for alert channels.
    
    Alert channels handle the delivery of alerts to various destinations
    such as email, webhooks, messaging services, etc.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize alert channel.
        
        Parameters
        ----------
        name : str
            Channel name
        config : dict, optional
            Channel configuration
        """
        self.name = name
        self.config = config or {}
        self._logger = Logger(f"{self.__class__.__name__}[{name}]")
        self._enabled = True
        self._filter_levels: Optional[Set[AlertLevel]] = None
        
    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """
        Send an alert through this channel.
        
        Parameters
        ----------
        alert : Alert
            Alert to send
            
        Returns
        -------
        bool
            True if sent successfully
        """
        pass
        
    async def initialize(self) -> None:
        """Initialize the channel (optional)."""
        pass
        
    async def shutdown(self) -> None:
        """Shutdown the channel (optional)."""
        pass
        
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the channel."""
        self._enabled = enabled
        
    def set_filter_levels(self, levels: Set[AlertLevel]) -> None:
        """Set alert levels to filter."""
        self._filter_levels = levels
        
    def should_send(self, alert: Alert) -> bool:
        """Check if alert should be sent through this channel."""
        if not self._enabled:
            return False
            
        if self._filter_levels and alert.level not in self._filter_levels:
            return False
            
        return True


class AlertConfig(NautilusConfig):
    """Configuration for alert manager."""
    
    storage_enabled: bool = True
    storage_namespace: str = "alerts"
    max_alerts_per_minute: int = 100
    deduplication_window_seconds: int = 60
    batch_window_seconds: float = 1.0
    retry_attempts: int = 3
    retry_delay: float = 1.0


class AlertManager:
    """
    Manages alerts and notification channels.
    
    This manager provides:
    - Alert creation and routing
    - Multiple notification channels
    - Alert deduplication
    - Rate limiting
    - Batch sending
    - Alert storage
    """
    
    def __init__(
        self,
        config: AlertConfig,
        storage: Optional[StorageProvider] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """
        Initialize alert manager.
        
        Parameters
        ----------
        config : AlertConfig
            Manager configuration
        storage : StorageProvider, optional
            Storage provider for alerts
        event_bus : EventBus, optional
            Event bus for alert events
        """
        self._config = config
        self._storage = storage
        self._event_bus = event_bus or get_event_bus()
        self._logger = Logger(self.__class__.__name__)
        
        # Channels
        self._channels: Dict[str, AlertChannel] = {}
        
        # Rate limiting
        self._alert_counts: Dict[str, List[datetime]] = defaultdict(list)
        
        # Deduplication
        self._recent_alerts: Dict[str, datetime] = {}
        
        # Batch processing
        self._pending_alerts: List[Alert] = []
        self._batch_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats = {
            "sent": 0,
            "failed": 0,
            "rate_limited": 0,
            "deduplicated": 0,
        }
        
    async def initialize(self) -> None:
        """Initialize alert manager."""
        # Initialize channels
        for channel in self._channels.values():
            await channel.initialize()
            
        # Start batch processor
        if self._config.batch_window_seconds > 0:
            self._batch_task = asyncio.create_task(self._batch_processor())
            
        self._logger.info("Alert manager initialized")
        
    async def shutdown(self) -> None:
        """Shutdown alert manager."""
        # Stop batch processor
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
                
        # Process remaining alerts
        if self._pending_alerts:
            await self._process_batch()
            
        # Shutdown channels
        for channel in self._channels.values():
            await channel.shutdown()
            
        self._logger.info("Alert manager shutdown")
        
    def add_channel(self, channel: AlertChannel) -> None:
        """
        Add a notification channel.
        
        Parameters
        ----------
        channel : AlertChannel
            Channel to add
        """
        self._channels[channel.name] = channel
        self._logger.info(f"Added alert channel: {channel.name}")
        
    def remove_channel(self, name: str) -> None:
        """
        Remove a notification channel.
        
        Parameters
        ----------
        name : str
            Channel name to remove
        """
        if name in self._channels:
            del self._channels[name]
            self._logger.info(f"Removed alert channel: {name}")
            
    async def send_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        source: str,
        immediate: bool = False,
        **metadata,
    ) -> Optional[str]:
        """
        Send an alert.
        
        Parameters
        ----------
        level : AlertLevel
            Alert severity
        title : str
            Alert title
        message : str
            Alert message
        source : str
            Alert source
        immediate : bool
            Whether to send immediately (skip batching)
        **metadata
            Additional metadata
            
        Returns
        -------
        str or None
            Alert ID if sent, None if filtered
        """
        # Create alert
        alert = Alert.create(
            level=level,
            title=title,
            message=message,
            source=source,
            **metadata,
        )
        
        # Check rate limit
        if not self._check_rate_limit(source):
            self._stats["rate_limited"] += 1
            self._logger.warning(f"Rate limit exceeded for source: {source}")
            return None
            
        # Check deduplication
        if self._is_duplicate(alert):
            self._stats["deduplicated"] += 1
            self._logger.debug(f"Duplicate alert filtered: {alert.title}")
            return None
            
        # Store alert if enabled
        if self._config.storage_enabled and self._storage:
            await self._store_alert(alert)
            
        # Send immediately or batch
        if immediate or self._config.batch_window_seconds <= 0:
            await self._send_to_channels(alert)
        else:
            self._pending_alerts.append(alert)
            
        return alert.alert_id
        
    async def send_event_alert(
        self,
        event: TradingEvent,
        level: Optional[AlertLevel] = None,
    ) -> Optional[str]:
        """
        Send alert for a trading event.
        
        Parameters
        ----------
        event : TradingEvent
            Trading event
        level : AlertLevel, optional
            Override alert level
            
        Returns
        -------
        str or None
            Alert ID if sent
        """
        # Determine alert level
        if level is None:
            if hasattr(event, "severity"):
                level = AlertLevel(event.severity)
            elif event.event_type == EventType.ERROR:
                level = AlertLevel.ERROR
            elif event.event_type == EventType.WARNING:
                level = AlertLevel.WARNING
            else:
                level = AlertLevel.INFO
                
        # Create alert from event
        title = f"{event.event_type.value}: {getattr(event, 'title', event.event_type.value)}"
        message = getattr(event, "message", str(event))
        source = getattr(event, "source", "system")
        
        # Include event data as metadata
        metadata = event.__dict__.copy()
        metadata["event_type"] = event.event_type.value
        
        return await self.send_alert(
            level=level,
            title=title,
            message=message,
            source=source,
            **metadata,
        )
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert manager statistics."""
        stats = self._stats.copy()
        stats["channels"] = list(self._channels.keys())
        stats["pending_alerts"] = len(self._pending_alerts)
        return stats
        
    async def _send_to_channels(self, alert: Alert) -> None:
        """Send alert to all applicable channels."""
        tasks = []
        
        for channel in self._channels.values():
            if channel.should_send(alert):
                tasks.append(self._send_with_retry(channel, alert))
                
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes and failures
            for result in results:
                if isinstance(result, Exception):
                    self._stats["failed"] += 1
                    self._logger.error(f"Alert send failed: {result}")
                else:
                    self._stats["sent"] += 1
                    
    async def _send_with_retry(
        self,
        channel: AlertChannel,
        alert: Alert,
    ) -> bool:
        """Send alert with retry logic."""
        for attempt in range(self._config.retry_attempts):
            try:
                result = await channel.send_alert(alert)
                if result:
                    return True
                    
            except Exception as e:
                if attempt == self._config.retry_attempts - 1:
                    raise
                    
                self._logger.warning(
                    f"Alert send attempt {attempt + 1} failed: {e}, retrying..."
                )
                await asyncio.sleep(self._config.retry_delay * (attempt + 1))
                
        return False
        
    async def _batch_processor(self) -> None:
        """Process alerts in batches."""
        while True:
            await asyncio.sleep(self._config.batch_window_seconds)
            
            if self._pending_alerts:
                await self._process_batch()
                
    async def _process_batch(self) -> None:
        """Process pending alerts."""
        alerts = self._pending_alerts.copy()
        self._pending_alerts.clear()
        
        # Group by level for batch sending
        grouped = defaultdict(list)
        for alert in alerts:
            grouped[alert.level].append(alert)
            
        # Send each group
        for level, level_alerts in grouped.items():
            if len(level_alerts) == 1:
                await self._send_to_channels(level_alerts[0])
            else:
                # Create summary alert
                summary = Alert.create(
                    level=level,
                    title=f"{len(level_alerts)} {level.value} alerts",
                    message=self._format_batch_message(level_alerts),
                    source="alert_manager",
                    alerts=len(level_alerts),
                )
                await self._send_to_channels(summary)
                
    def _format_batch_message(self, alerts: List[Alert]) -> str:
        """Format batch alert message."""
        lines = []
        for alert in alerts[:10]:  # Limit to 10
            lines.append(f"• {alert.title}: {alert.message}")
            
        if len(alerts) > 10:
            lines.append(f"... and {len(alerts) - 10} more")
            
        return "\n".join(lines)
        
    def _check_rate_limit(self, source: str) -> bool:
        """Check if source is within rate limit."""
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - 60  # 1 minute window
        
        # Clean old entries
        self._alert_counts[source] = [
            dt for dt in self._alert_counts[source]
            if dt.timestamp() > cutoff
        ]
        
        # Check limit
        if len(self._alert_counts[source]) >= self._config.max_alerts_per_minute:
            return False
            
        # Add current
        self._alert_counts[source].append(now)
        return True
        
    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is a duplicate."""
        # Create dedup key
        key = f"{alert.source}:{alert.title}:{alert.level.value}"
        
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - self._config.deduplication_window_seconds
        
        # Check if seen recently
        if key in self._recent_alerts:
            last_seen = self._recent_alerts[key]
            if last_seen.timestamp() > cutoff:
                return True
                
        # Update last seen
        self._recent_alerts[key] = now
        
        # Clean old entries periodically
        if len(self._recent_alerts) > 1000:
            self._recent_alerts = {
                k: v for k, v in self._recent_alerts.items()
                if v.timestamp() > cutoff
            }
            
        return False
        
    async def _store_alert(self, alert: Alert) -> None:
        """Store alert in storage."""
        try:
            await self._storage.save_json(
                key=alert.alert_id,
                data=alert.to_dict(),
                namespace=self._config.storage_namespace,
            )
        except Exception as e:
            self._logger.error(f"Failed to store alert: {e}")