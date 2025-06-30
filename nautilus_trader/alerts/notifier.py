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
Real-time alert notification system.
"""

import asyncio
import json
import smtplib
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set

import httpx

from nautilus_trader.common.component import Component
from nautilus_trader.common.logging import Logger
from nautilus_trader.cryptography.key_manager import KeyManager


class AlertLevel(Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    
    # Trading alerts
    POSITION_RISK = "position_risk"
    DRAWDOWN = "drawdown"
    EXECUTION_ERROR = "execution_error"
    ORDER_REJECTED = "order_rejected"
    
    # System alerts
    HIGH_CPU = "high_cpu"
    HIGH_MEMORY = "high_memory"
    DISK_SPACE = "disk_space"
    NETWORK_ISSUE = "network_issue"
    
    # Security alerts
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    MFA_FAILURE = "mfa_failure"
    API_ABUSE = "api_abuse"
    
    # AI alerts
    AI_SERVICE_DOWN = "ai_service_down"
    AI_ANALYSIS_FAILED = "ai_analysis_failed"
    AI_CONFIDENCE_LOW = "ai_confidence_low"


class Alert:
    """Represents an alert."""
    
    def __init__(
        self,
        alert_type: AlertType,
        level: AlertLevel,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> None:
        """Initialize alert."""
        self.alert_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.alert_type = alert_type
        self.level = level
        self.title = title
        self.message = message
        self.details = details or {}
        self.source = source or "system"
        self.timestamp = datetime.now(timezone.utc)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "type": self.alert_type.value,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "details": self.details,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""
    
    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Send alert through this channel."""
        pass
        
    @abstractmethod
    def is_available(self) -> bool:
        """Check if channel is available."""
        pass


class EmailChannel(NotificationChannel):
    """Email notification channel."""
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: List[str],
        use_tls: bool = True,
    ) -> None:
        """Initialize email channel."""
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        self.use_tls = use_tls
        self._available = True
        
    async def send(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{alert.level.value.upper()}] {alert.title}"
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)
            
            # Create HTML content
            html_content = f"""
            <html>
                <body>
                    <h2 style="color: {self._get_color(alert.level)};">
                        {alert.title}
                    </h2>
                    <p><strong>Type:</strong> {alert.alert_type.value}</p>
                    <p><strong>Level:</strong> {alert.level.value}</p>
                    <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                    <p><strong>Source:</strong> {alert.source}</p>
                    
                    <h3>Message</h3>
                    <p>{alert.message}</p>
                    
                    {self._format_details(alert.details)}
                    
                    <hr>
                    <p style="font-size: 12px; color: #666;">
                        Alert ID: {alert.alert_id}<br>
                        Sent by Nautilus Trader Alert System
                    </p>
                </body>
            </html>
            """
            
            # Attach HTML
            part = MIMEText(html_content, "html")
            msg.attach(part)
            
            # Send email
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._send_email_sync,
                msg,
            )
            
            return True
            
        except Exception as e:
            print(f"Failed to send email alert: {e}")
            self._available = False
            return False
            
    def _send_email_sync(self, msg: MIMEMultipart) -> None:
        """Send email synchronously."""
        if self.use_tls:
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port)
            
        server.login(self.username, self.password)
        server.send_message(msg)
        server.quit()
        
    def _get_color(self, level: AlertLevel) -> str:
        """Get color for alert level."""
        colors = {
            AlertLevel.INFO: "#0066cc",
            AlertLevel.WARNING: "#ff9900",
            AlertLevel.ERROR: "#cc3300",
            AlertLevel.CRITICAL: "#990000",
        }
        return colors.get(level, "#333333")
        
    def _format_details(self, details: Dict[str, Any]) -> str:
        """Format alert details as HTML."""
        if not details:
            return ""
            
        html = "<h3>Details</h3><ul>"
        for key, value in details.items():
            html += f"<li><strong>{key}:</strong> {value}</li>"
        html += "</ul>"
        return html
        
    def is_available(self) -> bool:
        """Check if email channel is available."""
        return self._available


class WebhookChannel(NotificationChannel):
    """Webhook notification channel."""
    
    def __init__(
        self,
        webhook_url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ) -> None:
        """Initialize webhook channel."""
        self.webhook_url = webhook_url
        self.headers = headers or {}
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        self._available = True
        
    async def send(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        try:
            # Prepare payload
            payload = {
                "alert": alert.to_dict(),
                "formatted_message": self._format_message(alert),
            }
            
            # Send webhook
            response = await self._client.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
            )
            
            if response.status_code >= 200 and response.status_code < 300:
                return True
            else:
                print(f"Webhook returned status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Failed to send webhook alert: {e}")
            self._available = False
            return False
            
    def _format_message(self, alert: Alert) -> str:
        """Format alert message for webhook."""
        emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }.get(alert.level, "📢")
        
        message = f"{emoji} **{alert.title}**\n\n"
        message += f"**Type:** {alert.alert_type.value}\n"
        message += f"**Level:** {alert.level.value}\n"
        message += f"**Time:** {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        message += f"**Source:** {alert.source}\n\n"
        message += f"{alert.message}\n"
        
        if alert.details:
            message += "\n**Details:**\n"
            for key, value in alert.details.items():
                message += f"- {key}: {value}\n"
                
        return message
        
    def is_available(self) -> bool:
        """Check if webhook channel is available."""
        return self._available
        
    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


class AlertNotifier(Component):
    """
    Real-time alert notification system.
    
    Manages alert routing, throttling, and delivery through multiple channels.
    """
    
    def __init__(
        self,
        channels: List[NotificationChannel],
        throttle_window_seconds: int = 300,  # 5 minutes
        max_alerts_per_window: int = 10,
        enable_file_logging: bool = True,
        log_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize alert notifier.
        
        Parameters
        ----------
        channels : List[NotificationChannel]
            Notification channels to use
        throttle_window_seconds : int
            Time window for throttling (seconds)
        max_alerts_per_window : int
            Maximum alerts per throttle window
        enable_file_logging : bool
            Whether to log alerts to file
        log_dir : str, optional
            Directory for alert logs
        """
        super().__init__()
        
        self._channels = channels
        self._throttle_window = throttle_window_seconds
        self._max_alerts = max_alerts_per_window
        self._enable_file_logging = enable_file_logging
        
        # Setup logging directory
        if log_dir:
            self._log_dir = Path(log_dir).expanduser()
        else:
            self._log_dir = Path("~/.nautilus/alert_logs").expanduser()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        # Alert tracking
        self._alert_history: Deque[Alert] = deque(maxlen=1000)
        self._alert_counts: Dict[str, List[datetime]] = {}
        
        # Channel routing rules
        self._routing_rules: Dict[AlertLevel, List[NotificationChannel]] = {
            AlertLevel.INFO: [],  # No notifications for info by default
            AlertLevel.WARNING: channels[:],
            AlertLevel.ERROR: channels[:],
            AlertLevel.CRITICAL: channels[:],  # All channels for critical
        }
        
    async def send_alert(
        self,
        alert_type: AlertType,
        level: AlertLevel,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> bool:
        """
        Send an alert.
        
        Parameters
        ----------
        alert_type : AlertType
            Type of alert
        level : AlertLevel
            Severity level
        title : str
            Alert title
        message : str
            Alert message
        details : Dict[str, Any], optional
            Additional details
        source : str, optional
            Source of the alert
            
        Returns
        -------
        bool
            True if alert was sent successfully
        """
        # Create alert
        alert = Alert(
            alert_type=alert_type,
            level=level,
            title=title,
            message=message,
            details=details,
            source=source,
        )
        
        # Check throttling
        if self._is_throttled(alert):
            self.log.warning(f"Alert throttled: {alert.title}")
            return False
            
        # Add to history
        self._alert_history.append(alert)
        
        # Log to file
        if self._enable_file_logging:
            self._log_alert(alert)
            
        # Get channels for this alert level
        channels = self._routing_rules.get(level, [])
        
        # Send through available channels
        success = False
        for channel in channels:
            if channel.is_available():
                try:
                    result = await channel.send(alert)
                    if result:
                        success = True
                except Exception as e:
                    self.log.error(f"Failed to send alert through channel: {e}")
                    
        # Update throttle tracking
        self._update_throttle_count(alert)
        
        return success
        
    def _is_throttled(self, alert: Alert) -> bool:
        """Check if alert should be throttled."""
        key = f"{alert.alert_type.value}:{alert.level.value}"
        
        if key not in self._alert_counts:
            return False
            
        # Remove old entries
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self._throttle_window)
        self._alert_counts[key] = [
            dt for dt in self._alert_counts[key]
            if dt > cutoff_time
        ]
        
        # Check count
        return len(self._alert_counts[key]) >= self._max_alerts
        
    def _update_throttle_count(self, alert: Alert) -> None:
        """Update throttle count for alert."""
        key = f"{alert.alert_type.value}:{alert.level.value}"
        
        if key not in self._alert_counts:
            self._alert_counts[key] = []
            
        self._alert_counts[key].append(alert.timestamp)
        
    def _log_alert(self, alert: Alert) -> None:
        """Log alert to file."""
        try:
            # Create daily log file
            date_str = alert.timestamp.strftime("%Y%m%d")
            log_file = self._log_dir / f"alerts_{date_str}.jsonl"
            
            # Append alert
            with open(log_file, "a") as f:
                json.dump(alert.to_dict(), f)
                f.write("\n")
                
        except Exception as e:
            self.log.error(f"Failed to log alert to file: {e}")
            
    def set_routing_rule(
        self,
        level: AlertLevel,
        channels: List[NotificationChannel],
    ) -> None:
        """Set which channels to use for a specific alert level."""
        self._routing_rules[level] = channels
        
    def get_alert_history(
        self,
        alert_type: Optional[AlertType] = None,
        level: Optional[AlertLevel] = None,
        hours: int = 24,
    ) -> List[Alert]:
        """
        Get alert history.
        
        Parameters
        ----------
        alert_type : AlertType, optional
            Filter by alert type
        level : AlertLevel, optional
            Filter by level
        hours : int
            How many hours of history to return
            
        Returns
        -------
        List[Alert]
            Filtered alert history
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        alerts = []
        for alert in self._alert_history:
            if alert.timestamp < cutoff_time:
                continue
                
            if alert_type and alert.alert_type != alert_type:
                continue
                
            if level and alert.level != level:
                continue
                
            alerts.append(alert)
            
        return alerts
        
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts."""
        summary = {
            "total_alerts_24h": 0,
            "by_level": {},
            "by_type": {},
            "throttled_count": 0,
        }
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        for alert in self._alert_history:
            if alert.timestamp >= cutoff_time:
                summary["total_alerts_24h"] += 1
                
                # Count by level
                level_key = alert.level.value
                summary["by_level"][level_key] = summary["by_level"].get(level_key, 0) + 1
                
                # Count by type
                type_key = alert.alert_type.value
                summary["by_type"][type_key] = summary["by_type"].get(type_key, 0) + 1
                
        # Count throttled alerts
        for timestamps in self._alert_counts.values():
            recent_timestamps = [dt for dt in timestamps if dt >= cutoff_time]
            if len(recent_timestamps) >= self._max_alerts:
                summary["throttled_count"] += len(recent_timestamps) - self._max_alerts
                
        return summary