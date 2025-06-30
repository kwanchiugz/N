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
Security audit logging for compliance and monitoring.

This module provides comprehensive audit logging for security events,
authentication attempts, and configuration changes.
"""

import json
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from pathlib import Path

from nautilus_trader.common.component import Logger
from nautilus_trader.core.uuid import UUID4


class SecurityEventType(Enum):
    """Security event types for audit logging."""
    
    # Authentication events
    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILED = "LOGIN_FAILED"
    LOGOUT = "LOGOUT"
    MFA_SUCCESS = "MFA_SUCCESS"
    MFA_FAILED = "MFA_FAILED"
    MFA_ENABLED = "MFA_ENABLED"
    MFA_DISABLED = "MFA_DISABLED"
    
    # Session events
    SESSION_CREATED = "SESSION_CREATED"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    SESSION_REVOKED = "SESSION_REVOKED"
    TOKEN_REFRESHED = "TOKEN_REFRESHED"
    
    # Access control events
    ACCESS_GRANTED = "ACCESS_GRANTED"
    ACCESS_DENIED = "ACCESS_DENIED"
    PERMISSION_CHANGED = "PERMISSION_CHANGED"
    
    # Configuration events
    CONFIG_CHANGED = "CONFIG_CHANGED"
    KEY_ROTATED = "KEY_ROTATED"
    CREDENTIALS_UPDATED = "CREDENTIALS_UPDATED"
    
    # Trading events
    ORDER_PLACED = "ORDER_PLACED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    TRADE_EXECUTED = "TRADE_EXECUTED"
    
    # Strategy events
    STRATEGY_START = "STRATEGY_START"
    STRATEGY_STOP = "STRATEGY_STOP"
    
    # Risk events
    RISK_LIMIT_EXCEEDED = "RISK_LIMIT_EXCEEDED"
    
    # Security events
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    INVALID_API_KEY = "INVALID_API_KEY"
    IP_WHITELIST_VIOLATION = "IP_WHITELIST_VIOLATION"


class SecurityEvent:
    """
    Represents a security audit event.
    
    Parameters
    ----------
    event_type : SecurityEventType
        Type of security event.
    user_id : str, optional
        User associated with the event.
    ip_address : str, optional
        IP address of the client.
    user_agent : str, optional
        User agent string.
    session_id : str, optional
        Session ID if applicable.
    details : dict, optional
        Additional event details.
    severity : str, default "INFO"
        Event severity (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    
    """
    
    def __init__(
        self,
        event_type: SecurityEventType,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "INFO",
    ) -> None:
        self.event_id = UUID4().value
        self.event_type = event_type
        self.timestamp = datetime.now(timezone.utc)
        self.user_id = user_id
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.session_id = session_id
        self.details = details or {}
        self.severity = severity
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "session_id": self.session_id,
            "details": self.details,
            "severity": self.severity,
        }
        
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())


class SecurityAuditLogger:
    """
    Logs security events for audit and compliance.
    
    This logger provides:
    - Structured security event logging
    - File-based audit trail
    - Real-time alerts for critical events
    - Compliance reporting support
    
    Parameters
    ----------
    log_dir : str
        Directory for audit log files.
    max_file_size : int, default 100MB
        Maximum size of log file before rotation.
    retention_days : int, default 90
        Number of days to retain logs.
    enable_alerts : bool, default True
        Whether to enable real-time alerts.
    
    """
    
    def __init__(
        self,
        log_dir: str,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        retention_days: int = 90,
        enable_alerts: bool = True,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        self._max_file_size = max_file_size
        self._retention_days = retention_days
        self._enable_alerts = enable_alerts
        
        self._log = Logger(self.__class__.__name__)
        self._current_file = None
        self._file_handle = None
        
        # Alert thresholds
        self._alert_thresholds = {
            SecurityEventType.LOGIN_FAILED: 5,  # Alert after 5 failed logins
            SecurityEventType.MFA_FAILED: 3,
            SecurityEventType.RATE_LIMIT_EXCEEDED: 10,
            SecurityEventType.SUSPICIOUS_ACTIVITY: 1,
        }
        
        # Event counters for alerting
        self._event_counters: Dict[str, Dict[SecurityEventType, int]] = {}
        
        self._open_log_file()
        
    def _open_log_file(self) -> None:
        """Open or create audit log file."""
        date_str = datetime.now().strftime("%Y%m%d")
        self._current_file = self._log_dir / f"security_audit_{date_str}.jsonl"
        
        # Close existing file handle
        if self._file_handle:
            self._file_handle.close()
            
        # Open new file in append mode
        self._file_handle = open(self._current_file, 'a', encoding='utf-8')
        
    def log_event(self, event: SecurityEvent) -> None:
        """
        Log a security event.
        
        Parameters
        ----------
        event : SecurityEvent
            The event to log.
        
        """
        # Check if we need to rotate file
        if self._current_file.stat().st_size > self._max_file_size:
            self._rotate_log_file()
            
        # Write event to file
        self._file_handle.write(event.to_json() + '\n')
        self._file_handle.flush()
        
        # Log to system logger based on severity
        if event.severity == "CRITICAL":
            self._log.error(f"CRITICAL SECURITY EVENT: {event.event_type.value}", color=3)
        elif event.severity == "ERROR":
            self._log.error(f"Security event: {event.event_type.value}")
        elif event.severity == "WARNING":
            self._log.warning(f"Security event: {event.event_type.value}")
        else:
            self._log.info(f"Security event: {event.event_type.value}")
            
        # Check for alerts
        if self._enable_alerts:
            self._check_alerts(event)
            
    def _rotate_log_file(self) -> None:
        """Rotate log file when size limit reached."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = self._current_file.with_suffix(f'.{timestamp}.jsonl')
        
        self._file_handle.close()
        self._current_file.rename(new_name)
        
        self._open_log_file()
        self._log.info(f"Rotated audit log to {new_name}")
        
    def _check_alerts(self, event: SecurityEvent) -> None:
        """Check if event should trigger an alert."""
        if event.event_type not in self._alert_thresholds:
            return
            
        # Track event counts per user
        user_key = event.user_id or event.ip_address or "unknown"
        
        if user_key not in self._event_counters:
            self._event_counters[user_key] = {}
            
        if event.event_type not in self._event_counters[user_key]:
            self._event_counters[user_key][event.event_type] = 0
            
        self._event_counters[user_key][event.event_type] += 1
        
        # Check threshold
        count = self._event_counters[user_key][event.event_type]
        threshold = self._alert_thresholds[event.event_type]
        
        if count >= threshold:
            self._trigger_alert(event, count)
            # Reset counter
            self._event_counters[user_key][event.event_type] = 0
            
    def _trigger_alert(self, event: SecurityEvent, count: int) -> None:
        """Trigger security alert."""
        alert_msg = (
            f"SECURITY ALERT: {event.event_type.value} "
            f"threshold exceeded ({count} occurrences) "
            f"for user/IP: {event.user_id or event.ip_address}"
        )
        
        self._log.error(alert_msg, color=3)
        
        # In production, this would send alerts via:
        # - Email
        # - Slack/Discord
        # - PagerDuty
        # - SIEM integration
        
    def log_login_attempt(
        self,
        user_id: str,
        success: bool,
        ip_address: str,
        user_agent: str,
        reason: Optional[str] = None,
    ) -> None:
        """Log login attempt."""
        event = SecurityEvent(
            event_type=SecurityEventType.LOGIN_SUCCESS if success else SecurityEventType.LOGIN_FAILED,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details={"reason": reason} if reason else None,
            severity="INFO" if success else "WARNING",
        )
        self.log_event(event)
        
    def log_mfa_attempt(
        self,
        user_id: str,
        success: bool,
        ip_address: str,
        session_id: str,
    ) -> None:
        """Log MFA verification attempt."""
        event = SecurityEvent(
            event_type=SecurityEventType.MFA_SUCCESS if success else SecurityEventType.MFA_FAILED,
            user_id=user_id,
            ip_address=ip_address,
            session_id=session_id,
            severity="INFO" if success else "WARNING",
        )
        self.log_event(event)
        
    def log_config_change(
        self,
        user_id: str,
        config_type: str,
        old_value: Any,
        new_value: Any,
    ) -> None:
        """Log configuration change."""
        # Sanitize sensitive values
        if "password" in config_type.lower() or "secret" in config_type.lower():
            old_value = "***"
            new_value = "***"
            
        event = SecurityEvent(
            event_type=SecurityEventType.CONFIG_CHANGED,
            user_id=user_id,
            details={
                "config_type": config_type,
                "old_value": str(old_value),
                "new_value": str(new_value),
            },
            severity="WARNING",
        )
        self.log_event(event)
        
    def log_trading_event(
        self,
        event_type: SecurityEventType,
        user_id: str,
        instrument_id: str,
        details: Dict[str, Any],
    ) -> None:
        """Log trading-related security event."""
        event = SecurityEvent(
            event_type=event_type,
            user_id=user_id,
            details={
                "instrument_id": instrument_id,
                **details,
            },
            severity="INFO",
        )
        self.log_event(event)
        
    def get_user_events(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[Dict[str, Any]]:
        """
        Get security events for a user.
        
        Parameters
        ----------
        user_id : str
            User ID to query.
        start_date : datetime, optional
            Start date for query.
        end_date : datetime, optional
            End date for query.
            
        Returns
        -------
        list[dict]
            List of events.
        
        """
        events = []
        
        # In production, this would query a database
        # For now, read from recent log files
        for log_file in sorted(self._log_dir.glob("security_audit_*.jsonl")):
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get("user_id") == user_id:
                            event_time = datetime.fromisoformat(event["timestamp"])
                            
                            if start_date and event_time < start_date:
                                continue
                            if end_date and event_time > end_date:
                                continue
                                
                            events.append(event)
                    except json.JSONDecodeError:
                        continue
                        
        return events
        
    def cleanup_old_logs(self) -> int:
        """
        Remove old audit logs.
        
        Returns
        -------
        int
            Number of files removed.
        
        """
        cutoff_time = time.time() - (self._retention_days * 24 * 60 * 60)
        removed = 0
        
        for log_file in self._log_dir.glob("security_audit_*.jsonl"):
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                removed += 1
                
        if removed > 0:
            self._log.info(f"Removed {removed} old audit log files")
            
        return removed
        
    def close(self) -> None:
        """Close the audit logger."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None