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
Alert channel implementations.
"""

import json
import asyncio
from typing import Any, Dict, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiosmtplib
import aiohttp

from nautilus_trader.alerts.base import Alert, AlertChannel, AlertLevel


class ConsoleAlertChannel(AlertChannel):
    """
    Console alert channel for development and debugging.
    
    Prints alerts to stdout with color coding.
    """
    
    # ANSI color codes
    COLORS = {
        AlertLevel.DEBUG: "\033[36m",    # Cyan
        AlertLevel.INFO: "\033[32m",     # Green
        AlertLevel.WARNING: "\033[33m",  # Yellow
        AlertLevel.ERROR: "\033[31m",    # Red
        AlertLevel.CRITICAL: "\033[35m", # Magenta
    }
    RESET = "\033[0m"
    
    def __init__(self, name: str = "console", use_colors: bool = True) -> None:
        """
        Initialize console channel.
        
        Parameters
        ----------
        name : str
            Channel name
        use_colors : bool
            Whether to use ANSI colors
        """
        super().__init__(name)
        self.use_colors = use_colors
        
    async def send_alert(self, alert: Alert) -> bool:
        """Print alert to console."""
        try:
            if self.use_colors:
                color = self.COLORS.get(alert.level, "")
                print(f"{color}[{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"{alert}{self.RESET}")
            else:
                print(f"[{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {alert}")
                
            # Print metadata if present
            if alert.metadata:
                print(f"  Metadata: {json.dumps(alert.metadata, indent=2)}")
                
            return True
            
        except Exception as e:
            self._logger.error(f"Console alert failed: {e}")
            return False


class EmailAlertChannel(AlertChannel):
    """
    Email alert channel using SMTP.
    
    Supports both plain SMTP and SMTP with TLS/SSL.
    """
    
    def __init__(
        self,
        name: str,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: list[str],
        use_tls: bool = True,
        use_ssl: bool = False,
    ) -> None:
        """
        Initialize email channel.
        
        Parameters
        ----------
        name : str
            Channel name
        smtp_host : str
            SMTP server host
        smtp_port : int
            SMTP server port
        username : str
            SMTP username
        password : str
            SMTP password
        from_email : str
            Sender email address
        to_emails : list[str]
            Recipient email addresses
        use_tls : bool
            Use TLS encryption
        use_ssl : bool
            Use SSL encryption
        """
        config = {
            "smtp_host": smtp_host,
            "smtp_port": smtp_port,
            "username": username,
            "password": password,
            "from_email": from_email,
            "to_emails": to_emails,
            "use_tls": use_tls,
            "use_ssl": use_ssl,
        }
        super().__init__(name, config)
        
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.config["from_email"]
            msg["To"] = ", ".join(self.config["to_emails"])
            msg["Subject"] = f"[{alert.level.value.upper()}] {alert.title}"
            
            # Create body
            body = self._format_email_body(alert)
            msg.attach(MIMEText(body, "html"))
            
            # Send email
            async with aiosmtplib.SMTP(
                hostname=self.config["smtp_host"],
                port=self.config["smtp_port"],
                use_tls=self.config["use_ssl"],
            ) as server:
                if self.config["use_tls"] and not self.config["use_ssl"]:
                    await server.starttls()
                    
                await server.login(
                    self.config["username"],
                    self.config["password"],
                )
                
                await server.send_message(msg)
                
            self._logger.info(f"Email alert sent: {alert.title}")
            return True
            
        except Exception as e:
            self._logger.error(f"Email alert failed: {e}")
            return False
            
    def _format_email_body(self, alert: Alert) -> str:
        """Format alert as HTML email."""
        level_colors = {
            AlertLevel.DEBUG: "#17a2b8",
            AlertLevel.INFO: "#28a745",
            AlertLevel.WARNING: "#ffc107",
            AlertLevel.ERROR: "#dc3545",
            AlertLevel.CRITICAL: "#e83e8c",
        }
        
        color = level_colors.get(alert.level, "#000000")
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="background-color: {color}; color: white; padding: 10px; border-radius: 5px;">
                <h2>{alert.title}</h2>
            </div>
            <div style="padding: 20px;">
                <p><strong>Level:</strong> {alert.level.value.upper()}</p>
                <p><strong>Source:</strong> {alert.source}</p>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                <hr>
                <p>{alert.message}</p>
        """
        
        if alert.metadata:
            html += """
                <hr>
                <h3>Additional Information:</h3>
                <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 3px;">
            """
            html += json.dumps(alert.metadata, indent=2)
            html += """
                </pre>
            """
            
        html += """
            </div>
        </body>
        </html>
        """
        
        return html


class WebhookAlertChannel(AlertChannel):
    """
    Webhook alert channel for HTTP/HTTPS endpoints.
    
    Sends alerts as JSON payloads to configured URLs.
    """
    
    def __init__(
        self,
        name: str,
        url: str,
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        auth_token: Optional[str] = None,
    ) -> None:
        """
        Initialize webhook channel.
        
        Parameters
        ----------
        name : str
            Channel name
        url : str
            Webhook URL
        method : str
            HTTP method (GET, POST, PUT)
        headers : dict, optional
            Additional headers
        timeout : float
            Request timeout in seconds
        auth_token : str, optional
            Bearer token for authentication
        """
        config = {
            "url": url,
            "method": method.upper(),
            "headers": headers or {},
            "timeout": timeout,
        }
        
        if auth_token:
            config["headers"]["Authorization"] = f"Bearer {auth_token}"
            
        super().__init__(name, config)
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self) -> None:
        """Initialize HTTP session."""
        self._session = aiohttp.ClientSession(
            headers=self.config["headers"],
            timeout=aiohttp.ClientTimeout(total=self.config["timeout"]),
        )
        
    async def shutdown(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        if not self._session:
            await self.initialize()
            
        try:
            # Prepare payload
            payload = {
                "alert": alert.to_dict(),
                "channel": self.name,
            }
            
            # Send request
            async with self._session.request(
                method=self.config["method"],
                url=self.config["url"],
                json=payload,
            ) as response:
                response.raise_for_status()
                
            self._logger.info(f"Webhook alert sent: {alert.title}")
            return True
            
        except Exception as e:
            self._logger.error(f"Webhook alert failed: {e}")
            return False


class SlackAlertChannel(AlertChannel):
    """
    Slack alert channel using webhooks.
    
    Sends formatted messages to Slack channels.
    """
    
    def __init__(
        self,
        name: str,
        webhook_url: str,
        channel: Optional[str] = None,
        username: str = "Nautilus Alerts",
        icon_emoji: str = ":warning:",
    ) -> None:
        """
        Initialize Slack channel.
        
        Parameters
        ----------
        name : str
            Channel name
        webhook_url : str
            Slack webhook URL
        channel : str, optional
            Override default channel
        username : str
            Bot username
        icon_emoji : str
            Bot icon emoji
        """
        config = {
            "webhook_url": webhook_url,
            "channel": channel,
            "username": username,
            "icon_emoji": icon_emoji,
        }
        super().__init__(name, config)
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self) -> None:
        """Initialize HTTP session."""
        self._session = aiohttp.ClientSession()
        
    async def shutdown(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        if not self._session:
            await self.initialize()
            
        try:
            # Map alert levels to colors
            colors = {
                AlertLevel.DEBUG: "#36a64f",
                AlertLevel.INFO: "#2eb886",
                AlertLevel.WARNING: "#ffa500",
                AlertLevel.ERROR: "#ff0000",
                AlertLevel.CRITICAL: "#8b0000",
            }
            
            # Create Slack message
            payload = {
                "username": self.config["username"],
                "icon_emoji": self.config["icon_emoji"],
                "attachments": [
                    {
                        "color": colors.get(alert.level, "#000000"),
                        "title": alert.title,
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Level",
                                "value": alert.level.value.upper(),
                                "short": True,
                            },
                            {
                                "title": "Source",
                                "value": alert.source,
                                "short": True,
                            },
                        ],
                        "footer": "Nautilus Trader",
                        "ts": int(alert.timestamp.timestamp()),
                    }
                ],
            }
            
            if self.config["channel"]:
                payload["channel"] = self.config["channel"]
                
            # Add metadata fields
            if alert.metadata:
                for key, value in list(alert.metadata.items())[:5]:  # Limit to 5
                    payload["attachments"][0]["fields"].append({
                        "title": key,
                        "value": str(value),
                        "short": True,
                    })
                    
            # Send to Slack
            async with self._session.post(
                self.config["webhook_url"],
                json=payload,
            ) as response:
                response.raise_for_status()
                
            self._logger.info(f"Slack alert sent: {alert.title}")
            return True
            
        except Exception as e:
            self._logger.error(f"Slack alert failed: {e}")
            return False


class TelegramAlertChannel(AlertChannel):
    """
    Telegram alert channel using Bot API.
    
    Sends messages to Telegram chats or channels.
    """
    
    def __init__(
        self,
        name: str,
        bot_token: str,
        chat_id: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False,
    ) -> None:
        """
        Initialize Telegram channel.
        
        Parameters
        ----------
        name : str
            Channel name
        bot_token : str
            Telegram bot token
        chat_id : str
            Chat ID to send messages to
        parse_mode : str
            Message parse mode (HTML, Markdown)
        disable_notification : bool
            Disable sound notifications
        """
        config = {
            "bot_token": bot_token,
            "chat_id": chat_id,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification,
        }
        super().__init__(name, config)
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self) -> None:
        """Initialize HTTP session."""
        self._session = aiohttp.ClientSession()
        
    async def shutdown(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to Telegram."""
        if not self._session:
            await self.initialize()
            
        try:
            # Format message
            if self.config["parse_mode"] == "HTML":
                message = self._format_html_message(alert)
            else:
                message = self._format_markdown_message(alert)
                
            # Telegram API URL
            url = f"https://api.telegram.org/bot{self.config['bot_token']}/sendMessage"
            
            # Send message
            payload = {
                "chat_id": self.config["chat_id"],
                "text": message,
                "parse_mode": self.config["parse_mode"],
                "disable_notification": self.config["disable_notification"],
            }
            
            async with self._session.post(url, json=payload) as response:
                response.raise_for_status()
                
            self._logger.info(f"Telegram alert sent: {alert.title}")
            return True
            
        except Exception as e:
            self._logger.error(f"Telegram alert failed: {e}")
            return False
            
    def _format_html_message(self, alert: Alert) -> str:
        """Format alert as HTML for Telegram."""
        # Map levels to emojis
        emojis = {
            AlertLevel.DEBUG: "🔍",
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }
        
        emoji = emojis.get(alert.level, "📢")
        
        message = f"""
{emoji} <b>{alert.title}</b>

<b>Level:</b> {alert.level.value.upper()}
<b>Source:</b> {alert.source}
<b>Time:</b> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

{alert.message}
"""
        
        if alert.metadata:
            message += "\n<b>Details:</b>\n"
            for key, value in list(alert.metadata.items())[:5]:
                message += f"• {key}: {value}\n"
                
        return message.strip()
        
    def _format_markdown_message(self, alert: Alert) -> str:
        """Format alert as Markdown for Telegram."""
        emojis = {
            AlertLevel.DEBUG: "🔍",
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }
        
        emoji = emojis.get(alert.level, "📢")
        
        message = f"""
{emoji} *{alert.title}*

*Level:* {alert.level.value.upper()}
*Source:* {alert.source}
*Time:* {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

{alert.message}
"""
        
        if alert.metadata:
            message += "\n*Details:*\n"
            for key, value in list(alert.metadata.items())[:5]:
                message += f"• {key}: {value}\n"
                
        return message.strip()