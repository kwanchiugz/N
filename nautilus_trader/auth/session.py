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
Session management with JWT tokens.

This module provides secure session management using JWT tokens with
refresh token support and automatic expiration.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import jwt

from nautilus_trader.common.component import Logger
from nautilus_trader.cryptography.encryption import DataEncryptor
from nautilus_trader.core.uuid import UUID4


class Session:
    """
    Represents an authenticated session.
    
    Parameters
    ----------
    session_id : str
        Unique session identifier.
    user_id : str
        User identifier.
    ip_address : str
        Client IP address.
    user_agent : str
        Client user agent.
    created_at : datetime
        Session creation time.
    expires_at : datetime
        Session expiration time.
    is_mfa_verified : bool, default False
        Whether MFA was verified.
    metadata : dict, optional
        Additional session metadata.
    
    """
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        ip_address: str,
        user_agent: str,
        created_at: datetime,
        expires_at: datetime,
        is_mfa_verified: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.session_id = session_id
        self.user_id = user_id
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.created_at = created_at
        self.expires_at = expires_at
        self.is_mfa_verified = is_mfa_verified
        self.metadata = metadata or {}
        self.last_activity = created_at
        
    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now(timezone.utc) > self.expires_at
        
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)


class SessionManager:
    """
    Manages user sessions with JWT tokens.
    
    This class provides:
    - JWT token generation and validation
    - Refresh token support
    - Session tracking and management
    - Automatic cleanup of expired sessions
    
    Parameters
    ----------
    secret_key : str
        Secret key for JWT signing.
    access_token_ttl : int, default 900
        Access token TTL in seconds (15 minutes).
    refresh_token_ttl : int, default 86400
        Refresh token TTL in seconds (24 hours).
    algorithm : str, default "HS256"
        JWT signing algorithm.
    
    """
    
    def __init__(
        self,
        secret_key: str,
        access_token_ttl: int = 900,  # 15 minutes
        refresh_token_ttl: int = 86400,  # 24 hours
        algorithm: str = "HS256",
    ) -> None:
        self._secret_key = secret_key
        self._access_token_ttl = access_token_ttl
        self._refresh_token_ttl = refresh_token_ttl
        self._algorithm = algorithm
        self._log = Logger(self.__class__.__name__)
        
        # Active sessions
        self._sessions: Dict[str, Session] = {}
        
        # Refresh token mapping
        self._refresh_tokens: Dict[str, str] = {}  # token -> session_id
        
    def create_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str,
        is_mfa_verified: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, str]:
        """
        Create a new session with tokens.
        
        Parameters
        ----------
        user_id : str
            User identifier.
        ip_address : str
            Client IP address.
        user_agent : str
            Client user agent.
        is_mfa_verified : bool, default False
            Whether MFA was verified.
        metadata : dict, optional
            Additional session metadata.
            
        Returns
        -------
        tuple[str, str, str]
            Session ID, access token, and refresh token.
        
        """
        # Generate session ID
        session_id = UUID4().value
        
        # Create session
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=self._refresh_token_ttl)
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            created_at=now,
            expires_at=expires_at,
            is_mfa_verified=is_mfa_verified,
            metadata=metadata,
        )
        
        # Store session
        self._sessions[session_id] = session
        
        # Generate tokens
        access_token = self._generate_access_token(session)
        refresh_token = self._generate_refresh_token(session)
        
        # Map refresh token
        self._refresh_tokens[refresh_token] = session_id
        
        self._log.info(
            f"Created session for user {user_id} from {ip_address} "
            f"(MFA: {is_mfa_verified})"
        )
        
        return session_id, access_token, refresh_token
        
    def _generate_access_token(self, session: Session) -> str:
        """Generate JWT access token."""
        now = datetime.now(timezone.utc)
        exp = now + timedelta(seconds=self._access_token_ttl)
        
        payload = {
            "sub": session.user_id,
            "sid": session.session_id,
            "iat": now,
            "exp": exp,
            "mfa": session.is_mfa_verified,
            "ip": session.ip_address,
        }
        
        return jwt.encode(payload, self._secret_key, algorithm=self._algorithm)
        
    def _generate_refresh_token(self, session: Session) -> str:
        """Generate JWT refresh token."""
        now = datetime.now(timezone.utc)
        exp = session.expires_at
        
        payload = {
            "sub": session.user_id,
            "sid": session.session_id,
            "iat": now,
            "exp": exp,
            "type": "refresh",
        }
        
        return jwt.encode(payload, self._secret_key, algorithm=self._algorithm)
        
    def validate_access_token(self, token: str) -> Optional[Session]:
        """
        Validate access token and return session.
        
        Parameters
        ----------
        token : str
            JWT access token.
            
        Returns
        -------
        Session or None
            The session if valid, None otherwise.
        
        """
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self._secret_key,
                algorithms=[self._algorithm],
            )
            
            # Get session
            session_id = payload.get("sid")
            if not session_id or session_id not in self._sessions:
                return None
                
            session = self._sessions[session_id]
            
            # Verify session not expired
            if session.is_expired:
                del self._sessions[session_id]
                return None
                
            # Verify IP address matches
            if payload.get("ip") != session.ip_address:
                self._log.warning(
                    f"IP address mismatch for session {session_id}: "
                    f"expected {session.ip_address}, got {payload.get('ip')}"
                )
                return None
                
            # Update activity
            session.update_activity()
            
            return session
            
        except jwt.ExpiredSignatureError:
            self._log.debug("Access token expired")
            return None
        except jwt.InvalidTokenError as e:
            self._log.warning(f"Invalid access token: {e}")
            return None
            
    def refresh_tokens(self, refresh_token: str) -> Optional[Tuple[str, str]]:
        """
        Refresh access token using refresh token.
        
        Parameters
        ----------
        refresh_token : str
            JWT refresh token.
            
        Returns
        -------
        tuple[str, str] or None
            New access token and refresh token if valid.
        
        """
        try:
            # Decode refresh token
            payload = jwt.decode(
                refresh_token,
                self._secret_key,
                algorithms=[self._algorithm],
            )
            
            # Verify it's a refresh token
            if payload.get("type") != "refresh":
                return None
                
            # Get session
            session_id = payload.get("sid")
            if not session_id or session_id not in self._sessions:
                return None
                
            # Remove old refresh token mapping
            if refresh_token in self._refresh_tokens:
                del self._refresh_tokens[refresh_token]
                
            session = self._sessions[session_id]
            
            # Verify session not expired
            if session.is_expired:
                del self._sessions[session_id]
                return None
                
            # Generate new tokens
            new_access_token = self._generate_access_token(session)
            new_refresh_token = self._generate_refresh_token(session)
            
            # Update refresh token mapping
            self._refresh_tokens[new_refresh_token] = session_id
            
            # Update activity
            session.update_activity()
            
            self._log.info(f"Refreshed tokens for session {session_id}")
            
            return new_access_token, new_refresh_token
            
        except jwt.ExpiredSignatureError:
            self._log.debug("Refresh token expired")
            return None
        except jwt.InvalidTokenError as e:
            self._log.warning(f"Invalid refresh token: {e}")
            return None
            
    def revoke_session(self, session_id: str) -> bool:
        """
        Revoke a session.
        
        Parameters
        ----------
        session_id : str
            Session ID to revoke.
            
        Returns
        -------
        bool
            True if session was revoked.
        
        """
        if session_id in self._sessions:
            # Remove session
            del self._sessions[session_id]
            
            # Remove associated refresh tokens
            tokens_to_remove = [
                token for token, sid in self._refresh_tokens.items()
                if sid == session_id
            ]
            for token in tokens_to_remove:
                del self._refresh_tokens[token]
                
            self._log.info(f"Revoked session {session_id}")
            return True
            
        return False
        
    def revoke_user_sessions(self, user_id: str) -> int:
        """
        Revoke all sessions for a user.
        
        Parameters
        ----------
        user_id : str
            User ID whose sessions to revoke.
            
        Returns
        -------
        int
            Number of sessions revoked.
        
        """
        sessions_to_revoke = [
            sid for sid, session in self._sessions.items()
            if session.user_id == user_id
        ]
        
        count = 0
        for session_id in sessions_to_revoke:
            if self.revoke_session(session_id):
                count += 1
                
        if count > 0:
            self._log.info(f"Revoked {count} sessions for user {user_id}")
            
        return count
        
    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions.
        
        Returns
        -------
        int
            Number of sessions cleaned up.
        
        """
        expired = [
            sid for sid, session in self._sessions.items()
            if session.is_expired
        ]
        
        count = 0
        for session_id in expired:
            if self.revoke_session(session_id):
                count += 1
                
        if count > 0:
            self._log.info(f"Cleaned up {count} expired sessions")
            
        return count
        
    def get_active_sessions(self, user_id: Optional[str] = None) -> list[Session]:
        """
        Get active sessions.
        
        Parameters
        ----------
        user_id : str, optional
            Filter by user ID.
            
        Returns
        -------
        list[Session]
            List of active sessions.
        
        """
        sessions = list(self._sessions.values())
        
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
            
        return sessions