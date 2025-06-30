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
Enhanced session management with storage abstraction.

This module provides session management using the new storage abstraction layer,
enabling flexible storage backends (Redis, Database, File, S3).
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple, List
import uuid

import jwt

from nautilus_trader.common.component import Logger
from nautilus_trader.storage import StorageProvider, StorageFactory, StorageType
from nautilus_trader.core.uuid import UUID4


class Session:
    """
    Represents an authenticated session with storage support.
    """
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        created_at: datetime,
        last_activity: datetime,
        device_info: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        is_active: bool = True,
        is_mfa_verified: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = created_at
        self.last_activity = last_activity
        self.device_info = device_info or {}
        self.ip_address = ip_address
        self.is_active = is_active
        self.is_mfa_verified = is_mfa_verified
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for storage."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "device_info": self.device_info,
            "ip_address": self.ip_address,
            "is_active": self.is_active,
            "is_mfa_verified": self.is_mfa_verified,
            "metadata": self.metadata,
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create session from dictionary."""
        return cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            device_info=data.get("device_info", {}),
            ip_address=data.get("ip_address"),
            is_active=data.get("is_active", True),
            is_mfa_verified=data.get("is_mfa_verified", False),
            metadata=data.get("metadata", {}),
        )


class SessionManagerV2:
    """
    Enhanced session manager with flexible storage backends.
    
    This manager supports:
    - Multiple storage backends (Redis, Database, File, S3)
    - JWT token management
    - Session persistence and recovery
    - Distributed session management
    """
    
    def __init__(
        self,
        secret_key: str,
        storage_provider: StorageProvider,
        access_token_expire: int = 900,  # 15 minutes
        refresh_token_expire: int = 86400,  # 24 hours
        max_sessions_per_user: int = 5,
        session_timeout: int = 3600,  # 1 hour idle timeout
    ) -> None:
        """
        Initialize enhanced session manager.
        
        Parameters
        ----------
        secret_key : str
            Secret key for JWT signing
        storage_provider : StorageProvider
            Storage backend for sessions
        access_token_expire : int
            Access token expiration in seconds
        refresh_token_expire : int
            Refresh token expiration in seconds
        max_sessions_per_user : int
            Maximum concurrent sessions per user
        session_timeout : int
            Session idle timeout in seconds
        """
        self._secret_key = secret_key
        self._storage = storage_provider
        self._access_expire = access_token_expire
        self._refresh_expire = refresh_token_expire
        self._max_sessions = max_sessions_per_user
        self._session_timeout = session_timeout
        self._logger = Logger(self.__class__.__name__)
        
        # Storage namespaces
        self._session_namespace = "sessions"
        self._user_namespace = "user_sessions"
        self._token_namespace = "refresh_tokens"
        
    async def initialize(self) -> None:
        """Initialize storage connection."""
        await self._storage.connect()
        self._logger.info("Session manager initialized with storage backend")
        
    async def shutdown(self) -> None:
        """Shutdown storage connection."""
        await self._storage.disconnect()
        self._logger.info("Session manager shutdown")
        
    async def create_session(
        self,
        user_id: str,
        device_info: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        require_mfa: bool = True,
    ) -> Tuple[str, str, str]:
        """
        Create a new session with tokens.
        
        Parameters
        ----------
        user_id : str
            User identifier
        device_info : dict, optional
            Device information
        ip_address : str, optional
            Client IP address
        require_mfa : bool
            Whether MFA is required
            
        Returns
        -------
        tuple[str, str, str]
            Session ID, access token, and refresh token
        """
        # Check and enforce session limit
        await self._enforce_session_limit(user_id)
        
        # Create new session
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            device_info=device_info,
            ip_address=ip_address,
            is_active=True,
            is_mfa_verified=not require_mfa,  # If MFA not required, mark as verified
        )
        
        # Store session
        await self._storage.save_json(
            key=session_id,
            data=session.to_dict(),
            namespace=self._session_namespace,
        )
        
        # Update user sessions index
        await self._add_user_session(user_id, session_id)
        
        # Generate tokens
        access_token = self._generate_access_token(session)
        refresh_token = self._generate_refresh_token(session)
        
        # Store refresh token mapping
        await self._storage.save(
            key=refresh_token,
            data=session_id,
            namespace=self._token_namespace,
        )
        
        self._logger.info(
            f"Created session {session_id} for user {user_id} "
            f"from {ip_address or 'unknown'}"
        )
        
        return session_id, access_token, refresh_token
        
    async def validate_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate access token and return payload.
        
        Parameters
        ----------
        token : str
            JWT access token
            
        Returns
        -------
        dict or None
            Token payload if valid
        """
        try:
            payload = jwt.decode(
                token,
                self._secret_key,
                algorithms=["HS256"],
            )
            
            # Check if session exists and is active
            session_id = payload.get("session_id")
            session = await self._get_session(session_id)
            
            if not session or not session.is_active:
                return None
                
            # Check idle timeout
            idle_time = (datetime.now(timezone.utc) - session.last_activity).total_seconds()
            if idle_time > self._session_timeout:
                await self.invalidate_session(session_id)
                return None
                
            # Update last activity
            session.last_activity = datetime.now(timezone.utc)
            await self._update_session(session)
            
            return payload
            
        except jwt.ExpiredSignatureError:
            self._logger.debug("Access token expired")
            return None
        except jwt.InvalidTokenError as e:
            self._logger.warning(f"Invalid access token: {e}")
            return None
            
    async def refresh_tokens(self, refresh_token: str) -> Optional[Tuple[str, str]]:
        """
        Refresh access and refresh tokens.
        
        Parameters
        ----------
        refresh_token : str
            JWT refresh token
            
        Returns
        -------
        tuple[str, str] or None
            New access and refresh tokens if valid
        """
        try:
            # Verify refresh token
            payload = jwt.decode(
                refresh_token,
                self._secret_key,
                algorithms=["HS256"],
            )
            
            # Get session ID from refresh token mapping
            session_id = await self._storage.load(
                key=refresh_token,
                namespace=self._token_namespace,
            )
            
            if not session_id:
                return None
                
            # Get session
            session = await self._get_session(session_id)
            
            if not session or not session.is_active:
                return None
                
            # Delete old refresh token
            await self._storage.delete(
                key=refresh_token,
                namespace=self._token_namespace,
            )
            
            # Generate new tokens
            new_access_token = self._generate_access_token(session)
            new_refresh_token = self._generate_refresh_token(session)
            
            # Store new refresh token mapping
            await self._storage.save(
                key=new_refresh_token,
                data=session_id,
                namespace=self._token_namespace,
            )
            
            # Update session activity
            session.last_activity = datetime.now(timezone.utc)
            await self._update_session(session)
            
            self._logger.info(f"Refreshed tokens for session {session_id}")
            
            return new_access_token, new_refresh_token
            
        except jwt.ExpiredSignatureError:
            self._logger.debug("Refresh token expired")
            # Clean up expired token mapping
            await self._storage.delete(
                key=refresh_token,
                namespace=self._token_namespace,
            )
            return None
        except Exception as e:
            self._logger.error(f"Token refresh failed: {e}")
            return None
            
    async def invalidate_session(self, session_id: str) -> bool:
        """
        Invalidate a session.
        
        Parameters
        ----------
        session_id : str
            Session identifier
            
        Returns
        -------
        bool
            True if session was invalidated
        """
        session = await self._get_session(session_id)
        if not session:
            return False
            
        # Mark as inactive
        session.is_active = False
        await self._update_session(session)
        
        # Remove from user sessions
        await self._remove_user_session(session.user_id, session_id)
        
        # Clean up refresh tokens
        refresh_tokens = await self._storage.list_keys(
            namespace=self._token_namespace,
            pattern="*",
        )
        
        for token in refresh_tokens:
            stored_session_id = await self._storage.load(
                key=token,
                namespace=self._token_namespace,
            )
            if stored_session_id == session_id:
                await self._storage.delete(
                    key=token,
                    namespace=self._token_namespace,
                )
                
        self._logger.info(f"Invalidated session {session_id}")
        return True
        
    async def invalidate_user_sessions(self, user_id: str) -> int:
        """
        Invalidate all sessions for a user.
        
        Parameters
        ----------
        user_id : str
            User identifier
            
        Returns
        -------
        int
            Number of sessions invalidated
        """
        session_ids = await self._get_user_sessions(user_id)
        count = 0
        
        for session_id in session_ids:
            if await self.invalidate_session(session_id):
                count += 1
                
        if count > 0:
            self._logger.info(f"Invalidated {count} sessions for user {user_id}")
            
        return count
        
    async def get_user_sessions(self, user_id: str) -> List[Session]:
        """
        Get all active sessions for a user.
        
        Parameters
        ----------
        user_id : str
            User identifier
            
        Returns
        -------
        list[Session]
            List of active sessions
        """
        session_ids = await self._get_user_sessions(user_id)
        sessions = []
        
        for session_id in session_ids:
            session = await self._get_session(session_id)
            if session and session.is_active:
                sessions.append(session)
                
        return sessions
        
    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired and inactive sessions.
        
        Returns
        -------
        int
            Number of sessions cleaned up
        """
        all_sessions = await self._storage.list_keys(namespace=self._session_namespace)
        count = 0
        now = datetime.now(timezone.utc)
        
        for session_id in all_sessions:
            session = await self._get_session(session_id)
            if not session:
                continue
                
            # Check if inactive or timed out
            idle_time = (now - session.last_activity).total_seconds()
            if not session.is_active or idle_time > self._session_timeout:
                await self.invalidate_session(session_id)
                count += 1
                
        if count > 0:
            self._logger.info(f"Cleaned up {count} expired sessions")
            
        return count
        
    async def update_session_mfa(self, session_id: str, mfa_verified: bool) -> bool:
        """
        Update MFA verification status for a session.
        
        Parameters
        ----------
        session_id : str
            Session identifier
        mfa_verified : bool
            MFA verification status
            
        Returns
        -------
        bool
            True if updated successfully
        """
        session = await self._get_session(session_id)
        if not session:
            return False
            
        session.is_mfa_verified = mfa_verified
        await self._update_session(session)
        
        self._logger.info(
            f"Updated MFA status for session {session_id}: {mfa_verified}"
        )
        return True
        
    # Helper methods
    def _generate_access_token(self, session: Session) -> str:
        """Generate JWT access token."""
        now = datetime.now(timezone.utc)
        exp = now + timedelta(seconds=self._access_expire)
        
        payload = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "exp": exp,
            "iat": now,
            "mfa": session.is_mfa_verified,
        }
        
        return jwt.encode(payload, self._secret_key, algorithm="HS256")
        
    def _generate_refresh_token(self, session: Session) -> str:
        """Generate JWT refresh token."""
        now = datetime.now(timezone.utc)
        exp = now + timedelta(seconds=self._refresh_expire)
        
        payload = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "exp": exp,
            "iat": now,
            "type": "refresh",
        }
        
        return jwt.encode(payload, self._secret_key, algorithm="HS256")
        
    async def _get_session(self, session_id: str) -> Optional[Session]:
        """Get session from storage."""
        try:
            data = await self._storage.load_json(
                key=session_id,
                namespace=self._session_namespace,
            )
            return Session.from_dict(data)
        except:
            return None
            
    async def _update_session(self, session: Session) -> None:
        """Update session in storage."""
        await self._storage.save_json(
            key=session.session_id,
            data=session.to_dict(),
            namespace=self._session_namespace,
        )
        
    async def _get_user_sessions(self, user_id: str) -> List[str]:
        """Get user session IDs from storage."""
        try:
            return await self._storage.load_json(
                key=f"user:{user_id}",
                namespace=self._user_namespace,
            )
        except:
            return []
            
    async def _add_user_session(self, user_id: str, session_id: str) -> None:
        """Add session to user's session list."""
        sessions = await self._get_user_sessions(user_id)
        if session_id not in sessions:
            sessions.append(session_id)
            await self._storage.save_json(
                key=f"user:{user_id}",
                data=sessions,
                namespace=self._user_namespace,
            )
            
    async def _remove_user_session(self, user_id: str, session_id: str) -> None:
        """Remove session from user's session list."""
        sessions = await self._get_user_sessions(user_id)
        if session_id in sessions:
            sessions.remove(session_id)
            await self._storage.save_json(
                key=f"user:{user_id}",
                data=sessions,
                namespace=self._user_namespace,
            )
            
    async def _enforce_session_limit(self, user_id: str) -> None:
        """Enforce maximum sessions per user."""
        sessions = await self._get_user_sessions(user_id)
        
        if len(sessions) >= self._max_sessions:
            # Get session details to find oldest
            session_objects = []
            for sid in sessions:
                session = await self._get_session(sid)
                if session:
                    session_objects.append(session)
                    
            # Sort by creation time
            session_objects.sort(key=lambda s: s.created_at)
            
            # Invalidate oldest sessions
            sessions_to_remove = len(sessions) - self._max_sessions + 1
            for i in range(sessions_to_remove):
                await self.invalidate_session(session_objects[i].session_id)
                
            self._logger.info(
                f"Enforced session limit for user {user_id}: "
                f"removed {sessions_to_remove} oldest sessions"
            )