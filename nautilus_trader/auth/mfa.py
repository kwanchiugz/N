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
Multi-factor authentication (MFA) implementation.

This module provides TOTP-based two-factor authentication for enhanced security.
"""

import base64
import io
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import pyotp
import qrcode
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from nautilus_trader.common.component import Logger
from nautilus_trader.cryptography.encryption import DataEncryptor
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.model.identifiers import TraderId


class MFAProvider(ABC):
    """
    Abstract base class for MFA providers.
    """
    
    @abstractmethod
    def generate_secret(self, user_id: str) -> str:
        """Generate a new MFA secret for a user."""
        ...
        
    @abstractmethod
    def verify_token(self, user_id: str, token: str, secret: str) -> bool:
        """Verify an MFA token."""
        ...
        
    @abstractmethod
    def get_provisioning_uri(self, user_id: str, secret: str, issuer: str) -> str:
        """Get provisioning URI for QR code generation."""
        ...


class TOTPProvider(MFAProvider):
    """
    Time-based One-Time Password (TOTP) provider for 2FA.
    
    This provider implements RFC 6238 TOTP algorithm for secure
    two-factor authentication.
    
    Parameters
    ----------
    encryptor : DataEncryptor
        Encryptor for securing stored secrets.
    window : int, default 1
        Number of time windows to check (for clock drift tolerance).
    interval : int, default 30
        Time interval in seconds for TOTP generation.
    digits : int, default 6
        Number of digits in the generated token.
    
    """
    
    def __init__(
        self,
        encryptor: DataEncryptor,
        window: int = 1,
        interval: int = 30,
        digits: int = 6,
    ) -> None:
        self._encryptor = encryptor
        self._window = window
        self._interval = interval
        self._digits = digits
        self._log = Logger(self.__class__.__name__)
        
        # Store encrypted secrets (in production, use secure database)
        self._user_secrets: dict[str, str] = {}
        
    def generate_secret(self, user_id: str) -> str:
        """
        Generate a new TOTP secret for a user.
        
        Parameters
        ----------
        user_id : str
            The user identifier.
            
        Returns
        -------
        str
            Base32-encoded secret key.
        
        """
        PyCondition.not_empty(user_id, "user_id")
        
        # Generate random secret
        secret = pyotp.random_base32()
        
        # Encrypt and store
        encrypted_secret = self._encryptor.encrypt_string(secret)
        self._user_secrets[user_id] = encrypted_secret
        
        self._log.info(f"Generated new TOTP secret for user: {user_id}")
        return secret
        
    def verify_token(self, user_id: str, token: str, secret: Optional[str] = None) -> bool:
        """
        Verify a TOTP token.
        
        Parameters
        ----------
        user_id : str
            The user identifier.
        token : str
            The token to verify.
        secret : str, optional
            The secret to use. If not provided, uses stored secret.
            
        Returns
        -------
        bool
            True if token is valid, False otherwise.
        
        """
        PyCondition.not_empty(user_id, "user_id")
        PyCondition.not_empty(token, "token")
        
        try:
            # Get secret
            if secret is None:
                if user_id not in self._user_secrets:
                    self._log.warning(f"No secret found for user: {user_id}")
                    return False
                    
                encrypted_secret = self._user_secrets[user_id]
                secret = self._encryptor.decrypt_string(encrypted_secret)
                
            # Create TOTP instance
            totp = pyotp.TOTP(
                secret,
                interval=self._interval,
                digits=self._digits,
            )
            
            # Verify with time window for clock drift
            valid = totp.verify(token, valid_window=self._window)
            
            if valid:
                self._log.info(f"Valid TOTP token for user: {user_id}")
            else:
                self._log.warning(f"Invalid TOTP token for user: {user_id}")
                
            return valid
            
        except Exception as e:
            self._log.error(f"Error verifying TOTP token: {e}")
            return False
            
    def get_provisioning_uri(self, user_id: str, secret: str, issuer: str = "NautilusTrader") -> str:
        """
        Get provisioning URI for QR code generation.
        
        Parameters
        ----------
        user_id : str
            The user identifier.
        secret : str
            The TOTP secret.
        issuer : str, default "NautilusTrader"
            The issuer name for the authenticator app.
            
        Returns
        -------
        str
            The provisioning URI.
        
        """
        totp = pyotp.TOTP(secret, interval=self._interval, digits=self._digits)
        return totp.provisioning_uri(name=user_id, issuer_name=issuer)
        
    def generate_qr_code(self, user_id: str, secret: str, issuer: str = "NautilusTrader") -> bytes:
        """
        Generate QR code image for authenticator app setup.
        
        Parameters
        ----------
        user_id : str
            The user identifier.
        secret : str
            The TOTP secret.
        issuer : str, default "NautilusTrader"
            The issuer name for the authenticator app.
            
        Returns
        -------
        bytes
            PNG image data of the QR code.
        
        """
        uri = self.get_provisioning_uri(user_id, secret, issuer)
        
        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to bytes
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
        
    def generate_backup_codes(self, user_id: str, count: int = 10) -> list[str]:
        """
        Generate backup codes for account recovery.
        
        Parameters
        ----------
        user_id : str
            The user identifier.
        count : int, default 10
            Number of backup codes to generate.
            
        Returns
        -------
        list[str]
            List of backup codes.
        
        """
        PyCondition.positive_int(count, "count")
        
        codes = []
        for i in range(count):
            # Generate unique code based on user_id and index
            code_input = f"{user_id}:{i}:{time.time()}".encode()
            
            # Use PBKDF2 to generate code
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=4,  # 4 bytes = 8 hex chars
                salt=user_id.encode(),
                iterations=100000,
            )
            key = kdf.derive(code_input)
            code = key.hex().upper()
            codes.append(f"{code[:4]}-{code[4:]}")
            
        self._log.info(f"Generated {count} backup codes for user: {user_id}")
        return codes
        
    def enable_mfa(self, user_id: str) -> Tuple[str, bytes]:
        """
        Enable MFA for a user.
        
        Parameters
        ----------
        user_id : str
            The user identifier.
            
        Returns
        -------
        tuple[str, bytes]
            The secret and QR code image data.
        
        """
        # Generate new secret
        secret = self.generate_secret(user_id)
        
        # Generate QR code
        qr_code = self.generate_qr_code(user_id, secret)
        
        self._log.info(f"MFA enabled for user: {user_id}")
        return secret, qr_code
        
    def disable_mfa(self, user_id: str) -> bool:
        """
        Disable MFA for a user.
        
        Parameters
        ----------
        user_id : str
            The user identifier.
            
        Returns
        -------
        bool
            True if MFA was disabled, False if user not found.
        
        """
        if user_id in self._user_secrets:
            del self._user_secrets[user_id]
            self._log.info(f"MFA disabled for user: {user_id}")
            return True
        return False
        
    def is_mfa_enabled(self, user_id: str) -> bool:
        """
        Check if MFA is enabled for a user.
        
        Parameters
        ----------
        user_id : str
            The user identifier.
            
        Returns
        -------
        bool
            True if MFA is enabled.
        
        """
        return user_id in self._user_secrets