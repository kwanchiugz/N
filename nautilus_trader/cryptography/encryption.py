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
Data encryption module using AES-256-GCM for secure data handling.

This module provides encryption and decryption capabilities for sensitive data
such as API keys, trading data, and configuration values.
"""

import base64
import os
from typing import Union

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.exceptions import InvalidTag

from nautilus_trader.common.component import Logger
from nautilus_trader.core.correctness import PyCondition


class DataEncryptor:
    """
    Provides AES-256-GCM encryption for sensitive data.
    
    This class implements secure encryption using:
    - AES-256 in GCM mode for authenticated encryption
    - Random IV generation for each encryption
    - PBKDF2 for key derivation when using passwords
    
    Parameters
    ----------
    key : bytes, optional
        The 256-bit encryption key. If not provided, will be derived from password.
    password : str, optional
        Password for key derivation. Required if key is not provided.
    salt : bytes, optional
        Salt for key derivation. If not provided, a random salt will be generated.
    iterations : int, default 100_000
        Number of iterations for PBKDF2 key derivation.
    
    Raises
    ------
    ValueError
        If neither key nor password is provided.
    
    """
    
    def __init__(
        self,
        key: bytes | None = None,
        password: str | None = None,
        salt: bytes | None = None,
        iterations: int = 100_000,
    ) -> None:
        if key is None and password is None:
            raise ValueError("Either 'key' or 'password' must be provided")
        
        self._log = Logger(self.__class__.__name__)
        
        if key is not None:
            PyCondition.equal(len(key), 32, "key length", "32 bytes")
            self._key = key
        else:
            # Derive key from password
            if salt is None:
                salt = os.urandom(32)  # 256-bit salt
            self._salt = salt
            self._key = self._derive_key(password, salt, iterations)
            
        self._backend = default_backend()
        
    def _derive_key(self, password: str, salt: bytes, iterations: int) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256-bit key
            salt=salt,
            iterations=iterations,
            backend=self._backend,
        )
        return kdf.derive(password.encode('utf-8'))
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data using AES-256-GCM.
        
        Parameters
        ----------
        data : str or bytes
            The data to encrypt.
            
        Returns
        -------
        bytes
            The encrypted data with IV and authentication tag.
            Format: IV (12 bytes) || Ciphertext || Tag (16 bytes)
        
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        # Generate random IV for each encryption
        iv = os.urandom(12)  # 96-bit IV for GCM
        
        cipher = Cipher(
            algorithms.AES(self._key),
            modes.GCM(iv),
            backend=self._backend,
        )
        encryptor = cipher.encryptor()
        
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Return IV + ciphertext + tag
        return iv + ciphertext + encryptor.tag
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data encrypted with AES-256-GCM.
        
        Parameters
        ----------
        encrypted_data : bytes
            The encrypted data including IV and authentication tag.
            
        Returns
        -------
        bytes
            The decrypted data.
            
        Raises
        ------
        ValueError
            If the data is corrupted or authentication fails.
        
        """
        if len(encrypted_data) < 28:  # IV(12) + min_ciphertext(0) + tag(16)
            raise ValueError("Invalid encrypted data: too short")
            
        # Extract components
        iv = encrypted_data[:12]
        tag = encrypted_data[-16:]
        ciphertext = encrypted_data[12:-16]
        
        cipher = Cipher(
            algorithms.AES(self._key),
            modes.GCM(iv, tag),
            backend=self._backend,
        )
        decryptor = cipher.decryptor()
        
        try:
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            return plaintext
        except InvalidTag:
            raise ValueError("Decryption failed: invalid authentication tag")
    
    def encrypt_string(self, data: str) -> str:
        """
        Encrypt a string and return base64-encoded result.
        
        Parameters
        ----------
        data : str
            The string to encrypt.
            
        Returns
        -------
        str
            Base64-encoded encrypted data.
        
        """
        encrypted = self.encrypt(data)
        return base64.b64encode(encrypted).decode('ascii')
    
    def decrypt_string(self, encrypted_data: str) -> str:
        """
        Decrypt a base64-encoded encrypted string.
        
        Parameters
        ----------
        encrypted_data : str
            Base64-encoded encrypted data.
            
        Returns
        -------
        str
            The decrypted string.
        
        """
        encrypted_bytes = base64.b64decode(encrypted_data.encode('ascii'))
        decrypted_bytes = self.decrypt(encrypted_bytes)
        return decrypted_bytes.decode('utf-8')
    
    @classmethod
    def generate_key(cls) -> bytes:
        """
        Generate a new random 256-bit encryption key.
        
        Returns
        -------
        bytes
            A 32-byte random key suitable for AES-256.
        
        """
        return os.urandom(32)
    
    @classmethod
    def generate_salt(cls) -> bytes:
        """
        Generate a new random 256-bit salt for key derivation.
        
        Returns
        -------
        bytes
            A 32-byte random salt.
        
        """
        return os.urandom(32)