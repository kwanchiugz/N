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
Secure key management for API credentials and sensitive data.

This module provides secure storage and retrieval of API keys, passwords,
and other sensitive configuration data using encryption and environment variables.
"""

import os
import json
from typing import Dict, Optional, Any
from pathlib import Path

from nautilus_trader.cryptography.encryption import DataEncryptor
from nautilus_trader.common.component import Logger
from nautilus_trader.core.correctness import PyCondition


class KeyManager:
    """
    Manages secure storage and retrieval of API keys and credentials.
    
    This class provides:
    - Encrypted storage of credentials
    - Environment variable fallback
    - In-memory caching with encryption
    - Key rotation support
    
    Parameters
    ----------
    key_file : str, optional
        Path to encrypted key storage file.
    master_key : bytes, optional
        Master encryption key. If not provided, derived from NAUTILUS_MASTER_KEY env var.
    use_env_fallback : bool, default True
        Whether to fall back to environment variables if key not found.
    
    """
    
    def __init__(
        self,
        key_file: str | None = None,
        master_key: bytes | None = None,
        use_env_fallback: bool = True,
    ) -> None:
        self._log = Logger(self.__class__.__name__)
        self._use_env_fallback = use_env_fallback
        
        # Initialize encryptor
        if master_key is None:
            # Try to get from environment
            master_key_env = os.environ.get("NAUTILUS_MASTER_KEY")
            if master_key_env:
                # Decode from hex
                master_key = bytes.fromhex(master_key_env)
            else:
                # Generate new key and log warning
                master_key = DataEncryptor.generate_key()
                self._log.warning(
                    "No master key provided. Generated new key. "
                    "Set NAUTILUS_MASTER_KEY environment variable to: "
                    f"{master_key.hex()}"
                )
        
        self._encryptor = DataEncryptor(key=master_key)
        
        # Key storage
        self._key_file = Path(key_file) if key_file else None
        self._keys: Dict[str, str] = {}  # Encrypted keys in memory
        
        # Load existing keys if file exists
        if self._key_file and self._key_file.exists():
            self._load_keys()
    
    def _load_keys(self) -> None:
        """Load encrypted keys from file."""
        try:
            with open(self._key_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self._encryptor.decrypt(encrypted_data)
            self._keys = json.loads(decrypted_data.decode('utf-8'))
            
            self._log.info(f"Loaded {len(self._keys)} encrypted keys")
        except Exception as e:
            self._log.error(f"Failed to load keys: {e}")
            self._keys = {}
    
    def _save_keys(self) -> None:
        """Save encrypted keys to file."""
        if not self._key_file:
            return
            
        try:
            # Ensure directory exists
            self._key_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Encrypt and save
            data = json.dumps(self._keys).encode('utf-8')
            encrypted_data = self._encryptor.encrypt(data)
            
            with open(self._key_file, 'wb') as f:
                f.write(encrypted_data)
                
            self._log.info(f"Saved {len(self._keys)} encrypted keys")
        except Exception as e:
            self._log.error(f"Failed to save keys: {e}")
    
    def set_key(self, name: str, value: str) -> None:
        """
        Store an encrypted key.
        
        Parameters
        ----------
        name : str
            The key name (e.g., 'BINANCE_API_KEY').
        value : str
            The key value to encrypt and store.
        
        """
        PyCondition.not_empty(name, "name")
        PyCondition.not_empty(value, "value")
        
        # Encrypt the value
        encrypted_value = self._encryptor.encrypt_string(value)
        self._keys[name] = encrypted_value
        
        # Save to file
        self._save_keys()
        
        self._log.info(f"Stored encrypted key: {name}")
    
    def get_key(self, name: str, default: str | None = None) -> str | None:
        """
        Retrieve a decrypted key.
        
        Parameters
        ----------
        name : str
            The key name to retrieve.
        default : str, optional
            Default value if key not found.
            
        Returns
        -------
        str or None
            The decrypted key value or default.
        
        """
        # Check encrypted storage first
        if name in self._keys:
            try:
                encrypted_value = self._keys[name]
                return self._encryptor.decrypt_string(encrypted_value)
            except Exception as e:
                self._log.error(f"Failed to decrypt key {name}: {e}")
        
        # Fall back to environment variable
        if self._use_env_fallback:
            env_value = os.environ.get(name)
            if env_value:
                self._log.debug(f"Using environment variable for {name}")
                return env_value
        
        return default
    
    def get_key_required(self, name: str) -> str:
        """
        Retrieve a required decrypted key.
        
        Parameters
        ----------
        name : str
            The key name to retrieve.
            
        Returns
        -------
        str
            The decrypted key value.
            
        Raises
        ------
        RuntimeError
            If the key is not found.
        
        """
        value = self.get_key(name)
        if value is None:
            raise RuntimeError(f"Required key '{name}' not found")
        return value
    
    def delete_key(self, name: str) -> bool:
        """
        Delete a stored key.
        
        Parameters
        ----------
        name : str
            The key name to delete.
            
        Returns
        -------
        bool
            True if key was deleted, False if not found.
        
        """
        if name in self._keys:
            del self._keys[name]
            self._save_keys()
            self._log.info(f"Deleted key: {name}")
            return True
        return False
    
    def list_keys(self) -> list[str]:
        """
        List all stored key names.
        
        Returns
        -------
        list[str]
            List of key names (values remain encrypted).
        
        """
        return list(self._keys.keys())
    
    def rotate_key(self, name: str, new_value: str) -> None:
        """
        Rotate (update) an existing key.
        
        Parameters
        ----------
        name : str
            The key name to rotate.
        new_value : str
            The new key value.
            
        """
        old_exists = name in self._keys
        self.set_key(name, new_value)
        
        if old_exists:
            self._log.info(f"Rotated key: {name}")
    
    def export_env_template(self, output_file: str) -> None:
        """
        Export a template .env file with key names.
        
        Parameters
        ----------
        output_file : str
            Path to write the template file.
        
        """
        template_lines = ["# Nautilus Trader Environment Variables\n"]
        template_lines.append(f"# Generated from encrypted key storage\n\n")
        
        for key_name in sorted(self._keys.keys()):
            template_lines.append(f"{key_name}=\n")
        
        with open(output_file, 'w') as f:
            f.writelines(template_lines)
            
        self._log.info(f"Exported environment template to {output_file}")