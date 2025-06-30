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
DeepSeek AI provider implementation.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from nautilus_trader.ai.config import DeepSeekConfig
from nautilus_trader.ai.providers.base import AIProvider, AIResponse
from nautilus_trader.adapters.env_secure import get_env_key
from nautilus_trader.common.enums import LogColor


class DeepSeekProvider(AIProvider):
    """
    DeepSeek AI provider implementation.
    
    Parameters
    ----------
    config : DeepSeekConfig
        The DeepSeek configuration.
    
    """
    
    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__(config)
        self._config: DeepSeekConfig = config
        self._client: Optional[httpx.AsyncClient] = None
        self._api_key: Optional[str] = None
        
    async def initialize(self) -> None:
        """Initialize the DeepSeek provider."""
        # Get API key from secure storage
        if self._config.api_key:
            self._api_key = self._config.api_key
        else:
            self._api_key = get_env_key("DEEPSEEK_API_KEY")
            
        # Set base URL
        base_url = self._config.base_url or "https://api.deepseek.com"
        
        # Initialize HTTP client with security headers
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._config.timeout,
        )
        
        self._log.info("DeepSeek provider initialized", LogColor.GREEN)
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> AIResponse:
        """
        Get completion from DeepSeek.
        
        Parameters
        ----------
        prompt : str
            The user prompt.
        system_prompt : str, optional
            The system prompt.
        **kwargs
            Additional parameters (temperature, max_tokens, etc.).
            
        Returns
        -------
        AIResponse
            The AI response.
        
        """
        if not self._client:
            raise RuntimeError("Provider not initialized")
            
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Build request payload
        payload = {
            "model": kwargs.get("model", self._config.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self._config.temperature),
            "max_tokens": kwargs.get("max_tokens", self._config.max_tokens),
            "top_p": kwargs.get("top_p", self._config.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self._config.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self._config.presence_penalty),
            "stream": False,
        }
        
        try:
            # Make API request
            response = await self._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Extract content and usage
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            
            return AIResponse(
                content=content,
                model=data["model"],
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                metadata={
                    "finish_reason": data["choices"][0].get("finish_reason"),
                },
            )
            
        except httpx.HTTPStatusError as e:
            self._log.error(f"DeepSeek API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            self._log.error(f"DeepSeek request failed: {e}")
            raise
            
    async def complete_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> AIResponse:
        """
        Get completion with function calling from DeepSeek.
        
        Parameters
        ----------
        prompt : str
            The user prompt.
        tools : list[dict]
            Tool/function definitions.
        system_prompt : str, optional
            The system prompt.
        **kwargs
            Additional parameters.
            
        Returns
        -------
        AIResponse
            The AI response with tool calls.
        
        """
        if not self._client:
            raise RuntimeError("Provider not initialized")
            
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Build request payload with tools
        payload = {
            "model": kwargs.get("model", self._config.model),
            "messages": messages,
            "tools": tools,
            "tool_choice": kwargs.get("tool_choice", "auto"),
            "temperature": kwargs.get("temperature", self._config.temperature),
            "max_tokens": kwargs.get("max_tokens", self._config.max_tokens),
            "stream": False,
        }
        
        try:
            # Make API request
            response = await self._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Extract content, tool calls, and usage
            message = data["choices"][0]["message"]
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])
            usage = data.get("usage", {})
            
            return AIResponse(
                content=content,
                model=data["model"],
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                metadata={
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "tool_calls": tool_calls,
                },
            )
            
        except httpx.HTTPStatusError as e:
            self._log.error(f"DeepSeek API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            self._log.error(f"DeepSeek request failed: {e}")
            raise
            
    async def shutdown(self) -> None:
        """Shutdown the DeepSeek provider."""
        if self._client:
            await self._client.aclose()
            self._client = None
            
        self._log.info("DeepSeek provider shutdown")