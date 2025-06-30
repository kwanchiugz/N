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
Base AI provider interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from nautilus_trader.ai.config import AIProviderConfig
from nautilus_trader.common.component import Logger


class AIResponse:
    """
    Container for AI provider responses.
    
    Parameters
    ----------
    content : str
        The response content.
    model : str
        The model used.
    usage : dict
        Token usage information.
    metadata : dict, optional
        Additional metadata.
    
    """
    
    def __init__(
        self,
        content: str,
        model: str,
        usage: Dict[str, int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.content = content
        self.model = model
        self.usage = usage
        self.metadata = metadata or {}
        
    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.usage.get("total_tokens", 0)


class AIProvider(ABC):
    """
    Abstract base class for AI providers.
    
    Parameters
    ----------
    config : AIProviderConfig
        The provider configuration.
    
    """
    
    def __init__(self, config: AIProviderConfig) -> None:
        self._config = config
        self._log = Logger(self.__class__.__name__)
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider."""
        ...
        
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> AIResponse:
        """
        Get completion from the AI provider.
        
        Parameters
        ----------
        prompt : str
            The user prompt.
        system_prompt : str, optional
            The system prompt.
        **kwargs
            Additional provider-specific parameters.
            
        Returns
        -------
        AIResponse
            The AI response.
        
        """
        ...
        
    @abstractmethod
    async def complete_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> AIResponse:
        """
        Get completion with tool/function calling.
        
        Parameters
        ----------
        prompt : str
            The user prompt.
        tools : list[dict]
            Tool definitions.
        system_prompt : str, optional
            The system prompt.
        **kwargs
            Additional provider-specific parameters.
            
        Returns
        -------
        AIResponse
            The AI response with tool calls.
        
        """
        ...
        
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the provider."""
        ...